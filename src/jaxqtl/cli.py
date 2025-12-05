import argparse as ap
import logging
import re
import sys

import pandas as pd
import polars as pl

import jax
import jax.numpy as jnp

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM, LinearModel
from jaxqtl.infer.permutations import ACAT, BetaPermutation
from jaxqtl.infer.solve import CGSolve, CholeskySolve, QRSolve
from jaxqtl.infer.spa import GaussianCGF, NegativeBinomialCGF, PoissonCGF
from jaxqtl.infer.stderr import FisherInfoError, HuberError
from jaxqtl.infer.utils import ScoreTest, SpaTest, WaldTest
from jaxqtl.io.data import align_pheno_covar, create_readydata
from jaxqtl.io.geno import PlinkData, VCFData
from jaxqtl.io.pheno import edger_cpm, ExpressionData, inverse_normal_transform
from jaxqtl.io.utils import read_offset_tsvlike, read_plink_style_tsvlike
from jaxqtl.log import get_logger
from jaxqtl.map.cis import map_cis


class _SplitAction(ap.Action):
    """Parse comma or space delimited command args into a list.
    Useful for pheno/pheno-col-num covar/covar-col-num.
    """

    def __init__(self, *, type=str, **kwargs):
        super().__init__(**kwargs)
        self.cast = type

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            raw = " ".join(values)
        else:
            raw = values
        # split on commas or whitespace
        items = re.split(r"[\s,]+", raw.strip())
        # drop empties
        final = []
        for it in items:
            if not it:
                continue
            try:
                final.append(self.cast(it))
            except ValueError:
                raise ap.ArgumentError(self, f"invalid {self.cast.__name__!r} value: {it!r}")

        setattr(namespace, self.dest, final)


def _create_common_subp(subp, name, help):
    common_p = subp.add_parser(name, help=help)

    # geno arguments
    geno_group = common_p.add_mutually_exclusive_group(required=True)
    geno_group.add_argument("--geno", help="Prefix to PLINK triplet")
    geno_group.add_argument("--bfile", help="Prefix to PLINK triplet")
    geno_group.add_argument("--vcf", help="Path to VCF data")

    # pheno / covariate arguments
    common_p.add_argument("--pheno", help="Path to phenotypes", required=True)
    common_p.add_argument("--covar", help="Path to covariate data")
    covar_group = common_p.add_mutually_exclusive_group()
    covar_group.add_argument(
        "--covar-name",
        nargs="+",
        action=_SplitAction,
        help="Covariate name(s) to include (comma/space delimited). All other covariates are discarded during analysis",
    )
    covar_group.add_argument(
        "--rm-covar",
        nargs="+",
        action=_SplitAction,
        help="Covariate name(s) to exclude (comma/space delimited). All other covariates are included during analysis",
    )
    common_p.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="Standardize covariates",
    )
    common_p.add_argument(
        "--one-hot",
        action="store_true",
        default=False,
        help="Encode string/categorical covariates using one-hot encoding",
    )

    # offset options. can only select one; otherwse we don't have an offset
    offset_group = common_p.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--offset",
        help="Path to log offset in tsv format (no header) with two columns: iid and log(library size)",
    )
    offset_group.add_argument(
        "--offset-name",
        help="Covariate name to use as fixed offset",
    )
    offset_group.add_argument(
        "--set-offset-from-libsize",
        action="store_true",
        default=False,
        help="Compute log(library size) on the fly and use as fixed as fixed offset",
    )

    # statistical test arguments
    common_p.add_argument("--model", choices=["gaussian", "poisson", "nb"], default="nb", help="eQTL model")
    common_p.add_argument(
        "--test",
        choices=["wald", "score"],
        default="score",
        help="Test to perform during scan. We recommend 'score' for cis mapping and 'wald' for nominal mapping",
    )
    common_p.add_argument(
        "--robust-se",
        action="store_true",
        default=False,
        help="Compute Robust/Huber standard errors for GLM rather than Fisher Information",
    )
    common_p.add_argument(
        "--spa",
        action="store_true",
        default=False,
        help="Whether to perform SPA correction for p-values computed from score statistics. Not applicable for Wald.",
    )

    # filtering arguments
    sample_group = common_p.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--keep",
        help="Path to file of iids to analyze. All other iids are discarded during current analysis.",
    )
    sample_group.add_argument(
        "--exclude",
        help="Path to file of iids to exclude from analysis. All other iids are kept during current analysis.",
    )

    common_p.add_argument(
        "--prop-cutoff", type=float, help="keep individual with gene expression below this proportion threshold"
    )
    common_p.add_argument(
        "--express-percent",
        type=float,
        default=0.0,
        help="Keep genes with expression levels above specified value",
    )
    gene_group = common_p.add_mutually_exclusive_group()
    gene_group.add_argument(
        "--gene-list",
        help="Path to gene list (no header). All other genes will be discarded during analysis",
    )
    gene_group.add_argument(
        "--genes",
        nargs="+",
        action=_SplitAction,
        help="Gene name(s) to analyze (comma/space delimited). All other genes will be discarded during analysis",
    )
    common_p.add_argument("--condition", help="Include specified variant as a covariate during analysis")

    chrom_group = common_p.add_mutually_exclusive_group()
    chrom_group.add_argument(
        "--chr",
        help="Excludes all variants (and pheno) not on specified chromosome",
    )
    chrom_group.add_argument(
        "--autosome",
        action="store_true",
        default=False,
        help="Excludes all unplaced and non-autosomal variants",
    )
    common_p.add_argument("--window", type=int, default=500000, help="One sided window size (bps) with respect to TSS")

    # inference/runtime arguments
    common_p.add_argument(
        "--acat",
        default=False,
        action="store_true",
        help="Whether to perform ACAT for gene-level p-values rather than Beta approximation to permutation testing",
    )
    common_p.add_argument(
        "--nperm",
        type=int,
        default=1000,
        help="Number of permutations to perform to bootstrap Beta approximation to permutation testing",
    )
    common_p.add_argument("--max-iter", type=int, default=1000, help="Maximum number of iterations for GLM inference")
    common_p.add_argument("--tol", type=float, default=1e-3, help="Tolerance for termination during GLM inference")
    common_p.add_argument("--step-size", type=float, default=1.0, help="Initial step-size during GLM inference")

    common_p.add_argument("--seed", type=int, default=0, help="Seed for PRNG initialization.")
    common_p.add_argument(
        "--perm-pheno",
        action="store_true",
        default=False,
        help="Permute phenotype for type I error calibration",
    )
    common_p.add_argument(
        "--solver",
        choices=["cholesky", "cg", "qr"],
        default="cholesky",
        help="The linear solver to use during model fitting",
    )

    common_p.add_argument(
        "-p",
        "--platform",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="Machine platform: cpu, gpu or tpu",
    )
    common_p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    common_p.add_argument("--out", "-o", type=str, help="out file prefix")
    return common_p


def _compute_expression_pcs(args, log):
    pheno = ExpressionData.from_bedfile(args.pheno)

    # todo: add more sanity checking
    if args.num_pcs < 1:
        raise ValueError("Number of PCS must be at least 1")

    if args.covar:
        covar = read_plink_style_tsvlike(args.covar)
    else:
        covar = None

    if args.offset:
        offset = read_offset_tsvlike(args.offset)
    else:
        offset = None

    pheno, covar, offset = align_pheno_covar(pheno, covar, args.offset, args.set_offset_from_libsize)
    if args.transform == "tmm":
        tmm_counts_df = edger_cpm(pheno, normalized_lib_sizes=True)
        pheno = inverse_normal_transform(tmm_counts_df)
    elif args.transform == "log1p":
        pheno = jnp.log1p(pheno)  # prevent log(0)
    elif args.transform == "offset":
        raise NotImplementedError("'offset' transform not implemented yet.")
    else:
        raise ValueError("Invalid transform {args.transform}. Only 'log1p', 'tmm', and 'offset' are accepted.")

    from jax.experimental.sparse.linalg import lobpcg_standard

    n = pheno.shape[0]
    k = args.num_pcs
    pheno = (pheno - pheno.mean(axis=0)) / pheno.std(axis=0)  # standardize genes
    theta, U, i = lobpcg_standard(pheno, jnp.eye(n, k))
    df_u = pd.DataFrame(data=U, index=pheno.index, columns=[f"EPC{i}" for i in range(k)])

    if covar:
        covar = pd.concat((covar, df_u))
    covar.to_csv(args.out)

    return 0


def _cis_scan(args, log):
    dat, family, glm, test, perm_test = _common_setup(args, log)

    if dat.num_genes < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    df_cis = map_cis(
        dat,
        test=test,
        perm_test=perm_test,
        mode="cis",
        window=args.window,
        log=log,
        seed=args.seed,
    )
    log.info("Finished cis-scan. Writing results.")
    test_str = test.name
    adj_name = perm_test.name
    df_cis.write_csv(f"{args.out}.cis.{test_str}.{adj_name}.tsv", separator="\t")
    log.info("Finished! Thank you!")

    return 0


def _nominal_scan(args, log):
    dat, family, glm, test, perm_test = _common_setup(args, log)
    if dat.num_genes < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    df_nominal = map_cis(
        dat,
        test=test,
        perm_test=perm_test,
        mode="nominal",
        window=args.window,
        log=log,
        seed=args.seed,
    )

    log.info("Finished nominal cis-scan. Writing results.")
    test_str = test.name
    adj_name = perm_test.name
    # ztd compression?
    df_nominal.write_parquet(f"{args.out}.nominal.{test_str}.{adj_name}.parquet.gz", compression="gzip")
    log.info("Finished! Thank you!")

    return 0


def _trans_scan(args, log):
    # TBD
    # dat, family, glm, test, perm_test = _common_setup(args, log)
    # out_df.to_csv(args.out + ".trans_score.tsv.gz", sep="\t", index=False)
    return 0


def _common_setup(args, log):
    # Set up the distributional family and corresponding cumulative generating function (CGF) here.
    # We only use CGF if --spa is set, but may as well set up thin objects here so we don't need to re-enumerate
    # later.
    if args.model == "poisson":
        family = Poisson()
        cgf = PoissonCGF()
    elif args.model == "nb":
        family = NegativeBinomial()
        cgf = NegativeBinomialCGF()
    elif args.model == "gaussian":
        family = Gaussian()
        cgf = GaussianCGF()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Whether to use Huber-style sandwich estimator or FisherInfo (ie Asymptotic) SEs
    # This really only matters when doing Wald test
    if args.robust_se:
        se_estimator = HuberError()
    else:
        se_estimator = FisherInfoError()

    # Power-users may want to explore diff solvers
    if args.solver == "cholesky":
        solver = CholeskySolve()
    elif args.solver == "cg":
        solver = CGSolve()
    elif args.solver == "qr":
        solver = QRSolve()
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    # GLM under Gaussian assumptions is a single step under the IRLS, but that adds a bunch of overhead.
    # So we use this simpler interface instead for Gaussian case
    if isinstance(family, Gaussian):
        glm = LinearModel(
            family=family,
            solver=solver,
        )
    else:
        glm = GLM(
            family=family,
            solver=solver,
            max_iter=args.max_iter,
            tol=args.tol,
            step_size=args.step_size,
        )

    # Set up our hypothesis testing framework. Score, SPA (which is Score + SPA), or Wald test.
    if args.test == "score":
        if args.spa:
            # cgf set up top
            test = SpaTest(model=glm, std_err=se_estimator, cgf=cgf)
        else:
            test = ScoreTest(model=glm, std_err=se_estimator)
    elif args.test == "wald":
        if args.spa:
            log.warning("`--spa` is only compatible with `--test score`. Found `--test wald`")
        test = WaldTest(model=glm, std_err=se_estimator)
    else:
        raise ValueError("Unknown test method: {args.test_method}")

    # Set up our within-gene multiple testing correction framework here: ACAT (fast) or Beta-Permutations.
    if args.acat:
        perm_test = ACAT()
    else:
        # for lm wald test, use t distribution during permutation
        use_tdist = isinstance(family, Gaussian)
        perm_test = BetaPermutation(max_perm_direct=args.nperm, use_tdist=use_tdist)

    if args.keep is not None:
        log.info("Reading list of samples to keep for analyses.")
        inds_to_keep = pd.read_csv(args.keep, header=None, sep="\t").iloc[:, 0].to_list()
        log.info(f"Found {len(inds_to_keep)} samples to keep.")
    else:
        inds_to_keep = None

    if args.exclude is not None:
        log.info("Reading list of samples to exclude from analyses.")
        inds_to_exclude = pd.read_csv(args.exclude, header=None, sep="\t").iloc[:, 0].to_list()
        log.info(f"Found {len(inds_to_keep)} samples to exclude.")
    else:
        inds_to_exclude = None

    log.info("Reading genotype, phenotype, and covariate data")
    if args.bfile is not None:
        geno_data = PlinkData.load(args.bfile)
    elif args.vcf is not None:
        geno_data = VCFData.load(args.vcf)
    elif args.geno is not None:
        geno_data = PlinkData.load(args.geno)
        log.warn("`--geno PREFIX` is deprecated and will be removed in a future version. Use `--bfile PREFIX` instead")
    else:
        # we really shouldn't get here with mutex above
        raise ValueError("No valid genotype file specified.")

    # so we end up aligning everything at the end of this function, but better to reduce as we go
    # this should help speed up final data alignment a touch
    if inds_to_keep:
        geno_data = geno_data.filter_individuals(inds_to_keep, "keep")
    elif inds_to_exclude:
        geno_data = geno_data.filter_individuals(inds_to_exclude, "drop")

    if args.gene_list is not None:
        gene_keep_list = pd.read_csv(args.gene_list, header=None, sep="\t").iloc[:, 0].to_list()
    elif args.genes is not None:
        gene_keep_list = args.genes
    else:
        gene_keep_list = None

    gene_exclude_list = None
    expr_data = ExpressionData.from_bedfile(
        args.pheno, inds_to_keep, inds_to_exclude, gene_keep_list, gene_exclude_list
    )
    expr_data = expr_data.filter_by_percentage(args.express_percent)

    if args.covar is not None:
        covar = read_plink_style_tsvlike(args.covar, args.covar_name, args.rm_covar)

        # perform one-hot encoding for string-based columns, if specified
        if args.one_hot:
            cat = pl.selectors.string().exclude("iid")
            covar = covar.to_dummies(cat, drop_first=True).drop(cat)

        # standardize all numeric columns
        if args.standardize:
            num = pl.all().exclude("iid")

            # let's make sure to not standardize the offset if it was provided
            if args.offset_name:
                num = num.exclude(args.offset_name)

            covar = covar.with_columns((num - num.mean()) / num.std())
    else:
        covar = None

    # before filter gene list, calculate library size and set offset, or read in pre-computed log(offset)
    if args.offset:
        offset = read_offset_tsvlike(args.offset)
    elif args.offset_name:
        if covar is None:
            raise ValueError("Covariate file must be provided if `--offset-name` is specified.")
        offset = covar.select(pl.col("iid"), pl.col(args.offset_name))
        # drop the offset from the covariates data
        covar = covar.drop(args.offset_name)
    elif args.set_offset_from_libsize:
        offset = expr_data.offset_from_libsize
    else:
        offset = None

    # take the genotype, expression, covariates, and offset and align by iid for valid analyses
    # lump those into single object for easier passing around
    data = create_readydata(
        geno_data,
        expr_data,
        covar,
        offset,
    )
    log.info("Finished reading and aligning genotype, phenotype, covariate data.")

    return data, family, glm, test, perm_test


def main(args):
    argp = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    # build association scan parser from 'common' parser
    cis_p = _create_common_subp(
        subp,
        "cis",
        help="Perform cis-eQTL scans and report the lead hit per tested gene",
    )
    cis_p.set_defaults(func=_cis_scan)

    nominal_p = _create_common_subp(
        subp,
        "nominal",
        help="Perform cis-eQTL scans and report all association stats per tested gene.",
    )
    nominal_p.set_defaults(func=_cis_scan)

    trans_p = _create_common_subp(subp, "trans", help="Perform a trans-eQTL scan.")
    trans_p.set_defaults(func=_nominal_scan)

    gepcs_p = subp.add_parser("compute-pcs", help="Compute gene expression principal components")
    gepcs_p.add_argument("--pheno", help="Path to phenotypes", required=True)
    gepcs_p.add_argument(
        "--num-pcs",
        type=int,
        required=True,
        help="Number of principal components to compute",
    )
    gepcs_p.add_argument("--covar", help="Path to covariate data. If included GE PCs will be appended.")
    gepcs_p.add_argument(
        "--transform",
        choices=["tmm", "log1p", "none"],
        default="none",
        help="Transformation to perform on observed gene expression before computing PCs.",
    )
    gepcs_p.add_argument(
        "-p",
        "--platform",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="Machine platform: cpu, gpu or tpu",
    )
    gepcs_p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    gepcs_p.add_argument("--out", "-o", type=str, help="out file prefix")
    gepcs_p.set_defaults(func=_compute_expression_pcs)

    args = argp.parse_args(args)

    jax.config.update("jax_platform_name", args.platform)
    if args.platform == "tpu":
        # TPU not support complex 64, only 16 and 32
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)

    log = get_logger(__name__, args.out)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # strange bug with TPU error cropping up on local machines...
    jax_logger = logging.getLogger("jax._src.xla_bridge")
    jax_logger.propagate = False

    # launch w/e task was selected
    if hasattr(args, "func"):
        args.func(args, log)
    else:
        argp.print_help()

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
