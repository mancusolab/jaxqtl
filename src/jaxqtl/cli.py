import argparse as ap
import logging
import re
import sys

import pandas as pd

import jax
import jax.numpy as jnp

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM, LinearModel
from jaxqtl.infer.permutations import ACAT, BetaPermutation
from jaxqtl.infer.solve import CGSolve, CholeskySolve, QRSolve
from jaxqtl.infer.stderr import FisherInfoError, HuberError
from jaxqtl.infer.utils import ScoreTest, WaldTest
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader, VCFReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_logger
from jaxqtl.map.cis import map_cis, write_parqet
from jaxqtl.map.nominal import map_nominal


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
        "--qvalue",
        action="store_true",
        default=False,
        help="Include q-values for downstream FDR correction",
    )

    # filtering arguments
    common_p.add_argument(
        "--keep",
        help="Path to file of iids to analyze. All other iids are discarded during current analysis",
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


def _cis_scan(args, log):
    dat, family, glm, offset, test, perm_test = _common_setup(args, log)

    if dat.pheno_meta.gene_map.shape[0] < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    """
    glm: GLM,
    test: HypothesisTest,
    perm_test: AbstractPermutation,
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    random_tiebreak: bool = False,
    sig_level: float = 0.05,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: Optional[ArrayLike] = None,
    offset: ArrayLike = 0.0,
    compute_qvalue: bool = False,
    verbose: bool = True,
    log=None,
    """
    outdf_cis_score = map_cis(
        dat,
        glm=glm,
        test=test,
        perm_test=perm_test,
        standardize=args.standardize,
        window=args.window,
        offset=offset,
        compute_qvalue=args.qvalue,
        log=log,
        seed=args.seed,
    )
    test_str = "score" if isinstance(test, ScoreTest) else "wald"
    outdf_cis_score.to_csv(args.out + f".cis.{test_str}.tsv.gz", sep="\t", index=False)

    return 0


def _nominal_scan(args, log):
    dat, family, glm, offset, test, perm_test = _common_setup(args, log)
    if dat.pheno_meta.gene_map.shape[0] < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    out_df = map_nominal(
        dat,
        test=test,
        standardize=args.standardize,
        log=log,
        window=args.window,
        offset_eta=offset,
        cond_snp=args.cond_snp,
    )
    test_str = "score" if isinstance(test, ScoreTest) else "wald"
    write_parqet(outdf=out_df, method=test_str, out_path=args.out)

    return 0


def _trans_scan(args, log):
    dat, family, glm, offset, test, perm_test = _common_setup(args, log)
    out_df = map_nominal(
        dat,
        family=family,
        offset_eta=offset,
        test=test,
        mode="trans",
        standardize=args.standardize,
        robust_se=args.robust,
        log=log,
        max_iter=args.max_iter,
        cond_snp=args.cond_snp,
    )
    out_df.to_csv(args.out + ".trans_score.tsv.gz", sep="\t", index=False)
    return 0


def _common_setup(args, log):
    if args.model == "poisson":
        family = Poisson()
    elif args.model == "nb":
        family = NegativeBinomial()
    elif args.model == "gaussian":
        family = Gaussian()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.robust_se:
        se_estimator = HuberError()
    else:
        se_estimator = FisherInfoError()

    if args.solver == "cholesky":
        solver = CholeskySolve()
    elif args.solver == "cg":
        solver = CGSolve()
    elif args.solver == "qr":
        solver = QRSolve()
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

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

    if args.test == "score":
        test = ScoreTest(model=glm, std_err=se_estimator)
    elif args.test == "wald":
        test = WaldTest(model=glm, std_err=se_estimator)
    else:
        raise ValueError("Unknown test method: {args.test_method}")

    if args.acat:
        perm_test = ACAT()
    else:
        # for lm wald test, use t distribution during permutation
        use_tdist = isinstance(family, Gaussian)
        perm_test = BetaPermutation(max_perm_direct=args.nperm, use_tdist=use_tdist)

    # raw genotype data and impute for genotype data
    if args.bfile is not None:
        geno_reader = PlinkReader()
        prefix = args.bfile
    elif args.vcf is not None:
        geno_reader = VCFReader()
        prefix = args.vcf
    elif args.geno is not None:
        geno_reader = PlinkReader()
        prefix = args.geno
        log.warn("`--geno PREFIX` is deprecated and will be removed in a future version. Use `--bfile PREFIX` instead")
    else:
        # we really shouldn't get here with mutex above
        raise ValueError("No valid genotype file specified.")

    if args.keep is not None:
        log.info("Reading list of samples to keep for analyses.")
        inds_to_keep = pd.read_csv(args.keep, header=None, sep="\t").iloc[:, 0].to_list()
        log.info(f"Found {len(inds_to_keep)} samples to keep")
    else:
        inds_to_keep = None

    log.info("Reading genotype, phenotype, and covariate data")
    # todo: we should pass in the list of samples here to restrict before reading in all geno data
    geno, bim, sample_info = geno_reader(prefix)

    if args.gene_list is not None:
        gene_list = pd.read_csv(args.gene_list, header=None, sep="\t").iloc[:, 0].to_list()
    else:
        gene_list = None

    # todo: same as above, but for gene_list
    pheno_reader = PheBedReader()
    pheno = pheno_reader(args.pheno)
    covar = covar_reader(args.covar, args.covar_name, args.rm_covar)

    dat = create_readydata(geno, bim, sample_info, pheno, covar, autosomal_only=args.autosome, ind_list=inds_to_keep)
    log.info("Finished reading and aligning genotype, phenotype, covariate data.")

    # before filter gene list, calculate library size and set offset, or read in pre-computed log(offset)
    if args.set_offset_from_libsize:
        total_libsize = jnp.array(dat.pheno.count.sum(axis=1))
        offset = jnp.log(total_libsize)
    elif args.offset:
        # todo: use args.offset_name if passed in; otherwise take first column after iid
        offset = pd.read_csv(args.offset, names=["iid", "eta"], sep="\t", index_col="iid")
        offset = offset.loc[offset.index.isin(dat.pheno.count.index)].sort_index()
        offset = jnp.array(offset)
    else:
        offset = 0.

    # filter gene list
    dat.filter_gene(gene_list=gene_list, geneexpr_percent_cutoff=args.express_percent)

    return dat, family, glm, offset, test, perm_test


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

    nominal_p = _create_common_subp(subp, "trans", help="Perform a trans-eQTL scan.")
    nominal_p.set_defaults(func=_nominal_scan)

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
