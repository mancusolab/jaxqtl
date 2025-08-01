import argparse as ap
import logging
import re
import sys

import numpy as np
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
        "--platform",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="platform: cpu, gpu or tpu",
    )
    common_p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    common_p.add_argument("--out", "-o", type=str, help="out file prefix")
    return common_p


def main(args):
    argp = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )

    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    # build association scan parser from 'common' parser
    cis_p = _create_common_subp(subp, "cis", help="Perform cis-eQTL scans and report the lead hit per tested gene")
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

    # launch w/e task was selected
    if hasattr(args, "func"):
        args.func(args)
    else:
        argp.print_help()

    return 0


def _cis_scan(args, log):
    dat, family, glm, offset, test, perm_test = _common_setup(args, log)

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
    )
    test_str = "score" if isinstance(test, ScoreTest) else "wald"
    outdf_cis_score.to_csv(args.out + f".cis.{test_str}.tsv.gz", sep="\t", index=False)

    return 0


def _nominal_scan(args, log):
    dat, family, glm, offset, test = _common_setup(args, log)

    out_df = map_nominal(
        dat,
        test=test,
        family=family,
        standardize=args.standardize,
        log=log,
        window=args.window,
        offset_eta=offset,
        robust_se=args.robust,
        max_iter=args.max_iter,
        cond_snp=args.cond_snp,
    )
    write_parqet(outdf=out_df, method="wald", out_path=args.out)

    return 0


def _trans_scan(args, log):
    dat, family, glm, offset, test = _common_setup(args, log)
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

    if args.robust:
        se_estimator = HuberError()
    else:
        se_estimator = FisherInfoError()

    if args.solver == "cholesky":
        solver = CholeskySolve()
    elif args.solver == "cg":
        solver = CGSolve()
    elif args.solver == "qr":
        solver = QRSolve()

    if not isinstance(family, Gaussian):
        glm = GLM(
            family=family,
            solver=solver,
            std_err=se_estimator,
            max_iter=args.max_iter,
            tol=args.tol,
            step_size=args.step_size,
        )
    else:
        glm = LinearModel(
            family=family,
            solver=solver,
            std_err=se_estimator,
        )
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

    geno, bim, sample_info = geno_reader(prefix)
    pheno_reader = PheBedReader()
    pheno = pheno_reader(args.pheno)
    covar = covar_reader(args.covar, args.add_covar, args.covar_test, args.rm_covar)
    if args.genelist is not None:
        genelist = pd.read_csv(args.genelist, header=None, sep="\t").iloc[:, 0].to_list()
    else:
        genelist = None
    if args.keep is not None:
        indList = pd.read_csv(args.indlist, header=None, sep="\t").iloc[:, 0].to_list()
    else:
        indList = None
    dat = create_readydata(geno, bim, pheno, covar, autosomal_only=args.autosomal_only, ind_list=indList)

    # before filter gene list, calculate library size and set offset, or read in pre-computed log(offset)
    if args.offset is None:
        total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
        offset = jnp.log(total_libsize)
    else:
        offset = pd.read_csv(args.offset, names=["iid", "eta"], sep="\t", index_col="iid")
        offset = offset.loc[offset.index.isin(dat.pheno.count.index)].sort_index()
        offset = jnp.array(offset)

    if isinstance(family, Gaussian) or args.no_offset is True:
        # dat.transform_y(mode='log1p')  # log1p
        # note: use pre-processed file as in tensorqtl
        offset = jnp.zeros_like(offset)

    # filter gene list
    dat.filter_gene(gene_list=genelist, geneexpr_percent_cutoff=args.express_percent)

    if args.acat:
        perm_test = ACAT()
    else:
        perm_test = BetaPermutation(args.nperm)

    # permute gene expression for type I error calibration
    if args.perm_pheno:
        np.random.seed(args.perm_seed)
        perm_idx = np.random.permutation(np.arange(0, len(dat.pheno.count)))
        dat.pheno.count = dat.pheno.count.iloc[perm_idx]
        offset = offset[perm_idx]
    if dat.pheno_meta.gene_map.shape[0] < 1:
        log.info("No gene exist.")
        sys.exit()
    # for lm wald test, use t distribution during permutation
    if args.test_method == "score":
        test = ScoreTest(model=glm)
    elif args.test_method == "wald":
        test = WaldTest(model=glm, max_iter=args.max_iter, tol=args.tol, step_size=args.step_size)

    return dat, family, glm, offset, test, perm_test


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
