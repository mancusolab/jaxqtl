# pattern: Imperative Shell

import argparse as ap
import logging
import re
import sys

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import jax

from .distribution import Gaussian, NegativeBinomial, Poisson
from .hypothesis import (
    ACAT,
    BetaPermutation,
    GaussianCGF,
    NegativeBinomialCGF,
    PoissonCGF,
    ScoreTest,
    SpaTest,
    WaldTest,
)
from .infer import (
    CGSolve,
    CholeskySolve,
    FisherInfoError,
    GeneralizedLinearModel,
    HuberError,
    LinearModel,
    QRSolve,
)
from .io import (
    ExpressionData,
    load_genotype_dataset,
    read_offset_tsvlike,
    read_plink_style_tsvlike,
    read_single_column_file,
)
from .log import get_logger
from .map import get_trans_schemas, map_cis, map_trans
from .map.data import ReadyDataState


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
    geno_group.add_argument("--geno", help="deprecated; use genoio-native inputs: --bfile, --pfile, --vcf, or --bgen.")
    geno_group.add_argument("--bfile", help="Prefix to PLINK1 BED/BIM/FAM triplets.")
    geno_group.add_argument("--pfile", help="Prefix to PLINK2 PGEN/PVAR/PSAM triplets.")
    geno_group.add_argument("--vcf", help="Path to an indexed VCF/BCF genotype file.")
    geno_group.add_argument("--bgen", help="Path to a BGEN genotype file.")

    # pheno / covariate arguments
    common_p.add_argument("--pheno", help="Path to phenotypes", required=True)
    common_p.add_argument("--covar", help="Path to covariate data", required=True)
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
        "--normalize-covar",
        action="store_true",
        default=False,
        help="Normalize covariates to have zero mean and unit variance.",
    )
    common_p.add_argument(
        "--one-hot",
        action="store_true",
        default=False,
        help=(
            "Encode string/categorical covariates using one-hot encoding."
            " The category corresponding to the first observation will be dropped for co-linearity reasons."
        ),
    )
    common_p.add_argument(
        "--no-intercept",
        action="store_true",
        default=False,
        help=(
            "By default jaxQTL appends an intercept to the covariates to handle a shared mean term in the response."
            " Set `--no-intercept` to disable this behavior."
        ),
    )

    # offset options. can only select one; otherwse we don't have an offset
    offset_group = common_p.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--offset",
        help=(
            "Path to offset file in tsv format."
            "Expects exactly two columns (with header). The first should be iid-like and second the offset name"
        ),
    )
    offset_group.add_argument(
        "--offset-name-from-covar",
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
        help=(
            "Whether to perform SPA correction for p-values computed from score statistics."
            " Not applicable for `--test wald` and not necessary for `--model gaussian`."
        ),
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
        "--min-indiv-expr-pct",
        type=float,
        default=None,
        help=(
            "Exclude individuals that have fewer than specified percentage of genes with "
            "non-zero expression (e.g., '0.1')"
        ),
    )
    common_p.add_argument(
        "--min-gene-expr-pct",
        type=float,
        default=0.0,
        help="Exclude genes expressed in fewer than specified percentage of individuals (e.g., '0.1')",
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
    """
    gene_group.add_argument(
        "--rm-genes",
        nargs="+",
        action=_SplitAction,
        help="Gene name(s) to exclude (comma/space delimited). All other genes will be included during analysis",
    )
    """
    # common_p.add_argument("--condition", help="Include specified variant as a covariate during analysis")

    """
    # functionality not supported yet
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
    """
    common_p.add_argument("--window", type=int, default=500_000, help="One sided window size (bps) with respect to TSS")

    # inference/runtime arguments
    common_p.add_argument(
        "--acat",
        default=False,
        action="store_true",
        help="Perform ACAT for gene-level p-values rather than Beta approximation to permutation testing",
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

    common_p.add_argument("--seed", type=int, default=0, help="Seed for PRNG initialization")
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
    common_p.add_argument("--out", "-o", type=str, default="jaxqtl", help="out file prefix")
    return common_p


def _compute_expression_pcs(args, log):
    log.info("Reading phenotype and filtering")
    expr_data = ExpressionData.from_bedfile(args.pheno)
    expr_data = expr_data.filter_genes_by_percentage(args.min_gene_expr_pct)

    # todo: this needs a ton of work; we should allow for include/exclusion of genes/phenotypes and samples/individuals
    # wondering if we should support this functionality at all, as it could induce a good bit of downstream maintenance
    if args.num_pcs < 1:
        raise ValueError("Number of PCS must be at least 1")

    """
    if args.offset:
        offset = read_offset_tsvlike(args.offset)
    else:
        offset = None
    """
    import jax.random as rdm

    key = rdm.key(args.seed)
    log.info(f"Computing {args.num_pcs} gene expression principal components")
    df_pcs = expr_data.compute_pcs(args.num_pcs, key, args.transform)
    log.info(f"Finished computing {args.num_pcs} gene expression principal components")

    if args.covar:
        log.info("Reading covariate data and appending principal components")
        covar = read_plink_style_tsvlike(args.covar)
        df_pcs = covar.join(df_pcs, on="iid", how="left")

    log.info("Writing results.")
    df_pcs.write_csv(args.out, separator="\t")

    return 0


def _cis_scan(args, log):
    dat, family, glm, test, perm_test = _common_setup(args, log)

    if dat.num_genes < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    log.info("Starting cis-scan.")
    test_str = test.name
    adj_name = perm_test.name
    cis_out = f"{args.out}.cis.{test_str}.{adj_name}.parquet.gz"
    wrote_results = _write_scan_results(
        map_cis(
            dat,
            snp_test=test,
            gene_test=perm_test,
            mode="cis",
            window=args.window,
            verbose=args.verbose,
            log=log,
            seed=args.seed,
        ),
        cis_out,
    )
    if wrote_results:
        log.info(f"Finished cis-scan. Wrote results to {cis_out}.")
    else:
        log.warning("Finished cis-scan. No results to output!")

    return 0


def _write_scan_results(chunks, output_path):
    writer = None
    schema = None
    wrote_results = False
    try:
        for df in chunks:
            table = df.to_arrow()
            if writer is None:
                # The first result fixes the Parquet schema; columns differ by scan mode and model family.
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema, compression="gzip")
            # Cast later chunks to the first schema so row groups stay consistent.
            writer.write_table(table.cast(schema))
            wrote_results = True
    finally:
        if writer is not None:
            writer.close()

    return wrote_results


def _nominal_scan(args, log):
    dat, family, glm, test, perm_test = _common_setup(args, log)
    if dat.num_genes < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    log.info("Starting nominal cis-scan.")
    test_str = test.name
    nominal_out = f"{args.out}.nominal.{test_str}.parquet.gz"
    wrote_results = _write_scan_results(
        map_cis(
            dat,
            snp_test=test,
            gene_test=perm_test,
            mode="nominal",
            window=args.window,
            verbose=args.verbose,
            log=log,
            seed=args.seed,
        ),
        nominal_out,
    )
    if wrote_results:
        log.info(f"Finished nominal cis-scan. Wrote results to {nominal_out}.")
    else:
        log.warning("Finished nominal cis-scan. No results to output!")

    return 0


def _trans_scan(args, log):
    data, family, glm, test, perm_test = _common_setup(args, log)
    if data.num_genes < 1:
        log.info("No gene exists after filtering. Exiting.")
        return 0

    # convert types from python to pyarrow types
    type_map: dict[type, pa.DataType] = {int: pa.int64(), float: pa.float64(), str: pa.string(), bool: pa.bool_()}
    var_schema, stats_schema = get_trans_schemas()
    var_schema_pa = pa.schema([(col, type_map[col_type]) for col, col_type in var_schema.items()])
    stats_schema_pa = pa.schema([(col, type_map[col_type]) for col, col_type in stats_schema.items()])

    test_str = test.name
    var_out = f"{args.out}.trans.{test_str}.variant.info.parquet.gz"
    stats_out = f"{args.out}.trans.{test_str}.sumstats.parquet.gz"
    with (
        pq.ParquetWriter(var_out, var_schema_pa) as var_writer,
        pq.ParquetWriter(stats_out, stats_schema_pa) as stats_writer,
    ):
        for tables in map_trans(data, test, chunk_size=2500, verbose=args.verbose, log=log, seed=args.seed):
            if tables is None:
                break

            var_df, stats_df = tables
            var_writer.write(var_df.to_arrow().cast(var_schema_pa))
            stats_writer.write(stats_df.to_arrow().cast(stats_schema_pa))

    return 0


def _load_genotype_data(args, log):
    if args.bfile is not None:
        return load_genotype_dataset("bfile", args.bfile)
    if args.pfile is not None:
        return load_genotype_dataset("pfile", args.pfile)
    if args.vcf is not None:
        return load_genotype_dataset("vcf", args.vcf)
    if args.bgen is not None:
        return load_genotype_dataset("bgen", args.bgen)
    if args.geno is not None:
        raise ValueError("`--geno` is deprecated. Use genoio-native inputs: --bfile, --pfile, --vcf, or --bgen.")

    raise ValueError("No valid genotype file specified.")


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
        reg_model = LinearModel(
            family=family,
            solver=solver,
        )
    else:
        reg_model = GeneralizedLinearModel(
            family=family,
            solver=solver,
            max_iter=args.max_iter,
            tol=args.tol,
            step_size=args.step_size,
        )

    # Set up our hypothesis testing framework. Score, SPA (which is Score + SPA), or Wald test.
    if args.test == "score":
        if args.spa:
            if not isinstance(family, Gaussian):
                # cgf set up top
                test = SpaTest(model=reg_model, std_err=se_estimator, cgf=cgf)
            else:
                msg = (
                    "Found `--spa` together with `--model gaussian`."
                    " SPA adjustment is unnecessary as normality assumptions are met. Skipping `--spa` adjustment"
                )
                log.warning(msg)
                test = ScoreTest(model=reg_model, std_err=se_estimator)
        else:
            test = ScoreTest(model=reg_model, std_err=se_estimator)
    elif args.test == "wald":
        if args.spa:
            log.warning("`--spa` is only compatible with `--test score`. Found `--test wald`")
        test = WaldTest(model=reg_model, std_err=se_estimator)
    else:
        raise ValueError("Unknown test method: {args.test_method}")

    # Set up our within-gene multiple testing correction framework here: ACAT (fast) or Beta-Permutations.
    if args.acat:
        # we only do multiple testing adjustment when in cis mode.
        if args.cmd != "cis":
            log.warning("`--acat` is only compatible with `cis` subcommand. Ignoring.")
        perm_test = ACAT()
    else:
        # for lm wald test, use t distribution during permutation
        use_tdist = isinstance(family, Gaussian)
        perm_test = BetaPermutation(max_perm_direct=args.nperm, use_tdist=use_tdist)

    if args.keep is not None:
        log.info("Reading list of samples to keep for analyses.")
        inds_to_keep = read_single_column_file(args.keep)
        log.info(f"Found {len(inds_to_keep)} samples to keep.")
    else:
        inds_to_keep = None

    if args.exclude is not None:
        log.info("Reading list of samples to exclude from analyses.")
        inds_to_exclude = read_single_column_file(args.exclude)
        log.info(f"Found {len(inds_to_exclude)} samples to exclude.")
    else:
        inds_to_exclude = None

    log.info("Reading genotype, phenotype, and covariate data")
    geno_data = _load_genotype_data(args, log)

    if args.gene_list is not None:
        gene_keep_list = read_single_column_file(args.gene_list)
    elif args.genes is not None:
        gene_keep_list = args.genes
    else:
        gene_keep_list = None

    gene_exclude_list = None
    expr_data = ExpressionData.from_bedfile(
        args.pheno, inds_to_keep, inds_to_exclude, gene_keep_list, gene_exclude_list
    )
    expr_data = expr_data.filter_genes_by_percentage(args.min_gene_expr_pct)
    if args.min_indiv_expr_pct:
        expr_data = expr_data.filter_individuals_by_percentage(args.min_indiv_expr_pct)

    covar = read_plink_style_tsvlike(args.covar, args.covar_name, args.rm_covar)

    # perform one-hot encoding for string-based columns, if specified
    if args.one_hot:
        cat = pl.selectors.string().exclude("iid")
        covar = covar.to_dummies(cat, drop_first=True).drop(cat)

    # normalize all numeric columns to have mean 0 and var 1
    if args.normalize_covar:
        num = pl.all().exclude("iid")

        # let's make sure to not standardize the offset if it was provided, as we haven't yet extracted it
        if args.offset_name_from_covar:
            num = num.exclude(args.offset_name_from_covar)

        covar = covar.with_columns((num - num.mean()) / num.std())

    # we add an intercept column to the covariates by default if no normalization is performed
    # but we allow users to disable this
    if not args.no_intercept:
        covar = covar.with_columns(pl.lit(1.0).alias("intercept"))

    # before filter gene list, calculate library size and set offset, or read in pre-computed offset
    if args.offset:
        offset = read_offset_tsvlike(args.offset)
    elif args.offset_name_from_covar:
        offset = covar.select(pl.col("iid"), pl.col(args.offset_name_from_covar))
        # drop the offset from the covariates data
        covar = covar.drop(args.offset_name_from_covar)
    elif args.set_offset_from_libsize:
        offset = expr_data.offset_from_libsize
    else:
        offset = None

    # take the genotype, expression, covariates, and offset and align by iid for valid analyses
    # lump those into single object for easier passing around
    data = ReadyDataState.from_data(
        geno_data,
        expr_data,
        covar,
        offset,
        keep_samples=inds_to_keep,
        drop_samples=inds_to_exclude,
    )
    log.info("Finished reading and aligning genotype, phenotype, covariate data.")

    return data, family, reg_model, test, perm_test


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
    nominal_p.set_defaults(func=_nominal_scan)

    trans_p = _create_common_subp(subp, "trans", help="Perform a trans-eQTL scan.")
    trans_p.set_defaults(func=_trans_scan)

    gepcs_p = subp.add_parser(
        "compute-pcs",
        help=(
            "Compute gene expression principal components."
            " This uses a randomized probabilistic PCA algorithm and will be dependent on `--seed`."
        ),
    )
    gepcs_p.add_argument("--pheno", help="Path to phenotypes", required=True)
    gepcs_p.add_argument(
        "--num-pcs",
        type=int,
        help="Number of principal components to compute",
    )
    gepcs_p.add_argument("--covar", help="Path to covariate data", required=True)
    gepcs_p.add_argument(
        "--transform",
        choices=["tmm", "log1p"],
        default=None,
        help="Transformation to perform on observed gene expression before computing PCs.",
    )
    gepcs_p.add_argument(
        "--min-gene-expr-pct",
        type=float,
        default=0.0,
        help="Keep genes with expression levels above specified value",
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
    gepcs_p.add_argument("--seed", type=int, default=0, help="Seed for PRNG initialization.")
    gepcs_p.add_argument(
        "--out",
        "-o",
        type=str,
        default="jaxqtl.princ_comp.tsv",
        help="Path to output computed gene expression principal components (and covariate data, if specified).",
    )
    gepcs_p.set_defaults(func=_compute_expression_pcs)

    args = argp.parse_args(args)

    # we need to set 64bit support before any other option
    if args.platform != "tpu":
        jax.config.update("jax_enable_x64", True)
    else:
        # TPU not support complex 64, only 16 and 32
        jax.config.update("jax_enable_x64", False)

    jax.config.update("jax_platform_name", args.platform)

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
        log.info("Finished! Thank you!")
    else:
        argp.print_help()

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
