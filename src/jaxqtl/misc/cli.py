#!/usr/bin/env python3
import argparse as ap
import logging
import os
import sys
from importlib import metadata

import numpy as np

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import H5AD, PheBedReader, SingleCellFilter
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.cis import map_cis
from jaxqtl.map.nominal import map_nominal
from jaxqtl.post import rfunc

sys.path.insert(1, os.path.dirname(__file__))


def get_logger(name, path=None):
    """get logger for jaxqtl progress"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)

        log_format = "[%(asctime)s - %(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console.setFormatter(formatter)

        if path is not None:
            disk_log_stream = open("{}.log".format(path), "w")
            disk_handler = logging.StreamHandler(disk_log_stream)
            logger.addHandler(disk_handler)
            disk_handler.setFormatter(formatter)

    return logger


def run_map(args, log):
    """Wrapper for run QTL mapping"""

    # read data
    dat = create_readydata(
        args.geno,
        args.pheno,
        args.covar,
        geno_reader=PlinkReader(),
        pheno_reader=PheBedReader(),
    )

    # TODO: filter for certain chromosomes

    log.info("Begin mapping")

    if args.mode == "cis":
        mapcis_df = map_cis(dat, family=Poisson())
        qval, pi0 = rfunc.qvalue(
            mapcis_df["pval_beta"], lambda_qvalue=args.qvalue_lambda
        )
        mapcis_df["qval"] = qval
        mapcis_df.to_csv(
            os.path.join(args.output_prefix, ".bed.gz"),
            index=False,
            sep="\t",
        )
    elif args.mode == "cis_nominal":
        map_nominal(dat, family=Poisson(), out_path=args.output_prefix)

    log.info("Finished mapping")

    return 0


def run_format(args, log):
    log.info("Read in raw count data")
    pheno_reader = H5AD()
    count_mat = pheno_reader(args.pheno)

    log.info("Filtering and creating pseudo bulk count data")
    count_df = pheno_reader.process(count_mat, SingleCellFilter)

    log.info("Splitting and writing in bed")
    pheno_reader.write_bed(
        count_df,
        gtf_bed_path=args.gtf,
        out_dir=args.output_prefix,
        celltype_path=args.celltype,
    )

    log.info("Finished.")
    return 0


def create_format_parser(subparser):
    subparser.add_argument("--pheno", type=str, help="Phenotypes in H5AD format")
    subparser.add_argument("--gtf", type=str, help="gtf file for annotating gene TSS")
    subparser.add_argument("--celltype", type=str, help="cell type list")
    subparser.add_argument("--out", type=str, help="output directory")

    return subparser


def create_map_parser(subparser):
    subparser.add_argument("geno", help="Genotypes file in PLINK format")
    subparser.add_argument("pheno", help="Phenotype file in BED format")
    subparser.add_argument(
        "--mode",
        default="cis",
        choices=["cis", "cis_nominal", "cis_independent", "cis_susie", "trans"],
        help="Mapping mode. Default: cis",
    )
    subparser.add_argument(
        "--covar",
        default=None,
        help="Covariates file, tab-delimited, covariates x samples",
    )
    subparser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of permutations. Default: 10000",
    )
    subparser.add_argument(
        "--chr",
        default=None,
        nargs="+",
        type=str,
        help="Only mapping for some chromosomes",
    )
    subparser.add_argument(
        "--interaction", default=None, type=str, help="Interaction term(s)"
    )
    subparser.add_argument(
        "--cis_output",
        default=None,
        type=str,
        help="Output from 'cis' mode with q-values. Required for independent cis-QTL mapping.",
    )
    subparser.add_argument(
        "--phenotype_groups",
        default=None,
        type=str,
        help="Phenotype groups. Header-less TSV with two columns: phenotype_id, group_id",
    )
    subparser.add_argument(
        "--window",
        default=500000,
        type=np.int32,
        help="Cis-window size (one side), in bases. Default: 500000.",
    )
    subparser.add_argument(
        "--pval_threshold",
        default=None,
        type=np.float64,
        help="Output only significant phenotype-variant pairs with a p-value below threshold. "
        "Default: 1e-5 for trans-QTL",
    )
    subparser.add_argument(
        "--maf_threshold",
        default=0,
        type=np.float64,
        help="Include only genotypes with minor allele frequency >= maf_threshold. Default: 0",
    )
    subparser.add_argument(
        "--maf_threshold_interaction",
        default=0.05,
        type=np.float64,
        help="MAF threshold for interactions, applied to lower and upper half of samples",
    )
    subparser.add_argument(
        "--dosages",
        action="store_true",
        help="Load dosages instead of genotypes (only applies to PLINK2 bgen input).",
    )
    subparser.add_argument(
        "--load_split",
        action="store_true",
        help="Load genotypes into memory separately for each chromosome.",
    )
    subparser.add_argument(
        "--disable_beta_approx",
        action="store_true",
        help="Disable Beta-distribution approximation of empirical p-values (not recommended).",
    )
    subparser.add_argument(
        "--warn_monomorphic",
        action="store_true",
        help="Warn if monomorphic variants are found.",
    )
    subparser.add_argument(
        "--fdr", default=0.05, type=np.float64, help="FDR for cis-QTLs"
    )
    subparser.add_argument(
        "--qvalue_lambda",
        default=None,
        type=np.float64,
        help="lambda parameter for pi0est in qvalue.",
    )
    subparser.add_argument(
        "--seed", default=None, type=int, help="Seed for permutations."
    )
    subparser.add_argument(
        "-o", "--output_prefix", default=".", help="Output directory"
    )

    return subparser


def _main(argsv):
    # top level parser
    global_parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    global_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="If this is true, set logger to be debug mode if debug=True. Default=False",
    )

    subparsers = global_parser.add_subparsers(required=True)

    # use different sub parsers for: 1) format input; 2) run_mapping; [3) finemap]
    format_parser = subparsers.add_parser(
        "format",
        description="create bed files from data matrix format such as H5AD",
        help="Subcommands: format input or run QTL mapping",
    )
    format_parser = create_format_parser(format_parser)
    format_parser.set_defaults(func=run_format)

    map_parser = subparsers.add_parser(
        "map",
        description="run mapping (cis or trans)",
        help="Subcommands: format input or run QTL mapping",
    )
    map_parser = create_map_parser(map_parser)
    map_parser.set_defaults(func=run_map)

    version = metadata.version("jaxQTL")

    masthead = "===================================" + os.linesep
    masthead += f"             jaxQTL v{version}             " + os.linesep
    masthead += "===================================" + os.linesep

    # parse arguments
    args = global_parser.parse_args(argsv)

    log = get_log()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    config.update("jax_enable_x64", True)

    args.func(args, log)


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
