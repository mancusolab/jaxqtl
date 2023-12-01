import argparse as ap
import logging
import sys

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import NegativeBinomial as smNB

import jax.numpy as jnp
from jax import random as rdm
from jax.config import config

# import statsmodels.api as sm
from jaxtyping import ArrayLike

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.utils import ScoreTest, WaldTest, _setup_G_y
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import ReadyDataState, create_readydata
from jaxqtl.log import get_log
from jaxqtl.map import (  # _get_geno_info,
    map_cis,
    map_fit_intercept_only,
    map_nominal,
    write_parqet,
)


def get_logger(name, path=None):
    """get logger for factorgo progress"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)

        # if need millisecond use : %(asctime)s.%(msecs)03d
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


def cis_scan_wald_sm(
    X: ArrayLike, G: ArrayLike, y: ArrayLike, offset_eta: ArrayLike = 0.0
):
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """
    # statsmodel result
    p_vec = []

    # print(sm_res.summary())
    for snp in G.T:
        M = jnp.append(X, snp.reshape((len(snp), 1)), axis=1)
        sm_mod = smNB(
            np.array(y),
            np.array(M),
            offset=np.array(offset_eta).reshape((len(offset_eta),)),
        ).fit(maxiter=100)
        p_vec.append(sm_mod.pvalues[-2])  # alpha estimate

    return p_vec


def map_cis_sm(
    dat: ReadyDataState,
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    nperm: int = 1000,
):
    """eQTL Mapping for all cis-SNP gene pairs

    append_intercept: add a column of ones in front for intercepts in design matrix
    standardize: on covariates

    Returns:
        score test statistics and p value (no effect estimates)
        write out parquet file by chrom for efficient data storage and retrieval
    """

    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    X = dat.covar
    n, k = X.shape

    gene_info = dat.pheno_meta

    # append genotype as the last column
    if standardize:
        X = X / jnp.std(X, axis=0)

    if append_intercept:
        X = jnp.hstack((jnp.ones((n, 1)), X))

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            if verbose:
                log.info(
                    "No cis-SNPs found for %s over region %s:%s-%s. Skipping.",
                    gene_name,
                    str(chrom),
                    str(lstart),
                    str(rend),
                )
            continue

        if verbose:
            log.info(
                "Performing cis-qtl scan (statsmodel) for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        sm_p = cis_scan_wald_sm(X, G, y, offset_eta)

        key = rdm.PRNGKey(1)
        key, key_init = rdm.split(key, 2)

        for i in range(nperm):
            key, key_init = rdm.split(key, 2)
            perm_idx = rdm.choice(
                key, jnp.arange(len(y)), shape=(len(y), 1), replace=False
            )
            sm_p = cis_scan_wald_sm(X, G, y[perm_idx], offset_eta[perm_idx])

        if verbose:
            log.info(
                "Finished cis-qtl scan (statsmodel) for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

    return sm_p


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("-geno", type=str, help="Genotype prefix, eg. chr17")
    argp.add_argument("-covar", type=str, help="Covariate path")
    argp.add_argument("-pheno", type=str, help="Pheno path")
    argp.add_argument(
        "-model", type=str, choices=["gaussian", "poisson", "NB"], help="Model"
    )
    argp.add_argument("-genelist", type=str, help="Path to gene list (no header)")
    argp.add_argument("-offset", type=str, help="Path to log offset (no header)")
    argp.add_argument(
        "-indlist", type=str, help="Path to individual list (no header); default is all"
    )
    argp.add_argument(
        "-mode",
        type=str,
        choices=["nominal", "cis", "fitnull"],
        help="Cis or nominal mapping",
    )
    argp.add_argument(
        "-test-method", type=str, choices=["wald", "score"], help="Wald or score test"
    )
    argp.add_argument("-window", type=int, default=500000)
    argp.add_argument("-nperm", type=int, default=1000)
    argp.add_argument("--perm-seed", type=int, default=1)
    argp.add_argument("-addpc", type=int, default=2, help="Add expression PCs")
    argp.add_argument(
        "--robust",
        action="store_true",
        default=False,
        help="Robust SE",
    )
    argp.add_argument(
        "--perm-count",
        action="store_true",
        default=False,
        help="Permute count for type I error calibration",
    )
    argp.add_argument(
        "--qvalue",
        action="store_true",
        default=False,
        help="Add q value",
    )
    argp.add_argument(
        "--statsmodel",
        action="store_true",
        default=False,
        help="Run statsmodel",
    )
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("-out", type=str, help="out file prefix")

    args = argp.parse_args(args)  # a list a strings

    platform = "cpu"
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", platform)

    log = get_logger(__name__, args.out)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    if args.model == "poisson":
        family = Poisson()
    elif args.model == "NB":
        family = NegativeBinomial()
    elif args.model == "gaussian":
        family = Gaussian()
    else:
        log.info("Please choose either poisson or gaussian.")

    # raw genotype data and impute for genotype data
    geno_reader = PlinkReader()
    geno, bim, sample_info = geno_reader(args.geno)

    pheno_reader = PheBedReader()
    pheno = pheno_reader(args.pheno)

    covar = covar_reader(args.covar)

    genelist = pd.read_csv(args.genelist, header=None, sep="\t").iloc[:, 0].to_list()

    if args.indlist is not None:
        indList = pd.read_csv(args.indlist, header=None, sep="\t").iloc[:, 0].to_list()
    else:
        indList = None

    dat = create_readydata(
        geno, bim, pheno, covar, autosomal_only=True, ind_list=indList
    )

    # before filter gene list, calculate library size and set offset
    if args.offset is None:
        total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
        offset_eta = jnp.log(total_libsize)
    else:
        offset_eta = pd.read_csv(args.offset, header=None, sep="\t").iloc[:, 0]
        offset_eta = jnp.array(offset_eta).reshape((len(offset_eta), 1))

    # filter genes with no expressions at all
    dat.filter_gene(geneexpr_percent_cutoff=0.0)

    # add expression PCs to covar, genotype PC should appended to covar outside jaxqtl
    if args.addpc > 0:
        dat.add_covar_pheno_PC(k=args.addpc)

    if isinstance(family, Gaussian):
        # dat.transform_y(mode='log1p')  # log1p
        offset_eta = jnp.zeros(offset_eta.shape)

    # filter gene list
    dat.filter_gene(gene_list=genelist)

    # permute gene expression for type I error calibration
    if args.perm_count:
        np.random.seed(args.perm_seed)
        # assume one gene
        perm_idx = np.random.permutation(np.arange(0, len(dat.pheno.count)))
        dat.pheno.count.iloc[:, 0] = dat.pheno.count.iloc[:, 0][perm_idx]
        offset_eta = offset_eta[perm_idx]

    if dat.pheno_meta.gene_map.shape[0] < 1:
        log.info("No gene exist.")
        sys.exit()

    if args.mode == "cis":
        if args.test_method == "score":
            outdf_cis_score = map_cis(
                dat,
                family=family,
                test=ScoreTest(),
                standardize=True,
                window=args.window,
                offset_eta=offset_eta,
                compute_qvalue=args.qvalue,
                n_perm=args.nperm,
                log=log,
            )
            outdf_cis_score.to_csv(
                args.out + ".cis_score.tsv.gz", sep="\t", index=False
            )
        elif args.test_method == "wald":
            outdf_cis_wald = map_cis(
                dat,
                family=family,
                test=WaldTest(),
                standardize=True,
                window=args.window,
                offset_eta=offset_eta,
                compute_qvalue=args.qvalue,
                robust_se=args.robust,
                n_perm=args.nperm,
                log=log,
            )
            outdf_cis_wald.to_csv(args.out + ".cis_wald.tsv.gz", sep="\t", index=False)

    elif args.mode == "nominal":
        if args.test_method == "score":
            out_df = map_nominal(
                dat,
                test=ScoreTest(),
                family=family,
                standardize=True,
                log=log,
                window=args.window,
                offset_eta=offset_eta,
            )
            write_parqet(outdf=out_df, method="score", out_path=args.out)
        elif args.test_method == "wald":
            out_df = map_nominal(
                dat,
                family=family,
                test=WaldTest(),
                standardize=True,
                log=log,
                window=args.window,
                offset_eta=offset_eta,
                robust_se=args.robust,
            )
            write_parqet(outdf=out_df, method="wald", out_path=args.out)

    elif args.mode == "fitnull":
        mapcis_intercept_only_mu = map_fit_intercept_only(
            dat, family=family, offset_eta=offset_eta
        )
        mapcis_intercept_only_mu.to_csv(
            args.out + ".fit.resid.tsv.gz", sep="\t", index=False
        )
    else:
        log.info("please select available methods.")
        sys.exit()

    # for speed comparison, doesn't output results
    if args.statsmodel:
        map_cis_sm(
            dat,
            standardize=True,
            log=log,
            window=args.window,
            offset_eta=offset_eta,
            nperm=args.nperm,
        )

    return 0


# user call this script will treat it like a program
if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:])
    )  # grab all arguments; first arg is alway the name of the script
