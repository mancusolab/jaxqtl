#!/usr/bin/env python3
import argparse as ap
import logging
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import jax
import jax.numpy as jnp

from jaxtyping import ArrayLike

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.utils import ScoreTest, WaldTest
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata, ReadyDataState
from jaxqtl.log import get_log, get_logger
from jaxqtl.map.cis import map_cis, write_parqet
from jaxqtl.map.nominal import fit_intercept_only, map_nominal, map_nominal_covar
from jaxqtl.map.utils import _get_geno_info, _setup_G_y


def cis_scan_score_sm(X: ArrayLike, G: ArrayLike, y: ArrayLike, offset_eta: ArrayLike = 0.0):
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """
    # statsmodel result
    sm_glm = sm.GLM(
        np.array(y),
        np.array(X),
        family=sm.families.Poisson(),
        offset=np.array(offset_eta).squeeze(),
    )
    sm_res = sm_glm.fit()

    chi2_vec = []
    p_vec = []

    # print(sm_res.summary())
    for snp in G.T:
        chi2, sm_p, _ = sm_res.score_test(params_constrained=sm_res.params, exog_extra=snp)
        chi2_vec.append(chi2)
        p_vec.append(sm_p)

    return chi2_vec, p_vec


def map_cis_nominal_score_sm(
    dat: ReadyDataState,
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
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

    af = []
    ma_count = []
    Z = []
    nominal_p = []
    num_var_cis = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"])

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

        chi2, sm_p = cis_scan_score_sm(X, G, y, offset_eta)
        g_info = _get_geno_info(G)

        if verbose:
            log.info(
                "Finished cis-qtl scan (statsmodel) for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        var_df["phenotype_id"] = gene_name
        var_df["tss"] = start_min
        var_df_all = pd.concat([var_df_all, var_df], ignore_index=True)
        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        af.append(g_info.af)
        ma_count.append(g_info.ma_count)

        nominal_p.append(sm_p)
        Z.append(chi2)
        num_var_cis.append(var_df.shape[0])

    # write result
    start_row = 0
    end_row = 0
    outdf = var_df_all
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm"], axis=1)

    # add additional columns
    outdf["af"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["Z"] = np.NaN

    for idx, _ in gene_mapped_list.iterrows():
        end_row += num_var_cis[idx]
        outdf.loc[np.arange(start_row, end_row), "af"] = af[idx]
        outdf.loc[np.arange(start_row, end_row), "ma_count"] = ma_count[idx]
        outdf.loc[np.arange(start_row, end_row), "pval_nominal"] = nominal_p[idx]
        outdf.loc[np.arange(start_row, end_row), "Z"] = Z[idx]
        start_row = end_row

    return outdf


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("-geno", type=str, help="Genotype prefix, eg. chr17")
    argp.add_argument("-covar", type=str, help="Covariate path")
    argp.add_argument("-add-covar", type=str, help="Covariate path for additional covariates")
    argp.add_argument("-covar-test", type=str, help="Covariate to test")
    argp.add_argument("-pheno", type=str, help="Pheno path")
    argp.add_argument("-model", type=str, choices=["gaussian", "poisson", "NB"], help="Model")
    argp.add_argument("-genelist", type=str, help="Path to gene list (no header)")
    argp.add_argument("-offset", type=str, help="Path to log offset (no header)")
    argp.add_argument("-indlist", type=str, help="Path to individual list (no header); default is all")
    argp.add_argument(
        "-mode",
        type=str,
        choices=["nominal", "cis", "fitnull", "covar", "trans", "estimate_ld_only"],
        help="Cis or nominal mapping",
    )
    argp.add_argument(
        "--platform",
        "-p",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        help="platform, cpu, gpu or tpu",
    )
    argp.add_argument("-test-method", type=str, choices=["wald", "score"], help="Wald or score test")
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
        "--perm-pheno",
        action="store_true",
        default=False,
        help="Permute phenotype for type I error calibration",
    )
    argp.add_argument(
        "--qvalue",
        action="store_true",
        default=False,
        help="Add q value",
    )
    argp.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="Standardize covariates",
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

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.platform)

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

    covar = covar_reader(args.covar, args.add_covar, args.covar_test)

    genelist = pd.read_csv(args.genelist, header=None, sep="\t").iloc[:, 0].to_list()

    if args.indlist is not None:
        indList = pd.read_csv(args.indlist, header=None, sep="\t").iloc[:, 0].to_list()
    else:
        indList = None

    dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True, ind_list=indList)

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
        dat.add_covar_pheno_PC(k=args.addpc, add_covar=args.add_covar)

    if isinstance(family, Gaussian):
        # dat.transform_y(mode='log1p')  # log1p
        # note: use pre-processed file as in tensorqtl
        offset_eta = jnp.zeros_like(offset_eta)

    # filter gene list
    dat.filter_gene(gene_list=genelist)

    # permute gene expression for type I error calibration
    if args.perm_pheno:
        np.random.seed(args.perm_seed)
        perm_idx = np.random.permutation(np.arange(0, len(dat.pheno.count)))
        dat.pheno.count = dat.pheno.count.iloc[perm_idx]
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
                standardize=args.standardize,
                window=args.window,
                offset_eta=offset_eta,
                n_perm=args.nperm,
                compute_qvalue=args.qvalue,
                log=log,
            )
            outdf_cis_score.to_csv(args.out + ".cis_score.tsv.gz", sep="\t", index=False)
        elif args.test_method == "wald":
            outdf_cis_wald = map_cis(
                dat,
                family=family,
                test=WaldTest(),
                standardize=args.standardize,
                window=args.window,
                offset_eta=offset_eta,
                n_perm=args.nperm,
                robust_se=args.robust,
                compute_qvalue=args.qvalue,
                log=log,
            )
            outdf_cis_wald.to_csv(args.out + ".cis_wald.tsv.gz", sep="\t", index=False)

    elif args.mode == "nominal":
        if args.test_method == "score":
            out_df = map_nominal(
                dat,
                family=family,
                test=ScoreTest(),
                standardize=args.standardize,
                window=args.window,
                offset_eta=offset_eta,
                log=log,
            )
            write_parqet(outdf=out_df, method="score", out_path=args.out)
        elif args.test_method == "wald":
            out_df = map_nominal(
                dat,
                test=WaldTest(),
                family=family,
                standardize=args.standardize,
                log=log,
                window=args.window,
                offset_eta=offset_eta,
                robust_se=args.robust,
            )
            write_parqet(outdf=out_df, method="wald", out_path=args.out)

    elif args.mode == "estimate_ld_only":
        _ = map_nominal(
            dat,
            family=family,
            standardize=args.standardize,
            window=args.window,
            offset_eta=offset_eta,
            log=log,
            mode=args.mode,
        )
        log.info("write out LD matrix.")

    elif args.mode == "trans":
        # genotype for trans-SNPs are read in from plink files, no trans-cutter
        if args.test_method == "score":
            out_df = map_nominal(
                dat,
                family=family,
                offset_eta=offset_eta,
                test=ScoreTest(),
                mode="trans",
                standardize=args.standardize,
                robust_se=args.robust,
                log=log,
            )
            out_df.to_csv(args.out + ".trans_score.tsv.gz", sep="\t", index=False)
        elif args.test_method == "wald":
            out_df = map_nominal(
                dat,
                family=family,
                offset_eta=offset_eta,
                test=WaldTest(),
                mode="trans",
                standardize=args.standardize,
                robust_se=args.robust,
                log=log,
            )
            out_df.to_csv(args.out + ".trans_wald.tsv.gz", sep="\t", index=False)

    elif args.mode == "covar":
        if args.test_method == "score":
            out_df = map_nominal_covar(
                dat,
                family=family,
                test=ScoreTest(),
                offset_eta=offset_eta,
                standardize=args.standardize,
                robust_se=args.robust,
                log=log,
            )
            out_df.to_csv(args.out + ".cis_score.tsv.gz", sep="\t", index=False)
        elif args.test_method == "wald":
            out_df = map_nominal_covar(
                dat,
                family=family,
                test=WaldTest(),
                offset_eta=offset_eta,
                standardize=args.standardize,
                robust_se=args.robust,
                log=log,
            )
            out_df.to_csv(args.out + ".cis_wald.tsv.gz", sep="\t", index=False)

    elif args.mode == "fitnull":
        pass
        out_df = fit_intercept_only(dat, family=family, offset_eta=offset_eta, robust_se=False, log=log)
        out_df.to_csv(args.out + ".intercept_only." + args.model + ".tsv.gz", sep="\t", index=False)
    else:
        log.info("please select available methods.")
        sys.exit()

    if args.statsmodel:
        out_df = map_cis_nominal_score_sm(dat, standardize=True, log=log, window=args.window, offset_eta=offset_eta)
        write_parqet(outdf=out_df, method="sm_score", out_path=args.out)

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
