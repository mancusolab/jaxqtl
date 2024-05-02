#!/usr/bin/env python3
import argparse as ap
import logging
import os
import sys

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from jaxqtl.io.expr import GeneMetaData
from jaxqtl.io.geno import PlinkReader
from jaxqtl.log import get_log, get_logger


def cov_scan_sm(X: ArrayLike, G: ArrayLike, y: ArrayLike, model: str = "logit"):
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """
    beta_vec = []
    bse_vec = []
    p_vec = []
    converged = []
    reg_vec = []

    # print(sm_res.summary())
    for snp in G.T:
        M = np.hstack((X, snp[:, np.newaxis]))

        if model == "logit":
            sm_glm = sm.Logit(
                np.array(y),
                np.array(M),
            )
        elif model == "probit":
            sm.Probit._continuous_ok = True
            sm_glm = sm.Probit(np.array(y), np.array(M))

        try:
            sm_res = sm_glm.fit(disp=False, warn_convergence=False, maxiter=1000)
            reg = 0
        except Exception:
            sm_res = sm_glm.fit_regularized(disp=False, warn_convergence=False, maxiter=1000)
            reg = 1

        beta_vec.append(sm_res.params[-1])
        bse_vec.append(sm_res.bse[-1])
        p_vec.append(sm_res.pvalues[-1])
        converged.append(sm_res.converged)
        reg_vec.append(reg)

    return beta_vec, bse_vec, p_vec, converged, reg_vec


def _cis_window_cutter(
    geno: ArrayLike, var_info: pd.DataFrame, chrom: str, start: int, end: int
) -> Tuple[Array, pd.DataFrame]:
    """
    return variant list in cis for given gene
    Map is a pandas data frame

    plink bim file is 1-based
    the map file is hg19,
    emsemble use 1-based
    vcf file is one-based

    gene_name = 'ENSG00000250479', start: 24110630
    GenomicRanges example: https://biocpy.github.io/GenomicRanges/

    Returns:
        Genotype matrix for cis-variants
    """

    cis_var_info = var_info.loc[
        (var_info["chrom"] == str(chrom)) & (var_info["pos"] >= start) & (var_info["pos"] <= end)
    ]

    # subset G to cis variants (nxp)
    G_tocheck = jnp.take(geno, jnp.array(cis_var_info.i), axis=1)

    # check monomorphic: G.T[:, [0]] find first occurrence on all genotype, var x 1
    mono_var = (G_tocheck.T == G_tocheck.T[:, [0]]).all(1)  # bool (var, ), show whether given variant is monomorphic
    not_mono_var = jnp.invert(mono_var)  # reverse False and True (same as "~" operator)
    G = G_tocheck[:, not_mono_var]  # take genotype that are NOT monomorphic
    cis_var_info = cis_var_info.loc[not_mono_var.tolist()]

    # note: if no variants taken, then G has shape (n,0), cis_var_info has shape (0, 7); both 2-dim
    return G, cis_var_info


def map_annot_sm(
    geno: ArrayLike,
    var_info: pd.DataFrame,
    annot: pd.DataFrame,
    celltype: str,
    pip_prefix: str,
    pip_suffix: str,
    gene_info: GeneMetaData,
    log=None,
    window: int = 500000,
    verbose: bool = True,
    ldscore: Optional[pd.DataFrame] = None,
    model: str = "logit",
    pip_cs: str = "pip",
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

    out_columns = [
        "phenotype_id",
        "annot",
        "chrom",
        "num_var",
        "slope",
        "slope_se",
        "pval_nominal",
        "model_converged",
        "reg",
    ]

    phenotype_id = []
    chrom_list = []
    annot_list = []
    slope = []
    slope_se = []
    nominal_p = []
    converged_list = []
    num_var_cis = []
    reg_list = []

    # all annotation names
    annot_names = annot.columns

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # read geno on one chromosome and pull cis G (nxM) and y for this gene
        _, var_df = _cis_window_cutter(geno, var_info, str(chrom), lstart, rend)

        # read in pip_y: replace wald with score if needed
        file_path = f"{pip_prefix}/{gene_name}.{celltype}.{pip_suffix}"
        if os.path.exists(file_path):
            pip_y = pd.read_csv(file_path, sep="\t")
        else:
            pip_suffix = pip_suffix.replace("wald", "score")
            file_path = f"{pip_prefix}/{gene_name}.{celltype}.{pip_suffix}"
            if os.path.exists(file_path):
                pip_y = pd.read_csv(file_path, sep="\t")
            else:
                continue

        # choose outcome
        if pip_cs == "pip":
            y = np.array(pip_y[['pip']])
        elif pip_cs == "cs":
            # sum over CS column if exist
            cs_col = [col for col in pip_y if col.startswith('cs')]
            if len(cs_col) > 0:
                y = (np.array(pip_y[cs_col]).sum(axis=1) > 0) * 1
                y = y.reshape(-1, 1)
            else:
                continue

        # TODO: use var_df to subset cis variant in annot file
        cov = annot.iloc[var_df.i.values]

        # PIP is only produced for one SNP if multi-allele, so need to remove duplicate in annotation
        cov = np.array(cov[var_df.snp.values.isin(pip_y.snp.values)])

        # remove cov all zeros
        take_annot = cov.sum(0) > 0
        cov = cov[:, take_annot]

        # skip if no cis SNPs found
        if cov.shape[1] == 0:
            if verbose:
                log.info(
                    "No annot found for %s over region %s:%s-%s. Skipping.",
                    gene_name,
                    str(chrom),
                    str(lstart),
                    str(rend),
                )
            continue

        if verbose:
            log.info(
                "Performing cis-cov scan (statsmodel) for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        X = jnp.ones_like(y)
        if ldscore is not None:
            ld = ldscore.iloc[var_df.i.values]
            ld = ld[ld.SNP.isin(pip_y.snp.values)]
            ld = np.array(ld['L2']).reshape(-1, 1)
            X = np.hstack((X, ld))

        beta, se, p, converged, reg = cov_scan_sm(X, cov, y, model)

        if verbose:
            log.info(
                "Finished cis-qtl scan (statsmodel) for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        # combine meta results
        annot_mapped = annot_names[take_annot]
        annot_list.extend(annot_mapped)

        phenotype_id.extend(np.repeat(gene_name, len(annot_mapped)))
        num_var_cis.extend(np.repeat(var_df.shape[0], len(annot_mapped)))
        chrom_list.extend(np.repeat(chrom, len(annot_mapped)))

        # combine results
        slope.extend(beta)
        slope_se.extend(se)
        nominal_p.extend(p)
        converged_list.extend(converged)  # whether  model converged
        reg_list.extend(reg)

    # write result
    result_out = [
        phenotype_id,
        annot_list,
        chrom_list,
        num_var_cis,
        slope,
        slope_se,
        nominal_p,
        converged_list,
        reg_list,
    ]

    result_df = pd.DataFrame(result_out, index=out_columns).T

    return result_df


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("--geno", type=str, help="Genotype prefix, eg. chr17")
    argp.add_argument("--chr", type=str, help="which chromosome")
    argp.add_argument("--ldscore", type=str, help="path to ldscore")
    argp.add_argument("--model", type=str, help="logit or probit")
    argp.add_argument("--celltype", type=str, help="Cell type, eg. chr17")
    argp.add_argument("--pip-prefix", type=str, help="pip prefix for pip file")
    argp.add_argument("--pip-suffix", type=str, help="pip suffix for pip file")
    argp.add_argument("--pip-cs", type=str, help="run regression on PIP or CS")
    argp.add_argument("--genelist", type=str, help="Path to gene list (no header)")
    argp.add_argument("--annot", type=str, help="path to annotation file")
    argp.add_argument("--gene-meta", type=str, help="Path to gene list, contains phenotype_id, chr, start, end")
    argp.add_argument("--window", type=int, default=500000)
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("--out", type=str, help="out file prefix")

    args = argp.parse_args(args)  # a list a strings

    log = get_logger(__name__, args.out)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # test this
    # geno_path="../example/local/NK_new/chr22"
    # annot_path="../example/local/chr22.annot.gz"
    # gene_meta_path="../example/local/ENSG00000189269_pos.tsv"

    # raw genotype data and impute for genotype data
    geno_reader = PlinkReader()
    geno, bim, sample_info = geno_reader(args.geno)
    geno = jnp.array(geno)

    # read ldscore
    if args.ldscore is not None:
        ldscore = pd.read_csv(args.ldscore, sep="\t")
    else:
        ldscore = None

    annot = pd.read_csv(args.annot, sep="\t")

    # contains chr, start, end
    pos_df = pd.read_csv(args.gene_meta, sep="\t")
    genelist = pd.read_csv(args.genelist, header=None, sep="\t").iloc[:, 0].to_list()
    pos_df = pos_df.loc[pos_df.phenotype_id.isin(genelist)]

    gene_info = GeneMetaData(pos_df)

    out_df = map_annot_sm(
        geno=geno,
        var_info=bim,
        annot=annot,
        ldscore=ldscore,
        celltype=args.celltype,
        pip_prefix=args.pip_prefix,
        pip_suffix=args.pip_suffix,
        gene_info=gene_info,
        log=log,
        window=args.window,
        model=args.model,
        pip_cs=args.pip_cs,
    )

    out_df.to_csv(
        f"{args.out}/chr{args.chr}.{args.celltype}.{args.model}.{args.pip_cs}.tss{args.adj_tss}.annot_res.gz",
        index=False,
        sep="\t",
    )

    return 0


def run_cli():
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
