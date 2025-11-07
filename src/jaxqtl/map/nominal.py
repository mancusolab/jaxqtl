from typing import Optional

import numpy as np
import pandas as pd

import jax.numpy.linalg as jnpla

from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..infer.utils import HypothesisTest
from ..io.readfile import ReadyDataState
from ..log import get_log
from .utils import _get_geno_info, _setup_G_y


def map_nominal(
    dat: ReadyDataState,
    test: HypothesisTest,
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    mode: Optional[str] = None,
    cond_snp: Optional[str] = None,
) -> pd.DataFrame:
    """cis eQTL Mapping for all cis-SNP gene pairs

    :param dat: data input containing genotype array, bim, gene count data, gene meta data (tss), and covariates
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param test: approach for hypothesis test, default to ScoreTest()
    :param log: logger for QTL progress
    :param append_intercept: `True` if want to append intercept, `False` otherwise
    :param standardize: True` if want to standardize covariates data
    :param window: window size (bp) of one side for cis scope, default to 500000,
        meaning in total 1Mb from left to right
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param robust_se: `True` if use huber white robust estimator for standard errors for nominal mapping (not used here)
        default to `False`
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :return: data frame of nominal mapping for cisSNPs - gene pairs
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
    slope = []
    slope_se = []
    nominal_p = []
    converged = []
    num_var_cis = []
    alpha = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"])

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G (nxM) and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend, mode)

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
                "Performing cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        # add conditional SNP
        if cond_snp is not None:
            cond_snp_idx = dat.bim.i[dat.bim.snp == cond_snp].values
            cond_snp_vec = dat.geno[:, cond_snp_idx]
            X_add_cov = jnp.append(X, cond_snp_vec, axis=1)
            result = test(X_add_cov, G, y, offset_eta)
        else:
            result = test(X, G, y, offset_eta)

        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )
        g_info = _get_geno_info(G)
        var_df["phenotype_id"] = gene_name
        var_df["tss"] = start_min
        var_df_all = pd.concat([var_df_all, var_df], ignore_index=True)
        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        af.append(g_info.af)
        ma_count.append(g_info.ma_count)

        slope.append(result.beta)
        slope_se.append(result.se)
        nominal_p.append(result.p)
        converged.append(result.converged)  # whether full model converged
        num_var_cis.append(var_df.shape[0])
        alpha.append(result.alpha)

    # write result
    start_row = 0
    end_row = 0
    outdf = var_df_all
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm"], axis=1)

    # add additional columns
    outdf["af"] = np.nan
    outdf["ma_count"] = np.nan
    outdf["pval_nominal"] = np.nan
    outdf["slope"] = np.nan
    outdf["slope_se"] = np.nan
    outdf["converged"] = np.nan
    outdf["alpha"] = np.nan

    for idx, _ in gene_mapped_list.iterrows():
        end_row += num_var_cis[idx]
        outdf.loc[np.arange(start_row, end_row), "af"] = af[idx]
        outdf.loc[np.arange(start_row, end_row), "ma_count"] = ma_count[idx]
        outdf.loc[np.arange(start_row, end_row), "pval_nominal"] = nominal_p[idx]
        outdf.loc[np.arange(start_row, end_row), "slope"] = slope[idx]
        outdf.loc[np.arange(start_row, end_row), "slope_se"] = slope_se[idx]
        outdf.loc[np.arange(start_row, end_row), "converged"] = converged[idx]
        outdf.loc[np.arange(start_row, end_row), "alpha"] = alpha[idx]
        start_row = end_row

    return outdf
