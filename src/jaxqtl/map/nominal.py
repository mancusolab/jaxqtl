import numpy as np
import pandas as pd

from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..families.distribution import ExponentialFamily
from ..infer.stderr import FisherInfoError, HuberError
from ..infer.utils import HypothesisTest, ScoreTest
from ..io.readfile import ReadyDataState
from ..log import get_log
from .utils import _get_geno_info, _setup_G_y


def map_nominal(
    dat: ReadyDataState,
    family: ExponentialFamily,
    test: HypothesisTest = ScoreTest(),
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    max_iter: int = 500,
):
    """eQTL Mapping for all cis-SNP gene pairs

    append_intercept: add a column of ones in front for intercepts in design matrix
    standardize: on covariates

    Returns:
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
    slope = []
    slope_se = []
    nominal_p = []
    converged = []
    num_var_cis = []
    alpha = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"])
    se_estimator = FisherInfoError() if robust_se else HuberError()
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
                "Performing cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result = test(X, G, y, family, offset_eta, se_estimator, max_iter)

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
    outdf["af"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["slope"] = np.NaN
    outdf["slope_se"] = np.NaN
    outdf["converged"] = np.NaN
    outdf["alpha"] = np.NaN

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
