from typing import Optional

import numpy as np
import pandas as pd

import jax.numpy.linalg as jnpla

from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..families.distribution import ExponentialFamily
from ..infer.glm import GLM
from ..infer.stderr import FisherInfoError, HuberError
from ..infer.utils import HypothesisTest, ScoreTest, WaldTest
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
    max_iter: int = 1000,
    mode: Optional[str] = None,
    prop_cutoff: Optional[float] = None,
    ld_out: str = "./gene",
) -> pd.DataFrame:
    """cis eQTL Mapping for all cis-SNP gene pairs

    :param dat: data input containing genotype array, bim, gene count data, gene meta data (tss), and covariates
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param test: approach for hypothesis test, default to ScoreTest()
    :param log: logger for QTL progress
    :param append_intercept: `True` if want to append intercept, `False` otherwise
    :param standardize: True` if want to standardize covariates data
    :param window: window size (bp) of one side for cis scope, default to 500000, meaning in total 1Mb from left to right
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param robust_se: `True` if use huber white robust estimator for standard errors for nominal mapping (not used here), default to `False`
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
    se_estimator = HuberError() if robust_se else FisherInfoError()

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G (nxM) and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend, mode)

        # filter by cutoff
        if prop_cutoff is not None:
            lib_size = np.exp(offset_eta)
            keep_iid = ((y / lib_size) >= prop_cutoff).squeeze()
            G = G[keep_iid]
            y = y[keep_iid]
            X = X[keep_iid]
            offset_eta = offset_eta[keep_iid]
            if y.shape[0] < 2:
                log.info("only 1 person left")
                exit()

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

        # calculate in-sample LD for cis-SNPs (weighted by GLM null model output, i.e., Gt W G)
        if mode == "estimate_ld_only":
            # only available for one gene
            R_wt_df = _calc_LD(G, X, result.weights)
            R_wt_df.to_csv(ld_out + ".ld_wt.tsv.gz", sep="\t", index=False, header=False)
            del R_wt_df
            continue

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


def _calc_LD(G, X, wts):
    w_half_X = X * jnp.sqrt(wts)
    w_X = X * wts

    # project covariates from G
    infor_inv = jnpla.inv(w_half_X.T @ w_half_X)
    G_resid = G - jnpla.multi_dot([X, infor_inv, w_X.T, G])
    w_G_resid = G_resid * jnp.sqrt(wts)

    w_G_resid = (w_G_resid - jnp.mean(w_G_resid, axis=0)) / jnp.std(w_G_resid, axis=0)
    GtWG = (w_G_resid).T @ w_G_resid
    R_wt = GtWG / jnp.diag(GtWG)

    R_wt_df = pd.DataFrame.from_records(R_wt)
    return R_wt_df


def map_nominal_covar(
    dat: ReadyDataState,
    family: ExponentialFamily,
    test: HypothesisTest = WaldTest(),
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    max_iter: int = 1000,
):
    """test association between gene expression and other covariates

    :param dat: data input containing genotype array, bim, gene count data, gene meta data (tss), and covariates
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param test: approach for hypothesis test, default to ScoreTest()
    :param log: logger for QTL progress
    :param append_intercept: `True` if want to append intercept, `False` otherwise
    :param standardize: True` if want to standardize covariates data
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param robust_se: `True` if use huber white robust estimator for standard errors for nominal mapping (not used here), default to `False`
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :return: data frame of nominal mapping for cisSNPs - gene pairs
    """
    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    X = dat.covar[:, :-1]  # separate the column from the rest
    n, _ = X.shape
    cov = dat.covar[:, -1].reshape((n, 1))

    gene_info = dat.pheno_meta

    # append genotype as the last column
    if standardize:
        X = X / jnp.std(X, axis=0)

    if append_intercept:
        X = jnp.hstack((jnp.ones((n, 1)), X))

    out_columns = ["phenotype_id", "chrom", "slope", "slope_se", "pval_nominal", "model_converged", "alpha_cov"]

    phenotype_id = []
    chrom_list = []
    slope = []
    slope_se = []
    nominal_p = []
    converged = []
    alpha = []
    se_estimator = HuberError() if robust_se else FisherInfoError()

    for gene in gene_info:
        gene_name, chrom, _, _ = gene

        # pull cis G (sample x gene) and y for this gene
        y = dat.pheno[gene_name]  # __getitem__

        # skip if no cis SNPs found
        if verbose:
            log.info("Performing scan for %s", gene_name)

        result = test(X, cov, y, family, offset_eta, se_estimator, max_iter)

        if verbose:
            log.info("Finished cis-qtl scan for %s", gene_name)

        # combine results
        phenotype_id.append(gene_name)
        slope.append(result.beta.item())
        slope_se.append(result.se.item())
        nominal_p.append(result.p.item())
        converged.append(result.converged.item())  # whether full model converged
        alpha.append(result.alpha.item())
        chrom_list.append(chrom)

    # write result
    result_out = [phenotype_id, chrom_list, slope, slope_se, nominal_p, converged, alpha]

    result_df = pd.DataFrame(result_out, index=out_columns).T

    return result_df


def fit_intercept_only(
    dat: ReadyDataState,
    family: ExponentialFamily,
    log=None,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    max_iter: int = 1000,
):
    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    n = dat.pheno.count.shape[0]
    X = jnp.ones((n, 1))

    gene_info = dat.pheno_meta

    out_columns = ["phenotype_id", "chrom", "slope", "model_converged", "alpha_cov"]

    phenotype_id = []
    chrom_list = []
    slope = []
    converged = []
    alpha = []
    se_estimator = HuberError() if robust_se else FisherInfoError()

    for gene in gene_info:
        gene_name, chrom, _, _ = gene

        # pull cis G (sample x gene) and y for this gene
        y = dat.pheno[gene_name]  # __getitem__

        # skip if no cis SNPs found
        if verbose:
            log.info("Performing scan for %s", gene_name)

        glm = GLM(family=family, max_iter=max_iter)

        eta, alpha_n = glm.calc_eta_and_dispersion(X, y, offset_eta)
        glmstate = glm.fit(
            X,
            y,
            offset_eta=offset_eta,
            init=eta,
            alpha_init=alpha_n,
            se_estimator=se_estimator,
        )

        if verbose:
            log.info("Finished cis-qtl scan for %s", gene_name)

        # combine results
        phenotype_id.append(gene_name)
        slope.append(glmstate.beta)
        converged.append(glmstate.converged)  # whether full model converged
        alpha.append(glmstate.alpha)
        chrom_list.append(chrom)

    # write result
    result_out = [phenotype_id, chrom_list, slope, converged, alpha]

    result_df = pd.DataFrame(result_out, index=out_columns).T

    return result_df
