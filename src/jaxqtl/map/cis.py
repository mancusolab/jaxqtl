from typing import Literal, Optional

import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray

from ..infer.permutations import AbstractPermutation, BetaPermutation, PermutationResult
from ..infer.utils import HypothesisTest, TestResult
from ..io.data import ReadyDataState
from ..log import get_log
from ..post.qvalue import calculate_qval, estimate_sig_threshold


class _ResultsAggregator:
    """
    Single internal class to unify dealing with cis results or nominal results
    """

    def __init__(self):
        self.frames: list = []

    def add_row(self, row: dict):
        # cheap 1-row DataFrame, but *only* created when needed
        self.frames.append(pl.DataFrame([row]))

    def add_df(self, df: pl.DataFrame):
        self.frames.append(df)

    def to_df(self):
        return pl.concat(self.frames, how="vertical")


def map_cis(
    data: ReadyDataState,
    test: HypothesisTest,
    perm_test: AbstractPermutation,
    mode: Literal["cis", "nominal"] = "cis",
    window: int = 500000,
    sig_level: float = 0.05,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: Optional[ArrayLike] = None,
    verbose: bool = True,
    log=None,
    seed: int = 123,
) -> pl.DataFrame:
    """Cis eQTL mapping for each gene, report lead variant

    Run cis-eQTL mapping by fitting specified GLM model, such as Poisson and Negative Binomial.
    To test association between each SNP and gene expression, choose either score test (much faster) or
    wald test.
    For each gene, calculate the corrected p value using permutation to estimate the null distribution of
    minimum p values.

    :param data: data input containing genotype array, bim, gene count data, gene meta data (tss), and covariates
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param test: approach for hypothesis test, default to ScoreTest()
    :param append_intercept: `True` if want to append intercept, `False` otherwise
    :param standardize: `True` if want to standardize covariates data
    :param seed: seed for permutation, default to 123
    :param window: window size (bp) of one side for cis scope, default to 500000,
        meaning in total 1Mb from left to right
    :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie, `False` if pick the first occurrence
        default to `False`
    :param sig_level: alpha significance level at each SNP level (not used), default to 0.05
    :param fdr_level: FDR level specified for across genes, default to 0.05 (not used if compute_qvalue=`False`)
    :param pi0: specified probability of null (optional) when compute_qvalue=`True`
    :param qvalue_lambda: an array of lambda value to fit a smooth spline (Optional)
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param n_perm: number of permutation to estimate min p distribution for each gene using beta approximation approach
        default to 1000
    :param compute_qvalue: `True` if add qvalue for genes, default to `False`
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param log: logger for QTL progress
    :return: data frame of QTL mapping results
    """
    if log is None:
        log = get_log()

    key = rdm.key(seed)
    results = _ResultsAggregator()
    for i, cis_data in enumerate(data.iter_cis(window)):
        gene_name = cis_data.gene_name
        chrom = cis_data.chrom
        lstart = cis_data.start
        rend = cis_data.end

        # skip if no cis SNPs found
        if cis_data.num_snps == 0:
            if verbose:
                log.warning(f"No cis-SNPs found for {gene_name} over region {chrom}:{lstart}-{rend}. Skipping.")
            continue

        # skip if no variation in y
        y_var = jnp.var(cis_data.y)
        if y_var == 0 or jnp.isnan(y_var):
            if verbose:
                log.warning(f"No variation found in for {gene_name}. Skipping.")
            continue

        if verbose:
            log.info(f"Performing cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        if mode == "cis":
            key, p_key, s_key = rdm.split(key, 3)
            test_result, perm_result = map_cis_single(
                cis_data.X, cis_data.G, cis_data.y, cis_data.offset, test, perm_test, p_key, sig_level
            )
            result = _process_cis_result(cis_data, test_result, perm_result, s_key)
            results.add_row(result)
        else:
            test_result = eqx.filter_jit(test)(cis_data.X, cis_data.G, cis_data.y, cis_data.offset)
            # result = _process_nominal_result(cis_data, test_result)
            # results.add_df(result)

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # clear caches every 50 genes
        if (i + 1) % 50 == 0:
            if verbose:
                log.debug("Clearing JAX JIT-caches")
            jax.clear_caches()  # clear up caches

    result_df = results.to_df()

    # if we're in cis-mode (ie take only top hits), compute q-values for FDR correction
    if mode == "cis":
        log.info("Computing q-values")
        p_values = result_df.get_column("pval_adj").to_numpy()
        q_values, pi0 = calculate_qval(p_values, log, pi0, lam=qvalue_lambda)
        result_df = result_df.with_columns(pl.Series("qval", q_values))
        num_sig = (q_values <= fdr_level).sum()
        p_thold = estimate_sig_threshold(q_values, p_values, fdr_level)

        log.info(f"  * Proportion of significant phenotypes (1-pi0): {1 - pi0:.2f}")
        log.info(f"  * QTL phenotypes @ FDR {fdr_level:.3f}: {num_sig}")
        log.info(f"  * min p-value threshold @ FDR {fdr_level}: {p_thold:.3e}")
        if isinstance(perm_test, BetaPermutation):
            # could this update be done for ACAT also?
            from scipy import stats

            beta_shape1 = result_df["beta_shape1"].values
            beta_shape2 = result_df["beta_shape2"].values
            result_df["pval_nominal_threshold"] = stats.beta.ppf(p_thold, beta_shape1, beta_shape2)

    return result_df


@eqx.filter_jit
def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    test: HypothesisTest,
    perm: AbstractPermutation,
    key: PRNGKeyArray,
    sig_level: float = 0.05,
) -> tuple[TestResult, PermutationResult]:
    """Fit GLM, perform hypothesis testing for each variant, and then compute gene-level adjustment of p-values"""
    test_result = test(X, G, y, offset)
    perm_result = perm(X, G, y, offset, test_result, test, key, sig_level)

    return test_result, perm_result


def _process_cis_result(cis_data, test_result, perm_result, key):
    """Process the results for a gene under the cis-scan and format for output"""

    # get info at lead hit, and lead snp
    minp = jnp.nanmin(test_result.p)
    ties_ind = jnp.argwhere(test_result.p == minp).squeeze()  # why does this add extra axis?
    if ties_ind.ndim > 0:
        vdx = rdm.choice(key, ties_ind, replace=False)
    else:
        vdx = ties_ind

    adj_pvalue, aux = perm_result

    # this is kind of hacky but if aux is not None we did a beta-approximation
    if aux is not None:
        beta_params, nc_estimate, opt_status = aux
        beta_k = float(beta_params.k)
        beta_n = float(beta_params.n)
        beta_converged = beta_params.converged
        opt_status = bool(opt_status)
        nc_estimate = float(nc_estimate)
        lead_adj_pvalue = float(adj_pvalue[vdx])
        method = "BETA"
    else:
        beta_k = float("nan")
        beta_n = float("nan")
        beta_converged = True
        opt_status = True
        nc_estimate = float("nan")
        lead_adj_pvalue = float(adj_pvalue)
        method = "ACAT"

    snp = cis_data.get_snp_info(int(vdx))
    result = {
        "phenotype_id": cis_data.gene_name,
        "chrom": cis_data.chrom,
        "num_var": cis_data.num_snps,
        "variant_id": snp.id,
        "a1": snp.a1,
        "a0": snp.a0,
        "pos": snp.pos,
        "tss_distance": snp.tss_distance,
        "ma_count": snp.ma_count,
        "af": snp.af,
        "beta_shape1": beta_k,
        "beta_shape2": beta_n,
        "beta_converged": beta_converged,
        "opt_status": opt_status,
        "nc_estimate": nc_estimate,
        "effect": float(test_result.beta[vdx]),
        "effect_se": float(test_result.se[vdx]),
        "pval_nominal": float(test_result.p[vdx]),
        "pval_adj": lead_adj_pvalue,
        "adj_method": method,
        "alpha": float(test_result.alpha[vdx]),
        "model_converged": bool(test_result.converged[vdx]),
    }

    return result
