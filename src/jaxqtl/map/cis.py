from logging import Logger
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray

from ..families.distribution import NegativeBinomial
from ..infer.permutations import AbstractPermutation, PermutationResult
from ..infer.utils import HypothesisTest, TestResult
from ..io.data import CisData, ReadyDataState
from ..log import get_log


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
    window: int = 500_000,
    verbose: bool = True,
    log: Optional[Logger] = None,
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
    :param window: window size (bp) of one side for cis scope, default to 500000,
        meaning in total 1Mb from left to right
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param log: logger for QTL progress
    :param seed: seed for permutation, default to 123
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
                cis_data.X,
                cis_data.G,
                cis_data.y,
                cis_data.offset,
                test,
                perm_test,
                p_key,
            )
            result = _process_cis_result(cis_data, test_result, perm_result, s_key)
            results.add_row(result)
        else:
            test_result = eqx.filter_jit(test)(cis_data.X, cis_data.G, cis_data.y, cis_data.offset)
            result = _process_nominal_result(cis_data, test_result)
            results.add_df(result)

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # clear caches every 50 genes
        if (i + 1) % 50 == 0:
            if verbose:
                log.debug("Clearing JAX JIT-caches")
            jax.clear_caches()  # clear up caches

    result_df = results.to_df()

    # if we didn't fit a negative binomial, just drop the alpha column as its const 0
    # its usually a code-smell to refer to chained attributes (ie something.something.something), but w/e
    if not isinstance(test.model.family, NegativeBinomial):
        result_df = result_df.drop("nb_alpha")

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
) -> tuple[TestResult, PermutationResult]:
    """Fit GLM, perform hypothesis testing for each variant, and then compute gene-level adjustment of p-values"""
    test_result = test(X, G, y, offset)
    perm_result = perm(X, G, y, offset, test_result, test, key)

    return test_result, perm_result


def _process_cis_result(
    cis_data: CisData,
    test_result: TestResult,
    perm_result: tuple[ArrayLike, Any],
    key: PRNGKeyArray,
):
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

    snp = cis_data.get_snp_info(vdx)
    result = {
        "phenotype_id": cis_data.gene_name,
        "chrom": cis_data.chrom,
        "num_var": cis_data.num_snps,
        "snp": snp.id,
        "a1": snp.a1,
        "a0": snp.a0,
        "pos": snp.pos,
        "tss_distance": snp.tss_distance,
        "af": snp.af,
        "ma_count": snp.ma_count,
        "beta_shape1": beta_k,
        "beta_shape2": beta_n,
        "beta_converged": beta_converged,
        "opt_status": opt_status,
        "nc_estimate": nc_estimate,
        "effect": float(test_result.beta[vdx]),
        "effect_se": float(test_result.se[vdx]),
        "pvalue": float(test_result.p[vdx]),
        "pvalue_adj": lead_adj_pvalue,
        "adj_method": method,
        "nb_alpha": float(test_result.alpha[vdx]),
        "model_converged": bool(test_result.converged[vdx]),
    }
    # if we did ACAT [we need to make this more robust...], drop the beta-perm related columns to save disk space
    if aux is None:
        for beta_perm_col in ["beta_shape1", "beta_shape2", "beta_converged", "opt_status", "nc_estimate"]:
            result.pop(beta_perm_col, None)

    return result


def _process_nominal_result(cis_data: CisData, test_result: TestResult) -> pl.DataFrame:
    region_df = cis_data.get_cis_info()
    region_df = region_df.with_columns(
        pl.lit(cis_data.gene_name).alias("phenotype_id"),
        pl.Series("effect", np.asarray(test_result.beta)),
        pl.Series("se", np.asarray(test_result.se)),
        pl.Series("pvalue", np.asarray(test_result.p)),
        pl.Series("nb_alpha", np.asarray(test_result.alpha)),
        pl.Series("model_converged", np.asarray(test_result.converged)),
    )
    # put pheno id in front
    region_df = region_df.select(pl.col("phenotype_id"), pl.all().exclude("phenotype_id"))
    return region_df
