from logging import Logger
from typing import Any, Literal

import numpy as np
import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray

from ..families.distribution import NegativeBinomial
from ..infer.aggregate import AbstractAggregateTest, PermutationResult
from ..infer.utils import HypothesisTest, TestResult
from ..io.data import CisData, ReadyDataState
from ..log import get_log


class _ResultsAggregator:
    """Accumulate per-gene or per-variant result frames into a single DataFrame."""

    def __init__(self):
        self.frames: list = []

    def __len__(self) -> int:
        return len(self.frames)

    def add_row(self, row: dict):
        # cheap 1-row DataFrame, but *only* created when needed
        self.frames.append(pl.DataFrame([row]))

    def add_df(self, df: pl.DataFrame):
        self.frames.append(df)

    def to_df(self):
        return pl.concat(self.frames, how="vertical")


def map_cis(
    data: ReadyDataState,
    snp_test: HypothesisTest,
    gene_test: AbstractAggregateTest,
    mode: Literal["cis", "nominal"] = "cis",
    window: int = 500_000,
    verbose: bool = True,
    log: Logger | None = None,
    seed: int = 123,
) -> pl.DataFrame:
    r"""Run cis or nominal eQTL mapping per gene and return summary statistics.

    **Arguments:**

    - `data`: Genotype/expression/covariate bundle aligned on IID.
    - `snp_test`: Hypothesis test to apply per variant (score or Wald).
    - `gene_test`: Gene-level multiple testing adjustment for the cis mode.
    - `mode`: `"cis"` (per-gene lead SNP with multiple testing adjustment) or `"nominal"` (all variant stats).
    - `window`: Cis window size in base pairs on each side of a gene TSS/stop.
    - `verbose`: Whether to emit progress logging.
    - `log`: Optional logger to use; defaults to module logger.
    - `seed`: PRNG seed for permutation and tie-breaking.

    **Returns:**

    A `pl.DataFrame` of concatenated cis or nominal results.
    """
    if log is None:
        log = get_log()

    if mode not in ["cis", "nominal"]:
        raise ValueError("`mode` must be 'cis' or 'nominal'")

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
            # cis-mode tests each variant, and then performs either permutations or ACAT to compute a gene-level
            # calibrated p-value
            key, p_key, s_key = rdm.split(key, 3)
            test_result, perm_result = map_cis_single(
                cis_data.X,
                cis_data.G,
                cis_data.y,
                cis_data.offset,
                snp_test,
                gene_test,
                p_key,
            )
            result = _process_cis_result(cis_data, test_result, perm_result, s_key)
            results.add_row(result)
        else:
            # nominal mode only performs variant-level testing
            test_result = eqx.filter_jit(snp_test)(cis_data.X, cis_data.G, cis_data.y, cis_data.offset)
            result = _process_nominal_result(cis_data, test_result)
            results.add_df(result)

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # clear caches every 50 genes
        if (i + 1) % 50 == 0:
            if verbose:
                log.debug("Clearing JAX JIT-caches")
            jax.clear_caches()  # clear up caches

    if len(results) > 0:
        result_df = results.to_df()

        # if we didn't fit a negative binomial, just drop the alpha column as its const 0
        # its usually a code-smell to refer to chained attributes (ie something.something.something), but w/e
        if not isinstance(snp_test.model.family, NegativeBinomial):
            result_df = result_df.drop("nb_alpha")
    else:
        log.warning("All genes were skipped!")
        result_df = None

    return result_df


@eqx.filter_jit
def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    snp_test: HypothesisTest,
    gene_test: AbstractAggregateTest,
    key: PRNGKeyArray,
) -> tuple[TestResult, PermutationResult]:
    r"""Fit GLM, test variants, and compute gene-level permutation adjustment.

    **Arguments:**

    - `X`: Covariate matrix of shape ``(n, p)``.
    - `G`: Genotype matrix of shape ``(n, m)``.
    - `y`: Response vector of length ``n``.
    - `offset`: Offset vector broadcastable to ``y``.
    - `test`: Hypothesis test callable producing per-variant statistics.
    - `perm`: Aggregate permutation test for multiple-testing correction.
    - `key`: PRNG key for permutation randomness.

    **Returns:**

    Per-variant stats plus permutation-adjusted p-values as a tuple of
        ([`jaxqtl.TestResult`][], [`jaxqtl.PermutationResult`][]).
    """
    test_result = snp_test(X, G, y, offset)
    perm_result = gene_test(X, G, y, offset, test_result, snp_test, key)

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
        shape_k = float(beta_params.k)
        shape_n = float(beta_params.n)
        nc_estimate = float(nc_estimate)
        perm_converged = bool(beta_params.converged) and bool(opt_status)
        lead_adj_pvalue = float(adj_pvalue[vdx])
        method = "BETA"
    else:
        shape_k = float("nan")
        shape_n = float("nan")
        nc_estimate = float("nan")
        perm_converged = True
        lead_adj_pvalue = float(adj_pvalue)
        method = "ACAT"

    snp = cis_data.get_snp_info(vdx)
    if jnp.ndim(test_result.disp) > 0:
        nb_alpha = float(test_result.disp[vdx])
    else:
        nb_alpha = float(test_result.disp)

    if jnp.ndim(test_result.converged) > 0:
        glm_converged = bool(test_result.converged[vdx])
    else:
        glm_converged = bool(test_result.converged)

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
        "shape1": shape_k,
        "shape2": shape_n,
        "nc_estimate": nc_estimate,
        "perm_converged": perm_converged,
        "beta": float(test_result.beta[vdx]),
        "se": float(test_result.se[vdx]),
        "pvalue": float(test_result.p[vdx]),
        "pvalue_adj": lead_adj_pvalue,
        "adj_method": method,
        "nb_alpha": nb_alpha,
        "model_converged": glm_converged,
    }
    # if we did ACAT [we need to make this more robust...], drop the beta-perm related columns to save disk space
    if aux is None:
        for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
            result.pop(beta_perm_col, None)

    return result


def _process_nominal_result(cis_data: CisData, test_result: TestResult) -> pl.DataFrame:
    region_df = cis_data.get_cis_info()

    if jnp.ndim(test_result.disp) > 0:
        nb_alpha = np.asarray(test_result.disp)
    else:
        nb_alpha = np.full_like(test_result.beta, test_result.disp)

    if jnp.ndim(test_result.converged) > 0:
        glm_converged = np.asarray(test_result.converged)
    else:
        glm_converged = np.full_like(test_result.beta, test_result.converged)

    region_df = region_df.with_columns(
        pl.lit(cis_data.gene_name).alias("phenotype_id"),
        pl.Series("beta", np.asarray(test_result.beta)),
        pl.Series("se", np.asarray(test_result.se)),
        pl.Series("pvalue", np.asarray(test_result.p)),
        pl.Series("nb_alpha", nb_alpha),
        pl.Series("model_converged", glm_converged),
    )
    # put pheno id in front
    region_df = region_df.select(pl.col("phenotype_id"), pl.all().exclude("phenotype_id"))
    return region_df
