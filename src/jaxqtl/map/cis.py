from logging import Logger
from typing import Any, Literal

import numpy as np
import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..distribution import NegativeBinomial
from ..hypothesis import AbstractAggregateTest, AbstractHypothesisTest, PermutationResult, TestResult
from ..log import get_log
from .data import CisData, ReadyDataState


# Keep Parquet row groups reasonably sized without rebuilding genome-wide result tables.
_MAP_CIS_BATCH_ROWS = 10_000
_NO_FINITE_PVALUES = "no_finite_pvalues"


def map_cis(
    data: ReadyDataState,
    snp_test: AbstractHypothesisTest,
    gene_test: AbstractAggregateTest,
    mode: Literal["cis", "nominal"] = "cis",
    window: int = 500_000,
    verbose: bool = True,
    log: Logger | None = None,
    seed: int = 123,
):
    r"""Yield cis or nominal eQTL mapping results in bounded DataFrame chunks.

    **Arguments:**

    - `data`: Genotype/expression/covariate bundle aligned on IID.
    - `snp_test`: Hypothesis test to apply per variant (score or Wald).
    - `gene_test`: Gene-level p-value aggregation for cis mode. It is ignored in
      nominal mode.
    - `mode`: `"cis"` (per-gene lead SNP with multiple testing adjustment) or `"nominal"` (all variant stats).
    - `window`: Cis window size in base pairs on each side of a gene TSS/stop.
    - `verbose`: Whether to emit progress logging.
    - `log`: Optional logger to use; defaults to module logger.
    - `seed`: PRNG seed for permutation and tie-breaking.

    **Returns:**

    An iterator of `pl.DataFrame` chunks. Each chunk may contain one or more genes.

    **Failure Modes:**

    Genes with no variants in the requested window or no phenotype variance are
    skipped. If every gene is skipped, the iterator yields one empty frame with the
    mode-specific schema.

    In cis mode, a tested gene with no finite SNP-level p-values is retained as an invalid result row. Its lead and
    association fields are null, `result_valid` is false, and `failure_reason` is `"no_finite_pvalues"`.

    **Raises:**

    - `ValueError`: If `mode` is not `"cis"` or `"nominal"`.
    """
    if log is None:
        log = get_log()

    if mode not in ["cis", "nominal"]:
        raise ValueError("`mode` must be 'cis' or 'nominal'")

    # Only cis mode needs PRNG state: permutations and lead-SNP tie breaking both consume keys.
    key = rdm.key(seed)
    include_nb_alpha = isinstance(snp_test.model.family, NegativeBinomial)
    pending = []
    pending_rows = 0
    yielded = False
    for i, cis_data in enumerate(data.iter_cis(window)):
        gene_name = cis_data.gene_name
        chrom = cis_data.chrom
        lstart = cis_data.start
        rend = cis_data.end

        if _should_skip_cis_data(cis_data, verbose, log):
            continue

        if verbose:
            log.info(f"Performing cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        if mode == "cis":
            # cis mode tests variants, then computes a gene-level calibrated p-value.
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
            result_record = _process_cis_result(cis_data, test_result, perm_result, s_key)
            if not result_record["result_valid"]:
                log.warning(
                    f"No finite p-values for {gene_name} over region {chrom}:{lstart}-{rend}; "
                    "emitting an invalid result row."
                )

            result_schema = _empty_cis_columns(gene_test)
            if not include_nb_alpha:
                result_record.pop("nb_alpha")
                result_schema.pop("nb_alpha")
            result = pl.DataFrame([result_record], schema=result_schema)
        else:
            test_result = eqx.filter_jit(snp_test)(cis_data.X, cis_data.G, cis_data.y, cis_data.offset)
            result = _process_nominal_result(cis_data, test_result)
            # Keep the output schema model-specific. Non-NB tests carry a constant placeholder alpha.
            if not include_nb_alpha:
                result = result.drop("nb_alpha")

        pending.append(result)
        pending_rows += result.height

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # Repeated per-gene shapes can leave stale compiled functions resident after many genes.
        if (i + 1) % 50 == 0:
            if verbose:
                log.debug("Clearing JAX JIT-caches")
            jax.clear_caches()

        # Flush by row count rather than gene count: cis emits one row per gene, while nominal emits one row per SNP.
        if pending_rows >= _MAP_CIS_BATCH_ROWS:
            yielded = True
            yield pl.concat(pending, how="vertical")
            pending = []
            pending_rows = 0

    if pending:
        yielded = True
        yield pl.concat(pending, how="vertical")

    if not yielded:
        log.warning("All genes were skipped!")
        yield _empty_result_frame(mode, snp_test, gene_test)


def _should_skip_cis_data(cis_data: CisData, verbose: bool, log: Logger) -> bool:
    gene_name = cis_data.gene_name
    chrom = cis_data.chrom
    lstart = cis_data.start
    rend = cis_data.end

    if cis_data.num_snps == 0:
        if verbose:
            log.warning(f"No cis-SNPs found for {gene_name} over region {chrom}:{lstart}-{rend}. Skipping.")
        return True

    y_var = jnp.var(cis_data.y)
    if y_var == 0 or jnp.isnan(y_var):
        if verbose:
            log.warning(f"No variation found in for {gene_name}. Skipping.")
        return True

    return False


def _empty_result_frame(mode: Literal["cis", "nominal"], snp_test, gene_test) -> pl.DataFrame:
    columns = _empty_cis_columns(gene_test) if mode == "cis" else _empty_nominal_columns()
    if not isinstance(snp_test.model.family, NegativeBinomial):
        columns.pop("nb_alpha")

    return pl.DataFrame(schema=columns)


def _empty_nominal_columns() -> dict[str, Any]:
    return {
        "phenotype_id": pl.Utf8,
        "chrom": pl.Utf8,
        "snp": pl.Utf8,
        "pos": pl.Int64,
        "a1": pl.Utf8,
        "a0": pl.Utf8,
        "tss_distance": pl.Int64,
        "af": pl.Float64,
        "ma_count": pl.Int64,
        "beta": pl.Float64,
        "se": pl.Float64,
        "pvalue": pl.Float64,
        "nb_alpha": pl.Float64,
        "model_converged": pl.Boolean,
    }


def _empty_cis_columns(gene_test) -> dict[str, Any]:
    columns = {
        "phenotype_id": pl.Utf8,
        "chrom": pl.Utf8,
        "num_var": pl.Int64,
        "snp": pl.Utf8,
        "a1": pl.Utf8,
        "a0": pl.Utf8,
        "pos": pl.Int64,
        "tss_distance": pl.Int64,
        "af": pl.Float64,
        "ma_count": pl.Int64,
        "shape1": pl.Float64,
        "shape2": pl.Float64,
        "nc_estimate": pl.Float64,
        "perm_converged": pl.Boolean,
        "beta": pl.Float64,
        "se": pl.Float64,
        "pvalue": pl.Float64,
        "pvalue_adj": pl.Float64,
        "adj_method": pl.Utf8,
        "nb_alpha": pl.Float64,
        "model_converged": pl.Boolean,
        "result_valid": pl.Boolean,
        "failure_reason": pl.Utf8,
    }

    if getattr(gene_test, "name", None) == "acat":
        for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
            columns.pop(beta_perm_col)

    return columns


@eqx.filter_jit
def map_cis_single(
    X: Array,
    G: Array,
    y: Array,
    offset: Array,
    snp_test: AbstractHypothesisTest,
    gene_test: AbstractAggregateTest,
    key: PRNGKeyArray,
) -> tuple[TestResult, PermutationResult]:
    r"""Test variants and compute a gene-level adjusted p-value.

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `G`: Genotype matrix with shape `(n, m)`.
    - `y`: Response vector with shape `(n,)`.
    - `offset`: Offset vector broadcastable to `y`.
    - `snp_test`: Hypothesis test producing per-variant statistics.
    - `gene_test`: Gene-level p-value aggregation method.
    - `key`: PRNG key for permutation randomness.

    **Returns:**

    A tuple `(test_result, aggregate_result)` containing per-variant statistics and
    the gene-level p-value with method-specific auxiliary diagnostics.
    """
    test_result = snp_test(X, G, y, offset)
    perm_result = gene_test(X, G, y, offset, test_result, snp_test, key)

    return test_result, perm_result


def _process_cis_result(
    cis_data: CisData,
    test_result: TestResult,
    perm_result: tuple[Array, Any],
    key: PRNGKeyArray,
):
    """Process the results for a gene under the cis-scan and format for output."""

    pvalues = np.asarray(test_result.p)
    finite_idx = np.flatnonzero(np.isfinite(pvalues))
    adj_pvalue, aux = perm_result

    if finite_idx.size == 0:
        method = "BETA" if aux is not None else "ACAT"
        result = {
            "phenotype_id": cis_data.gene_name,
            "chrom": cis_data.chrom,
            "num_var": cis_data.num_snps,
            "snp": None,
            "a1": None,
            "a0": None,
            "pos": None,
            "tss_distance": None,
            "af": None,
            "ma_count": None,
            "shape1": None,
            "shape2": None,
            "nc_estimate": None,
            "perm_converged": None,
            "beta": None,
            "se": None,
            "pvalue": None,
            "pvalue_adj": None,
            "adj_method": method,
            "nb_alpha": None,
            "model_converged": None,
            "result_valid": False,
            "failure_reason": _NO_FINITE_PVALUES,
        }
        if aux is None:
            for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
                result.pop(beta_perm_col)
        return result

    finite_pvalues = pvalues[finite_idx]
    minp = finite_pvalues.min()
    ties_ind = finite_idx[finite_pvalues == minp]
    if ties_ind.size > 1:
        vdx_int = int(rdm.choice(key, jnp.asarray(ties_ind), replace=False))
    else:
        vdx_int = int(ties_ind[0])

    adj_pvalue = jnp.asarray(adj_pvalue)

    # this is kind of hacky but if aux is not None we did a beta-approximation
    if aux is not None:
        beta_params, nc_estimate, opt_status = aux
        shape_k = float(beta_params.k)
        shape_n = float(beta_params.n)
        nc_estimate = float(nc_estimate)
        perm_converged = bool(beta_params.converged) and bool(opt_status)
        lead_adj_pvalue = float(adj_pvalue[vdx_int])
        method = "BETA"
    else:
        shape_k = float("nan")
        shape_n = float("nan")
        nc_estimate = float("nan")
        perm_converged = True
        lead_adj_pvalue = float(adj_pvalue)
        method = "ACAT"

    snp = cis_data.get_snp_info(vdx_int)
    if jnp.ndim(test_result.disp) > 0:
        nb_alpha = float(test_result.disp[vdx_int])
    else:
        nb_alpha = float(test_result.disp)

    if jnp.ndim(test_result.converged) > 0:
        glm_converged = bool(test_result.converged[vdx_int])
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
        "beta": float(test_result.beta[vdx_int]),
        "se": float(test_result.se[vdx_int]),
        "pvalue": float(test_result.p[vdx_int]),
        "pvalue_adj": lead_adj_pvalue,
        "adj_method": method,
        "nb_alpha": nb_alpha,
        "model_converged": glm_converged,
        "result_valid": True,
        "failure_reason": None,
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
