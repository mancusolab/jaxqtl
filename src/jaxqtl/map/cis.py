# pattern: Functional Core

from logging import Logger
from typing import Any, Literal

import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..distribution import NegativeBinomial
from ..hypothesis import (
    AbstractAggregateTest,
    AbstractHypothesisTest,
    TestResult,
)
from ..log import get_log
from ._scan import (
    AssociationScan,
    full_scan as map_cis_single,  # noqa: F401 -- preserve full-window entry point
    select_lead_variant,
)
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
    *,
    tss_centered: bool = False,
):
    r"""Yield cis or nominal eQTL mapping results in bounded DataFrame chunks.

    **Arguments:**

    - `data`: Genotype/expression/covariate bundle aligned on IID.
    - `snp_test`: Hypothesis test to apply per variant (score or Wald).
    - `gene_test`: Gene-level p-value aggregation for cis mode. It is ignored in
      nominal mode.
    - `mode`: `"cis"` (per-gene lead SNP with multiple testing adjustment) or `"nominal"` (all variant stats).
    - `window`: Cis window size in base pairs.
    - `verbose`: Whether to emit progress logging.
    - `log`: Optional logger to use; defaults to module logger.
    - `seed`: PRNG seed for permutation and tie-breaking.
    - `tss_centered`: Center the window on the TSS when true. Otherwise, extend
      the window upstream of the TSS and downstream of the TES.

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
    execution = AssociationScan(snp_test, gene_test if mode == "cis" else None)
    blocked = execution.blocked
    cache_clear_interval = execution.cache_clear_interval
    pending = []
    pending_rows = 0
    yielded = False
    options = {"tss_centered": True} if tss_centered else {}
    if blocked:
        options["host_genotypes"] = True
    cis_iterator = data.iter_cis(window, **options)
    for i, cis_data in enumerate(cis_iterator):
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
            test_result, perm_result = execution.run(cis_data.X, cis_data.G, cis_data.y, cis_data.offset, p_key)
            result_record = _process_cis_result(
                cis_data, test_result, perm_result, s_key, gene_test=gene_test, host_genotypes=blocked
            )
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
            test_result, _ = execution.observed(cis_data.X, cis_data.G, cis_data.y, cis_data.offset)
            result = _process_nominal_result(cis_data, test_result)
            # Keep the output schema model-specific. Non-NB tests carry a constant placeholder alpha.
            if not include_nb_alpha:
                result = result.drop("nb_alpha")

        pending.append(result)
        pending_rows += result.height

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # Repeated per-gene shapes can leave stale compiled functions resident after many genes.
        if cache_clear_interval is not None and (i + 1) % cache_clear_interval == 0:
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
        "negloglikelihood": pl.Float64,
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
        "negloglikelihood": pl.Float64,
        "model_converged": pl.Boolean,
        "result_valid": pl.Boolean,
        "failure_reason": pl.Utf8,
    }

    if not gene_test.has_calibration:
        for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
            columns.pop(beta_perm_col)

    return columns


def _process_cis_result(
    cis_data: CisData,
    test_result: TestResult,
    perm_result: tuple[Array, Any],
    key: PRNGKeyArray,
    *,
    gene_test: AbstractAggregateTest,
    host_genotypes: bool = False,
):
    """Process the results for a gene under the cis-scan and format for output."""

    # Output formatting is a host boundary: indexing device arrays here would
    # compile a gather for every window length, defeating fixed-block scoring.
    test_result = jax.device_get(test_result)
    vdx_int = select_lead_variant(test_result.p, key)
    adj_pvalue, aux = perm_result

    if vdx_int is None:
        method = gene_test.adjustment_method
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
            "negloglikelihood": None,
            "model_converged": None,
            "result_valid": False,
            "failure_reason": _NO_FINITE_PVALUES,
        }
        if not gene_test.has_calibration:
            for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
                result.pop(beta_perm_col)
        return result

    adj_pvalue = jax.device_get(adj_pvalue)

    if gene_test.has_calibration:
        beta_params, nc_estimate, opt_status = aux
        shape_k = float(beta_params.k)
        shape_n = float(beta_params.n)
        nc_estimate = float(nc_estimate)
        perm_converged = bool(beta_params.converged) and bool(opt_status)
        lead_adj_pvalue = float(adj_pvalue[vdx_int])
        method = gene_test.adjustment_method
    else:
        shape_k = float("nan")
        shape_n = float("nan")
        nc_estimate = float("nan")
        perm_converged = True
        lead_adj_pvalue = float(adj_pvalue)
        method = gene_test.adjustment_method

    if host_genotypes:
        # Blocked tests use host-prepared blocks. Slice the selected
        # genotype here without specializing JAX on window width. Other paths
        # keep device slicing, avoiding a new full-window transfer from accelerators.
        cis_data = eqx.tree_at(lambda data: data.G, cis_data, jax.device_get(cis_data.G))
    snp = cis_data.get_snp_info(vdx_int)
    if jnp.ndim(test_result.disp) > 0:
        nb_alpha = float(test_result.disp[vdx_int])
    else:
        nb_alpha = float(test_result.disp)

    if jnp.ndim(test_result.converged) > 0:
        glm_converged = bool(test_result.converged[vdx_int])
    else:
        glm_converged = bool(test_result.converged)

    if jnp.ndim(test_result.negloglikelihood) > 0:
        negloglikelihood = float(test_result.negloglikelihood[vdx_int])
    else:
        negloglikelihood = float(test_result.negloglikelihood)

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
        "negloglikelihood": negloglikelihood,
        "model_converged": glm_converged,
        "result_valid": True,
        "failure_reason": None,
    }
    # Aggregation metadata defines the output schema explicitly.
    if not gene_test.has_calibration:
        for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
            result.pop(beta_perm_col, None)

    return result


def _process_nominal_result(cis_data: CisData, test_result: TestResult) -> pl.DataFrame:
    region_df = cis_data.get_cis_info()

    # Polars broadcasts scalar fit metadata; host vectors retain their dtype.
    result = jax.device_get(test_result)
    columns = []
    for name, values in (
        ("beta", result.beta),
        ("se", result.se),
        ("pvalue", result.p),
        ("nb_alpha", result.disp),
        ("negloglikelihood", result.negloglikelihood),
        ("model_converged", result.converged),
    ):
        if jnp.ndim(values) == 0:
            scalar = values.item() if hasattr(values, "item") else values
            columns.append(pl.lit(scalar).alias(name))
        else:
            columns.append(pl.Series(name, values))
    region_df = region_df.with_columns(pl.lit(cis_data.gene_name).alias("phenotype_id"), *columns)
    # put pheno id in front
    region_df = region_df.select(pl.col("phenotype_id"), pl.all().exclude("phenotype_id"))
    return region_df
