# pattern: Imperative Shell
"""Orchestrate cis scans, select gene-level leads, and format tabular output."""

from logging import Logger
from typing import Any, Literal

import polars as pl

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..distribution import NegativeBinomial
from ..hypothesis import (
    AbstractAggregateTest,
    AbstractHypothesisTest,
    ACAT,
    BetaPermutation,
    PermutationReference,
    TestResult,
)
from ..log import get_log
from ._scan import (
    _HostBuffers,
    AssociationScan,
    full_scan,
)
from .data import CisData, ReadyDataState


# Keep Parquet row groups reasonably sized without rebuilding genome-wide result tables.
_MAP_CIS_BATCH_ROWS = 10_000
_NO_FINITE_PVALUES = "no_finite_pvalues"


@eqx.filter_jit
def _lead_candidates(pvalues, width):
    valid = (jnp.arange(pvalues.shape[0]) < width) & jnp.isfinite(pvalues)
    minimum = jnp.min(jnp.where(valid, pvalues, jnp.inf))
    return minimum, valid & (pvalues == minimum)


def select_lead_variant(pvalues: ArrayLike, key: PRNGKeyArray) -> int | None:
    """Return the index of a minimum finite p-value, or None if none are finite.

    Ties are sampled uniformly using `key`. Fixed blocks avoid compiling for
    each gene's variant count; only candidate indices are collected on the host.
    Different tie counts can still specialize the random-choice kernel.
    """
    minimum = float("inf")
    indices = []
    for start, stop, block in _HostBuffers(pvalues).blocks(2048):
        value, mask = jax.device_get(_lead_candidates(block, jnp.asarray(stop - start)))
        value = float(value)
        if value < minimum:
            minimum, indices = value, []
        if value == minimum:
            indices.extend(start + i for i, selected in enumerate(mask.tolist()) if selected)
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    return int(rdm.choice(key, jnp.asarray(indices), replace=False))


@eqx.filter_jit
def _finalize(aggregation, state, reference):
    return aggregation.finalize(state, reference)


@eqx.filter_jit
def _finalize_lead(aggregation, result, reference, valid):
    statistic = aggregation.statistic(result)
    return aggregation.finalize(jnp.where(valid, statistic, jnp.nan), reference)


def _select_variant(result: TestResult, index: int) -> TestResult:
    """Slice host result arrays while retaining shared scalar fit metadata."""
    return jax.tree.map(lambda value: value if jnp.ndim(value) == 0 else value[index], result)


def _finish_cis_scan(aggregation, result, state, reference, lead_key):
    """Select one lead for both gene-level finalization and output."""
    lead = select_lead_variant(result.p, lead_key)
    if isinstance(aggregation, BetaPermutation):
        # Select on the host: device indexing would compile for each window width.
        host_result = jax.device_get(result)
        # A missing lead uses a valid placeholder; the compiled finalizer replaces its statistic with NaN.
        selected = jax.device_put(_select_variant(host_result, 0 if lead is None else lead))
        adjusted = _finalize_lead(aggregation, selected, reference, jnp.asarray(lead is not None))
    else:
        adjusted = _finalize(aggregation, state, reference)
    return result, adjusted, lead


def _run_cis_scan(execution, X, G, y, offset, key, lead_key=None):
    """Execute a gene and return its scalar adjusted p-value and selected lead."""
    aggregation = execution.aggregation
    if aggregation is None:
        raise ValueError("gene-level scans require an aggregation")
    if not execution.blocked:
        result, state, reference = full_scan(X, G, y, offset, execution.test, aggregation, key)
    else:
        permutation = isinstance(aggregation, BetaPermutation)
        result, state = execution.observed(X, G, y, offset, reduction=None if permutation else aggregation)
        reference = None
        if permutation:
            maxima = execution.permutation_maxima(X, G, y, offset, key)
            reference = PermutationReference(maxima, X.shape[0] - X.shape[1] - 1)
    return _finish_cis_scan(aggregation, result, state, reference, key if lead_key is None else lead_key)


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    snp_test: AbstractHypothesisTest,
    gene_test: AbstractAggregateTest,
    key: PRNGKeyArray,
) -> tuple[TestResult, tuple[Array, Any]]:
    """Return variant test results and one gene-level p-value with diagnostics.

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `G`: Genotype matrix with shape `(n, m)`, with at least one variant.
    - `y`: Outcome vector with shape `(n,)`.
    - `offset`: Scalar offset or vector with shape `(n,)`.
    - `snp_test`: Score, SPA, or Wald hypothesis test.
    - `gene_test`: Gene-level aggregation method.
    - `key`: Key used for permutations and random lead-SNP tie breaking.

    **Returns:**

    `(test_result, (gene_pvalue, diagnostics))`. The gene p-value is scalar for
    both ACAT and BetaPermutation. BetaPermutation returns a fitted calibration;
    ACAT returns `None` for diagnostics.

    Full-window numerical kernels are compiled; this host orchestration wrapper
    is not JIT-transformable. Use `map_cis` for fixed-block scans over many genes.
    Apply `BetaPermutation.adjust` separately to calibrate additional SNP statistics.
    """
    result, state, reference = full_scan(X, G, y, offset, snp_test, gene_test, key)
    result, adjusted, _ = _finish_cis_scan(gene_test, result, state, reference, key)
    return result, adjusted


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
    pending = []
    pending_rows = 0
    yielded = False
    options = {"tss_centered": True} if tss_centered else {}
    if blocked:
        options["host_genotypes"] = True

    cis_iterator = data.iter_cis(window, **options)
    for cis_data in cis_iterator:
        gene_name = cis_data.gene_name
        chrom = cis_data.chrom
        start = cis_data.start
        end = cis_data.end

        if _should_skip_cis_data(cis_data, verbose, log):
            continue

        if verbose:
            log.info(f"Performing cis-qtl scan for {gene_name} over region {chrom}:{start}-{end}")

        if mode == "cis":
            # cis mode tests variants, then computes a gene-level calibrated p-value.
            key, p_key, s_key = rdm.split(key, 3)
            test_result, gene_result, lead = _run_cis_scan(
                execution, cis_data.X, cis_data.G, cis_data.y, cis_data.offset, p_key, s_key
            )
            result_record = _process_cis_result(
                cis_data, test_result, gene_result, lead, gene_test=gene_test, host_genotypes=blocked
            )
            if not result_record["result_valid"]:
                log.warning(
                    f"No finite p-values for {gene_name} over region {chrom}:{start}-{end}; "
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
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{start}-{end}")

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
    start = cis_data.start
    end = cis_data.end

    if cis_data.num_snps == 0:
        if verbose:
            log.warning(f"No cis-SNPs found for {gene_name} over region {chrom}:{start}-{end}. Skipping.")
        return True

    y_var = jnp.var(cis_data.y)
    if y_var == 0 or jnp.isnan(y_var):
        if verbose:
            log.warning(f"No variation found for {gene_name}. Skipping.")
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

    if not isinstance(gene_test, BetaPermutation):
        for beta_perm_col in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
            columns.pop(beta_perm_col)

    return columns


def _process_cis_result(
    cis_data: CisData,
    test_result: TestResult,
    gene_result: tuple[Array, Any],
    lead: int | None,
    *,
    gene_test: AbstractAggregateTest,
    host_genotypes: bool = False,
):
    """Format one gene using the lead index already selected by cis orchestration."""

    if isinstance(gene_test, ACAT):
        method = "ACAT"
    elif isinstance(gene_test, BetaPermutation):
        method = "BETA"
    else:
        raise TypeError(f"unsupported aggregation for cis output: {type(gene_test).__name__}")

    # Output formatting is a host boundary: indexing device arrays here would
    # compile a gather for every window length, defeating fixed-block scoring.
    test_result = jax.device_get(test_result)
    adj_pvalue, aux = gene_result

    if lead is None:
        # Derive null fields from the output schema so invalid and valid rows stay aligned.
        result = dict.fromkeys(_empty_cis_columns(gene_test))
        result.update(
            phenotype_id=cis_data.gene_name,
            chrom=cis_data.chrom,
            num_var=cis_data.num_snps,
            adj_method=method,
            result_valid=False,
            failure_reason=_NO_FINITE_PVALUES,
        )
        return result

    lead_adj_pvalue = float(jax.device_get(adj_pvalue))

    if host_genotypes:
        # Blocked tests use host-prepared blocks. Slice the selected
        # genotype here without specializing JAX on window width. Other paths
        # keep device slicing, avoiding a new full-window transfer from accelerators.
        cis_data = eqx.tree_at(lambda data: data.G, cis_data, jax.device_get(cis_data.G))
    snp = cis_data.get_snp_info(lead)
    selected = _select_variant(test_result, lead)

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
        "beta": float(selected.beta),
        "se": float(selected.se),
        "pvalue": float(selected.p),
        "pvalue_adj": lead_adj_pvalue,
        "adj_method": method,
        "nb_alpha": float(selected.disp),
        "negloglikelihood": float(selected.negloglikelihood),
        "model_converged": bool(selected.converged),
        "result_valid": True,
        "failure_reason": None,
    }
    if isinstance(gene_test, BetaPermutation):
        beta_params, reference_estimate, reference_converged = aux
        result.update(
            shape1=float(beta_params.k),
            shape2=float(beta_params.n),
            nc_estimate=float(reference_estimate),
            perm_converged=bool(beta_params.converged) and bool(reference_converged),
        )

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
    # Keep phenotype identifiers first in the exported table.
    region_df = region_df.select(pl.col("phenotype_id"), pl.all().exclude("phenotype_id"))
    return region_df
