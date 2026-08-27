# pattern: Functional Core

"""JAX-native expression normalization kernels."""

from typing import Any

import jax
import jax.numpy as jnp

from jax.scipy import stats as jsp_stats


def _as_inexact_array(values: Any) -> jax.Array:
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array.astype(jnp.float32)
    return array


def edger_calcnormfactors(
    counts: Any,
    ref: int | None = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    acutoff: float = -1e10,
) -> jax.Array:
    """Calculate edgeR-style TMM scaling factors for a gene-by-sample matrix."""
    counts_array = _as_inexact_array(counts)
    library_sizes = jnp.sum(counts_array, axis=0)
    normalized_counts = counts_array / library_sizes

    if ref is None:
        upper_quartiles = jnp.percentile(normalized_counts, 75.0, axis=0)
        reference_index = jnp.argmin(jnp.abs(upper_quartiles - jnp.mean(upper_quartiles)))
    else:
        reference_index = jnp.asarray(ref)

    reference_profile = normalized_counts[:, reference_index]
    log_ratios = jnp.log2(normalized_counts / reference_profile[:, None])
    average_log_expression = 0.5 * (jnp.log2(normalized_counts) + jnp.log2(reference_profile)[:, None])
    variances = (library_sizes - counts_array) / (library_sizes * counts_array)
    variances = variances + variances[:, reference_index, None]

    def factor_for_sample(
        log_ratios_column: jax.Array,
        average_log_expression_column: jax.Array,
        variances_column: jax.Array,
    ) -> jax.Array:
        finite = (
            jnp.isfinite(log_ratios_column)
            & jnp.isfinite(average_log_expression_column)
            & (average_log_expression_column > acutoff)
        )
        n_finite = jnp.sum(finite)

        def nonempty_factor(_: None) -> jax.Array:
            ranked_log_ratios = jsp_stats.rankdata(
                jnp.where(finite, log_ratios_column, jnp.inf),
                method="average",
            )
            ranked_expression = jsp_stats.rankdata(
                jnp.where(finite, average_log_expression_column, jnp.inf),
                method="average",
            )
            n_finite_float = n_finite.astype(counts_array.dtype)
            lower_log_ratio = jnp.floor(n_finite_float * logratio_trim) + 1.0
            upper_log_ratio = n_finite_float + 1.0 - lower_log_ratio
            lower_expression = jnp.floor(n_finite_float * sum_trim) + 1.0
            upper_expression = n_finite_float + 1.0 - lower_expression
            keep = (
                finite
                & (ranked_log_ratios >= lower_log_ratio)
                & (ranked_log_ratios <= upper_log_ratio)
                & (ranked_expression >= lower_expression)
                & (ranked_expression <= upper_expression)
            )
            numerator = jnp.sum(jnp.where(keep, log_ratios_column / variances_column, 0.0))
            denominator = jnp.sum(jnp.where(keep, 1.0 / variances_column, 0.0))
            safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
            return jnp.where(denominator > 0.0, 2.0 ** (numerator / safe_denominator), 1.0)

        return jax.lax.cond(
            n_finite > 0,
            nonempty_factor,
            lambda _: jnp.array(1.0, dtype=counts_array.dtype),
            operand=None,
        )

    factors = jax.vmap(factor_for_sample, in_axes=1)(log_ratios, average_log_expression, variances)
    return factors / jnp.exp(jnp.mean(jnp.log(factors)))


def edger_cpm(
    counts: Any,
    tmm: jax.Array | None = None,
    normalized_lib_sizes: bool = True,
) -> jax.Array:
    """Return edgeR-style TMM-normalized counts per million."""
    counts_array = _as_inexact_array(counts)
    library_sizes = jnp.sum(counts_array, axis=0)
    if normalized_lib_sizes:
        factors = edger_calcnormfactors(counts_array) if tmm is None else _as_inexact_array(tmm)
        library_sizes = library_sizes * factors
    return counts_array / library_sizes * 1e6


def inverse_normal_transform(values: Any) -> jax.Array:
    """Apply a rank-based inverse-normal transform along the final axis."""
    values_array = _as_inexact_array(values)
    ranks = jsp_stats.rankdata(values_array, axis=-1, method="average")
    return jsp_stats.norm.ppf(ranks / (values_array.shape[-1] + 1))
