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
    r"""Calculate edgeR-style TMM scaling factors for a gene-by-sample count matrix.

    TMM (trimmed mean of M-values) estimates a sample-specific scaling factor
    from finite log-ratios and average log-expression values. The returned
    factors have geometric mean one. This is the normalization-factor step used
    by [`edger_cpm`][].

    **Arguments:**

    - `counts`: Nonnegative count matrix with shape `(g, n)`, where rows are
      genes and columns are samples. Each sample must have a positive library
      size. Remove genes with zero counts in every sample before calling this
      function.
    - `ref`: Optional zero-based reference-sample column. If `None`, select the
      sample whose upper-quartile normalized expression is closest to the mean.
    - `logratio_trim`: Fraction trimmed from each tail of the log-ratio ranks.
    - `sum_trim`: Fraction trimmed from each tail of the average-expression
      ranks.
    - `acutoff`: Minimum average log-expression retained for factor estimation.

    **Returns:**

    A floating-point JAX array with shape `(n,)` containing one TMM factor per
    sample.

    **Failure Modes:**

    A sample with zero total counts produces nonfinite values. Invalid count
    values, such as negative counts or NaNs, can also produce nonfinite factors.
    """
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
    r"""Return edgeR-style TMM-normalized counts per million.

    The effective library size is the column sum of `counts` multiplied by its
    TMM factor. Set `normalized_lib_sizes=False` to use unadjusted library sizes.

    **Arguments:**

    - `counts`: Nonnegative gene-by-sample count matrix with shape `(g, n)`.
    - `tmm`: Optional TMM factors with shape `(n,)`. If `None`, factors are
      calculated from `counts` with [`edger_calcnormfactors`][].
    - `normalized_lib_sizes`: Whether to apply TMM factors to library sizes.

    **Returns:**

    A floating-point JAX array with shape `(g, n)` of counts per million.

    **Failure Modes:**

    A zero effective library size produces nonfinite values. Rows with zero
    counts in every sample should be removed before factor estimation.
    """
    counts_array = _as_inexact_array(counts)
    library_sizes = jnp.sum(counts_array, axis=0)
    if normalized_lib_sizes:
        factors = edger_calcnormfactors(counts_array) if tmm is None else _as_inexact_array(tmm)
        library_sizes = library_sizes * factors
    return counts_array / library_sizes * 1e6


def inverse_normal_transform(values: Any) -> jax.Array:
    r"""Apply a rank-based inverse-normal transform along the final axis.

    Ties receive their average rank. Each rank is scaled by $n + 1$, where $n$
    is the length of the final axis, then transformed with the standard-normal
    quantile function.

    **Arguments:**

    - `values`: Floating-point or integer array. For an expression matrix with
      shape `(g, n)`, rows are transformed independently across the `n` samples.

    **Returns:**

    A floating-point JAX array with the same shape as `values`.
    """
    values_array = _as_inexact_array(values)
    ranks = jsp_stats.rankdata(values_array, axis=-1, method="average")
    return jsp_stats.norm.ppf(ranks / (values_array.shape[-1] + 1))
