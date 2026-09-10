# pattern: Functional Core
"""Numerical reductions and calibration for scalar gene-level association tests."""

from abc import abstractmethod
from typing import ClassVar, Generic, NamedTuple, TypeAlias, TypeVar

import equinox as eqx
import jax.scipy.stats as jaxstats
import optimistix as optx

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike

from ..distribution import ncx2_sf, t_cdf
from ..infer import BetaParams, infer_beta_params
from ._base import TestResult


Aux = TypeVar("Aux")
ReductionStateT = TypeVar("ReductionStateT")
ReferenceT = TypeVar("ReferenceT")
#: Scalar gene-level p-value and method-specific diagnostics.
PermutationResult: TypeAlias = tuple[Array, Aux]


class BetaCalibration(NamedTuple):
    """Reusable gene calibration and its fit diagnostics.

    `reference_estimate` is Student's t degrees of freedom when `use_tdist=True`,
    otherwise a chi-squared noncentrality parameter. `beta_params` carries its own
    convergence flag; `reference_converged` describes the reference-distribution fit.
    """

    beta_params: BetaParams
    reference_estimate: Array
    reference_converged: Array


class CauchyState(NamedTuple):
    """Weighted Cauchy sum, exact-zero/one flags, and the whole-window SNP weight.

    Endpoint flags persist across blocks so incompatible exact p-values are detected
    even when they occur in different blocks.
    """

    statistic: Array
    any_zeros: Array
    any_ones: Array
    weight: Array


class PermutationReference(NamedTuple):
    """Finalization inputs: one maximum per permutation and residual degrees of freedom.

    `maxima` has shape `(num_permutations,)`. `dof` initializes the optional
    Student's t reference fit.
    """

    maxima: Array
    dof: int


class AbstractAggregateTest(eqx.Module, Generic[ReductionStateT, ReferenceT, Aux]):
    """Numerical contract for reducing SNP blocks to one gene-level p-value.

    `statistic` selects values from a TestResult; `init` and `update` accumulate
    blocks; `finalize` returns a scalar p-value and diagnostics. The generic types
    describe the reduction state, finalization reference, and output diagnostics.
    Scheduling, hypothesis fitting, and lead selection belong to the mapper.
    """

    block_size: ClassVar[int | None] = None

    @abstractmethod
    def statistic(self, result: TestResult) -> Array:
        """Select the per-variant values used by this aggregation inside the compiled kernel."""
        ...

    @abstractmethod
    def init(self, dtype, *, num_variants: ArrayLike) -> ReductionStateT:
        """Initialize fixed-size state in the statistic dtype.

        `num_variants` counts real variants across the entire window, excluding padding.
        It determines ACAT weights and is unused by maximum-statistic reduction.
        """
        ...

    @abstractmethod
    def update(self, state: ReductionStateT, values: Array, valid: Array) -> ReductionStateT:
        """Accumulate one block of values with a same-shape Boolean validity mask.

        `valid=False` excludes padding. The state structure, leaf shapes, and dtypes
        must remain unchanged across updates.
        """
        ...

    @abstractmethod
    def finalize(self, state: ReductionStateT, reference: ReferenceT) -> tuple[Array, Aux]:
        """Return one scalar gene-level p-value and diagnostics after all blocks are accumulated.

        For permutation calibration, state is the observed statistic to evaluate.
        Permutation reductions supply the reference without individually being finalized.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short method identifier used in output filenames."""
        ...


class BetaPermutation(AbstractAggregateTest[Array, PermutationReference, BetaCalibration]):
    r"""Permutation-based gene-level p-values via a Beta approximation.

    This method reduces permutation results to statistics $T_1, \dots, T_B$ (the maximum absolute z statistic
    across variants), converts them to permutation p-values $p_b$, then fits a Beta approximation
    $p_b \sim \mathrm{Beta}(k, n)$. Observed variant statistics are mapped through
    the same calibration and fitted Beta CDF. Cis mapping reports the adjusted value
    corresponding to the selected lead variant.

    The mapper generates permutations; this class reduces and calibrates their statistics.
    Reusing the same PRNG key with the same inputs in the mapper produces the same permutations.
    Floating-point results can still vary across JAX backends.

    **Attributes:**

    - `max_perm_direct`: Number of direct permutations. Defaults to 1000.
    - `max_iter_beta`: Maximum iterations for fitting the Beta approximation.
      Defaults to 1000.
    - `use_tdist`: Estimate a Student's t degrees-of-freedom adjustment when true;
      otherwise estimate noncentrality for a chi-squared reference distribution.
    """

    block_size: ClassVar[int | None] = 512

    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    use_tdist: bool = eqx.field(static=True, default=False)

    def statistic(self, result: TestResult) -> Array:
        """Return the per-variant z statistics."""
        return result.z

    def init(self, dtype, *, num_variants: ArrayLike) -> Array:
        """Initialize a scalar NaN maximum in `dtype`; `num_variants` is unused.

        NaN is retained if every real statistic is NaN.
        """
        return jnp.asarray(jnp.nan, dtype=dtype)

    def update(self, state: Array, values: Array, valid: Array) -> Array:
        """Accumulate absolute z statistics, ignoring NaNs and padded variants."""
        maximum = jnp.nanmax(jnp.where(valid, jnp.abs(values), jnp.nan))
        return jnp.fmax(state, maximum)

    def finalize(self, state: Array, reference: PermutationReference) -> tuple[Array, BetaCalibration]:
        """Calibrate a scalar lead z statistic against complete permutation maxima.

        Returns `(gene_pvalue, BetaCalibration)`. Vector input raises ValueError;
        use `adjust` to apply an existing calibration to additional SNPs.
        """
        if jnp.ndim(state) != 0:
            raise ValueError("gene-level finalization requires a scalar lead statistic; use adjust for SNP arrays")
        calibration = self.fit_calibration(reference.maxima, reference.dof)
        return self.adjust(state, calibration), calibration

    def fit_calibration(self, z_stats_perm: Array, dof: int) -> BetaCalibration:
        """Fit a reusable calibration from a complete permutation reference.

        `z_stats_perm` contains one maximum absolute statistic per permutation.
        `dof` initializes the Student's t fit and is unused for the chi-squared fit.
        Returns Beta parameters, the fitted reference parameter, and convergence
        diagnostics. Solver failures retain Optimistix's error behavior.
        """
        if self.use_tdist:
            stats = jnp.where(jnp.isnan(z_stats_perm), 0.0, -jnp.abs(z_stats_perm))
            reference_pvalue = lambda stat, x: t_cdf(stat, x)
            solver = optx.NelderMead(rtol=1e-4, atol=1e-4)
            reference_init = float(dof)
        else:
            stats = jnp.where(jnp.isnan(z_stats_perm), 0.0, z_stats_perm**2)
            reference_pvalue = lambda stat, x: ncx2_sf(stat, 1, x)
            solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
            reference_init = 0.1

        # Choose the reference parameter so the first Beta shape has moment estimate one.
        def _shape_cost(parameter, args):
            (stats,) = args
            pval = reference_pvalue(stats, parameter)
            mean = jnp.nanmean(pval)
            var = jnp.nanvar(pval)
            return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0

        reference_fit = optx.least_squares(
            _shape_cost,
            solver=solver,
            y0=reference_init,
            args=(stats,),
        )
        estimate = reference_fit.value
        reference_converged = reference_fit.result == optx.RESULTS.successful

        p_perm = reference_pvalue(stats, estimate)

        # Beta fitting requires probabilities strictly inside (0, 1).
        tiny = jnp.finfo(float).tiny
        eps = jnp.finfo(float).eps
        p_perm = jnp.clip(p_perm, tiny, 1 - eps)

        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)

        beta_init = jnp.array([k_init, n_init])
        beta_result = infer_beta_params(p_perm, beta_init, max_iter=self.max_iter_beta)

        return BetaCalibration(beta_result, estimate, reference_converged)

    def adjust(self, z: Array, calibration: BetaCalibration) -> Array:
        """Apply an existing calibration to scalar or array-valued z statistics.

        Returns adjusted p-values with the same shape as `z`, without refitting.
        They use the gene's permutation-maximum reference for multiple testing
        adjustment; this is distinct from marginal p-value calibration by SPA.
        """
        beta_result, estimate, _ = calibration
        adj_obs_p = t_cdf(-jnp.abs(z), estimate) if self.use_tdist else ncx2_sf(z**2, 1, estimate)
        return jaxstats.beta.cdf(adj_obs_p, beta_result.k, beta_result.n)

    @property
    def name(self) -> str:
        """Return the permutation identifier used in output filenames."""
        return "perm"


def _acat_components(pvalues: Array, valid: Array, weight: Array) -> tuple[Array, Array, Array]:
    """Return a block's weighted Cauchy sum and endpoint flags; weight is global.

    Only padding is masked. Nonfinite p-values for real variants retain the
    ordinary ACAT behavior, including propagation of NaN.
    """
    p = jnp.where(valid, pvalues, 0.5)
    contributions = jnp.where(
        p < 1e-16,
        weight * jnp.reciprocal(p * jnp.pi),
        weight * jnp.tan((0.5 - p) * jnp.pi),
    )
    return jnp.sum(jnp.where(valid, contributions, 0.0)), jnp.any(valid & (p == 0.0)), jnp.any(valid & (p == 1.0))


def _acat_pvalue(cct_stat: Array, any_zeros: Array, any_ones: Array) -> Array:
    """Finish ACAT after combining sums and endpoint flags across all blocks."""
    cct_stat = eqx.error_if(cct_stat, any_ones & any_zeros, "Cannot have both 0 and 1 p-values.")
    return jnp.where(
        cct_stat > 1e15,
        jnp.reciprocal(cct_stat * jnp.pi),
        jaxstats.cauchy.sf(cct_stat),
    )


class ACAT(AbstractAggregateTest[CauchyState, None, None]):
    r"""Aggregate p-values using ACAT.

    Given per-variant p-values $p_1, \dots, p_m$ and weights $w_i = 1/m$, the Cauchy combination statistic is
    $T = \sum_i w_i \tan\left(\left(\frac{1}{2} - p_i\right)\pi\right)$, with p-value
    $p = 1 - F_{\mathrm{Cauchy}(0,1)}(T)$.

    **Failure Modes:**

    A mixture containing both exact zero and exact one p-values cannot be ordered
    consistently in the Cauchy transform. The method reports this condition through
    `equinox.error_if`; behavior follows Equinox's transformed-runtime error policy.
    """

    block_size: ClassVar[int | None] = 2048

    def statistic(self, result: TestResult) -> Array:
        """Return the per-variant p-values."""
        return result.p

    def init(self, dtype, *, num_variants: ArrayLike) -> CauchyState:
        """Initialize a Cauchy accumulator in `dtype` with weight `1 / num_variants`.

        `num_variants` is the positive number of real SNPs in the whole window,
        including those processed in later blocks.
        """
        weight = jnp.asarray(1.0, dtype=dtype) / num_variants
        return CauchyState(jnp.zeros((), dtype=dtype), jnp.array(False), jnp.array(False), weight)

    def update(self, state: CauchyState, values: Array, valid: Array) -> CauchyState:
        """Accumulate masked p-value contributions without transferring values to the host."""
        statistic, zeros, ones = _acat_components(values, valid, state.weight)
        return CauchyState(state.statistic + statistic, state.any_zeros | zeros, state.any_ones | ones, state.weight)

    def finalize(self, state: CauchyState, reference: None = None) -> tuple[Array, None]:
        """Return `(gene_pvalue, None)` from the complete Cauchy accumulator.

        `reference` is unused. A mixture of exact zero and exact one p-values
        raises through Equinox's transformed-runtime error handling.
        """
        return _acat_pvalue(state.statistic, state.any_zeros, state.any_ones), None

    @property
    def name(self) -> str:
        """Return the ACAT identifier used in output filenames."""
        return "acat"
