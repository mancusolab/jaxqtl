# pattern: Functional Core

from abc import abstractmethod
from typing import ClassVar, Generic, NamedTuple, Protocol, TypeAlias, TypeVar

import equinox as eqx
import jax.random as rdm
import jax.scipy.stats as jaxstats
import optimistix as optx

from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..distribution import ncx2_sf, t_cdf
from ..infer import BetaParams, infer_beta_params
from ._base import AbstractHypothesisTest, TestResult


Aux = TypeVar("Aux")
#: Method-specific adjusted p-value output, scalar or variantwise, plus auxiliary diagnostics.
PermutationResult: TypeAlias = tuple[Array, Aux]


class BetaCalibration(NamedTuple):
    """Fitted Beta parameters, reference estimate, and reference-fit status."""

    beta_params: BetaParams
    reference_estimate: Array
    reference_converged: Array


class CauchyState(NamedTuple):
    """Weighted Cauchy sum and endpoint flags across real variants."""

    statistic: Array
    any_zeros: Array
    any_ones: Array


class ScanExecution(Protocol):
    """Operations used by aggregation workflows, supplied by the mapping layer.

    The executor owns block scheduling and compiled calls. Aggregations depend
    on this contract rather than a concrete mapping implementation.
    """

    def observed(
        self, X, G, y, offset, *, reduction: "ACAT | None" = None
    ) -> tuple[TestResult, CauchyState | None]: ...

    def permutation_maxima(self, X, G, y, offset, key: PRNGKeyArray) -> Array: ...

    def fit_calibration(self, maxima: Array, dof: int) -> BetaCalibration: ...

    def adjust(self, z: Array, calibration: BetaCalibration) -> Array: ...

    def finish_acat(self, state: CauchyState) -> Array: ...


class AbstractAggregateTest(eqx.Module, Generic[Aux]):
    r"""Abstract base class for gene-level aggregation in cis mapping."""

    block_size: ClassVar[int | None] = None
    adjustment_method: eqx.AbstractClassVar[str]
    has_calibration: eqx.AbstractClassVar[bool]

    @abstractmethod
    def scan(self, execution: ScanExecution, X: Array, G: Array, y: Array, offset: Array, key: PRNGKeyArray):
        """Coordinate this aggregation using the configured scan executor.

        This host entry point composes separately compiled kernels. Use
        ``aggregate`` for a transformable full-array calculation.
        """
        ...

    @abstractmethod
    def aggregate(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: AbstractHypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, Aux]:
        r"""Aggregate per-variant test results into a gene-level statistic.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` for the cis window.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `result`: Per-variant statistics from a single scan.
        - `test`: Hypothesis test used to generate `result`.
        - `key`: PRNG key for stochastic aggregation procedures.

        **Returns:**

        A tuple `(pvalue, aux)` where `aux` contains method-specific diagnostics.
        """
        ...

    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: AbstractHypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, Aux]:
        r"""Alias for [`jaxqtl.hypothesis.AbstractAggregateTest.aggregate`][].

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` for the cis window.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `result`: Per-variant statistics from a single scan.
        - `test`: Hypothesis test used to generate `result`.
        - `key`: PRNG key for stochastic aggregation procedures.

        **Returns:**

        A tuple `(pvalue, aux)` where `aux` contains method-specific diagnostics.
        """
        return self.aggregate(X, G, y, offset, result, test, key)

    @property
    @abstractmethod
    def name(self) -> str:
        r"""Return a short identifier for the aggregation method.

        **Arguments:**

        `None`

        **Returns:**

        A short string name for display and downstream metadata.
        """
        ...


class BetaPermutation(AbstractAggregateTest[BetaCalibration]):
    r"""Permutation-based gene-level p-values via a Beta approximation.

    This method generates permutation statistics $T_1, \dots, T_B$ (here based on a max score/z statistic across
    variants), converts them to permutation p-values $p_b$, then fits a Beta approximation
    $p_b \sim \mathrm{Beta}(k, n)$. Observed variant statistics are mapped through
    the same calibration and fitted Beta CDF. Cis mapping reports the adjusted value
    corresponding to the selected lead variant.

    Reusing the same PRNG key with the same inputs produces the same permutations.
    Floating-point results can still vary across JAX backends.

    **Attributes:**

    - `max_perm_direct`: Number of direct permutations. Defaults to 1000.
    - `max_iter_beta`: Maximum iterations for fitting the Beta approximation.
      Defaults to 1000.
    - `use_tdist`: Estimate a Student's t degrees-of-freedom adjustment when true;
      otherwise estimate noncentrality for a chi-squared reference distribution.
    """

    block_size: ClassVar[int | None] = 512
    adjustment_method: ClassVar[str] = "BETA"
    has_calibration: ClassVar[bool] = True

    def scan(self, execution: ScanExecution, X: Array, G: Array, y: Array, offset: Array, key: PRNGKeyArray):
        result, _ = execution.observed(X, G, y, offset)
        maxima = execution.permutation_maxima(X, G, y, offset, key)
        calibration = execution.fit_calibration(maxima, X.shape[0] - X.shape[1] - 1)
        return result, (execution.adjust(result.z, calibration), calibration)

    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    use_tdist: bool = eqx.field(static=True, default=False)

    def _run_permutations(
        self,
        X: Array,
        G: Array,
        y: Array,
        offset: Array,
        test: AbstractHypothesisTest,
        key: PRNGKeyArray,
    ):
        r"""Run direct permutations and return a vector of max statistics.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)`.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `test`: Hypothesis test to apply per permutation.
        - `key`: PRNG key.

        **Returns:**

        A 1D array of permutation max statistics.
        """
        X = jnp.asarray(X)
        G = jnp.asarray(G)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)

        def _func(key, x):
            key, p_key = rdm.split(key)
            perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
            if offset.ndim > 0:
                glmstate = test(X, G, y[perm_idx], offset[perm_idx])
            else:
                glmstate = test(X, G, y[perm_idx], offset)

            return key, jnp.nanmax(jnp.abs(glmstate.z))

        key, z_stats = lax.scan(_func, key, xs=None, length=self.max_perm_direct)

        return z_stats

    def aggregate(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: AbstractHypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, BetaCalibration]:
        r"""Compute variantwise adjusted p-values using a Beta approximation.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` for the cis window.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `result`: Per-variant statistics from a single scan.
        - `test`: Hypothesis test used to generate `result`.
        - `key`: PRNG key.

        **Returns:**

        A tuple `(pvalue, aux)`. `pvalue` contains one adjusted value per variant.
        `aux` is `(beta_params, calibration_estimate, optimizer_converged)`, where
        `calibration_estimate` is the fitted t degrees of freedom or chi-squared
        noncentrality parameter according to `use_tdist`.
        """
        X = jnp.asarray(X)
        G = jnp.asarray(G)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)
        z_stats_perm = self._run_permutations(X, G, y, offset, test, key)

        return self._calibrate(z_stats_perm, result.z, X.shape[0] - X.shape[1] - 1)

    def fit_calibration(self, z_stats_perm: Array, dof: int) -> BetaCalibration:
        """Fit the reference distribution and Beta parameters to complete permutation maxima."""
        if self.use_tdist:
            prep = lambda stat: -jnp.abs(stat)
            stats = jnp.where(jnp.isnan(z_stats_perm), 0.0, prep(z_stats_perm))
            sf = lambda stat, x: t_cdf(stat, x)
            solver = optx.NelderMead(rtol=1e-4, atol=1e-4)
            init = float(dof)
        else:
            prep = lambda stat: stat**2
            stats = jnp.where(jnp.isnan(z_stats_perm), 0.0, prep(z_stats_perm))
            sf = lambda stat, x: ncx2_sf(stat, 1, x)
            solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
            init = 0.1

        def _df_cost(nc, args):
            (stats,) = args
            pval = sf(stats, nc)
            mean = jnp.nanmean(pval)
            var = jnp.nanvar(pval)
            return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0

        res = optx.least_squares(
            _df_cost,
            solver=solver,
            y0=init,
            args=(stats,),
        )
        estimate = res.value
        opt_status = res.result == optx.RESULTS.successful

        p_perm = sf(stats, estimate)

        tiny = jnp.finfo(float).tiny
        eps = jnp.finfo(float).eps
        p_perm = jnp.clip(p_perm, tiny, 1 - eps)

        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)

        init = jnp.array([k_init, n_init])
        beta_result = infer_beta_params(p_perm, init, max_iter=self.max_iter_beta)

        return BetaCalibration(beta_result, estimate, opt_status)

    def adjust(self, z: Array, calibration: BetaCalibration) -> Array:
        """Apply one gene's calibration to observed statistics (which may be a SNP block)."""
        beta_result, estimate, _ = calibration
        adj_obs_p = t_cdf(-jnp.abs(z), estimate) if self.use_tdist else ncx2_sf(z**2, 1, estimate)
        return jaxstats.beta.cdf(adj_obs_p, beta_result.k, beta_result.n)

    def _calibrate(self, z_stats_perm: Array, observed_z: Array, dof: int) -> tuple[Array, BetaCalibration]:
        """Calibrate existing maxima without generating permutations or fitting the null again."""
        calibration = self.fit_calibration(z_stats_perm, dof)
        return self.adjust(observed_z, calibration), calibration

    @property
    def name(self) -> str:
        r"""Return the aggregation name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the method.
        """
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


class ACAT(AbstractAggregateTest[None]):
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
    adjustment_method: ClassVar[str] = "ACAT"
    has_calibration: ClassVar[bool] = False

    def scan(self, execution: ScanExecution, X: Array, G: Array, y: Array, offset: Array, key: PRNGKeyArray):
        result, state = execution.observed(X, G, y, offset, reduction=self)
        assert state is not None
        return result, (execution.finish_acat(state), None)

    def init(self, dtype) -> CauchyState:
        """Initialize a fixed-size accumulator in the contribution dtype."""
        return CauchyState(jnp.zeros((), dtype=dtype), jnp.array(False), jnp.array(False))

    def components(self, pvalues: Array, valid: Array, weight: Array) -> CauchyState:
        """Reduce a masked block with weights defined across the whole window."""
        return CauchyState(*_acat_components(pvalues, valid, weight))

    def update(self, state: CauchyState, block: CauchyState) -> CauchyState:
        """Merge one block's contributions without transferring values to the host."""
        return CauchyState(
            state.statistic + block.statistic, state.any_zeros | block.any_zeros, state.any_ones | block.any_ones
        )

    def finalize(self, state: CauchyState) -> Array:
        """Convert the complete accumulator to a gene-level p-value."""
        return _acat_pvalue(*state)

    def aggregate(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: AbstractHypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, None]:
        r"""Compute a gene-level p-value using ACAT.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` for the cis window.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `result`: Per-variant statistics from a single scan.
        - `test`: Hypothesis test used to generate `result`.
        - `key`: PRNG key (unused).

        **Returns:**

        A tuple `(pvalue, None)` containing the ACAT gene-level p-value.

        **Failure Modes:**

        If `result.p` contains both exact zero and exact one values, the method
        reports an error through `equinox.error_if`.
        """
        obs_p = result.p
        components = self.components(obs_p, jnp.ones_like(obs_p, dtype=bool), jnp.asarray(1.0 / len(obs_p)))
        return self.finalize(components), None

    @property
    def name(self) -> str:
        r"""Return the aggregation name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the method.
        """
        return "acat"
