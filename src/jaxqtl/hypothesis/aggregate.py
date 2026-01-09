from abc import abstractmethod
from typing import Generic, TypeAlias, TypeVar

import equinox as eqx
import jax.random as rdm
import jax.scipy.stats as jaxstats
import optimistix as optx

from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from ..families.utils import ncx2_sf, t_cdf
from ..infer.optimize import BetaParams, infer_beta_params
from .base import AbstractHypothesisTest, TestResult


Aux = TypeVar("Aux")
PermutationResult: TypeAlias = tuple[Scalar, Aux]


class AbstractAggregateTest(eqx.Module, Generic[Aux]):
    r"""Abstract base class for gene-level aggregation in cis mapping."""

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


class BetaPermutation(AbstractAggregateTest[tuple[BetaParams, float, bool]]):
    r"""Permutation-based gene-level p-values via a Beta approximation.

    This method generates permutation statistics $T_1, \dots, T_B$ (here based on a max score/z statistic across
    variants), converts them to permutation p-values $p_b$, then fits a Beta approximation
    $p_b \sim \mathrm{Beta}(k, n)$. The observed gene-level p-value is computed by mapping the observed statistic
    through the same calibration and applying the fitted Beta CDF.
    """

    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    use_tdist: bool = eqx.field(static=True, default=False)

    def _run_permutations(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
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
    ) -> tuple[Array, tuple[BetaParams, float, bool]]:
        r"""Compute a gene-level p-value using direct permutations and a Beta approximation.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` for the cis window.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
        - `result`: Per-variant statistics from a single scan.
        - `test`: Hypothesis test used to generate `result`.
        - `key`: PRNG key.

        **Returns:**

        A tuple `(pvalue, aux)` where `aux` includes fitted Beta parameters and diagnostics.
        """
        z_stats_perm = self._run_permutations(X, G, y, offset, test, key)

        n = X.shape[0]
        p = X.shape[1] + 1  # covariates plus genotype
        dof = n - p  # include intercept
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

        adj_obs_p = sf(prep(result.z), estimate)
        adj_p = jaxstats.beta.cdf(adj_obs_p, beta_result.k, beta_result.n)

        return adj_p, (beta_result, estimate, opt_status)

    @property
    def name(self) -> str:
        r"""Return the aggregation name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the method.
        """
        return "perm"


class ACAT(AbstractAggregateTest[None]):
    r"""Aggregate p-values using ACAT.

    Given per-variant p-values $p_1, \dots, p_m$ and weights $w_i = 1/m$, the Cauchy combination statistic is
    $T = \sum_i w_i \tan\left(\left(\frac{1}{2} - p_i\right)\pi\right)$, with p-value
    $p = 1 - F_{\mathrm{Cauchy}(0,1)}(T)$.
    """

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
        """
        obs_p = result.p
        any_ones = jnp.any(obs_p == 1.0)
        any_zeros = jnp.any(obs_p == 0.0)
        obs_p = eqx.error_if(obs_p, any_ones & any_zeros, "Cannot have both 0 and 1 p-values.")

        weight = 1.0 / len(obs_p)

        cct_stat = jnp.sum(
            jnp.where(
                obs_p < 1e-16,
                weight * jnp.reciprocal(obs_p * jnp.pi),
                weight * jnp.tan((0.5 - obs_p) * jnp.pi),
            )
        )

        pvalue = jnp.where(
            cct_stat > 1e15,
            jnp.reciprocal(cct_stat * jnp.pi),
            jaxstats.cauchy.sf(cct_stat),
        )

        return pvalue, None

    @property
    def name(self) -> str:
        r"""Return the aggregation name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the method.
        """
        return "acat"
