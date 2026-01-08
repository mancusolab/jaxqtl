from abc import abstractmethod
from typing import Generic, TypeAlias, TypeVar

import optimistix as optx

import equinox as eqx
import jax.random as rdm
import jax.scipy.stats as jaxstats

from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from ..families.utils import ncx2_sf, t_cdf
from ..infer.optimize import BetaParams, infer_beta_params
from .base import HypothesisTest, TestResult


Aux = TypeVar("Aux")
PermutationResult: TypeAlias = tuple[Scalar, Aux]


class AbstractAggregateTest(eqx.Module, Generic[Aux]):
    """Aggregate per-variant results to a gene-level p-value in cis mapping."""

    @abstractmethod
    def aggregate(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: HypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, Aux]:
        ...

    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: HypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, Aux]:
        return self.aggregate(X, G, y, offset, result, test, key)

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class BetaPermutation(AbstractAggregateTest[tuple[BetaParams, float, bool]]):
    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    use_tdist: bool = eqx.field(static=True, default=False)

    def _run_permutations(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        test: HypothesisTest,
        key: PRNGKeyArray,
    ):
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
        test: HypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, tuple[BetaParams, float, bool]]:
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
        return "perm"


class ACAT(AbstractAggregateTest[None]):
    def aggregate(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        test: HypothesisTest,
        key: PRNGKeyArray,
    ) -> tuple[Array, None]:
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
        return "acat"
