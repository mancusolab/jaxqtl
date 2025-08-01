from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import TypeAlias

import optimistix as optx

import equinox as eqx
import jax.random as rdm
import jax.scipy.stats as jaxstats

from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike, Scalar

from ..families.utils import ncx2_sf, t_cdf
from .glm import AbstractLinearModel
from .optimize import BetaParams, infer_beta
from .utils import HypothesisTest, TestResult


Aux = TypeVar("Aux")
PermutationResult: TypeAlias = tuple[Scalar, Aux]


class AbstractPermutation(eqx.Module, Generic[Aux]):
    """
    For a given cis-window around a gene (L variants), perform permutation test to
    identify (one candidate) eQTL for this gene.
    direct_perm performs native permutation with max_iters,
    i.e. for each permutated data, do cis-window scan
    """

    @abstractmethod
    def perm(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        glm: AbstractLinearModel,
        test: HypothesisTest,
        key: rdm.PRNGKey,
        sig_level: float = 0.05,
    ) -> tuple[Array, Aux]: ...

    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        glm: AbstractLinearModel,
        test: HypothesisTest,
        key: rdm.PRNGKey,
        sig_level: float = 0.05,
    ) -> Array:
        self.perm(X, G, y, offset, result, glm, test, key, sig_level)


class BetaPermutation(AbstractPermutation[tuple[BetaParams, float, bool]]):
    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    use_tdist: bool = eqx.field(static=True)

    def _run_permutations(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        offset: ArrayLike,
        glm: AbstractLinearModel,
        test: HypothesisTest,
        key: rdm.PRNGKey,
    ):
        def _func(key, x):
            key, p_key = rdm.split(key)
            perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
            glmstate = test(X, G, y[perm_idx], offset[perm_idx], glm)
            # Note: permute individual rows of G can still preserve LD of variants (columns)

            return key, jnp.nanmax(jnp.abs(glmstate.z))  # jnp.nanmin(glmstate.p)

        key, z_stats = lax.scan(_func, key, xs=None, length=self.max_perm_direct)

        return z_stats

    def perm(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        glm: AbstractLinearModel,
        test: HypothesisTest,
        key: rdm.PRNGKey,
        sig_level: float = 0.05,
    ) -> tuple[Array, tuple[BetaParams, float, bool]]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        z_stats_perm = self._run_permutations(X, G, y, offset, glm, test, key)

        n = X.shape[0]
        p = X.shape[1] + 1  # covariates plus genotype
        dof = n - p  # include intercept
        zsq = z_stats_perm**2
        if self.use_tdist:
            sf = lambda stat, x: t_cdf(stat, dof)
            init = float(dof)
        else:
            sf = lambda stat, x: ncx2_sf(stat, 1, x)
            init = 0.1

        def _df_cost(nc, args):
            (zsq,) = args
            """minimize abs(1-alpha) as a function of M_eff"""
            pval = sf(zsq, nc)
            mean = jnp.mean(pval)
            var = jnp.var(pval)
            return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0

        # learn non-central parameter
        res = optx.least_squares(
            _df_cost,
            solver=optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4),
            y0=init,
            args=(zsq,),
        )
        estimate = res.value
        opt_status = res.result == optx.RESULTS.successful

        # use non-central chi2 to recompute permutation pvals
        p_perm = sf(estimate)

        # init using method-of-moments
        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)

        # infer beta parameters numerically
        init = jnp.array([k_init, n_init])
        beta_result = infer_beta(p_perm, init, max_iter=self.max_iter)

        # compute final permutation pvalues from Beta(k, n) approximation
        adj_obs_p = sf(jnp.nanmax(result.z) ** 2, estimate)
        adj_p = jaxstats.beta.cdf(adj_obs_p, beta_result.k, beta_result.n)

        return adj_p, (beta_result, estimate, opt_status)


class ACAT(AbstractPermutation[None]):
    def perm(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        result: TestResult,
        glm: AbstractLinearModel,
        test: HypothesisTest,
        key: rdm.PRNGKey,
        sig_level: float = 0.05,
    ) -> tuple[Array, None]:
        obs_p = result.p
        any_ones = jnp.any(obs_p == 1.0)
        any_zeros = jnp.any(obs_p == 0.0)
        obs_p = eqx.error_if(obs_p, any_ones & any_zeros, "Cannot have both 0 and 1 p-values.")

        # split into 'large' and 'small' checks
        cct_stat = jnp.sum(
            jnp.where(
                obs_p < 1e-16,
                jnp.reciprocal(obs_p * jnp.pi),
                jnp.tan(0.5 - obs_p * jnp.pi),
            )
        )

        # numerics breaks down when stat gets too big; this threshold is fine if we're in 64bit mode
        # (which should always be case)
        pvalue = jnp.where(
            cct_stat > 1e15,
            jnp.reciprocal(cct_stat * jnp.pi),
            # first-order approximation when stat is large; higher-order terms are o(1)
            jaxstats.cauchy.sf(cct_stat),
        )

        return pvalue, None


def _acat(pvalues: ArrayLike) -> Array:
    """
    # ref: https://gist.github.com/ryananeff/c66cdf086979b13e855f2c3d0f3e54e1
    Aggregated Cauchy Assocaition Test
    A p-value combination method using the Cauchy distribution.

    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R

    Author: Ryan Neff

    Inputs:
        pvalues: <list or numpy array>
            The p-values you want to combine.
        weights: <list or numpy array>, default=None
            The weights for each of the p-values. If None, equal weights are used.

    Returns:
        pval: <float>
            The ACAT combined p-value.
    """
