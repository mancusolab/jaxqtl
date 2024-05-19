from abc import abstractmethod
from typing import NamedTuple, Optional, Tuple

import numpy
import numpy.random
import scipy

from scipy.stats import ncx2

import equinox as eqx
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.stats as jaxstats

from jax import grad, jit, lax, numpy as jnp
from jax.scipy.special import polygamma
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from ..families.distribution import ExponentialFamily
from ..families.utils import t_cdf
from .stderr import ErrVarEstimation, FisherInfoError
from .utils import HypothesisTest, ScoreTest


class BetaState(NamedTuple):
    adj_p: Array
    beta_res: Array
    true_nc: Array
    opt_status: Array


class InferBeta(eqx.Module):
    max_iter: int = 1000

    def __call__(self, z_stats_perm: ArrayLike, obs_z: ArrayLike, dof: Optional[ArrayLike]) -> BetaState:
        """Infer beta parameter for permutation"""
        return self.fit(z_stats_perm, obs_z, dof)

    @abstractmethod
    def fit(self, z_stats_perm: ArrayLike, obs_z: ArrayLike, dof: Optional[ArrayLike]) -> BetaState:
        pass


class InferBetaLM(InferBeta):
    def fit(self, z_stats_perm: ArrayLike, obs_z: ArrayLike, dof: ArrayLike) -> BetaState:
        def df_cost(zscore, dof):
            """minimize abs(1-alpha) as a function of M_eff"""
            pval = 2 * t_cdf(-jnp.fabs(zscore), dof)
            mean = jnp.mean(pval)
            var = jnp.var(pval)
            return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0

        # use normal distribution to compute p (approximate t in lm wald)
        p_perm = 2 * t_cdf(-jnp.fabs(z_stats_perm), dof)
        z_stats_perm = z_stats_perm[~jnp.isnan(p_perm)]  # remove z stats that will cause NAs

        # learn non-central parameter
        # true_dof = scipy.optimize.newton(lambda x: df_cost(z_stats_perm, x), dof, tol=1e-4, maxiter=50)
        res = scipy.optimize.minimize(
            lambda x: numpy.abs(df_cost(z_stats_perm, x)), dof, method='Nelder-Mead', tol=1e-4
        )
        true_dof = res.x[0]
        opt_status = res.success  # whether optimizer exited successfully

        # use non-central chi2 to recompute permutation pvals
        p_perm = 2 * t_cdf(-jnp.fabs(z_stats_perm), true_dof)

        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)

        init = jnp.array([k_init, n_init])

        # infer beta based on p_perm
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter)

        adj_obs_p = 2 * t_cdf(-jnp.fabs(obs_z), true_dof)
        adj_p = _calc_adjp_beta(adj_obs_p, beta_res[0:2])

        return BetaState(adj_p=adj_p, beta_res=beta_res, true_nc=true_dof, opt_status=opt_status)


class InferBetaGLM(InferBeta):
    def fit(self, z_stats_perm: ArrayLike, obs_z: ArrayLike, dof: Optional[ArrayLike]) -> BetaState:
        def df_cost(zscore, nc):
            """minimize abs(1-alpha) as a function of M_eff"""
            pval = ncx2.sf(zscore**2, 1, nc)
            mean = jnp.mean(pval)
            var = jnp.var(pval)
            return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0

        # use normal distribution to compute p (approximate t in lm wald)
        p_perm = 2 * norm.sf(jnp.fabs(z_stats_perm))
        z_stats_perm = z_stats_perm[~jnp.isnan(p_perm)]  # remove z stats that will cause NAs

        # learn non-central parameter
        # true_nc = scipy.optimize.newton(lambda x: df_cost(p_perm, x), 0.1, tol=1e-4, maxiter=50)
        res = scipy.optimize.minimize(
            lambda x: numpy.abs(df_cost(z_stats_perm, x)), 0.1, method='Nelder-Mead', tol=1e-4
        )
        true_nc = res.x[0]
        opt_status = res.success  # whether optimizer exited successfully

        # use non-central chi2 to recompute permutation pvals
        p_perm = ncx2.sf(z_stats_perm**2, 1, true_nc)

        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)

        init = jnp.array([k_init, n_init])

        # infer beta based on p_perm
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter)

        adj_obs_p = ncx2.sf(obs_z**2, 1, true_nc)
        adj_p = _calc_adjp_beta(adj_obs_p, beta_res[0:2])

        return BetaState(adj_p=adj_p, beta_res=beta_res, true_nc=true_nc, opt_status=opt_status)


class Permutation(eqx.Module):
    """
    For a given cis-window around a gene (L variants), perform permutation test to
    identify (one candidate) eQTL for this gene.
    direct_perm performs native permutation with max_iters,
    i.e. for each permutated data, do cis-window scan
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        obs_z: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        test: HypothesisTest = ScoreTest(),
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 500,
    ) -> Array:
        pass


class DirectPerm(Permutation):
    max_perm_direct: int = 10000

    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        obs_z: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        test: HypothesisTest = ScoreTest(),
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 500,
    ) -> Array:
        """

        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param G: genotype matrix
        :param obs_p: observed minimum p value for given gene (lead SNP p value); not used
        :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
        :param key_init: jax PRNGKey
        :param sig_level: alpha significance level at each SNP level (not used), default to 0.05
        :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
        :param test: approach for hypothesis test, default to ScoreTest()
        :param se_estimator: SE estimator using HuberError() or FisherInfoError()
        :param max_iter: maximum iterations for fitting GLM, default to 500
        :return: permutation p values
        """

        def _func(key, x):
            del x
            key, p_key = rdm.split(key)
            perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
            glmstate = test(X, G, y[perm_idx], family, offset_eta[perm_idx], se_estimator, max_iter)
            # Note: permute individual rows of G can still preserve LD of variants (columns)

            return key, jnp.nanmax(jnp.abs(glmstate.z))  # jnp.nanmin(glmstate.p)

        key, z_stats = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)

        # # TODO: write a for loop to write out pvals[-18] G, y, offset
        # key = key_init
        # for idx in range(self.max_perm_direct):
        #     key, p_key = rdm.split(key)
        #     perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
        #     glmstate = test(X, G, y[perm_idx], family, offset_eta[perm_idx], se_estimator, max_iter)
        #     pvals = jnp.nanmin(glmstate.p)  # G index 934
        #
        # import numpy as np; import jax; jax.debug.breakpoint()

        return z_stats


@eqx.filter_jit
def _calc_adjp_naive(obs_pval: ArrayLike, pval: ArrayLike) -> Array:
    """calculate adjusted minimum p value under beta distribution

    :param obs_pval: observed minimum p value
    :param pval: permutation minimum p values
    :return: adjusted minimum p value
    """
    return (jnp.sum(pval < obs_pval) + 1) / (len(pval) + 1)


@eqx.filter_jit
def infer_beta(
    p_perm: ArrayLike,
    init: ArrayLike,
    step_size=0.1,
    tol=1e-3,
    max_iter=500,
) -> Array:
    """Infer shape and scale parameter for beta distribution
    given p values from R permutations (strongest signals),
    use newton's method to estimate beta distribution parameters:
    p ~ Beta(k, n)

    :param p_perm: permutation minimum p values
    :param init: initial value for shape and scale
    :param step_size: step size to update parameters at each step, default to 0.1
    :param tol: tolerance for stopping, default to 0.001
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :return:
    """

    def loglik(params, p: ArrayLike) -> Array:
        return jnp.sum(jaxstats.beta.logpdf(p, params[0], params[1]))

    def info_and_christoffel(params: ArrayLike, p: ArrayLike) -> Tuple[Array, Array]:
        k, n = params

        # reuse terms
        pg_1k = polygamma(1, k)
        pg_1n = polygamma(1, n)
        pg_1kn = polygamma(1, k + n)

        pg_2k = polygamma(2, k)
        pg_2n = polygamma(2, n)
        pg_2kn = polygamma(2, k + n)

        # fisher information matrix
        i_kn = -pg_1kn
        i_k = pg_1k + i_kn
        i_n = pg_1n + i_kn

        info_mat = -len(p) * jnp.array([[i_k, i_kn], [i_kn, i_n]])

        # first sub-matrix of the unscaled 2nd-order Christoffell symbol
        i_kkn = pg_1n * pg_2kn
        i_k = -pg_1n * pg_2k + i_kkn + pg_1kn * pg_2k
        i_knn = i_kkn - pg_1kn * pg_2n

        # second sub-matrix of the unscaled 2nd-order Christoffell symbol
        i_nnk = pg_1k * pg_2kn
        i_nkk = i_nnk - pg_1kn * pg_2k
        i_n = -pg_1k * pg_2n + i_nnk + pg_1kn * pg_2n

        # scale for the 2nd-order Christoffel symbol
        scale = -pg_1k * pg_1n + (pg_1k + pg_1n) * pg_1kn

        # combine into single tensor
        sec_gamma = 0.5 * jnp.array([[[i_k, i_kkn], [i_kkn, i_knn]], [[i_nkk, i_nnk], [i_nnk, i_n]]]) / scale

        return info_mat, sec_gamma

    score_fn = grad(loglik)

    def body_fun(val: Tuple):
        old_lik, diff, num_iter, old_param = val
        # first order approx to RGD => NGD
        # direction = NatGrad
        info_mat, gamma = info_and_christoffel(old_param, p_perm)
        direction = jnla.solve(info_mat, score_fn(old_param, p_perm))

        # take second order approx to RGD
        adjustment = jnp.einsum("cab,a,b->c", gamma, direction, direction)
        new_param = old_param - step_size * direction - 0.5 * step_size**2 * adjustment

        new_lik = loglik(new_param, p_perm)
        diff = old_lik - new_lik

        return new_lik, diff, num_iter + 1, new_param

    def cond_fun(val: Tuple):
        old_lik, diff, num_iter, old_param = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_tuple = (10000.0, 1000.0, 0, init)
    lik, diff, num_iters, params = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter).astype(float)

    return jnp.array([params[0], params[1], converged])


@jit
def _calc_adjp_beta(p_obs: ArrayLike, params: ArrayLike) -> Array:
    """
    p_obs is the min p observed
    """
    k, n = params

    # TODO: sometimes give out-of-bound values
    p_adj = jaxstats.beta.cdf(p_obs, k, n)

    return p_adj


class BetaPerm(DirectPerm):
    max_perm_direct: int = 1000
    max_iter_beta: int = 1000

    def __call__(  # type: ignore
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        obs_z: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        test: HypothesisTest = ScoreTest(),
        se_estimator: ErrVarEstimation = FisherInfoError(),
        beta_estimator: InferBeta = InferBetaGLM(),
        max_iter: int = 1000,
    ) -> Tuple[Array, Array, Array, Array]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        z_stats_perm = super().__call__(
            X,
            y,
            G,
            obs_p,
            obs_z,
            family,
            key_init,
            sig_level,
            offset_eta,
            test,
            se_estimator,
            max_iter,
        )

        # # use normal distribution to compute p (approximate t in lm wald)
        # p_perm = 2 * norm.sf(jnp.fabs(z_stats_perm))
        # z_stats_perm = z_stats_perm[~jnp.isnan(p_perm)]  # remove z stats that will cause NAs
        #
        # # learn non-central parameter
        # # true_nc = scipy.optimize.newton(lambda x: df_cost(p_perm, x), 0.1, tol=1e-4, maxiter=50)
        # res = scipy.optimize.minimize(
        #     lambda x: numpy.abs(df_cost(z_stats_perm, x)), 0.1, method='Nelder-Mead', tol=1e-4
        # )
        # true_nc = res.x[0]
        # opt_status = res.success  # whether optimizer exited successfully
        #
        # # use non-central chi2 to recompute permutation pvals
        # p_perm = ncx2.sf(z_stats_perm**2, 1, true_nc)
        #
        # p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        # k_init = jnp.nan_to_num(p_mean * (p_mean * (1 - p_mean) / p_var - 1), nan=1.0)
        # n_init = jnp.nan_to_num(k_init * (1 / p_mean - 1), nan=1.0)
        #
        # init = jnp.array([k_init, n_init])
        #
        # # infer beta based on p_perm
        # beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)
        #
        # adj_obs_p = ncx2.sf(obs_z**2, 1, true_nc)
        # adj_p = _calc_adjp_beta(adj_obs_p, beta_res[0:2])

        n = X.shape[0]
        p = X.shape[1] + 1  # covariates plus genotype
        dof = n - p  # include intercept

        adj_p, beta_res, true_nc, opt_status = beta_estimator(z_stats_perm, obs_z, dof)

        return adj_p, beta_res, true_nc, opt_status


# def df_cost(zscore, nc):
#     """minimize abs(1-alpha) as a function of M_eff"""
#     # pval = 2 * norm.sf(jnp.fabs(zscore))
#     pval = ncx2.sf(zscore**2, 1, nc)
#     mean = jnp.mean(pval)
#     var = jnp.var(pval)
#     return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0
