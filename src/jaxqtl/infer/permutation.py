from abc import ABCMeta, abstractmethod
from typing import Tuple

import equinox as eqx
import scipy

import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.stats as jaxstats
from jax import Array, grad, jit, lax
from jax.scipy.special import polygamma
from jax.scipy.stats import chi2, norm
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.utils import cis_scan, cis_scan_score


class Permutation(eqx.Module, metaclass=ABCMeta):
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
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        robust_se: bool = False,
    ) -> Array:
        pass


class DirectPerm(Permutation):
    max_perm_direct: int

    def __init__(self, max_perm_direct: int = 10000):
        self.max_perm_direct = max_perm_direct

    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        robust_se: bool = False,
    ) -> Array:
        def _func(key, x):
            key, p_key = rdm.split(key)
            # y_p = rdm.permutation(p_key, y, axis=0)
            perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
            glmstate = cis_scan(
                X, G, y[perm_idx], family, offset_eta[perm_idx], robust_se
            )
            # jax.debug.print("min p: {}", glmstate.p.min())
            # allTS = jnp.abs(glmstate.beta / glmstate.se)

            # TODO: remove NA values before take min
            return key, glmstate.p.min()  # glmstate.p.min()

        key, pvals = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)
        # key, TS = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)
        # pvals = pval_from_Zstat(TS, 1.0)

        return pvals  # , TS


class PermutationScore(eqx.Module, metaclass=ABCMeta):
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
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        pass


class DirectPermScore(PermutationScore):
    max_perm_direct: int

    def __init__(self, max_perm_direct: int = 10000):
        self.max_perm_direct = max_perm_direct

    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        def _func(key, x):
            key, p_key = rdm.split(key)
            perm_idx = rdm.permutation(p_key, jnp.arange(0, len(y)))
            glmstate = cis_scan_score(X, G, y[perm_idx], family, offset_eta[perm_idx])

            # TODO: remove NA values before take min
            return (
                key,
                # glmstate.p.min(),
                jnp.abs(glmstate.Z).max(),
            )  # jnp.where(jnp.isnan(allp), jnp.inf, allp).min()

        # key, pvals = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)
        key, Z = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)

        return Z


@eqx.filter_jit
def _calc_adjp_naive(obs_pval: ArrayLike, pval: ArrayLike) -> Array:
    """
    obs_pval: the strongest nominal p value
    """
    return (jnp.sum(pval < obs_pval) + 1) / (len(pval) + 1)


@eqx.filter_jit
def infer_beta(
    p_perm: ArrayLike,
    init: ArrayLike,
    stepsize=1.0,
    tol=1e-3,
    max_iter=100,
) -> Array:
    """
    given p values from R permutations (strongest signals),
    use newton's method to estimate beta distribution parameters:
    p ~ Beta(k, n)
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
        sec_gamma = (
            0.5
            * jnp.array(
                [[[i_k, i_kkn], [i_kkn, i_knn]], [[i_nkk, i_nnk], [i_nnk, i_n]]]
            )
            / scale
        )

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
        new_param = old_param - stepsize * direction - 0.5 * stepsize ** 2 * adjustment

        new_lik = loglik(new_param, p_perm)
        diff = old_lik - new_lik

        return new_lik, diff, num_iter + 1, new_param

    def cond_fun(val: Tuple):
        old_lik, diff, num_iter, old_param = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_tuple = (10000.0, 1000.0, 0, init)
    lik, diff, num_iters, params = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter).astype(
        float
    )
    # jax.debug.print("num_iter = {num_iters}", num_iters=num_iters)
    return jnp.array([params[0], params[1], converged])


# infer beta from tensorqtl
# def calculate_beta_approx_pval(r2_perm, r2_nominal, dof_init, tol=1e-4):
#     """
#       r2_nominal: nominal max. r2 (scalar or array)
#       r2_perm:    array of max. r2 values from permutations
#       dof_init:   degrees of freedom
#     """
#     beta_shape1, beta_shape2, true_dof = fit_beta_parameters(r2_perm, dof_init, tol)
#     pval_true_dof = pval_from_corr(r2_nominal, true_dof)
#     pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
#     return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof


# def fit_beta_parameters(r2_perm, dof_init, tol=1e-4, return_minp=False):
#     """
#       r2_perm:    array of max. r2 values from permutations
#       dof_init:   degrees of freedom
#     """
#     try:
#         true_dof = scipy.optimize.newton(lambda x: df_cost(r2_perm, x), dof_init, tol=tol, maxiter=50)
#     except:
#         print('WARNING: scipy.optimize.newton failed to converge (running scipy.optimize.minimize)')
#         res = scipy.optimize.minimize(lambda x: np.abs(df_cost(r2_perm, x)), dof_init, method='Nelder-Mead', tol=tol)
#         true_dof = res.x[0]
#
#     pval = pval_from_corr(r2_perm, true_dof)
#     mean, var = np.mean(pval), np.var(pval)
#     beta_shape1 = mean * (mean * (1 - mean) / var - 1)
#     beta_shape2 = beta_shape1 * (1/mean - 1)
#     res = scipy.optimize.minimize(lambda s: beta_log_likelihood(pval, s[0], s[1]), [beta_shape1, beta_shape2],
#                                   method='Nelder-Mead', tol=tol)
#     beta_shape1, beta_shape2 = res.x
#     if return_minp:
#         return beta_shape1, beta_shape2, true_dof, pval
#     else:
#         return beta_shape1, beta_shape2, true_dof


# def df_cost(r2, dof):
#     """minimize abs(1-alpha) as a function of M_eff"""
#     pval = pval_from_corr(r2, dof)
#     mean = np.mean(pval)
#     var = np.var(pval)
#     return mean * (mean * (1.0-mean) / var - 1.0) - 1.0


@jit
def _calc_adjp_beta(p_obs: ArrayLike, params: ArrayLike) -> Array:
    """
    p_obs is a vector of nominal p value in cis window
    """
    k, n = params

    # TODO: sometimes give wrong values
    p_adj = jaxstats.beta.cdf(jnp.min(p_obs), k, n)

    return p_adj


class BetaPerm(DirectPerm):
    max_perm_direct: int
    max_iter_beta: int

    def __init__(self, max_perm_direct: int = 1000, max_iter_beta: int = 1000):
        self.max_iter_beta = max_iter_beta
        super().__init__(max_perm_direct)

    def __call__(  # type: ignore
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        robust_se: bool = False,
    ) -> Tuple[Array, Array]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        p_perm = super().__call__(
            X, y, G, obs_p, family, key_init, sig_level, offset_eta, robust_se
        )
        p_perm = p_perm[~jnp.isnan(p_perm)]  # remove NAs

        ####
        # TODO: calculate true df and adjust every p_perm accordingly
        # https://github.com/google/jaxopt/blob/main/jaxopt/_src/scipy_wrappers.py  #  Nelder-Mead
        # res = scipy.optimize.minimize(lambda x: np.abs(df_cost(TS, x)), dof_init, method='Nelder-Mead', tol=tol)
        # dof_init = 1.0
        # opt = jaxopt.ScipyMinimize(
        #     fun=lambda x: jnp.abs(df_cost(TS, x)),
        #     method="Newton-CG",
        #     tol=1e-3,
        #     maxiter=100,
        # )
        # opt_res = opt.run(init_params=dof_init)
        #
        # if opt_res.state.success:
        #     true_dof = opt_res.params
        #     p_perm = pval_from_Zstat(TS, true_dof)
        # else:
        #     true_dof = dof_init
        #####

        # init = jnp.ones(2)  # initialize with 1
        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
        n_init = k_init * (1 / p_mean - 1)
        init = jnp.array([k_init, n_init])

        # infer beta based on adjusted p_perm
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)

        adj_p = _calc_adjp_beta(obs_p, beta_res[0:2])

        return adj_p, beta_res


class BetaPermScore(DirectPermScore):
    max_perm_direct: int
    max_iter_beta: int

    def __init__(self, max_perm_direct: int = 1000, max_iter_beta: int = 1000):
        self.max_iter_beta = max_iter_beta
        super().__init__(max_perm_direct)

    def __call__(  # type: ignore
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init: rdm.PRNGKey,
        sig_level: float = 0.05,
        offset_eta: ArrayLike = 0.0,
        log=None,
    ) -> Tuple[Array, Array]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        # p_perm = super().__call__(
        #     X, y, G, obs_p, family, key_init, sig_level, offset_eta
        # )
        # p_perm = p_perm[~jnp.isnan(p_perm)]  # remove NAs

        # infer true_dof
        Z_perm = super().__call__(
            X, y, G, obs_p, family, key_init, sig_level, offset_eta
        )
        p_perm = pval_from_Zstat(Z_perm, 1.0)
        Z_perm = Z_perm[~jnp.isnan(p_perm)]

        dof_init = 1.0
        # try:
        #     true_dof = scipy.optimize.newton(lambda x: df_cost(Z_perm, x), dof_init, tol=1e-3,maxiter=50)
        # except:
        #     log.info(
        #         "WARNING: scipy.optimize.newton failed to converge (running scipy.optimize.minimize)"
        #     )
        res = scipy.optimize.minimize(
            lambda x: jnp.abs(df_cost(Z_perm, x)),
            dof_init,
            method="Nelder-Mead",
            tol=1e-3,
        )
        if res.success:
            true_dof = res.x.squeeze()
        else:
            log.info("Nelder-Mead not converge; use true_dof=1")
            true_dof = 1.0

        p_perm = pval_from_Zstat(Z_perm, true_dof)
        p_perm = p_perm[~jnp.isnan(p_perm)]
        #

        # init = jnp.ones(2)  # initialize with 1
        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
        n_init = k_init * (1 / p_mean - 1)
        init = jnp.array([k_init, n_init])

        # infer beta based on p_perm
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)

        # adj_p = _calc_adjp_beta(obs_p, beta_res[0:2])
        obs_p_true_dof = pval_from_Zstat(norm.ppf(obs_p / 2), true_dof)
        adj_p = _calc_adjp_beta(obs_p_true_dof, beta_res[0:2])

        return adj_p, beta_res


def pval_from_Zstat(TS: ArrayLike, dof: float = 1.0):
    # TS is the beta / se; use chi2(df)?
    return 1 - chi2.cdf(TS ** 2, dof)  # norm.cdf(-abs(TS)) * 2


def df_cost(TS, dof):
    """minimize abs(1-alpha) as a function of M_eff"""
    pval = pval_from_Zstat(TS, dof)
    mean = jnp.mean(pval)
    var = jnp.var(pval)
    return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0
