from abc import ABCMeta, abstractmethod
from typing import Tuple

import equinox as eqx

import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.stats as jaxstats
from jax import Array, grad, jit, lax
from jax.scipy.special import polygamma
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily

# from jaxqtl.infer.glm import GLM
from jaxqtl.infer.utils import cis_scan

# import jaxopt


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
    ) -> Tuple[Array, Array, Array]:
        def _func(key, x):
            key, p_key = rdm.split(key)
            y_p = rdm.permutation(p_key, y, axis=0)
            # glmstate_null = GLM(
            #     X=X,
            #     y=y_p,
            #     family=family,
            #     append=False,
            #     maxiter=100,
            # ).fit()
            glmstate = cis_scan(X, G, y_p, family, offset_eta)
            allTS = jnp.abs(glmstate.beta / glmstate.se)
            return key, allTS.max()  # glmstate.p.min()

        # key, pvals = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)
        key, TS = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)
        pvals = pval_from_Zstat(TS, 1.0)
        adj_p = _calc_adjp_naive(obs_p, pvals)

        return adj_p, pvals, TS


@jit
def _calc_adjp_naive(obs_pval: ArrayLike, pval: ArrayLike) -> Array:
    """
    obs_pval: the strongest nominal p value
    """
    return (jnp.sum(pval < obs_pval) + 1) / (len(pval) + 1)


@jit
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


@jit
def _calc_adjp_beta(p_obs: ArrayLike, params: ArrayLike) -> Array:
    """
    p_obs is a vector of nominal p value in cis window
    """
    k, n = params
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
    ) -> Tuple[Array, Array]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        _, p_perm, TS = super().__call__(X, y, G, obs_p, family, key_init, offset_eta)

        # TODO: calculate true df and adjust every p_perm accordingly
        # dof_init = 1.0
        # # https://github.com/google/jaxopt/blob/main/jaxopt/_src/scipy_wrappers.py  #  Nelder-Mead
        # # res = scipy.optimize.minimize(lambda x: np.abs(df_cost(TS, x)), dof_init, method='Nelder-Mead', tol=tol)
        # opt = jaxopt.ScipyMinimize(fun=lambda x: jnp.abs(df_cost(TS, x)),
        #                            method='Nelder-Mead', tol=1e-3, maxiter=1000)
        # opt_res = opt.run(init_params=dof_init)
        #
        # if opt_res.state.success:
        #     true_dof = opt_res.params
        # else:
        #     true_dof = dof_init
        # p_perm = pval_from_Zstat(TS, true_dof)

        # init = jnp.ones(2)  # initialize with 1
        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
        n_init = k_init * (1 / p_mean - 1)
        init = jnp.array([k_init, n_init])

        # infer beta based on adjusted p_perm
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)

        adj_p = _calc_adjp_beta(obs_p, beta_res[0:2])

        return adj_p, beta_res


def pval_from_Zstat(TS: ArrayLike, dof: float):
    # TS is the beta / se
    return 1 - jaxstats.chi2.cdf(jnp.square(TS), dof)


def df_cost(TS, dof):
    """minimize abs(1-alpha) as a function of M_eff"""
    pval = pval_from_Zstat(TS, dof)
    mean = jnp.mean(pval)
    var = jnp.var(pval)
    return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0
