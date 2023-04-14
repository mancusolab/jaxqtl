from abc import ABCMeta, abstractmethod
from typing import Tuple

import equinox as eqx

import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.stats as jaxstats
from jax import Array, grad, jit, lax
from jax.scipy.special import gammaln, polygamma
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.utils import cis_scan


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
    ) -> Tuple[Array, Array]:
        def _func(key, x):
            key, p_key = rdm.split(key)
            y_p = rdm.permutation(p_key, y, axis=0)
            glmstate = cis_scan(X, G, y_p, family)
            return key, glmstate.p[-1]

        key, pvals = lax.scan(_func, key_init, xs=None, length=self.max_perm_direct)

        adj_p = _calc_adjp_naive(obs_p, pvals)

        return adj_p, pvals


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

    def loglik(params: ArrayLike, p: ArrayLike) -> Array:
        k, n = params
        # cannot use jaxstats.beta.logpdf due to bug when deriv at 1, 1
        return (
            (k - 1) * jnp.sum(jnp.log(p))
            + (n - 1) * jnp.sum(jnp.log1p(-p))
            - len(p) * (gammaln(k) + gammaln(n) - gammaln(k + n))
        )

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
    ) -> Tuple[Array, Array]:
        """Perform permutation to estimate beta distribution parameters
        Repeat direct_perm for max_direct_perm times --> vector of lead p values
        Estimate Beta(k,n) using Newton's gradient descent, step size = 1
        Returns:
            k, n estimates
            adjusted p value for lead SNP
        """
        _, p_perm = super().__call__(
            X,
            y,
            G,
            obs_p,
            family,
            key_init,
        )
        # init = jnp.ones(2)  # initialize with 1
        p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
        k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
        n_init = k_init * (1 / p_mean - 1)
        init = jnp.array([k_init, n_init])

        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)

        adj_p = _calc_adjp_beta(obs_p, beta_res[0:2])

        return adj_p, beta_res
