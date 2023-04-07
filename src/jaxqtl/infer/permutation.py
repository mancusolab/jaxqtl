# TODO: add permutation
# 1) Direct permutation
# 2) beta distribution

from abc import ABCMeta, abstractmethod
from typing import Tuple  # , NamedTuple

import equinox as eqx

# import jax.debug
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.stats as jaxstats
from jax import Array, grad, hessian, jit, lax
from jax.scipy.special import gammaln
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

    def __init__(self, max_perm_direct: int = 100):
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

    # this -might- be sensitive to step size; if we get bad estimates due to step size
    # we can adjust using results from https://arxiv.org/abs/2002.10060
    def loglik(params: ArrayLike, p: ArrayLike, R: int) -> jnp.ndarray:
        k, n = params
        return (
            (k - 1) * jnp.sum(jnp.log(p))
            + (n - 1) * jnp.sum(jnp.log1p(-p))
            - R * (gammaln(k) + gammaln(n) - gammaln(k + n))
        )

    score_fn = grad(loglik)
    hess_fn = hessian(loglik)

    r = len(p_perm)

    def body_fun(val: Tuple):
        old_lik, diff, num_iter, old_param = val
        direction = jnla.solve(
            hess_fn(old_param, p_perm, r), score_fn(old_param, p_perm, r)
        )
        new_param = old_param - stepsize * direction

        new_lik = loglik(new_param, p_perm, r)
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
        init = jnp.ones(2)
        beta_res = infer_beta(p_perm, init, max_iter=self.max_iter_beta)

        adj_p = _calc_adjp_beta(obs_p, beta_res[0:2])

        return adj_p, beta_res
