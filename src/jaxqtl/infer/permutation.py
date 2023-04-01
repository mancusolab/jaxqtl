# TODO: add permutation
# 1) Direct permutation
# 2) beta distribution

from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
import jax.scipy.stats as jaxstats
from jax import Array, grad, random
from jax.config import config
from jax.scipy.special import gammaln
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.utils import cis_GLM

config.update("jax_enable_x64", True)


@register_pytree_node_class
class Permutation(ABC):
    """
    For a given cis-window around a gene (L variants), perform permutation test to
    identify (one candidate) eQTL for this gene.
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init,
        gene_idx: int,
        cis_idx: int,
        sig_level: float = 0.05,
        max_perm_direct=1000,
        max_perm_beta=1000,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def direct_perm(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init,
        cis_idx: int,
        max_perm_direct=1000,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        pvals = jnp.zeros((max_perm_direct,))
        for idx in range(max_perm_direct):
            key_init, key_perm = random.split(key_init)
            y = random.permutation(key_perm, y, axis=0)
            glmstate = cis_GLM(X, y, G, family, cis_idx)  # cis-scan
            pvals.at[idx].set(jnp.min(glmstate.p))  # take strongest signal

        adj_p = self.calc_adjp_naive(obs_p, pvals)

        return pvals, adj_p

    def calc_adjp_naive(self, obs_pval: ArrayLike, pval: ArrayLike) -> Array:
        """
        obs_pval: the strongest nominal p value
        """
        return (jnp.sum(pval < obs_pval) + 1) / (len(pval) + 1)

    def calc_adjp_beta(self, p_obs: ArrayLike, params: ArrayLike) -> Array:
        """
        p_obs is a vector of nominal p value in cis window
        """
        k, n = params
        p_adj = jaxstats.beta.cdf(jnp.min(p_obs), k, n)
        return p_adj

    def tree_flatten(self):
        children = ()
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


# class DirectPerm(Permutation):
#     """
#     For a given cis-window with L variants, permutate phenotype for R times,
#     each time, take the strongest signal -> R samples
#     calculate adjusted p = P(perm_p < obs_p) = (r+1)/(R+1), r = # stronger signals
#     """
#
#     def __call__(
#         self,
#         X: ArrayLike,
#         y: ArrayLike,
#         G: ArrayLike,
#         obs_p: ArrayLike,
#         family: ExponentialFamily,
#         key_init,
#         gene_idx: int,
#         cis_idx: int,
#         sig_level: float = 0.05,
#         max_perm_direct=1000,
#         max_perm_beta=1000,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#
#         pvals = jnp.zeros((max_perm_direct, ))
#         for idx in range(max_perm_direct):
#             key_init, key_perm = random.split(key_init)
#             y = random.permutation(key_perm, y, axis=0)
#             glmstate = cis_GLM(X, y, G, family, cis_idx)  # cis-scan
#             pvals.at[idx].set(jnp.min(glmstate.p))  # take strongest signal
#
#         adj_p = self.calc_adjp_naive(obs_p, pvals)
#
#         return pvals, adj_p


class BetaPerm(Permutation):
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        G: ArrayLike,
        obs_p: ArrayLike,
        family: ExponentialFamily,
        key_init,
        gene_idx: int,
        cis_idx: int,
        sig_level: float = 0.05,
        max_perm_direct=1000,
        max_perm_beta=1000,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        we perform R permutation, where R is around 50, 100, 1000
        """
        # pseudo code
        p_perm, _ = self.direct_perm(
            X, y, G, obs_p, family, key_init, cis_idx, max_perm_direct
        )
        init = jnp.array([1.0, 1.0])
        k, n = self._infer_beta(p_perm, init, max_iter=max_perm_beta)
        adj_p = self.calc_adjp_beta(obs_p, jnp.array([k, n]))
        return jnp.array([k, n]), adj_p, k, n

    @staticmethod
    def _infer_beta(
        p_perm: ArrayLike,
        init: ArrayLike,
        stepsize=1,
        tol=1e-3,
        max_iter=1000,
    ) -> jnp.ndarray:
        """
        given p values from R permutations (strongest signals),
        use newton's method to estimate beta distribution parameters:
        p ~ Beta(k, n)
        """

        def loglik(k: float, n: float, p: ArrayLike, R: int) -> jnp.ndarray:
            return (
                (k - 1) * jnp.sum(jnp.log(p))
                + (n - 1) * jnp.sum(jnp.log1p(-p))
                - R * (gammaln(k) + gammaln(n) - gammaln(k + n))
            )

        score_k_fn = grad(loglik, 0)
        score_n_fn = grad(loglik, 1)

        hess_k_fn = grad(score_k_fn, 0)
        hess_n_fn = grad(score_n_fn, 1)

        diff = 10000.0
        num_iters = 0
        old_k, old_n = init
        r = len(p_perm)

        while diff > tol and num_iters <= max_iter:
            new_k = old_k - stepsize * score_k_fn(old_k, old_n, p_perm, r) / hess_k_fn(
                old_k, old_n, p_perm, r
            )
            new_n = old_n - stepsize * score_n_fn(old_k, old_n, p_perm, r) / hess_n_fn(
                old_k, old_n, p_perm, r
            )
            old_lik = loglik(old_k, old_n, p_perm, r)
            new_lik = loglik(new_k, new_n, p_perm, r)
            diff = jnp.abs(old_lik - new_lik)

            if diff < tol:
                break

            old_k = new_k
            old_n = new_n
            num_iters += 1

        return jnp.array([new_k, new_n])
