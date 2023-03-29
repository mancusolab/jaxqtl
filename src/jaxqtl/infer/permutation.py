# TODO: add permutation
# 1) Direct permutation
# 2) Adpative permutation
# 3) beta distribution

from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
from jax import Array, grad, random
from jax.scipy.special import gammaln
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM


@register_pytree_node_class
class Permutation(ABC):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: ExponentialFamily,
        key_init,
        obs_pval,
        sig_level: float = 0.05,
        max_perm=10000,
    ) -> jnp.ndarray:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def calc_permp(self, obs_pval: ArrayLike, pval: ArrayLike) -> Array:
        return (jnp.sum(pval < obs_pval) + 1) / (len(pval) + 1)

    def tree_flatten(self):
        children = ()
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


class DirectPerm(Permutation):
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: ExponentialFamily,
        key_init,
        obs_pval,
        sig_level: float = 0.05,
        max_perm=10000,
    ) -> jnp.ndarray:
        """ "Permutation with fixed number of perm iters"""

        # pseudo code
        pvals = []
        key_init, key_perm = random.split(key_init)
        for idx in range(max_perm):
            Xperm = random.permutation(key_perm, X, axis=0)
            # not sure if we want to start new instance of GLM family
            glmstate = GLM.fit(Xperm)
            pvals.append(glmstate.p)

        return pvals


class BetaPerm(Permutation):
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: ExponentialFamily,
        key_init,
        obs_pval,
        sig_level: float = 0.05,
        max_perm=1000,
    ) -> jnp.ndarray:

        # pseudo code
        # DirectPerm(X, y ,family, key_init.)

        return


def infer_beta(
    R, perm_p, init: Tuple, stepsize=1, tol=1e-3, max_iter=1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def loglik(k: float, n: float, R: int, p: ArrayLike) -> jnp.ndarray:
        return (
            (k - 1) * jnp.sum(jnp.log(p))
            + (n - 1) * jnp.sum(jnp.log1p(-p))
            - R * (gammaln(k) + gammaln(n) + gammaln(k * n))
        )

    score_k_fn = grad(loglik, 0)
    score_n_fn = grad(loglik, 1)

    hess_k_fn = grad(score_k_fn, 0)
    hess_n_fn = grad(score_n_fn, 0)

    diff = 10000.0
    num_iters = 0
    old_k, old_n = init

    while diff > tol and num_iters <= max_iter:
        new_k = old_k - stepsize * score_k_fn(old_k, old_n, R, perm_p) / hess_k_fn(
            old_k, old_n, R, perm_p
        )
        new_n = old_n - stepsize * score_n_fn(old_k, old_n, R, perm_p) / hess_n_fn(
            old_k, old_n, R, perm_p
        )
        diff = jnp.array([new_k, new_n]) - jnp.array([old_k, old_n])

        if diff < 0:
            break

        old_k, old_n = new_k, new_n

    return new_k, new_n
