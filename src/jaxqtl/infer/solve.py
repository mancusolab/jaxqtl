from abc import ABC, abstractmethod

import jaxopt.linear_solve as ls

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax.tree_util import register_pytree_node, register_pytree_node_class

from jaxqtl.families.distribution import ExponentialFamily


@register_pytree_node_class
class LinearSolve(ABC):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        eta: jnp.ndarray,
        family: ExponentialFamily,
    ) -> jnp.ndarray:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def tree_flatten(self):
        children = ()
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


class QRSolve(LinearSolve):
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        eta: jnp.ndarray,
        family: ExponentialFamily,
    ) -> jnp.ndarray:

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k)
        w_r = w_half * r
        w_X = w_half * X

        Q, R = jnpla.qr(w_X)

        return jspla.solve_triangular(R, Q.T @ w_r)


class CholeskySolve(LinearSolve):
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        eta: jnp.ndarray,
        family: ExponentialFamily,
    ) -> jnp.ndarray:

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k)
        w_X = w_half * X

        XtWX = w_X.T @ w_X
        XtWy = (X * weight).T @ r
        factor = jspla.cho_factor(XtWX, lower=True)

        return jspla.cho_solve(factor, XtWy)


class CGSolve(LinearSolve):
    def __call__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        eta: jnp.ndarray,
        family: ExponentialFamily,
    ) -> jnp.ndarray:

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k)
        w_half_X = X * w_half

        def _matvec(beta):
            return w_half_X @ beta

        return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
