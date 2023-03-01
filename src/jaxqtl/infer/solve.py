from abc import ABC, abstractmethod

import jaxopt.linear_solve as ls

import jax.numpy as jnp
import jax.scipy.linalg as jspla

# from jax import grad
from jax.tree_util import register_pytree_node, register_pytree_node_class

from src.jaxqtl.infer.distribution import AbstractExponential


@register_pytree_node_class
class AbstractLinearSolve(ABC):
    @abstractmethod
    def __call__(
        self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    ) -> jnp.ndarray:
        # type hint: X, y input should be jnp.ndarray, output should be jnp.ndarray
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def tree_flatten(self):
        children = ()
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


class QRSolve(AbstractLinearSolve):
    def __call__(
        self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    ) -> jnp.ndarray:

        eta = model.eta(X, y)
        pred_k, mu_k, g_deriv_k, weight = model._calc_weight(
            X, y, eta
        )  # TODO how to get eta

        w_half = jnp.sqrt(weight)
        r = pred_k + g_deriv_k * (y - mu_k)
        w_r = w_half * r
        w_X = w_half[:, jnp.newaxis] * X

        Q, R = jnp.linalg.qr(w_X)

        return jspla.solve_triangular(R, Q.T @ w_r)


class CholeskySolve(AbstractLinearSolve):
    def __call__(
        self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    ) -> jnp.ndarray:

        eta = model.eta(X, y)
        pred_k, mu_k, g_deriv_k, weight = model._calc_weight(
            X, y, eta
        )  # TODO how to get eta

        w_half = jnp.sqrt(weight)
        r = pred_k + g_deriv_k * (y - mu_k)
        w_X = w_half[:, jnp.newaxis] * X

        XtWX = w_X.T @ w_X
        XtWy = w_X.T @ r
        factor = jspla.cho_factor(XtWX)

        return jspla.cho_solve(factor, XtWy)


class CGSolve(AbstractLinearSolve):
    def __call__(
        self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    ) -> jnp.ndarray:

        eta = model.eta(X, y)
        pred_k, mu_k, g_deriv_k, weight = model._calc_weight(
            X, y, eta
        )  # TODO how to get eta

        w_half = jnp.sqrt(weight)
        r = pred_k + g_deriv_k * (y - mu_k)

        def _matvec(beta):
            return w_half * (X @ beta)

        return ls.solve_normal_cg(_matvec, r, init=jnp.zeros(X.shape[1]))
