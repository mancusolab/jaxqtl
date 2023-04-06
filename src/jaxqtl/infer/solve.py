from abc import ABCMeta, abstractmethod

import equinox as eqx
import jaxopt.linear_solve as ls

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax import Array
from jax.typing import ArrayLike

from ..families.distribution import ExponentialFamily


class LinearSolve(eqx.Module, metaclass=ABCMeta):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
    ) -> Array:
        pass


class QRSolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
    ) -> Array:

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
    ) -> Array:

        # calculate dispersion only for NB model
        # family.alpha = family.calc_dispersion(
        #     y, family.glink.inverse(eta), family.alpha
        # )
        # import jax; jax.debug.breakpoint()
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
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
    ) -> Array:

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k)
        w_half_X = X * w_half

        def _matvec(beta):
            return w_half_X @ beta

        return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
