from abc import ABCMeta, abstractmethod

import equinox as eqx

# from jaxopt._src.linear_solve import solve_normal_cg
from jaxopt import linear_solve as ls

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax import Array
from jax.typing import ArrayLike

from ..families.distribution import ExponentialFamily

# import jax.debug


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
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        pass


class QRSolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
        w_half_r = w_half * r
        w_half_X = w_half * X

        Q, R = jnpla.qr(w_half_X)

        return jspla.solve_triangular(R, Q.T @ w_half_r)


class CholeskySolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        y: jnp.ndarray,
        eta: jnp.ndarray,
        family: ExponentialFamily,
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta

        XtWX = (X * weight).T @ X
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
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        """not converged for some cases in real data;
        Used jaxopt solve_normal_cg, not always gurantee convergence (not allow specify tol)
        !!! Don't use this. Need future fix
        """

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
        w_half_X = X * w_half

        def _matvec(beta):
            return w_half_X @ beta

        # import jax; jax.debug.breakpoint()
        # import pdb; pdb.set_trace()
        return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
