from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla

from jaxopt import linear_solve as ls
from jaxtyping import Array, ArrayLike


class LinearSolve(eqx.Module):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        pass


class QRSolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        w_half = jnp.sqrt(weights)
        w_half_r = w_half * r
        w_half_X = w_half * X

        Q, R = jnpla.qr(w_half_X)

        return jspla.solve_triangular(R, Q.T @ w_half_r)


class CholeskySolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        Xw = X * weights
        XtWX = Xw.T @ X
        XtWy = Xw.T @ r
        factor = jspla.cho_factor(XtWX, lower=True)

        return jspla.cho_solve(factor, XtWy)


class CGSolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        """not converged for some cases in real data;
        Used jaxopt solve_normal_cg, not always gurantee convergence (not allow specify tol)
        !!! Don't use this. Need future fix
        """
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half

        def _matvec(beta):
            return w_half_X @ beta

        # import jax; jax.debug.breakpoint()
        # import pdb; pdb.set_trace()
        return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
