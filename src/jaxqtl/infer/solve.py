from abc import ABCMeta, abstractmethod

import equinox as eqx
from jaxopt._src.linear_solve import solve_normal_cg

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
        # calculate dispersion only for NB model
        # family.alpha = family.calc_dispersion(
        #     y, family.glink.inverse(eta), family.alpha
        # )

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


# # CG solver
# def solve_normal_cg(matvec: Callable,
#                     b: Any,
#                     ridge: Optional[float] = None,
#                     init: Optional[Any] = None,
#                     **kwargs) -> Any:
#   """Solves the normal equation ``A^T A x = A^T b`` using conjugate gradient.
#
#   This can be used to solve Ax=b using conjugate gradient when A is not
#   hermitian, positive definite.
#
#   Args:
#     matvec: product between ``A`` and a vector.
#     b: pytree.
#     ridge: optional ridge regularization.
#     init: optional initialization to be used by normal conjugate gradient.
#     **kwargs: additional keyword arguments for solver.
#
#   Returns:
#     pytree with same structure as ``b``.
#   """
#
#   def _make_rmatvec(matvec, x):
#       transpose = jax.linear_transpose(matvec, x)
#       return lambda y: transpose(y)[0]
#
#   def _normal_matvec(matvec, x):
#       """Computes A^T A x from matvec(x) = A x."""
#       matvec_x, vjp = jax.vjp(matvec, x)
#       return vjp(matvec_x)[0]
#
#   if init is None:
#     example_x = b  # This assumes that matvec is a square linear operator.
#   else:
#     example_x = init
#
#   try:
#     rmatvec = _make_rmatvec(matvec, example_x)
#   except TypeError:
#     raise TypeError("The initialization `init` of solve_normal_cg is "
#                     "compulsory when `matvec` is nonsquare. It should "
#                     "have the same pytree structure as a solution. "
#                     "Typically, a pytree filled with zeros should work.")
#
#   def normal_matvec(x):
#     return _normal_matvec(matvec, x)
#
#   if ridge is not None:
#     normal_matvec = _make_ridge_matvec(normal_matvec, ridge=ridge)
#
#   Ab = rmatvec(b)  # A.T b
#
#   return jax.scipy.sparse.linalg.cg(normal_matvec, Ab, x0=init, **kwargs)[0]


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
        """not converged for some cases;
        Don't use this. Need future fix
        """

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_half = jnp.sqrt(weight)
        r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
        w_half_X = X * w_half

        def _matvec(beta):
            return w_half_X @ beta

        # import jax; jax.debug.breakpoint()
        # import pdb; pdb.set_trace()
        return solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
        # jax.scipy.sparse.linalg.cg(normal_matvec, Ab, x0=init, **kwargs)[0]
        # return jax.scipy.sparse.linalg.cg(normal_matvec, Ab, x0=init, **kwargs)[0]
        # return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
