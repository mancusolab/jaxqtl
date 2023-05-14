from abc import ABCMeta, abstractmethod
from typing import Tuple

import equinox as eqx

import jax.numpy as jnp

# import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax import Array
from jax.typing import ArrayLike

from ..families.distribution import ExponentialFamily

# import jax.debug
# import jaxopt.linear_solve as ls


class LinearSolve(eqx.Module, metaclass=ABCMeta):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        g: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        pass


# class QRSolve(LinearSolve):
#     def __call__(
#         self,
#         X: ArrayLike,
#         y: ArrayLike,
#         eta: ArrayLike,
#         family: ExponentialFamily,
#         stepsize: float = 1.0,
#         offset_eta: ArrayLike = 0.0,
#     ) -> Array:
#         # calculate dispersion only for NB model
#         # family.alpha = family.calc_dispersion(
#         #     y, family.glink.inverse(eta), family.alpha
#         # )
#
#         mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)
#
#         w_half = jnp.sqrt(weight)
#         r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
#         w_half_r = w_half * r
#         w_half_X = w_half * X
#
#         Q, R = jnpla.qr(w_half_X)
#
#         return jspla.solve_triangular(R, Q.T @ w_half_r)


# class CholeskySolve(LinearSolve):
#     def __call__(
#         self,
#         X: ArrayLike,
#         g: ArrayLike,
#         y: jnp.ndarray,
#         eta: jnp.ndarray,
#         family: ExponentialFamily,
#         stepsize: float = 1.0,
#         offset_eta: ArrayLike = 0.0,
#     ) -> Tuple[Array, Array, Array]:
#
#         mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)
#
#         r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
#
#         XtWX = (X * weight).T @ X
#         XtWy = (X * weight).T @ r
#         factor = jspla.cho_factor(XtWX, lower=True)
#
#         return jspla.cho_solve(factor, XtWy)


class FastSolve(LinearSolve):
    def __call__(
        self,
        X: ArrayLike,
        g: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        family: ExponentialFamily,
        stepsize: float = 1.0,
        offset_eta: ArrayLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        """
        X contains the covariate only, g is the additional column to add
        (XtWX)^-1 XtW (y - mu + eta)
        """
        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)

        w_X = weight * X
        w_g = weight * g
        u1 = w_X.T @ g

        XtWX = w_X.T @ X
        # XtWX = jnp.einsum('ij,ijk->jk', weight, XtX_batch)

        p = X.shape[1]

        factor = jspla.cho_factor(XtWX, lower=True)
        u2 = jspla.cho_solve(factor, u1)

        d = 1 / (w_g.T @ g - u1.T @ u2)
        # use einsum for outer product
        F_inv = jspla.cho_solve(
            factor, jnp.eye(p) + XtWX * d @ (jnp.einsum("ij,jk->ik", u2, u2.T))
        )
        u3 = d * u2

        # use block matrix to calculate (XtWX)^-1 XtW
        block_top = F_inv @ w_X.T - u3 @ (g * weight).T  # (p-1) x n
        block_bottom = -u3.T @ w_X.T + d * (g * weight).T  # 1 x n

        r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
        cov_beta = block_top @ r
        g_beta = block_bottom @ r

        # correct_inv = jspla.inv((jnp.hstack((X, g)) * weight).T @ jnp.hstack((X, g)))
        # import jax; jax.debug.breakpoint()
        infor_se = jnp.sqrt(jnp.append(jnp.diag(F_inv), d.squeeze()))

        return cov_beta, g_beta, infor_se


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


# class CGSolve(LinearSolve):
#     def __call__(
#         self,
#         X: ArrayLike,
#         y: ArrayLike,
#         eta: ArrayLike,
#         family: ExponentialFamily,
#         stepsize: float = 1.0,
#         offset_eta: ArrayLike = 0.0,
#     ) -> Array:
#
#         mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta)
#
#         w_half = jnp.sqrt(weight)
#         r = eta + g_deriv_k * (y - mu_k) * stepsize - offset_eta
#         w_half_X = X * w_half
#
#         def _matvec(beta):
#             return w_half_X @ beta
#
#         # import jax; jax.debug.breakpoint()
#         # import pdb; pdb.set_trace()
#         # res = solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
#         # jax.scipy.sparse.linalg.cg(normal_matvec, Ab, x0=init, **kwargs)[0]
#         return ls.solve_normal_cg(_matvec, r * w_half, init=jnp.zeros((X.shape[1], 1)))
