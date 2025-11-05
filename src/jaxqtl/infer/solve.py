from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
import lineax as lx

from jaxtyping import Array, ArrayLike


class LinearSolve(eqx.Module):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    @abstractmethod
    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        """Linear equation solver

        :param X: covariate data matrix (nxp)
        :param r: residuals
        :param weights: weights for each individual
        """
        pass

    @abstractmethod
    def lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
    ) -> Array:
        """Linear equation solver

        :param X: covariate data matrix (nxp)
        :param r: residuals
        """
        pass


class QRSolve(LinearSolve):
    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        w_half = jnp.sqrt(weights)
        w_half_r = w_half * r
        w_half_X = X * w_half[:, jnp.newaxis]

        Q, R = jnpla.qr(w_half_X)

        return jspla.solve_triangular(R, Q.T @ w_half_r)

    def lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
    ) -> Array:
        Q, R = jnpla.qr(X)
        return jspla.solve_triangular(R, Q.T @ r)


class CholeskySolve(LinearSolve):
    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        Xw = X * weights[:, jnp.newaxis]
        XtWX = Xw.T @ X
        XtWy = Xw.T @ r
        factor = jspla.cho_factor(XtWX, lower=True)

        return jspla.cho_solve(factor, XtWy)

    def lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
    ) -> Array:
        XtX = X.T @ X
        Xty = X.T @ r
        factor = jspla.cho_factor(XtX, lower=True)

        return jspla.cho_solve(factor, Xty)


class CGSolve(LinearSolve):
    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        """not converged for some cases in real data;
        Used jaxopt solve_normal_cg, not always gurantee convergence (not allow specify tol),
        Now switch lineax
        """
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half[:, jnp.newaxis]

        # Method 1: CG solve
        # cg_solver = lx.CG(atol=1e-5, rtol=1e-5)
        # XtWX = w_half_X.T @ w_half_X
        # operator = lx.MatrixLinearOperator(XtWX, lx.positive_semidefinite_tag)
        # b = (weights * X).T @ r
        # sol = lx.linear_solve(operator, b.squeeze(), solver=cg_solver)

        # Method 2 (faster): CG solve using normal equation which solve A^t A x = A^t b
        # Here we solve (XtWX) beta = XtW b, so A = X * sqrt(W), b = sqrt(W) * r
        ncg_solver = lx.NormalCG(atol=1e-5, rtol=1e-5)
        operator = lx.MatrixLinearOperator(w_half_X)
        b = w_half * r
        sol = lx.linear_solve(operator, b, solver=ncg_solver)

        return sol.value

    def lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
    ) -> Array:
        """not converged for some cases in real data;
        Used jaxopt solve_normal_cg, not always gurantee convergence (not allow specify tol),
        Now switch lineax
        """
        # Method 2 (faster): CG solve using normal equation which solve A^t A x = A^t b
        # Here we solve (XtWX) beta = XtW b, so A = X * sqrt(W), b = sqrt(W) * r
        ncg_solver = lx.NormalCG(atol=1e-5, rtol=1e-5)
        operator = lx.MatrixLinearOperator(X)
        sol = lx.linear_solve(operator, r, solver=ncg_solver)

        return sol.value
