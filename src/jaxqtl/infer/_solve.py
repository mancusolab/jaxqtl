from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
import lineax as lx

from jaxtyping import Array, ArrayLike


class LinearSolve(eqx.Module):
    r"""Base interface for linear solvers used inside IRLS/GLM fitting.

    During iteratively reweighted least squares (IRLS), each iteration reduces to a (weighted) least-squares solve.
    Implementations provide:

    - `wgt_lstsq`: weighted least squares.
    - `lstsq`: unweighted least squares.

    These are used by [`jaxqtl.infer.irls`][] and [`jaxqtl.infer.lstsq`][].
    """

    @abstractmethod
    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        r"""Solve a weighted least-squares problem.

        This computes $\hat\beta$ solving:
        $\hat\beta = \arg\min_\beta \sum_{i=1}^n w_i (r_i - x_i^\top \beta)^2$.

        **Arguments:**

        - `X`: Design matrix $X$ with shape `(n, p)`.
        - `r`: Working response vector $r$ with shape `(n,)`.
        - `weights`: Non-negative weights $w$ with shape `(n,)`.

        **Returns:**

        Coefficient vector $\hat\beta$ with shape `(p,)`.
        """
        pass

    @abstractmethod
    def lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
    ) -> Array:
        r"""Solve an unweighted least-squares problem.

        This computes $\hat\beta$ solving:
        $\hat\beta = \arg\min_\beta \lVert r - X\beta \rVert_2^2$.

        **Arguments:**

        - `X`: Design matrix $X$ with shape `(n, p)`.
        - `r`: Response vector $r$ with shape `(n,)`.

        **Returns:**

        Coefficient vector $\hat\beta$ with shape `(p,)`.
        """
        pass


class QRSolve(LinearSolve):
    r"""Solve least-squares problems using a QR decomposition.

    QR-based solvers are typically more numerically stable than normal-equation solvers for ill-conditioned designs.
    """

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
    r"""Solve least-squares problems via normal equations and a Cholesky factorization.

    This forms $(X^\top W X)\beta = X^\top W r$ (or $(X^\top X)\beta = X^\top r$) and solves using a Cholesky
    factorization. It is often fast for well-conditioned designs.
    """

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
    r"""Solve least-squares problems using conjugate gradients (via `lineax`).

    This is useful when forming $(X^\top W X)$ is expensive. The returned solution may be approximate depending on the
    stopping criteria used by the CG solver.
    """

    def wgt_lstsq(
        self,
        X: ArrayLike,
        r: ArrayLike,
        weights: ArrayLike,
    ) -> Array:
        r"""Solve weighted least squares using a normal-equation CG method.

        **Arguments:**

        - `X`: Design matrix $X$ with shape `(n, p)`.
        - `r`: Working response vector $r$ with shape `(n,)`.
        - `weights`: Non-negative weights $w$ with shape `(n,)`.

        **Returns:**

        Coefficient vector $\hat\beta$ with shape `(p,)`.
        """
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half[:, jnp.newaxis]

        # CG solve using normal equation which solve A^t A x = A^t b
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
        r"""Solve unweighted least squares using a normal-equation CG method.

        **Arguments:**

        - `X`: Design matrix $X$ with shape `(n, p)`.
        - `r`: Response vector $r$ with shape `(n,)`.

        **Returns:**

        Coefficient vector $\hat\beta$ with shape `(p,)`.
        """
        # CG solve using normal equation which solve A^t A x = A^t b
        ncg_solver = lx.NormalCG(atol=1e-5, rtol=1e-5)
        operator = lx.MatrixLinearOperator(X)
        sol = lx.linear_solve(operator, r, solver=ncg_solver)

        return sol.value
