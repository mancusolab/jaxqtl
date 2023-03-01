from abc import ABC  # , abstractmethod

# from typing import List, NamedTuple, Tuple, Union
from typing import Tuple

# import numpy as np
import pandas as pd

import jax.numpy as jnp
import jax.numpy.linalg as jnpla

# from jax import grad
from jax.tree_util import register_pytree_node, register_pytree_node_class

from src.jaxqtl.infer.distribution import AbstractExponential  # , Normal

# here lets consider implementing difference types of linear solvers
# e.g., cholesky, qr, conjugate gradient using either
# hessian or fisher-info matrix


@register_pytree_node_class
class AbstractLinearSolve(ABC):
    # @abstractmethod
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


# QR decomposition for OLS
class OLS(AbstractLinearSolve):
    """
    QR of X --> Q, R
    solve R @ X = Qt @ y

    use Cholesky: (Xt @ X)^-1 = (L @ Lt)^-1 = Lt^-1 @ Lt <-- find L and this invert it
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = jnp.asarray(X)  # will do this in parent class
        self.y = jnp.asarray(y)
        self.df = y.shape[0] - X.shape[1]

    def _fitQR(self) -> None:
        Q, R = jnp.linalg.qr(self.X)
        ytrans = Q.T @ self.y
        self.beta = jnpla.solve(R, ytrans)

    def _sumstats(self) -> None:
        self.pred = self.X @ self.beta  # or Q @ Qt @ y
        resid = jnp.sum(jnp.square(self.pred - self.y))
        self.resid_var = resid / self.df

        L = jnpla.inv(jnpla.cholesky(self.X.T @ self.X))
        RtRinv = L.T @ L
        self.beta_se = jnp.sqrt(jnp.diag(RtRinv) * self.resid_var)

    def __str__(self):
        return f"""
        Beta: {self.beta}
        SE: {self.beta_se}
        df: {self.df}
          """


class IRLS(AbstractLinearSolve):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        family: AbstractExponential,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        self.MAX_ITER = max_iter
        self.TOL = tol
        self.scale = 1  # placebolder; for normal this is variance

        self.X = jnp.asarray(X)  # parent function will do this
        self.y = jnp.asarray(y)
        self.nobs = y.shape[0]
        self.pfeature = X.shape[1]  # feature + 1 (intercept)
        self.df = self.nobs - self.pfeature

        self._hlink = family._hlink
        self._hlink_der = family._hlink_der
        self._glink = family._glink
        self._glink_inv = family._glink_inv
        self._glink_der = family._glink_der
        self._calc_scale = family._calc_scale

    def _calc_weight(
        self, X: jnp.ndarray, beta: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred_k = X @ beta  # (n,)
        mu_k = self._glink_inv(pred_k)
        num = self._hlink_der(pred_k)
        g_deriv_k = self._glink_der(mu_k)
        phi = self._calc_scale(pred_k, self.y, beta)
        weight_k = num / (
            g_deriv_k * phi
        )  # TODO: for normal, the self.scale should be residual variance
        return pred_k, mu_k, g_deriv_k, weight_k

    def _WLS(self, X: jnp.ndarray, y: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        w_half = jnp.sqrt(weight)
        w_y = w_half * y
        w_X = w_half[:, jnp.newaxis] * X

        Q, R = jnp.linalg.qr(w_X)
        ytrans = Q.T @ w_y
        beta = jnpla.solve(R, ytrans)
        return beta

    def _fit_irls(self) -> None:
        converged = False
        old_beta = self._WLS(self.X, self.y, jnp.ones((len(self.y))))

        for iter in range(self.MAX_ITER):
            pred_k, mu_k, g_deriv_k, weight_k = self._calc_weight(self.X, old_beta)
            new_y = pred_k + g_deriv_k * (self.y - mu_k)
            new_beta = self._WLS(self.X, new_y, weight_k)
            converged = jnp.allclose(new_beta, old_beta, atol=self.TOL)
            if converged:
                self.beta = new_beta
                _, _, _, self.weight = self._calc_weight(self.X, new_beta)
                infor = (self.X * self.weight[:, jnp.newaxis]).T @ self.X
                self.beta_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))
                self.converge_iter = iter + 1
                break
            else:
                old_beta = new_beta

    def __str__(self):
        return f"""
        Beta: {self.beta}
        SE: {self.beta_se}
        Converge after {self.converge_iter} iterations
          """


@register_pytree_node_class
class GLM(ABC):
    """
    example:
    model = jaxqtl.GLM(X, y, family=Normal, append=True)
    model.fit(method = "qr")

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        family: AbstractExponential,
        append: bool,
    ) -> None:
        self.nobs = len(y)
        self.X = jnp.asarray(X)  # preprocessed in previous steps
        if append is True:
            self.X = jnp.column_stack((jnp.ones((self.nobs, 1)), self.X))
        self.pfeatures = X.shape[1]  # include intercept term if any
        self.y = jnp.asarray(y)
        self.family = family

    def fit(self):
        # model = OLS(self.X, self.y)
        # model._fitQR()
        # model._sumstats()
        # print(model)

        model = IRLS(self.X, self.y, self.family)
        model._fit_irls()
        print(model)

    # @abstractmethod
    # def __call__(
    #     self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    # ) -> jnp.ndarray:
    #     pass

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
