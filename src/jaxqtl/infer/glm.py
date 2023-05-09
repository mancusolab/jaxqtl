from abc import ABCMeta
from typing import NamedTuple, Optional, Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.numpy import linalg as jnpla
from jax.scipy.stats import norm  # , t (not supported rn), chi2
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily, Gaussian
from jaxqtl.families.utils import t_cdf
from jaxqtl.infer.optimize import irls
from jaxqtl.infer.solve import CholeskySolve, LinearSolve


# change jnp.ndarray --> np.ndarray for mutable array
class GLMState(NamedTuple):
    beta: Array
    se: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    num_iters: Array
    converged: Array


class GLM(eqx.Module, metaclass=ABCMeta):
    """
    example:
    model = jaxqtl.GLM(X, y, family="Gaussian", solver="qr", append=True)
    res = model.fit()
    print(res)

    need check domain in glink function

     from statsmodel code:
     ============= ===== === ===== ====== ======= === ==== ====== ====== ====
     Family        ident log logit probit cloglog pow opow nbinom loglog logc
     ============= ===== === ===== ====== ======= === ==== ====== ====== ====
     Gaussian      x     x   x     x      x       x   x     x      x
     inv Gaussian  x     x                        x
     binomial      x     x   x     x      x       x   x           x      x
     Poisson       x     x                        x
     neg binomial  x     x                        x        x
     gamma         x     x                        x
     Tweedie       x     x                        x
     ============= ===== === ===== ====== ======= === ==== ====== ====== ====
    """

    X: ArrayLike
    y: ArrayLike
    family: ExponentialFamily
    solver: LinearSolve
    init: ArrayLike
    maxiter: int
    tol: float
    stepsize: float

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily = Gaussian(),
        solver: LinearSolve = CholeskySolve(),
        append: bool = True,
        maxiter: int = 100,
        tol: float = 1e-3,
        init: Optional[ArrayLike] = None,
        stepsize: float = 1.0,
    ) -> None:
        nobs = len(y)
        self.maxiter = maxiter
        self.tol = tol

        self.X = jnp.asarray(X)  # preprocessed in previous steps

        try:
            self.X.shape[1]
        except IndexError:
            self.X = self.X.reshape((len(self.X), 1))  # reshape 1D array

        if append:
            self.X = jnp.column_stack((jnp.ones((nobs, 1)), self.X))

        self.y = jnp.asarray(y).reshape((nobs, 1))

        self.family = family
        self.solver = solver
        self.init = init if init is not None else family.init_eta(self.y)
        self.stepsize = stepsize

    def wald_test(self, TS, df) -> Array:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
        """
        if isinstance(self.family, Gaussian):
            pval = t_cdf(-abs(TS), df) * 2  # follow t(n-p-1) for Gaussian
        else:
            pval = (
                norm.cdf(-abs(TS)) * 2
            )  # follow Normal(0, 1), this gives more accurate p value than chi2(1)
            # pval = 1 - chi2.cdf(jnp.square(TS), 1)  # equivalently chi2(df=1)

        return pval

    @staticmethod
    def score_test_add_g(
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        glm_null_res: GLMState,
        df: float,
    ) -> Array:
        """test for additional covariate g
        X is the full design matrix containing covariates and g
        calculate score in full model using the model fitted from null model
        """
        score_null = family.score(X, y, glm_null_res.mu)
        score_null_info = (X * glm_null_res.glm_wt).T @ X
        TS_chi2 = score_null.T @ jnpla.inv(score_null_info) @ score_null
        # pval = 1 - chi2.cdf(TS_chi2, df)
        pval = norm.cdf(-abs(jnp.sqrt(TS_chi2))) * 2
        return pval

    def sumstats(self, weight) -> Tuple[Array, Array]:
        infor = (self.X * weight).T @ self.X
        beta_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))
        return beta_se, infor

    def fit(self, offset_eta: ArrayLike = 0.0) -> GLMState:
        """Report Wald test p value"""
        beta, n_iter, converged = irls(
            self.X,
            self.y,
            self.family,
            self.solver,
            self.init,
            self.maxiter,
            self.tol,
            self.stepsize,
            offset_eta,
        )
        eta = self.X @ beta + offset_eta
        mu = self.family.glink.inverse(eta)

        _, _, weight = self.family.calc_weight(self.X, self.y, eta)

        beta_se, infor = self.sumstats(weight)
        beta = jnp.reshape(beta, (self.X.shape[1],))

        df = self.X.shape[0] - self.X.shape[1]
        TS = beta / beta_se

        pval_wald = self.wald_test(TS, df)

        return GLMState(
            beta,
            beta_se,
            pval_wald,
            eta,
            mu,
            weight,
            n_iter,
            converged,
        )
