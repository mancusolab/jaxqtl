from abc import ABCMeta
from typing import NamedTuple, Optional, Tuple

import equinox as eqx

# import jax.debug
from jax import Array, numpy as jnp
from jax.numpy import linalg as jnpla
from jax.scipy.stats import norm  # , t (not supported rn), chi2
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily, Gaussian
from jaxqtl.families.utils import t_cdf
from jaxqtl.infer.optimize import irls
from jaxqtl.infer.solve import CholeskySolve, LinearSolve

# import jax.scipy.linalg as jspla


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

    family: ExponentialFamily
    solver: LinearSolve
    maxiter: int
    tol: float
    stepsize: float

    def __init__(
        self,
        family: ExponentialFamily = Gaussian(),
        solver: LinearSolve = CholeskySolve(),
        maxiter: int = 100,
        tol: float = 1e-3,
        stepsize: float = 1.0,
    ) -> None:

        self.maxiter = maxiter
        self.tol = tol
        self.family = family
        self.solver = solver
        self.stepsize = stepsize

    def wald_test(self, TS: ArrayLike, df: int) -> Array:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
        """
        if isinstance(self.family, Gaussian):
            pval = t_cdf(-abs(TS), df) * 2  # follow t(n-p-1) for Gaussian
        else:
            pval = (
                norm.cdf(-abs(TS)) * 2
            )  # follow Normal(0, 1), this gives more accurate p value than chi2(1)

        return pval

    def score_test_add_g(
        self,
        g: ArrayLike,
        X_cov: ArrayLike,
        y: ArrayLike,
        glm_null_res: GLMState,
        offset_eta: ArrayLike = 0.0,
    ) -> Tuple[Array, Array]:
        """test for additional covariate g
        only require fit null model using fitted covariate only model + new vector g
        X is the full design matrix containing covariates and g
        calculate score in full model using the model fitted from null model
        """
        x_W = X_cov * glm_null_res.glm_wt
        score_null_info = (x_W).T @ X_cov
        g_regout = g - X_cov @ jnpla.inv(score_null_info) @ x_W.T @ g
        mu_k, g_deriv_k, weight = self.family.calc_weight(X_cov, y, glm_null_res.eta)
        resid = g_deriv_k * (y - mu_k) * self.stepsize - offset_eta

        w_g_regout = g_regout * glm_null_res.glm_wt
        Z = w_g_regout.T @ resid / jnp.sqrt(w_g_regout.T @ g_regout)
        pval = norm.cdf(-abs(Z)) * 2
        return Z, pval

    def sumstats(
        self, X: ArrayLike, y: ArrayLike, weight: ArrayLike, mu: ArrayLike
    ) -> Tuple[Array, Array]:
        infor = (X * weight).T @ X
        infor_inv = jnpla.inv(infor)
        infor_se = jnp.sqrt(jnp.diag(infor_inv))

        # huber sandwich
        huber_v = huber_var(self.family, X, y, mu, infor_inv)
        huber_se = jnp.sqrt(huber_v)

        return infor_se, huber_se

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        robust_se: bool = False,
        init: Optional[ArrayLike] = None,
    ) -> GLMState:

        # init = self.family.init_eta(y)
        """Report Wald test p value"""
        beta, n_iter, converged = irls(
            X,
            y,
            self.family,
            self.solver,
            init,
            self.maxiter,
            self.tol,
            self.stepsize,
            offset_eta,
        )
        eta = X @ beta + offset_eta
        mu = self.family.glink.inverse(eta)

        _, _, weight = self.family.calc_weight(X, y, eta)

        infor_se, huber_se = self.sumstats(X, y, weight, mu)

        df = X.shape[0] - X.shape[1]
        beta_se = jnp.where(robust_se, huber_se, infor_se)

        beta = beta.squeeze()  # (p,)
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


def huber_var(
    family: ExponentialFamily,
    X: ArrayLike,
    y: ArrayLike,
    mu: ArrayLike,
    infor_inv: ArrayLike,
) -> Array:
    """
    TODO: this will break
    """
    score_no_x = (y - mu) / family.scale(X, y, mu)
    Bs = (X * (score_no_x ** 2)).T @ X
    Vs = infor_inv @ Bs @ infor_inv
    return jnp.diag(Vs)
