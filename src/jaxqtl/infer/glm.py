from abc import ABCMeta
from typing import NamedTuple, Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.numpy import linalg as jnpla
from jax.scipy.stats import norm
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily, Gaussian
from jaxqtl.families.utils import t_cdf
from jaxqtl.infer.optimize import irls
from jaxqtl.infer.solve import CholeskySolve, LinearSolve


class GLMState(NamedTuple):
    beta: Array
    se: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    num_iters: Array
    converged: Array
    infor_inv: Array  # for score test
    resid: Array  # for score test, not the working resid!
    alpha: Array  # dispersion parameter in NB model


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
    max_iter: int
    tol: float
    step_size: float

    def __init__(
        self,
        family: ExponentialFamily = Gaussian(),
        solver: LinearSolve = CholeskySolve(),
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.family = family
        self.solver = solver
        self.step_size = step_size

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

    def sumstats(
        self,
        X: ArrayLike,
        y: ArrayLike,
        weight: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        alpha: ArrayLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        infor = (X * weight).T @ X
        infor_inv = jnpla.inv(infor)
        infor_se = jnp.sqrt(jnp.diag(infor_inv))

        # huber sandwich
        huber_v = huber_var(self.family, X, y, eta, mu, alpha)
        huber_se = jnp.sqrt(huber_v)

        return infor_se, huber_se, infor_inv

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        robust_se: bool = False,
        init: ArrayLike = None,
        alpha_init: ArrayLike = jnp.float64(
            jnp.zeros((1,))
        ),  # start with float64 for non-NB family
    ) -> GLMState:
        """Report Wald test p value"""
        beta, n_iter, converged, alpha = irls(
            X,
            y,
            self.family,
            self.solver,
            init,
            self.max_iter,
            self.tol,
            self.step_size,
            offset_eta,
            alpha_init,
        )

        eta = X @ beta + offset_eta
        mu = self.family.glink.inverse(eta)
        resid = (y - mu) * self.family.glink.deriv(
            mu
        )  # note: this is the working resid

        _, _, weight = self.family.calc_weight(X, y, eta, alpha)

        infor_se, huber_se, infor_inv = self.sumstats(X, y, weight, eta, mu, alpha)

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
            infor_inv,
            resid,
            alpha,
        )


def huber_var(
    family: ExponentialFamily,
    X: ArrayLike,
    y: ArrayLike,
    eta: ArrayLike,
    mu: ArrayLike,
    alpha: ArrayLike = jnp.float64(jnp.zeros((1,))),
) -> Array:
    """
    Huber white sandwich estimator using observed hessian
    """
    phi = 1.0
    # calculate observed hessian
    W = (
        1
        / phi
        * (
            family._hlink_score(eta, alpha) / family.glink.deriv(mu)
            - family._hlink_hess(eta, alpha) * (y - mu)
        )
    )
    hess_inv = jnpla.inv(-(X * W).T @ X)

    score_no_x = (y - mu) / (family.variance(mu, alpha) * family.glink.deriv(mu)) * phi
    Bs = (X * (score_no_x**2)).T @ X
    Vs = hess_inv @ Bs @ hess_inv

    return jnp.diag(Vs)
