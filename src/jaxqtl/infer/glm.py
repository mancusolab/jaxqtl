from typing import NamedTuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from ..families.distribution import (
    ExponentialFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from ..families.utils import t_cdf
from .optimize import irls
from .solve import CholeskySolve, LinearSolve
from .stderr import ErrVarEstimation, FisherInfoError


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


class GLM(eqx.Module):
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

    family: ExponentialFamily = Gaussian()
    solver: LinearSolve = CholeskySolve()
    max_iter: int = 1000
    tol: float = 1e-3
    step_size: float = 1.0

    def wald_test(self, TS: ArrayLike, df: int) -> Array:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
        """
        if isinstance(self.family, Gaussian):
            pval = t_cdf(-abs(TS), df) * 2  # follow t(n-p-1) for Gaussian
        else:
            pval = norm.cdf(-abs(TS)) * 2  # follow Normal(0, 1), this gives more accurate p value than chi2(1)

        return pval

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        init: ArrayLike = None,
        alpha_init: ScalarLike = 0.0,
        se_estimator: ErrVarEstimation = FisherInfoError(),
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
        resid = (y - mu) * self.family.glink.deriv(mu)  # note: this is the working resid

        _, _, weight = self.family.calc_weight(X, y, eta, alpha)

        resid_covar = se_estimator(self.family, X, y, eta, mu, weight, alpha)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))

        df = X.shape[0] - X.shape[1]
        beta = beta.squeeze()  # (p,)
        stat = beta / beta_se

        pval_wald = self.wald_test(stat, df)

        return GLMState(
            beta,
            beta_se,
            pval_wald,
            eta,
            mu,
            weight,
            n_iter,
            converged,
            resid_covar,
            resid,
            alpha,
        )

    def calc_eta_and_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
    ) -> Array:
        n, p = X.shape
        init_val = self.family.init_eta(y)
        if isinstance(self.family, NegativeBinomial):
            jaxqtl_pois = GLM(family=Poisson(), max_iter=self.max_iter)
            glm_state_pois = jaxqtl_pois.fit(X, y, init=init_val, offset_eta=offset_eta)

            # fit covariate-only model (null)
            alpha_init = n / jnp.sum((y / self.family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
            eta = glm_state_pois.eta
            disp = self.family.estimate_dispersion(X, y, eta, alpha=alpha_init)

            # convert disp to 0.1 if bad initialization
            disp = jnp.nan_to_num(disp, nan=0.1)
        else:
            eta = jnp.asarray(0.0)
            disp = jnp.asarray(0.0)

        return eta, disp
