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

    def sumstats(self, X: ArrayLike, y: ArrayLike, weight: ArrayLike, mu: ArrayLike) -> Tuple[Array, Array, Array]:
        infor = (X * weight).T @ X
        infor_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))

        # huber sandwich
        score = self.family.score(X, y, mu)
        huber_v = huber_var(self.family, X, y, weight, mu, score, infor)
        huber_se = jnp.sqrt(huber_v)

        return infor_se, huber_se, infor

    def fit(self, X, y, offset_eta: ArrayLike = 0.0, robust_se: bool = False) -> GLMState:

        init = self.family.init_eta(y)
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

        infor_se, huber_se, infor = self.sumstats(X, y, weight, mu)
        beta = jnp.reshape(beta, (X.shape[1],))

        df = X.shape[0] - X.shape[1]
        beta_se = jnp.where(robust_se, huber_se, infor_se)

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
    weight: ArrayLike,
    score: ArrayLike,
    infor: ArrayLike,
) -> Array:
    score_no_x = (y - mu) / family.scale(X, y, mu)
    Bs = (X * (score_no_x ** 2)).T @ X
    infor_inv = jnpla.inv(infor)
    Vs = infor_inv @ Bs @ infor_inv
    return jnp.diag(Vs)
