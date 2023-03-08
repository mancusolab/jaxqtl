from typing import NamedTuple, Tuple

from scipy.stats import norm, t  # , chi2

from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax.tree_util import register_pytree_node_class

from .families.distribution import Binomial, Gaussian, Poisson
from .optimize import irls
from .solve import CGSolve, CholeskySolve, QRSolve


class GLMState(NamedTuple):
    beta: jnp.ndarray
    se: jnp.ndarray
    p: jnp.ndarray
    num_iters: int
    converged: bool


@register_pytree_node_class
class GLM:
    """
    example:
    model = jaxqtl.GLM(X, y, family="Gaussian", solver="qr", append=True)
    res = model.fit()
    print(res)

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

    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: str,
        link: str = "canonical",  # [canonical, Identity, Log, Logit, ...]
        solver: str = "qr",
        append: bool = True,
        init: str = "default",  # [default or OLS]
        maxiter: int = 100,
        seed: int = None,
    ) -> None:
        nobs = len(y)
        self.seed = seed
        self.init = init
        self.maxiter = maxiter

        self.X = jnp.asarray(X)  # preprocessed in previous steps
        if append is True:
            self.X = jnp.column_stack((jnp.ones((nobs, 1)), self.X))
        self.y = jnp.asarray(y).reshape((nobs, 1))

        # TODO: how to simplify this
        if family == "Gaussian":
            self.family = Gaussian(link)
        elif family == "Binomial":
            self.family = Binomial(link)
        elif family == "Poisson":
            self.family = Poisson(link)
        else:
            print("no family found")

        if solver == "qr":
            self.solver = QRSolve()
        elif solver == "cholesky":
            self.solver = CholeskySolve()
        elif solver == "CG":
            self.solver = CGSolve()
        else:
            print("no solver found")

    def WaldTest(self) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
        """
        df = self.X.shape[0] - self.X.shape[1]
        TS = self.beta / self.beta_se

        if isinstance(self.family, Gaussian):
            pval = t.cdf(-abs(TS), df) * 2  # follow t(df) for Gaussian
        else:
            pval = norm.cdf(-abs(TS)) * 2  # follow Normal(0,1)
        return TS, pval, df

    def sumstats(self):
        _, _, weight = self.family.calc_weight(self.X, self.y, self.eta)
        infor = (self.X * weight).T @ self.X
        beta_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))
        return beta_se

    def fit(self):
        beta, self.n_iter, self.converged = irls(
            self.X, self.y, self.family, self.solver, self.seed, self.init, self.maxiter
        )
        self.eta = self.X @ beta
        self.beta_se = self.sumstats()
        self.beta = jnp.reshape(beta, (self.X.shape[1],))
        self.TS, self.pval, self.df = self.WaldTest()
        return GLMState(self.beta, self.beta_se, self.pval, self.n_iter, self.converged)

    def __str__(self) -> str:
        return f"""
        jaxQTL
        beta: {self.beta}
        se: {self.beta_se}
        p: {self.pval}
        converged: {self.converged} in {self.n_iter}
               """

    def tree_flatten(self):
        children = (self.X, self.y, self.family, self.solver)
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, False)
