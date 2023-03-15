from typing import NamedTuple, Optional, Tuple

import numpy as np
from scipy.stats import norm, t  # , chi2

from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax.tree_util import register_pytree_node_class

from jaxqtl.families.distribution import ExponentialFamily, Gaussian, Poisson
from jaxqtl.infer.optimize import irls
from jaxqtl.infer.solve import CGSolve, LinearSolve
from jaxqtl.load.readfile import read_data


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

    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: ExponentialFamily = Gaussian(),
        solver: LinearSolve = CGSolve(),
        append: bool = True,
        maxiter: int = 100,
        tol: float = 1e-3,
        init: Optional[jnp.ndarray] = None,
    ) -> None:
        nobs = len(y)
        self.maxiter = maxiter
        self.tol = tol

        self.X = jnp.asarray(X)  # preprocessed in previous steps
        if append is True:
            self.X = jnp.column_stack((jnp.ones((nobs, 1)), self.X))
        self.y = jnp.asarray(y).reshape((nobs, 1))

        self.family = family
        self.solver = solver
        self.init = init if init is not None else family.init_eta(self.y)

    def WaldTest(self) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
        """
        df = self.X.shape[0] - self.X.shape[1]
        TS = self.beta / self.beta_se

        if isinstance(self.family, Gaussian):
            pval = t.cdf(-abs(TS), df) * 2  # follow t(df) for Gaussian
        else:
            pval = norm.cdf(-abs(TS)) * 2  # follow Normal(0, 1)

        return TS, pval, df

    def sumstats(self):
        _, _, weight = self.family.calc_weight(self.X, self.y, self.eta)
        infor = (self.X * weight).T @ self.X
        beta_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))
        return beta_se

    def fit(self):
        beta, self.n_iter, self.converged = irls(
            self.X, self.y, self.family, self.solver, self.init, self.maxiter, self.tol
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
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, False)


def run_bigGLM(geno_path, pheno_path, family: ExponentialFamily):
    cell_type = "CD14-positive monocyte"
    dat = read_data(geno_path, pheno_path, cell_type)

    # TODO: order of genotype is not same as count matrix
    # TODO: use donor_id as family id when creating plink file
    G = dat.genotype  # n x p variants
    Xmat = dat.count.obs[["sex", "age"]].astype("float64")
    ycount = dat.count.X[:, 0]

    num_var = 1000  # G.shape[1]
    allbeta = np.zeros((4, num_var))
    allpval = np.zeros((4, num_var))

    for idx in range(num_var):
        Xmat["variant"] = G[:, idx]

        glmstate = GLM(
            X=Xmat,
            y=ycount,
            family=Poisson(),
            append=True,
            maxiter=100,
        ).fit()

        allbeta[:, idx] = glmstate.beta
        allpval[:, idx] = glmstate.p
