from typing import NamedTuple, Optional, Tuple

from scipy.stats import norm, t  # , chi2

from jax import Array, numpy as jnp
from jax.numpy import linalg as jnpla
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily, Gaussian, Poisson
from jaxqtl.infer.optimize import irls
from jaxqtl.infer.solve import CGSolve, LinearSolve


# change jnp.ndarray --> np.ndarray for mutable array
class GLMState(NamedTuple):
    beta: Array
    se: Array
    p: Array
    num_iters: Array
    converged: Array


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
        X: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily = Gaussian(),
        solver: LinearSolve = CGSolve(),
        append: bool = True,
        maxiter: int = 100,
        tol: float = 1e-3,
        init: Optional[ArrayLike] = None,
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

    def WaldTest(self) -> Tuple[Array, Array, int]:
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
        self.mu = self.family.glink.inverse(self.eta)
        self.beta_se = self.sumstats()
        self.beta = jnp.reshape(beta, (self.X.shape[1],))
        self.TS, self.pval, self.df = self.WaldTest()

        return GLMState(self.beta, self.beta_se, self.pval, self.n_iter, self.converged)

    def calc_resid(self, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.square(y - mu)

    def __str__(self) -> str:
        return f"""
        jaxQTL
        beta: {self.beta}
        se: {self.beta_se}
        alpha: {self.family.alpha}
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


def run_bigGLM(
    dat, family: ExponentialFamily = Poisson(), test_run: Optional[int] = None
):
    # TODO: order of genotype is not same as count matrix
    # TODO: use donor_id as family id when creating plink file
    # G = dat.genotype  # n x p variants
    Xmat = dat.count.obs[["sex", "age"]].astype("float64")
    num_params = Xmat.shape[1] + 1  # features + intercept
    num_genes = test_run if test_run is not None else dat.count.X.shape[1]

    # num_var = 1000  # G.shape[1]
    all_beta = jnp.zeros((num_params, num_genes))
    all_se = jnp.zeros((num_params, num_genes))
    all_pval = jnp.zeros((num_params, num_genes))
    all_num_iters = jnp.zeros((num_genes,))
    all_converged = jnp.zeros((num_genes,))

    for idx in range(num_genes):
        # Xmat["variant"] = G[:, idx] # append X with genotype
        ycount = dat.count.X[:, idx].astype("float64")

        glmstate = GLM(
            X=Xmat,
            y=ycount,
            family=family,
            append=True,
            maxiter=1000,
        ).fit()

        all_beta[:, idx] = glmstate.beta
        all_se[:, idx] = glmstate.se
        all_pval[:, idx] = glmstate.p
        all_num_iters[idx] = glmstate.num_iters
        all_converged[idx] = glmstate.converged

    return GLMState(all_beta, all_se, all_pval, all_num_iters, all_converged)
