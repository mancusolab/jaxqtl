from typing import NamedTuple

from jax import numpy as jnp
from jax.numpy import linalg as jnpla
from jax.tree_util import register_pytree_node_class

from .distribution import Binomial, Normal, Poisson
from .optimize import irls  # a function
from .solve import CGSolve, CholeskySolve, QRSolve


class GLMState(NamedTuple):
    beta: jnp.ndarray
    se: jnp.ndarray
    num_iters: int
    converged: bool


@register_pytree_node_class
class GLM:
    """
    example:
    model = jaxqtl.GLM(X, y, family="Gaussian", solver="qr", append=True)
    res = model.fit()
    print(res)
    """

    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: str,
        solver: str,
        append: bool,
        seed: int = 123,
    ) -> None:
        nobs = len(y)
        self.X = jnp.asarray(X)  # preprocessed in previous steps
        if append is True:
            self.X = jnp.column_stack((jnp.ones((nobs, 1)), self.X))
        self.y = jnp.asarray(y).reshape((nobs, 1))
        if family == "Gaussian":
            self.family = Normal()
        elif family == "Binomial":
            self.family = Binomial()
        elif family == "Poisson":
            self.family = Poisson()
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

        # key = random.PRNGKey(seed)
        # self.key, self.key_init = random.split(key, 2)

    def sumstats(self):
        _, _, weight = self.family.calc_weight(self.X, self.y, self.eta)
        infor = (self.X * weight).T @ self.X
        beta_se = jnp.sqrt(jnp.diag(jnpla.inv(infor)))
        return beta_se

    def fit(self):
        self.beta, self.n_iter, self.converged = irls(
            self.X, self.y, self.family, self.solver
        )
        self.eta = self.X @ self.beta
        self.beta_se = self.sumstats()
        self.beta = self.beta.reshape((self.X.shape[1],))
        return GLMState(self.beta, self.beta_se, self.n_iter, self.converged)

    def __str__(self):
        return f"""
        beta: {self.beta}
        se: {self.beta_se}
        converged: {self.converged} in {self.n_iter}
               """

    def tree_flatten(self):
        children = (self.X, self.y, self.family, self.solver)
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, False)
