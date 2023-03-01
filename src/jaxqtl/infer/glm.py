from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from .distribution import AbstractExponential
from .optimize import irls
from .solve import AbstractLinearSolve


@register_pytree_node_class
class GLM:
    """
    example:
    model = jaxqtl.GLM(X, y, family=Normal, append=True)
    model.fit(method = "qr")

    """

    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        family: AbstractExponential,
        solver: AbstractLinearSolve,
        append: bool,
    ) -> None:
        nobs = len(y)
        self.X = jnp.asarray(X)  # preprocessed in previous steps
        if append is True:
            self.X = jnp.column_stack((jnp.ones((nobs, 1)), self.X))
        self.y = jnp.asarray(y)
        self.family = family
        self.solver = solver

    def fit(self):
        beta, n_iter, converged = irls(self.X, self.y, self.family, self.solver)

    def tree_flatten(self):
        children = (self.X, self.y, self.family, self.solver)
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, False)
