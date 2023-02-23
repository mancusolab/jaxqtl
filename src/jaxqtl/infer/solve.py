from abc import ABC, abstractmethod

from distribution import AbstractExponential

import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

# from typing import List, NamedTuple, Tuple, Union


# here lets consider implementing difference types of linear solvers
# e.g., cholesky, qr, conjugate gradient using either
# hessian or fisher-info matrix


@register_pytree_node_class
class AbstractLinearSolve(ABC):
    @abstractmethod
    def __call__(
        self, X: jnp.ndarray, y: jnp.ndarray, model: AbstractExponential
    ) -> jnp.ndarray:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def tree_flatten(self):
        children = ()
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()
