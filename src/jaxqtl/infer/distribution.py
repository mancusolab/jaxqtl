from abc import ABC, abstractmethod

# import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

# from typing import List, NamedTuple, Tuple, Union


@register_pytree_node_class
class AbstractExponential(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def log_prob(self, y, pred):
        pass

    @abstractmethod
    def tree_flatten(self):
        pass

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


""" example usage...
class Normal(AbstractExponential):
    def __init__(self, scale: Union[jnp.ndarray, float] = 1.):
        self.scale = scale
"""
