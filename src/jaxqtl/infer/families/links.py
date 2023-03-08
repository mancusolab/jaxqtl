# from abc import ABC, abstractmethod
# typing.NamedTuple class is immutable (cannot change attribute values) [Chapter 7]
# from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class


@register_pytree_node_class
class Link:
    """
    Parent class for different link function g(mu) = eta
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def __call__(self, mu: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g(mu) = eta
        """
        pass

    def inverse(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g^-1(eta) = mu
        """
        pass

    def deriv(self, mu: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g'(mu)
        """
        pass

    def inverse_deriv(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g^{-1}'(eta)
        """
        pass

    def tree_flatten(self):
        pass

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Power(Link):
    def __init__(self, power=1.0):
        self.power = power

    def __call__(self, mu: jnp.ndarray):
        """
        Power transform link function
        g(mu) = mu ** self.power
        """
        if self.power == 1:
            return mu
        else:
            return jnp.power(mu, self.power)

    def inverse(self, eta):
        if self.power == 1.0:
            return eta
        else:
            return jnp.power(eta, 1.0 / self.power)

    def deriv(self, mu):
        if self.power == 1:
            # return vector of ones, with same shape as p (note: can we return one value instead?)
            return jnp.ones_like(mu)
        else:
            return self.power * jnp.power(mu, self.power - 1)

    def inverse_deriv(self, eta):
        if self.power == 1:
            return jnp.ones_like(eta)
        else:
            return jnp.power(eta, 1 / self.power - 1) / self.power


class Identity(Power):
    def __init__(self):
        super().__init__(power=1.0)


class Logit(Link):
    def __call__(self, mu: jnp.ndarray):
        """
        Power transform link function
        g(mu) = log(mu / (1-mu))
        """
        return jnp.log(mu / (1 - mu))

    def inverse(self, eta):
        return jnp.exp(-jnp.log1p(jnp.exp(-eta)))

    def deriv(self, mu):
        return jnp.exp(-jnp.log(mu) - jnp.log(1 - mu))

    def inverse_deriv(self, eta):
        z = jnp.exp(eta)
        return z / (1 + z) ** 2


class Log(Link):
    def __call__(self, mu: jnp.ndarray):
        """
        Power transform link function
        g(mu) = log(mu)
        """
        return jnp.log(mu)

    def inverse(self, eta):
        return jnp.exp(eta)

    def deriv(self, mu):
        return 1.0 / mu

    def inverse_deriv(self, eta):
        return jnp.exp(eta)
