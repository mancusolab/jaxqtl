from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class


@register_pytree_node_class
class Link(ABC):
    """
    Parent class for different link function g(mu) = eta
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def __call__(self, mu: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g(mu) = eta
        """
        pass

    @abstractmethod
    def inverse(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g^-1(eta) = mu
        """
        pass

    @abstractmethod
    def deriv(self, mu: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g'(mu)
        """
        pass

    @abstractmethod
    def inverse_deriv(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        calculate g^{-1}'(eta)
        """
        pass

    @abstractmethod
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
        return jnp.power(mu, self.power)

    def inverse(self, eta):
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

    def tree_flatten(self):
        children = (self.power,)
        aux = None
        return children, aux


class Identity(Power):
    def __init__(self):
        super().__init__(1.0)

    def tree_flatten(self):
        pass


class Logit(Link):
    def __call__(self, mu: jnp.ndarray):
        """
        Power transform link function
        g(mu) = log(mu / (1-mu))
        0 < mu < 1
        """
        return jnp.log(mu / (1 - mu))

    def inverse(self, eta):
        return jnp.exp(-jnp.log1p(jnp.exp(-eta)))

    def deriv(self, mu):
        return jnp.exp(-jnp.log(mu) - jnp.log(1 - mu))

    def inverse_deriv(self, eta):
        z = jnp.exp(eta)
        return z / (1 + z) ** 2

    def tree_flatten(self):
        children = ()
        aux = None
        return children, aux


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

    def tree_flatten(self):
        children = (self.power,)
        aux = None
        return children, aux


class NBlink(Link):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, mu: jnp.ndarray):
        z = mu * self.alpha
        return jnp.log(z / (z + 1))

    def inverse(self, eta):
        z = jnp.exp(eta)
        return z / (self.alpha * (1 - z))

    def deriv(self, mu):
        """
        1/(mu * (mu * alpha + 1)), mu > 0
        """
        term1 = -jnp.log(mu)
        term2 = -jnp.log(mu * self.alpha + 1)
        return jnp.exp(term1 + term2)

    def inverse_deriv(self, eta):
        z = jnp.exp(eta)
        return jnp.exp(z) / (self.alpha * (1 - jnp.exp(z)) ** 2)

    def tree_flatten(self):
        children = (self.alpha,)
        aux = None
        return children, aux
