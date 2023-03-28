from abc import ABC, abstractmethod

# import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax.typing import ArrayLike

from .utils import _clipped_expit, _grad_per_sample


@register_pytree_node_class
class Link(ABC):
    """
    Parent class for different link function g(mu) = eta
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def __call__(self, mu: ArrayLike) -> Array:
        """
        calculate g(mu) = eta
        """
        pass

    @abstractmethod
    def inverse(self, eta: ArrayLike) -> Array:
        """
        calculate g^-1(eta) = mu
        """
        pass

    @abstractmethod
    def deriv(self, mu: ArrayLike) -> Array:
        """
        calculate g'(mu)
        """
        pass

    @abstractmethod
    def inverse_deriv(self, eta: ArrayLike) -> Array:
        """
        calculate g^{-1}'(eta)
        """
        pass

    def tree_flatten(self):
        children = ()
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Power(Link):
    def __init__(self, power=1.0):
        self.power = power
        super(Power, self).__init__()

    def __call__(self, mu: ArrayLike) -> Array:
        return jnp.power(mu, self.power)

    def inverse(self, eta: ArrayLike) -> Array:
        return jnp.power(eta, 1.0 / self.power)

    def deriv(self, mu: ArrayLike) -> Array:
        """
        self.power * jnp.power(mu, self.power - 1)
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        """
        jnp.power(eta, 1 / self.power - 1) / self.power
        """
        return _grad_per_sample(self, eta)

    def tree_flatten(self):
        children = (self.power,)
        aux = ()
        return children, aux


class Identity(Link):
    def __init__(self):
        super(Identity, self).__init__()

    def __call__(self, mu: ArrayLike) -> Array:
        return mu

    def inverse(self, eta: ArrayLike) -> Array:
        return eta

    def deriv(self, mu: ArrayLike) -> Array:
        return jnp.ones_like(mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        return jnp.ones_like(eta)


class Logit(Link):
    def __call__(self, mu: ArrayLike) -> Array:
        """
        Power transform link function
        g(mu) = log(mu / (1-mu))
        need clip for mu: 0 < mu < 1
        """
        return jnp.log(mu / (1 - mu))

    def inverse(self, eta: ArrayLike) -> Array:
        """
        inverse of logistic sigmoid function
        expit(x) = 1/(1+exp(-x))
        """
        return _clipped_expit(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        """
        jnp.exp(-jnp.log(mu) - jnp.log(1 - mu))
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        """
        z = jnp.exp(eta)
        return z / (1 + z) ** 2
        """
        return _grad_per_sample(self.inverse, eta)


class Log(Link):
    def __call__(self, mu: ArrayLike) -> Array:
        return jnp.log(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        return jnp.exp(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        """
        1/mu
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        """
        jnp.exp(eta)
        """
        return _grad_per_sample(self.inverse, eta)


class NBlink(Link):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        super(NBlink, self).__init__()

    def __call__(self, mu: ArrayLike) -> Array:
        # z = mu * self.alpha
        return jnp.log(mu * self.alpha / (mu * self.alpha + 1))

    def inverse(self, eta: ArrayLike) -> Array:
        # z = jnp.exp(eta)
        return jnp.exp(eta) / (self.alpha * (1 - jnp.exp(eta)))

    def deriv(self, mu: ArrayLike) -> Array:
        """
        1/(mu * (mu * alpha + 1)), mu > 0

        term1 = -jnp.log(mu)
        term2 = -jnp.log1p(mu * self.alpha)
        jnp.exp(term1 + term2)
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        """
        z = jnp.exp(eta)
        jnp.exp(z) / (self.alpha * (1 - jnp.exp(z)) ** 2)
        """
        return _grad_per_sample(self.inverse, eta)

    def tree_flatten(self):
        children = (self.alpha,)
        aux = None
        return children, aux
