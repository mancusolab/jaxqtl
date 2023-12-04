from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jspec
from jaxtyping import Array, ArrayLike, Scalar

from .utils import _clipped_expit, _grad_per_sample


class Link(eqx.Module):
    """
    Parent class for different link function g(mu) = eta
    """

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


class Power(Link):
    power: Scalar = eqx.field(converter=jnp.asarray, default=1.0)

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
        return _grad_per_sample(self.inverse, eta)


class Identity(Link):
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
        return jspec.logit(mu)

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
    alpha: Scalar = eqx.field(converter=jnp.asarray, default=1.0)

    def __call__(self, mu: ArrayLike) -> Array:
        # pre-commit trigger
        mu_alpha = mu * self.alpha
        return jnp.log(mu_alpha / (mu_alpha + 1.0))

    def inverse(self, eta: ArrayLike) -> Array:
        """
        exp(eta) / (alpha * (1 - exp(eta)) = exp(eta) / (alpha - alpha * exp(eta))
         = - exp(eta) / (alpha * exp(eta) - alpha) = -1 / (alpha - alpha / exp(eta))
         = -1 / (alpha * (1 - 1 / exp(eta))) = -1 / (alpha * (1 - exp(-eta))
         = -1 / (alpha * -expm1(-eta)) = 1 / (alpha * expm1(-eta)
        """
        return 1.0 / (self.alpha * jnp.expm1(-eta))

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
