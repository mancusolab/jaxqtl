from abc import abstractmethod

import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.scipy.special as jspec

from jaxtyping import Array, ArrayLike, Scalar


class Link(eqx.Module):
    r"""Abstract base for GLM link functions mapping the mean parameter to the linear predictor
    $g: \mu \mapsto \eta$.
    """

    @abstractmethod
    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = \eta$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` (on the domain of the link).

        **Returns:**

        Linear predictor $\eta$.
        """
        pass

    @abstractmethod
    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = \mu$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$.
        """
        pass

    @abstractmethod
    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute the derivative $g'(\mu)$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        pass

    @abstractmethod
    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute the derivative of the inverse link $g^{-1}'(\eta)$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        pass


class Power(Link):
    r"""Power link with $g(\mu) = \mu^{p}$ on $\mu > 0$, configurable exponent $p$."""

    power: Scalar = eqx.field(converter=jnp.asarray, default=1.0)

    def __init__(self, power: Scalar = 1.0):
        r"""Create a power link.

        **Arguments:**

        - `power`: Exponent `p` in `$g(\mu) = \mu^{p}$`.

        **Returns:**

        `None`
        """
        self.power = jnp.asarray(power)

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = \mu^{p}$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        return jnp.power(mu, self.power)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = \eta^{1/p}$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$.
        """
        return jnp.power(eta, 1.0 / self.power)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$ for the power link.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta)$ for the power link.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        return _grad_per_sample(self.inverse, eta)


class Identity(Link):
    r"""Identity link with $g(\mu) = \mu$ for $\mu \in \mathbb{R}$."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = \mu$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        return mu

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = \eta$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$.
        """
        return eta

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = 1$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$`.

        **Returns:**

        Array of ones with the same shape as `mu`.
        """
        return jnp.ones_like(mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta) = 1$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Array of ones with the same shape as `eta`.
        """
        return jnp.ones_like(eta)


class Logit(Link):
    r"""Logit link with $g(\mu) = \log(\mu / (1-\mu))$ on $\mu \in (0, 1)$."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward logit link $g(\mu) = \log(\mu / (1-\mu))$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$(0, 1)$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        return jspec.logit(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = 1 / (1 + \exp(-\eta))$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$ with support on $(0, 1)$.
        """
        return _clipped_expit(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$ for the logit link.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$(0, 1)$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta)$ for the logit link.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        return _grad_per_sample(self.inverse, eta)


class Inverse(Link):
    r"""Inverse link with $g(\mu) = 1/\mu$ on $\mu > 0$."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = 1/\mu$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        return jnp.reciprocal(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = 1/\eta$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$.
        """
        return jnp.reciprocal(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$ for the inverse link.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta)$ for the inverse link.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        return _grad_per_sample(self.inverse, eta)


class Log(Link):
    r"""Log link with $g(\mu) = \log(\mu)$ on $\mu > 0$."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = \log(\mu)$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        return jnp.log(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta) = \exp(\eta)$.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$ with support on $\mathbb{R}_+$.
        """
        return jnp.exp(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$ for the log link.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta)$ for the log link.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        return _grad_per_sample(self.inverse, eta)


class NBlink(Link):
    r"""Negative Binomial-specific log link with
    $g(\mu) = \log(\mu \alpha / (\mu \alpha + 1))$ where $\alpha$ is dispersion.

    """

    alpha: Scalar = eqx.field(converter=jnp.asarray, default=1.0)

    def __init__(self, alpha: Scalar = 1.0):
        r"""Create a Negative Binomial-specific link parameterized by dispersion.

        **Arguments:**

        - `alpha`: Dispersion parameter `$\alpha$` used by the link.

        **Returns:**

        `None`
        """
        self.alpha = jnp.asarray(alpha)

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute the forward link $g(\mu) = \log(\mu \alpha / (\mu \alpha + 1))$.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Linear predictor $\eta$.
        """
        # pre-commit trigger
        mu_alpha = mu * self.alpha
        return jnp.log(mu_alpha / (mu_alpha + 1.0))

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute the inverse link $g^{-1}(\eta)$ for this parameterization.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Mean parameter $\mu$ with support on $\mathbb{R}_+$.
        """
        return jnp.reciprocal(self.alpha * jnp.expm1(-eta))

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$ for this Negative Binomial-specific link.

        **Arguments:**

        - `mu`: Mean parameter `$\mu$` with support on `$\mathbb{R}_+$`.

        **Returns:**

        Derivative $g'(\mu)$ evaluated at $\mu$.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}'(\eta)$ for this Negative Binomial-specific link.

        **Arguments:**

        - `eta`: Linear predictor `$\eta$`.

        **Returns:**

        Derivative $g^{-1}'(\eta)$ evaluated at $\eta$.
        """
        return _grad_per_sample(self.inverse, eta)


def _clipped_expit(x):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(nn.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


def _grad_per_sample(func, x):
    """Get gradient for each sample
    x.shape = (n,1), eg. x can be mu or eta
    need to convert x to (n,) in order to apply vmap and grad
    """
    grad_fn = jax.vmap(jax.grad(func), 0)
    return grad_fn(x)
