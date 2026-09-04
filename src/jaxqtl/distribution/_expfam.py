# pattern: Functional Core

from abc import abstractmethod
from typing import ClassVar, TYPE_CHECKING


# this is to fix a bug in type checking regarding AbstractClassVar and ClassVar
if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import jax.scipy.stats as jaxstats

from jax import lax
from jax.scipy.special import digamma, gammaln, polygamma, xlogy
from jaxtyping import Array, ArrayLike, ScalarLike

from ._links import (
    AbstractLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    NBLink,
    PowerLink,
)


class ExponentialFamily(eqx.Module):
    r"""Base interface for one-parameter exponential distribution and their GLM link. A natural exponential family has
    density $f(y \mid \theta, \phi) = \exp\left((y \theta - b(\theta))/\phi + c(y, \phi)\right)$,
    where $\theta$ is the natural parameter, $\phi$ is the dispersion/scale, $b(\theta)$ is the cumulant
    (log-partition) function, and $c(y, \phi)$ is the log base measure.

    The exponential dispersion model uses mean $\mu = b'(\theta)$ and variance function $V(\mu)$ with link
    mapping $g: \mu \mapsto \eta$. Subclasses specify $b(\cdot)$, $V(\cdot)$, and the link. For most cases,
    $V(\mu) := \phi b''(\theta)$; however, Negative Binomial models use $V(\mu) := \mu + \alpha \mu^2$ with
    overdispersion $\alpha$.

    !!! info

        Not all links are valid depending on the concrete class; this is checked automatically with a `ValueError`
        if invalid.

    """

    glink: eqx.AbstractVar[AbstractLink]
    _valid_links: AbstractClassVar[list[type[AbstractLink]]]
    _bounds: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._valid_links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        r"""Compute a dispersion/scale parameter (i.e. $\phi$) given predictors and the mean.

        **Arguments:**

        - `X`: Design matrix.
        - `y`: Observed response.
        - `mu`: Mean parameter for each observation.

        **Returns:**

        Dispersion estimate.
        """
        pass

    @abstractmethod
    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, disp: ScalarLike) -> Array:
        r"""Compute the negative log-likelihood at a given linear predictor `eta` and dispersion `disp`.

        **Arguments:**

        - `X`: Design matrix.
        - `y`: Observed response.
        - `eta`: Linear predictor.
        - `disp`: Dispersion/scale parameter.

        **Returns:**

        Negative log-likelihood.
        """
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Return the variance as a function of the mean and dispersion.

        **Arguments:**

        - `mu`: Mean parameter.
        - `disp`: Dispersion/scale parameter.

        **Returns:**

        Variance for each observation.
        """
        pass

    @abstractmethod
    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Draw a sample given a linear predictor and dispersion.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor.
        - `disp`: Dispersion/scale parameter.

        **Returns:**

        Simulated observations.
        """
        pass

    def calc_weight(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> tuple[Array, Array, Array]:
        r"""Compute mean, link derivative, and IRLS weights for observations.

        **Arguments:**

        - `eta`: Linear predictor.
        - `disp`: Dispersion/scale parameter.

        **Returns:**

        Tuple of (mu, link derivative, weights).
        """
        eta = jnp.asarray(eta)
        disp = jnp.asarray(disp)
        mu_k = jnp.clip(self.glink.inverse(eta), self._bounds[0], self._bounds[1])
        var_k = jnp.clip(self.variance(mu_k, disp), jnp.finfo(float).eps)
        g_deriv_k = self.glink.deriv(mu_k)
        weight_k = 1.0 / (var_k * g_deriv_k**2)

        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        r"""Provide a heuristic initializer for the linear predictor.

        **Arguments:**

        - `y`: Observed response.

        **Returns:**

        Initial linear predictor.
        """
        y = jnp.asarray(y)
        return self.glink((y + y.mean()) / 2)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        r"""Perform one dispersion update step. If not implemented, defaults to `disp` argument.

        **Arguments:**

        - `X`: Design matrix.
        - `y`: Observed response.
        - `eta`: Linear predictor.
        - `disp`: Current dispersion estimate.
        - `step_size`: Update step size.

        **Returns:**

        Updated dispersion estimate.
        """
        return jnp.asarray(disp)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size: ScalarLike = 1.0,
        tol: ScalarLike = 1e-3,
        max_iter: int = 1000,
    ) -> Array:
        r"""Iteratively estimate dispersion. If not implemented, defaults to `disp` argument

        **Arguments:**

        - `X`: Design matrix.
        - `y`: Observed response.
        - `eta`: Linear predictor.
        - `disp`: Initial dispersion estimate.
        - `step_size`: Update step size.
        - `tol`: Convergence tolerance.
        - `max_iter`: Maximum iterations.

        **Returns:**

        Estimated dispersion.
        """
        return jnp.asarray(disp)


class Gaussian(ExponentialFamily):
    r"""Normal exponential dispersion model with density
    $f(y \mid \mu, \phi) = (2\pi \phi)^{-1/2}\exp(-(y-\mu)^2/(2\phi))$. Dispersion $\phi > 0$ equals the variance,
    and the mean $\mu$ lies in $\mathbb{R}$.

    !!! info

        Valid links: [`jaxqtl.distribution.IdentityLink`][], [`jaxqtl.distribution.LogLink`][],
        [`jaxqtl.distribution.PowerLink`][].

    """

    glink: AbstractLink
    _valid_links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, PowerLink]
    _bounds: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    def __init__(self, glink: AbstractLink = IdentityLink()):
        r"""**Arguments:**

        - `glink`: [`jaxqtl.distribution.AbstractLink`][] mapping $\mu \mapsto \eta$
            (defaults to [`jaxqtl.distribution.IdentityLink`][]).
        """
        self.glink = glink

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        mu = jnp.asarray(mu)
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size: ScalarLike = 1.0,
        tol: ScalarLike = 1e-3,
        max_iter: int = 1000,
    ) -> Array:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        mu = self.glink.inverse(eta)
        rss = jnp.sum(jnp.square(mu - y))
        df = jnp.maximum(y.shape[0] - X.shape[1], 1)
        phi = rss / df
        return phi

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
    ) -> Array:
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        disp = jnp.asarray(disp)
        mu = self.glink.inverse(eta)
        sigma_sq = self.variance(mu, disp)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(sigma_sq)))
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        disp = jnp.asarray(disp)
        return jnp.ones_like(mu) * disp

    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample observations from the Gaussian model.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor `$\eta$`.
        - `disp`: Dispersion parameter `$\phi$` controlling the variance.

        **Returns:**

        Samples with the same shape as `eta`.
        """
        eta = jnp.asarray(eta)
        disp = jnp.asarray(disp)
        mu = self.glink.inverse(eta)
        return mu + rdm.normal(key, shape=mu.shape) * jnp.sqrt(disp)


class Gamma(ExponentialFamily):
    r"""Gamma exponential dispersion model with density
    $f(y \mid \mu, \phi) = y^{1/\phi-1}\exp(-y/(\mu\phi))/(\Gamma(1/\phi)(\mu\phi)^{1/\phi})$.
    Dispersion $\phi > 0$ scales the variance $\phi \mu^2$, and the mean $\mu$ lies in $\mathbb{R}_{+}$.

    !!! info

        Valid links: [`jaxqtl.distribution.IdentityLink`][], [`jaxqtl.distribution.InverseLink`][],
        [`jaxqtl.distribution.LogLink`][].

    """

    glink: AbstractLink
    _valid_links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, InverseLink, LogLink]
    _bounds: ClassVar[tuple[float, float]] = (float(jnp.finfo(float).eps), float("inf"))

    def __init__(self, glink: AbstractLink = InverseLink()):
        r"""**Arguments:**

        - `glink`: [`jaxqtl.distribution.AbstractLink`][] mapping $\mu \mapsto \eta$
            (defaults to [`jaxqtl.distribution.InverseLink`][]).
        """
        self.glink = glink

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        disp = jnp.asarray(disp)
        mu = jnp.clip(self.glink.inverse(eta), self._bounds[0])
        k = jnp.clip(1.0 / disp, self._bounds[0])
        theta = mu * disp  # scale

        # log f(y) = (k-1)log y - y/theta - k log theta - log Gamma(k)
        ll = (k - 1.0) * jnp.log(y) - (y / theta) - k * jnp.log(theta) - gammaln(k)
        return -ll

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        disp = jnp.asarray(disp)
        return disp * (mu**2)

    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample observations from the Gamma model.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor `$\eta$`.
        - `disp`: Dispersion parameter `$\phi$` controlling the variance `$\phi \mu^2$`.

        **Returns:**

        Samples with the same shape as `eta`.
        """
        eta = jnp.asarray(eta)
        mu = jnp.clip(self.glink.inverse(eta), self._bounds[0])
        disp = jnp.clip(jnp.asarray(disp), self._bounds[0])
        shape = jnp.reciprocal(disp)
        scale = mu * disp
        return rdm.gamma(key, shape, shape=mu.shape) * scale


class Binomial(ExponentialFamily):
    r"""Bernoulli/binomial ($n=1$) model with density $f(y \mid \mu) = \mu^{y}(1-\mu)^{1-y}$ and fixed dispersion 1.
    The mean $\mu$ lies in $[0, 1]$; there is no additional scale parameter beyond $\mu$.

    !!! info

        Valid links: [`jaxqtl.distribution.LogitLink`][], [`jaxqtl.distribution.LogLink`][],
        [`jaxqtl.distribution.IdentityLink`][].

    """

    glink: AbstractLink
    _valid_links: ClassVar[list[type[AbstractLink]]] = [
        LogitLink,
        LogLink,
        IdentityLink,
    ]  # Probit, Cauchy, LogC, CLogLog, LogLog
    _bounds: ClassVar[tuple[float, float]] = (0.0, 1.0)

    def __init__(self, glink: AbstractLink = LogitLink()):
        r"""**Arguments:**

        - `glink`: [`jaxqtl.distribution.AbstractLink`][] mapping $\mu \mapsto \eta$
            (defaults to [`jaxqtl.distribution.LogitLink`][]).
        """
        self.glink = glink

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        return mu * (1 - mu)

    def init_eta(self, y: ArrayLike) -> Array:
        y = jnp.asarray(y)
        return self.glink((y + 0.5) / 2.0)

    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample observations from the Bernoulli ($n=1$) model.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor `$\eta$`.
        - `disp`: Unused dispersion parameter (kept for API consistency).

        **Returns:**

        Samples with the same shape as `eta`.
        """
        eta = jnp.asarray(eta)
        mu = jnp.clip(self.glink.inverse(eta), self._bounds[0], self._bounds[1])
        return rdm.bernoulli(key, p=mu, shape=mu.shape).astype(float)


class Poisson(ExponentialFamily):
    r"""Poisson exponential family with density $f(y \mid \mu) = \exp(-\mu) \mu^{y}/y!$ and unit dispersion.
    The mean $\mu$ lies in $\mathbb{R}_{+}$; dispersion is fixed at 1.

    !!! info

        Valid links: [`jaxqtl.distribution.IdentityLink`][], [`jaxqtl.distribution.LogLink`][].

    """

    glink: AbstractLink = LogLink()
    _valid_links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink]  # Sqrt
    _bounds: ClassVar[tuple[float, float]] = (float(jnp.finfo(float).eps), float("inf"))

    def __init__(self, glink: AbstractLink = LogLink()):
        r"""**Arguments:**

        - `glink`: [`jaxqtl.distribution.AbstractLink`][] mapping $\mu \mapsto \eta$
            (defaults to [`jaxqtl.distribution.LogLink`][]).
        """
        self.glink = glink

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        mu = self.glink.inverse(eta)
        # Fractional abundance estimates are valid inputs, so do not apply the integer-support check in JAX's PMF.
        logprob = xlogy(y, mu) - gammaln(y + 1.0) - mu
        logprob = jnp.where(y < 0.0, -jnp.inf, logprob)
        logprob = jnp.sum(logprob)
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        return mu

    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample observations from the Poisson model.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor `$\eta$`.
        - `disp`: Unused dispersion parameter (kept for API consistency).

        **Returns:**

        Samples with the same shape as `eta`.
        """
        eta = jnp.asarray(eta)
        lam = self.glink.inverse(eta)
        return rdm.poisson(key, lam=lam)


_NB2_SERIES_SWITCH = 1e-3
_NB2_ALGDIV_R_SWITCH = 8.0
_NB2_LOG1P_REMAINDER_SERIES_SWITCH = 0.1
_NB2_EXPREL_SERIES_SWITCH = 1e-3
_NB2_DISPERSION_MIN = 1e-9
_NB2_DISPERSION_MAX = 1e9


def _nb2_exprel(value: Array) -> Array:
    """Evaluate ``expm1(value) / value`` with stable first and second derivatives at zero."""
    use_series = jnp.abs(value) <= _NB2_EXPREL_SERIES_SWITCH
    series = 1.0 / 5040.0
    series = 1.0 / 720.0 + value * series
    series = 1.0 / 120.0 + value * series
    series = 1.0 / 24.0 + value * series
    series = 1.0 / 6.0 + value * series
    series = 1.0 / 2.0 + value * series
    series = 1.0 + value * series
    safe_value = jnp.where(use_series, jnp.ones_like(value), value)
    direct = jnp.expm1(safe_value) / safe_value
    return jnp.where(use_series, series, direct)


def _nb2_log1p_over_x_series(x: Array) -> Array:
    """Approximate ``log1p(x) / x`` around zero through eighth order."""
    series = 1.0 / 9.0
    series = -1.0 / 8.0 + x * series
    series = 1.0 / 7.0 + x * series
    series = -1.0 / 6.0 + x * series
    series = 1.0 / 5.0 + x * series
    series = -1.0 / 4.0 + x * series
    series = 1.0 / 3.0 + x * series
    series = -1.0 / 2.0 + x * series
    return 1.0 + x * series


def _nb2_centered_lgamma_ratio_series_coefficients(y: Array) -> tuple[Array, ...]:
    """Return coefficients for the centered log-Gamma ratio expansion."""
    ym1 = y - 1.0
    c1 = y * ym1 / 2.0
    c2 = -y * ym1 * (2.0 * y - 1.0) / 12.0
    c3 = y**2 * ym1**2 / 12.0
    c4 = -y * ym1 * (2.0 * y - 1.0) * (3.0 * y**2 - 3.0 * y - 1.0) / 120.0
    c5 = y**2 * ym1**2 * (2.0 * y**2 - 2.0 * y - 1.0) / 60.0
    return c1, c2, c3, c4, c5


def _nb2_centered_lgamma_ratio_series(y: Array, alpha: Array) -> Array:
    """Expand ``gammaln(1/alpha + y) - gammaln(1/alpha) + y log(alpha)`` at zero."""
    c1, c2, c3, c4, c5 = _nb2_centered_lgamma_ratio_series_coefficients(y)
    return alpha * (c1 + alpha * (c2 + alpha * (c3 + alpha * (c4 + alpha * c5))))


def _nb2_one_plus_x_log1p_minus_x(x: Array) -> Array:
    """Evaluate ``(1 + x) * log1p(x) - x`` without cancellation near zero."""
    use_series = jnp.abs(x) <= _NB2_LOG1P_REMAINDER_SERIES_SWITCH
    series = 1.0 / 90.0
    series = -1.0 / 72.0 + x * series
    series = 1.0 / 56.0 + x * series
    series = -1.0 / 42.0 + x * series
    series = 1.0 / 30.0 + x * series
    series = -1.0 / 20.0 + x * series
    series = 1.0 / 12.0 + x * series
    series = -1.0 / 6.0 + x * series
    series = x**2 * (0.5 + x * series)
    safe_x = jnp.where(use_series, jnp.ones_like(x), x)
    direct = (1.0 + safe_x) * jnp.log1p(safe_x) - safe_x
    return jnp.where(use_series, series, direct)


def _nb2_algdiv_centered(y: Array, r: Array) -> Array:
    """Evaluate the centered log-Gamma ratio with the TOMS 708 expansion for ``r >= 8``."""
    c0 = 0.0833333333333333
    c1 = -0.00277777777760991
    c2 = 0.000793650666825390
    c3 = -0.000595202931351870
    c4 = 0.000837308034031215
    c5 = -0.00165322962780713

    h = y / r
    correction_weight = h / (1.0 + h)
    x = 1.0 / (1.0 + h)
    x2 = x * x
    s3 = 1.0 + x + x2
    s5 = 1.0 + x + x2 * s3
    s7 = 1.0 + x + x2 * s5
    s9 = 1.0 + x + x2 * s7
    s11 = 1.0 + x + x2 * s9
    t = (1.0 / r) ** 2
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0
    w *= correction_weight / r
    return r * _nb2_one_plus_x_log1p_minus_x(h) - 0.5 * jnp.log1p(h) - w


def _nb2_centered_lgamma_ratio(y: Array, alpha: Array) -> Array:
    """Evaluate the NB2 log-Gamma ratio using series, algdiv, or direct evaluation."""
    max_y = jnp.max(y)
    use_series = (alpha <= 0.0) | (alpha * max_y <= _NB2_SERIES_SWITCH)

    def series(operands):
        response, dispersion = operands
        return _nb2_centered_lgamma_ratio_series(response, dispersion)

    def nonseries(operands):
        response, dispersion = operands
        r = 1.0 / dispersion
        use_algdiv = r >= _NB2_ALGDIV_R_SWITCH

        def algdiv(values):
            response, r, _ = values
            return _nb2_algdiv_centered(response, r)

        def direct(values):
            response, r, dispersion = values
            return gammaln(r + response) - gammaln(r) + response * jnp.log(dispersion)

        return lax.cond(use_algdiv, algdiv, direct, (response, r, dispersion))

    return lax.cond(use_series, series, nonseries, (y, alpha))


def _nb2_mean_terms(y: Array, log_mu: Array, alpha: Array) -> Array:
    """Return the NB2 mean-dependent log-probability terms without exponentiating extreme means."""
    alpha_positive = alpha > 0.0
    alpha_safe = jnp.where(alpha_positive, alpha, jnp.ones_like(alpha))
    log_alpha = jnp.log(alpha_safe)
    log_x = log_alpha + log_mu
    use_series = ~alpha_positive | (log_x <= jnp.log(_NB2_SERIES_SWITCH))

    series_mu = jnp.exp(jnp.where(use_series, log_mu, jnp.zeros_like(log_mu)))
    finite_series_mu = jnp.isfinite(series_mu)
    series_x = alpha * jnp.where(finite_series_mu, series_mu, jnp.ones_like(series_mu))
    series_x = jnp.where(use_series & finite_series_mu & (alpha >= 0.0), series_x, jnp.zeros_like(series_x))
    series_ratio = _nb2_log1p_over_x_series(series_x)
    series_log_probability = log_mu - series_x * series_ratio
    series_mean_penalty = series_mu * series_ratio

    direct_log_x = jnp.where(use_series, jnp.zeros_like(log_x), log_x)
    direct_log_probability = -log_alpha - jax.nn.softplus(-direct_log_x)
    direct_mean_penalty = jax.nn.softplus(direct_log_x) / alpha_safe

    log_probability = jnp.where(use_series, series_log_probability, direct_log_probability)
    log_probability = jnp.where(y == 0.0, jnp.zeros_like(log_probability), log_probability)
    mean_penalty = jnp.where(use_series, series_mean_penalty, direct_mean_penalty)
    return y * log_probability - mean_penalty


def _nb2_horner_times_x(x: Array, coefficients: tuple[Array, ...]) -> Array:
    """Evaluate a coefficient sequence with no constant term using Horner's rule."""
    value = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = coefficient + x * value
    return x * value


def _nb2_mean_log_alpha_derivatives(y: Array, log_mu: Array, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """Return mean-term score, centered Hessian, and metric reductions."""
    alpha_positive = alpha > 0.0
    alpha_safe = jnp.where(alpha_positive, alpha, jnp.ones_like(alpha))
    log_alpha = jnp.log(alpha_safe)
    log_x = log_alpha + log_mu
    use_series = ~alpha_positive | (log_x <= jnp.log(_NB2_SERIES_SWITCH))

    series_mu = jnp.exp(jnp.where(use_series, log_mu, jnp.zeros_like(log_mu)))
    finite_series_mu = jnp.isfinite(series_mu)
    series_mu_safe = jnp.where(finite_series_mu, series_mu, jnp.zeros_like(series_mu))
    series_x = alpha * series_mu_safe
    series_x = jnp.where(use_series & alpha_positive, series_x, jnp.zeros_like(series_x))
    # Coefficients follow by differentiating
    # ``-y * log1p(x) - mu * log1p(x) / x`` with respect to ``log(alpha)``.
    score_coefficients = tuple(
        ((-1.0) ** order) * (y - order * series_mu_safe / (order + 1.0)) for order in range(1, 9)
    ) + (-y,)
    # The final coefficient is the ninth-order term contributed by ``-y * log1p(x)``.
    centered_hessian_coefficients = tuple(
        (order - 1) * coefficient for order, coefficient in enumerate(score_coefficients, start=1)
    )
    series_score = _nb2_horner_times_x(series_x, score_coefficients)
    series_centered_hessian = _nb2_horner_times_x(series_x, centered_hessian_coefficients)

    r = 1.0 / alpha_safe
    x_fraction = jax.nn.sigmoid(log_x)
    log1p_x = jax.nn.softplus(log_x)
    direct_score = r * (log1p_x - x_fraction) - y * x_fraction
    direct_centered_hessian = 2.0 * r * (x_fraction - log1p_x) + (r + y) * x_fraction**2

    score = jnp.where(use_series, series_score, direct_score)
    centered_hessian = jnp.where(use_series, series_centered_hessian, direct_centered_hessian)
    metric_fraction = jnp.where(alpha_positive, x_fraction, jnp.zeros_like(x_fraction))
    metric_fraction_squared = metric_fraction**2
    return (
        jnp.sum(score),
        jnp.sum(centered_hessian),
        jnp.sum(metric_fraction_squared),
        jnp.sum(metric_fraction_squared * metric_fraction),
    )


def _nb2_centered_lgamma_log_alpha_derivatives(y: Array, alpha: Array) -> tuple[Array, Array]:
    """Return the log-Gamma score and Hessian minus score for log-dispersion."""
    max_y = jnp.max(y)
    use_series = (alpha <= 0.0) | (alpha * max_y <= _NB2_SERIES_SWITCH)

    def series(operands):
        response, dispersion = operands
        c1, c2, c3, c4, c5 = _nb2_centered_lgamma_ratio_series_coefficients(response)
        score = dispersion * (
            c1 + dispersion * (2.0 * c2 + dispersion * (3.0 * c3 + dispersion * (4.0 * c4 + 5.0 * dispersion * c5)))
        )
        centered_hessian = dispersion**2 * (
            2.0 * c2 + dispersion * (6.0 * c3 + dispersion * (12.0 * c4 + 20.0 * dispersion * c5))
        )
        return jnp.sum(score), jnp.sum(centered_hessian)

    def nonseries(operands):
        response, dispersion = operands
        r = 1.0 / dispersion
        use_algdiv = r >= _NB2_ALGDIV_R_SWITCH

        def algdiv(values):
            response, dispersion = values

            def objective(alpha):
                return jnp.sum(_nb2_algdiv_centered(response, 1.0 / alpha))

            # Retain AD for this cancellation-resistant polynomial branch; it is
            # selected only in the intermediate regime and avoids duplicate formulas.
            alpha_score = jax.grad(objective)
            alpha_gradient, alpha_hessian = jax.value_and_grad(alpha_score)(dispersion)
            return dispersion * alpha_gradient, dispersion**2 * alpha_hessian

        def direct(values):
            response, dispersion = values
            r = 1.0 / dispersion
            # Analytic log-dispersion derivatives avoid tracing the full direct
            # likelihood through two nested automatic-differentiation passes.
            delta_digamma = digamma(r + response) - digamma(r)
            score = response - r * delta_digamma
            centered_hessian = (
                2.0 * r * delta_digamma + r**2 * (polygamma(1, r + response) - polygamma(1, r)) - response
            )
            return jnp.sum(score), jnp.sum(centered_hessian)

        return lax.cond(use_algdiv, algdiv, direct, (response, dispersion))

    return lax.cond(use_series, series, nonseries, (y, alpha))


def _nb2_log_alpha_derivatives(y: Array, log_mu: Array, alpha: Array) -> tuple[Array, Array, Array, Array]:
    """Return the NB2 NLL score, Hessian minus score, and metric reductions."""
    gamma_score, gamma_centered_hessian = _nb2_centered_lgamma_log_alpha_derivatives(y, alpha)
    mean_score, mean_centered_hessian, a2, a3 = _nb2_mean_log_alpha_derivatives(y, log_mu, alpha)
    return (
        -(gamma_score + mean_score),
        -(gamma_centered_hessian + mean_centered_hessian),
        a2,
        a3,
    )


def _nb2_riemannian_direction(score: Array, centered_hessian: Array, a2: Array, a3: Array) -> tuple[Array, Array]:
    """Return the variance-manifold Newton direction and ``A3 / A2``."""
    metric = 0.5 * a2
    metric_valid = jnp.isfinite(metric) & (metric > 0.0) & jnp.isfinite(a3)
    safe_a2 = jnp.where(metric_valid, a2, jnp.ones_like(a2))
    connection_complement = jnp.where(metric_valid, a3 / safe_a2, jnp.zeros_like(a3))
    connection_complement = jnp.clip(connection_complement, 0.0, 1.0)

    finite_score = jnp.isfinite(score)
    safe_score = jnp.where(finite_score, score, jnp.zeros_like(score))
    riemannian_hessian = centered_hessian + connection_complement * safe_score
    use_riemannian_hessian = jnp.isfinite(riemannian_hessian) & (riemannian_hessian > 0.0)
    curvature = jnp.where(use_riemannian_hessian, riemannian_hessian, metric)
    direction_valid = metric_valid & finite_score & jnp.isfinite(curvature) & (curvature > 0.0)
    safe_curvature = jnp.where(direction_valid, curvature, jnp.ones_like(curvature))
    direction = jnp.where(direction_valid, -safe_score / safe_curvature, jnp.zeros_like(score))
    return direction, connection_complement


class NegativeBinomial(ExponentialFamily):
    r"""NB2 parameterization with dispersion $\alpha$ (variance $\mu + \alpha \mu^2$) and density
    $f(y \mid \mu, \alpha) = \frac{\Gamma(y+r)}{\Gamma(r)\,y!}\left(\frac{r}{r+\mu}\right)^r
    \left(\frac{\mu}{r+\mu}\right)^y$ where $r = 1/\alpha$ for $\alpha > 0$. The dispersion satisfies
    $\alpha \geq 0$, the mean $\mu$ lies in $\mathbb{R}_{+}$, and $\alpha = 0$ evaluates the Poisson limit.

    !!! info

        Valid links: [`jaxqtl.distribution.IdentityLink`][], [`jaxqtl.distribution.LogLink`][],
        [`jaxqtl.distribution.NBLink`][], [`jaxqtl.distribution.PowerLink`][].

    """

    glink: AbstractLink = LogLink()
    _valid_links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, NBLink, PowerLink]  # CLogLog
    _bounds: ClassVar[tuple[float, float]] = (float(jnp.finfo(float).eps), float("inf"))

    def __init__(self, glink: AbstractLink = LogLink()):
        r"""**Arguments:**

        - `glink`: [`jaxqtl.distribution.AbstractLink`][] mapping $\mu \mapsto \eta$
            (defaults to [`jaxqtl.distribution.LogLink`][]).
        """
        self.glink = glink

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, disp: ScalarLike) -> Array:
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        alpha = jnp.asarray(disp)
        log_mu = self.glink.log_inverse(eta)
        logprob = _nb2_centered_lgamma_ratio(y, alpha) - gammaln(y + 1.0) + _nb2_mean_terms(y, log_mu, alpha)
        nll = -jnp.sum(logprob)
        invalid = (alpha < 0.0) | jnp.any(y < 0.0)
        return jnp.where(invalid, jnp.inf, nll)

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        disp = jnp.asarray(disp)
        return mu + disp * (mu**2)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.1,
        step_size: ScalarLike = 0.1,
    ) -> Array:
        r"""Take one variance-manifold modified-Newton step for NB2 dispersion.

        **Arguments:**

        - `X`: Design matrix, retained for the common family API but unused by this update.
        - `y`: Response vector.
        - `eta`: Linear predictor used to hold the fitted means fixed during the dispersion step.
        - `disp`: Current NB2 dispersion `$alpha$`.
        - `step_size`: Scalar multiplier for the tangent direction.

        **Returns:**

        The updated dispersion, clipped to `$[10^{-9}, 10^9]$`.

        **Failure Modes:**

        A nonpositive or nonfinite Riemannian Hessian uses the variance-manifold metric as a natural-gradient
        fallback. If that metric or the score is invalid, the clipped input dispersion is returned unchanged.
        """
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        alpha = jnp.clip(jnp.asarray(disp), _NB2_DISPERSION_MIN, _NB2_DISPERSION_MAX)
        step_size = jnp.asarray(step_size)
        log_mu = self.glink.log_inverse(eta)
        score, centered_hessian, a2, a3 = _nb2_log_alpha_derivatives(y, log_mu, alpha)
        direction, connection_complement = _nb2_riemannian_direction(score, centered_hessian, a2, a3)
        scaled_direction = step_size * direction
        candidate = alpha * (1.0 + scaled_direction * _nb2_exprel(connection_complement * scaled_direction))
        candidate = jnp.where(jnp.isnan(candidate), alpha, candidate)
        return jnp.clip(candidate, _NB2_DISPERSION_MIN, _NB2_DISPERSION_MAX)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size=1.0,
        tol=1e-3,
        max_iter=1000,
    ) -> Array:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        disp = jnp.asarray(disp)
        step_size = jnp.asarray(step_size)
        tol = jnp.asarray(tol)

        def body_fun(val: tuple):
            diff, num_iter, alpha_o = val
            alpha_n = self.update_dispersion(X, y, eta, alpha_o, step_size)
            diff = alpha_n - alpha_o

            return diff, num_iter + 1, alpha_n

        def cond_fun(val: tuple):
            diff, num_iter, alpha_o = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_tuple = (10000.0, 0, disp)
        diff, num_iters, disp = lax.while_loop(cond_fun, body_fun, init_tuple)

        return disp

    def sample(self, key, eta: ArrayLike, disp: ScalarLike = 0.1) -> Array:
        r"""Sample observations from the Negative Binomial model.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: Linear predictor `$\eta$`.
        - `disp`: Dispersion parameter `$\alpha$` where `$\mathrm{Var}(Y)=\mu+\alpha\mu^2$`.

        **Returns:**

        Samples with the same shape as `eta`.
        """
        eta = jnp.asarray(eta)
        mu = self.glink.inverse(eta)
        disp = jnp.asarray(disp)
        r = jnp.reciprocal(disp)

        # Gamma-Poisson mixture sampling for NB2:
        # lambda ~ Gamma(shape=r, scale=mu/r), y ~ Poisson(lambda)
        key_lam, key_y = rdm.split(key, 2)
        lam = rdm.gamma(key_lam, r, shape=mu.shape) * (mu / r)
        return rdm.poisson(key_y, lam=lam)
