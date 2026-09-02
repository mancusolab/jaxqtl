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
from jax.scipy.special import gammaln, xlogy
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


class NegativeBinomial(ExponentialFamily):
    r"""NB2 parameterization with dispersion $\alpha$ (variance $\mu + \alpha \mu^2$) and density
    $f(y \mid \mu, \alpha) = \frac{\Gamma(y+r)}{\Gamma(r)\,y!}\left(\frac{r}{r+\mu}\right)^r
    \left(\frac{\mu}{r+\mu}\right)^y$ where $r = 1/\alpha$. The dispersion satisfies $\alpha > 0$ and the mean $\mu$
    lies in $\mathbb{R}_{+}$.

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
        disp = jnp.asarray(disp)
        log_r = -jnp.log(disp)
        r = jnp.exp(log_r)
        log_mu = self.glink.log_inverse(eta)
        log_mu_plus_r = jnp.logaddexp(log_mu, log_r)

        log_p = log_mu - log_mu_plus_r
        log1m_p = log_r - log_mu_plus_r

        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = r * log1m_p + y * log_p
        return -jnp.sum(term1 + term2)

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        mu = jnp.asarray(mu)
        disp = jnp.asarray(disp)
        return mu + disp * (mu**2)

    def _log_alpha_score_and_hessian(self, X: Array, y: Array, eta: Array, log_alpha: Array) -> tuple[Array, Array]:
        """
        internally take exponential such as to take derivative wrt 1/alpha
        """

        def _ll(log_alpha_):
            alpha_ = jnp.exp(log_alpha_)
            return self.negloglikelihood(X, y, eta, alpha_)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)

    # def update_dispersion(
    #     self,
    #     X: ArrayLike,
    #     y: ArrayLike,
    #     eta: ArrayLike,
    #     disp: ScalarLike = 0.1,
    #     step_size: ScalarLike = 0.1,
    # ) -> Array:
    #     # alpha := disp
    #     # we optimize over log(alpha) isntead of alpha, but in theory we could compute exact geodesics on NB manifold
    #     # which could enable Riemannian optimization, but this is fine for now...
    #     X = jnp.asarray(X)
    #     y = jnp.asarray(y)
    #     eta = jnp.asarray(eta)
    #     disp = jnp.asarray(disp)
    #     step_size = jnp.asarray(step_size)
    #     log_alpha = jnp.log(disp)
    #     score, hess = self._log_alpha_score_and_hessian(X, y, eta, log_alpha)
    #     log_alpha_n = jnp.clip(log_alpha - step_size * (score / hess), jnp.log(1e-9), jnp.log(1e9))
    #
    #     return jnp.exp(log_alpha_n)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.1,
        step_size: ScalarLike = 0.1,
    ) -> Array:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        eta = jnp.asarray(eta)
        alpha = jnp.asarray(disp)
        step_size = jnp.asarray(step_size)

        # Keep compatibility with the current NB likelihood, which requires
        # strictly positive alpha. If negloglikelihood is later made safe at
        # alpha == 0, this can become a true boundary at zero.
        alpha = jnp.clip(alpha, 1e-9, 1e9)

        mu = jnp.clip(
            self.glink.inverse(eta),
            self._bounds[0],
            self._bounds[1],
        )

        # Exact derivatives of the NLL in theta = log(alpha).
        log_alpha = jnp.log(alpha)
        score, hess = self._log_alpha_score_and_hessian(X, y, eta, log_alpha)

        # Variance-manifold metric:
        #
        #   m_alpha = 1/2 sum_i [mu_i / (1 + alpha mu_i)]^2
        #
        # and associated connection.
        q = mu / (1.0 + alpha * mu)
        A2 = jnp.sum(q * q)
        A3 = jnp.sum(q * q * q)

        # Metric and Christoffel symbol in theta = log(alpha):
        #
        #   m_theta     = alpha^2 m_alpha
        #   Gamma_theta = 1 - alpha A3/A2
        metric = 0.5 * alpha**2 * A2
        gamma = 1.0 - alpha * (A3 / A2)

        # Covariant/Riemannian Hessian.
        riemann_hess = hess - gamma * score

        # Modified Riemannian Newton curvature.
        #
        # Flooring at the metric has two useful consequences:
        #   1. curvature is always positive;
        #   2. the Newton step can never be larger than the corresponding
        #      natural-gradient step solely because H_R is close to zero.
        curvature = jnp.maximum(riemann_hess, metric)

        # Tangent vector in theta coordinates.
        v_theta = -score / curvature

        # Convert to the alpha-coordinate tangent.
        v_alpha = alpha * v_theta

        # Cheap second-order retraction for the variance geometry.
        #
        # Gamma_alpha = -A3/A2, hence L = -1/Gamma_alpha = A2/A3.
        # This approaches:
        #
        #   alpha + v_alpha             near the Poisson boundary,
        #   alpha * exp(v_alpha/alpha)  for large alpha.
        L = A2 / A3
        alpha_new = alpha + L * jnp.expm1(step_size * v_alpha / L)

        return jnp.clip(alpha_new, 1e-9, 1e9)

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
