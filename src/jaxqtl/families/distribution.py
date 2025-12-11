from abc import abstractmethod
from typing import ClassVar, Tuple, Type

import equinox as eqx
import jax.debug
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jax import lax
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, ScalarLike

from .links import Identity, Inverse, Link, Log, Logit, NBlink, Power


class ExponentialFamily(eqx.Module):
    """
    Define parent class for exponential family distribution (One parameter EF for now).
    Provide all required link function relevant to generalized linear model (GLM).
    GLM: g(mu) = X @ b, where mu = E(Y|X)
    : hlink : h(X @ b) = b'-1 (g^-1(X @ b)) = theta, default is canonical link which returns identity function.
    : hlink_der : derivative of hlink function
    : glink : g(mu) = X @ b, canonical link is g = b'-1, allows user to provide other link function.
    : glink_inv : inverse of glink, where g^-1(X @ b) = mu
    : glink_der : derivative of glink
    : log_prob : log joint density of all observations
    """

    glink: Link
    _links: ClassVar[list[Type[Link]]]
    _bounds: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, disp: ScalarLike) -> Array:
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        pass

    def calc_weight(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        this is part of the Information matrix
        """
        mu_k = jnp.clip(self.glink.inverse(eta), self._bounds[0], self._bounds[1])
        var_k = jnp.clip(self.variance(mu_k, disp), jnp.finfo(float).eps)
        g_deriv_k = self.glink.deriv(mu_k)
        weight_k = 1.0 / (var_k * g_deriv_k**2)

        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        return jnp.asarray(0.0)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size: ScalarLike = 1.0,
        tol: ScalarLike = 1e-3,
        max_iter: int = 1000,
        offset_eta: ScalarLike = 0.0,
    ) -> Array:
        return jnp.asarray(1.0)

    def _hlink(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.asarray(eta)

    def _hlink_score(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.ones_like(eta)

    def _hlink_hess(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return jnp.zeros_like(eta)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: Link = Identity()
    _links: ClassVar[list[Type[Link]]] = [Identity, Log, Power]
    _bounds: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
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
        offset_eta: ScalarLike = 0.0,
    ) -> Array:
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
        disp: ScalarLike = 0.0,
    ) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.scale(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        return jnp.ones_like(mu) * disp


class Gamma(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: Link = Inverse()
    _links: ClassVar[list[Type[Link]]] = [Identity, Inverse, Log]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).eps, float("inf"))

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        mu = jnp.clip(self.glink.inverse(eta), self._bounds[0])
        k = jnp.clip(1.0 / disp, self._bounds[0])
        theta = mu * disp  # scale

        # log f(y) = (k-1)log y - y/theta - k log theta - log Gamma(k)
        ll = (k - 1.0) * jnp.log(y) - (y / theta) - k * jnp.log(theta) - gammaln(k)
        return -ll

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        return disp * (mu**2)


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    glink: Link = Logit()
    _links: ClassVar[list[Type[Link]]] = [
        Logit,
        Log,
        Identity,
    ]  # Probit, Cauchy, LogC, CLogLog, LogLog
    _bounds: ClassVar[tuple[float, float]] = (0.0, 1.0)

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
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        return mu * (1 - mu)

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + 0.5) / 2.0)


class Poisson(ExponentialFamily):
    glink: Link = Log()
    _links: ClassVar[list[Type[Link]]] = [Identity, Log]  # Sqrt
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).eps, float("inf"))

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        logprob = jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        return mu


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method
    Notation: alpha = 1/r = 1.
    Now only use Log link (not the canonical link of NB)
    """

    glink: Link = Log()
    _links: ClassVar[list[Type[Link]]] = [Identity, Log, NBlink, Power]  # CLogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).eps, float("inf"))

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, disp: ScalarLike) -> Array:
        log_r = -jnp.log(disp)
        r = jnp.exp(log_r)
        log_mu = jnp.log(self.glink.inverse(eta))
        log_mu_plus_r = jnp.logaddexp(log_mu, log_r)

        log_p = log_mu - log_mu_plus_r
        log1m_p = log_r - log_mu_plus_r

        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = r * log1m_p + y * log_p
        return -jnp.sum(term1 + term2)

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        return mu + disp * (mu**2)

    def _log_alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, log_alpha: ScalarLike
    ) -> Tuple[Array, Array]:
        """
        internally take exponential such as to take derivative wrt 1/alpha
        """

        def _ll(log_alpha_):
            alpha_ = jnp.exp(log_alpha_)
            return self.negloglikelihood(X, y, eta, alpha_)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.1,
        step_size: ScalarLike = 0.1,
    ) -> Array:
        # alpha := disp
        # we optimize over log(alpha) isntead of alpha, but in theory we could compute exact geodesics on NB manifold
        # which could enable Riemannian optimization, but this is fine for now...
        log_alpha = jnp.log(disp)
        score, hess = self._log_alpha_score_and_hessian(X, y, eta, log_alpha)
        log_alpha_n = jnp.clip(log_alpha - step_size * (score / hess), jnp.log(1e-8), jnp.log(1e10))

        return jnp.exp(log_alpha_n)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size=1.0,
        tol=1e-3,
        max_iter=1000,
        offset_eta=0.0,
    ) -> Array:
        def body_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            log_alpha_o = jnp.log(alpha_o)
            score, hess = self._log_alpha_score_and_hessian(X, y, eta, log_alpha_o)
            log_alpha_n = jnp.clip(log_alpha_o - step_size * (score / hess), jnp.log(1e-8), jnp.log(1e10))
            diff = jnp.exp(log_alpha_n) - jnp.exp(log_alpha_o)

            return diff, num_iter + 1, jnp.exp(log_alpha_n)

        def cond_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_tuple = (10000.0, 0, disp)
        diff, num_iters, disp = lax.while_loop(cond_fun, body_fun, init_tuple)

        return disp

    def _hlink(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        """
        Using log link in g function
        """
        return jnp.log1p(-1.0 / (disp * jnp.exp(eta)))

    def _hlink_score(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return 1.0 / (disp * jnp.exp(eta) + 1.0)

    def _hlink_hess(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return -disp * jnp.exp(eta) / (disp * jnp.exp(eta) + 1) ** 2
