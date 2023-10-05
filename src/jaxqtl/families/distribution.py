from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Type

import equinox as eqx
import numpy as np

import jax.debug
import jax.numpy as jnp
import jax.scipy.stats as jaxstats
from jax import Array
from jax.config import config
from jax.scipy.special import gammaln, xlog1py, xlogy
from jax.typing import ArrayLike

from .links import Identity, Link, Log, Logit, NBlink, Power

config.update("jax_enable_x64", True)


class ExponentialFamily(eqx.Module, metaclass=ABCMeta):
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
    _links: List[Type[Link]]

    def __init__(self, glink: Link):
        if not any([isinstance(glink, link) for link in self._links]):
            raise ValueError(f"Link {glink} is invalid for Family {self}")
        self.glink = glink

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def negloglikelihood(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ArrayLike
    ) -> Array:
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike, alpha: ArrayLike = 0.0) -> Array:
        pass

    def score(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """
        For canonical link, this is X^t (y - mu)/phi, phi is the self.scale
        """
        return -X.T @ (y - mu) / self.scale(X, y, mu)

    def random_gen(
        self, mu: ArrayLike, scale: float = 1.0, alpha: float = 0.0
    ) -> Array:
        pass

    def calc_weight(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ArrayLike = jnp.zeros((1,)),
    ) -> Tuple[Array, Array, Array]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        this is part of the Information matrix
        """
        mu_k = self.glink.inverse(eta)
        g_deriv_k = self.glink.deriv(mu_k)
        phi = self.scale(X, y, mu_k)
        # weight_k = self._hlink_score(eta, alpha) / (g_deriv_k * phi)
        weight_k = 1 / (phi * self.variance(mu_k, alpha) * g_deriv_k**2)
        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def calc_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ArrayLike = 0.01,
        step_size: ArrayLike = 1.0,
    ) -> Array:
        return jnp.zeros((1,))

    def _hlink(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        """
        If canonical link, then this is identify function
        """
        return eta

    def _hlink_score(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        """
        If canonical link, then this is identify function
        """
        return jnp.ones_like(eta)

    def _hlink_hess(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        return jnp.zeros_like(eta)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    _links = [Identity, Log, Power]

    def __init__(self, glink: Link = Identity()):
        super(Gaussian, self).__init__(glink)

    def random_gen(
        self, mu: ArrayLike, scale: float = 1.0, alpha: float = 0.0
    ) -> Array:
        y = np.random.normal(mu, scale)
        return y

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
        alpha: ArrayLike = jnp.zeros((1,)),
    ) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.scale(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))) -> Array:
        return jnp.ones_like(mu)


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    _links = [Logit, Log, Identity]  # Probit, Cauchy, LogC, CLogLog, LogLog

    def __init__(self, glink: Link = Logit()):
        super(Binomial, self).__init__(glink)

    def random_gen(
        self, mu: ArrayLike, scale: float = 1.0, alpha: float = 0.0
    ) -> Array:
        y = np.random.binomial(1, mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ArrayLike = jnp.zeros((1,)),
    ) -> Array:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))) -> Array:
        return mu - mu**2

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + 0.5) / 2)


class Poisson(ExponentialFamily):
    _links = [Identity, Log]  # Sqrt

    def __init__(self, glink: Link = Log()):
        super(Poisson, self).__init__(glink)

    def random_gen(
        self, mu: ArrayLike, scale: float = 1.0, alpha: float = 0.0
    ) -> Array:
        y = np.random.poisson(mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ArrayLike = jnp.zeros((1,)),
    ) -> Array:
        logprob = jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))) -> Array:
        return mu


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method
    Notation: alpha = 1/r = 1.
    Now only use Log link (not the canonical link of NB)
    """

    # alpha: float
    _links = [Identity, Log, NBlink, Power]  # CLogLog

    def __init__(
        self,
        glink: Link = Log(),
    ):
        super(NegativeBinomial, self).__init__(glink)

    def random_gen(
        self, mu: jnp.ndarray, scale: float = 1.0, alpha: float = 0.0
    ) -> np.ndarray:
        r = 1 / alpha
        p = mu / (mu + r)
        y = np.random.negative_binomial(r, 1 - p)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.array([1.0])

    def negloglikelihood(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ArrayLike
    ) -> Array:
        r = 1.0 / alpha
        mu = self.glink.inverse(eta)
        p = mu / (mu + r)
        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = xlog1py(r, -p) + xlogy(y, p)
        return -jnp.sum(term1 + term2)

    def variance(self, mu: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))) -> Array:
        return mu + alpha * (mu**2)

    def alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ArrayLike
    ) -> Tuple[Array, Array]:
        def _ll(alpha):
            return self.negloglikelihood(X, y, eta, alpha)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.grad(_alpha_score)

        return _alpha_score(alpha), _alpha_hess(alpha)

    def calc_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ArrayLike = 0.01,
        step_size: ArrayLike = 1.0,
    ) -> Array:
        # TODO: update alpha such that it is lower bounded by 1e-6
        #   should have either parameter or smarter update on Manifold
        score, hess = self.alpha_score_and_hessian(X, y, eta, alpha)
        alpha_n = jnp.maximum(alpha - step_size * (score / hess), 1e-6)

        return alpha_n

    def _set_alpha(self, alpha_n):
        self.alpha = alpha_n

    def _hlink(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        """
        Using log link in g function
        """
        return jnp.log1p(-1.0 / (alpha * jnp.exp(eta)))

    def _hlink_score(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        return 1.0 / (alpha * jnp.exp(eta) + 1.0)

    def _hlink_hess(self, eta: ArrayLike, alpha: ArrayLike = jnp.zeros((1,))):
        return -alpha * jnp.exp(eta) / (alpha * jnp.exp(eta) + 1) ** 2
