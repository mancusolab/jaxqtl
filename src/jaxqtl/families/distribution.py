from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import equinox as eqx
import numpy as np
from typing_extensions import Self

import jax.numpy as jnp
import jax.scipy.stats as jaxstats
from jax import Array, lax
from jax.scipy.special import digamma, gammaln, polygamma
from jax.typing import ArrayLike

from .links import Identity, Link, Log, Logit, NBlink, Power


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
    _links: List[Self]  # type: ignore

    def __init__(self, glink: Link):
        if not any([isinstance(glink, link) for link in self._links]):
            raise ValueError(f"Link {glink} is invalid for Family {self}")
        self.glink = glink

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def loglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    @abstractmethod
    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    @abstractmethod
    def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        pass

    def random_gen(self, mu: ArrayLike, scale: ArrayLike) -> Array:
        pass

    def calc_weight(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike
    ) -> Tuple[Array, Array, Array]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        """
        mu_k = self.glink.inverse(eta)
        g_deriv_k = self.glink.deriv(mu_k)
        phi = self.scale(X, y, mu_k)
        V_mu = self.variance(X, y, mu_k)
        weight_k = 1 / (jnp.square(g_deriv_k) * V_mu * phi)
        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def calc_dispersion(
        self, y: ArrayLike, mu: ArrayLike, alpha_old, tol=1e-3, max_iter=1000
    ) -> Array:
        return jnp.array([0.0])


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: Link
    _links = [Identity, Log, Power]

    def __init__(self, glink: Link = Identity()):
        super(Gaussian, self).__init__(glink)

    def random_gen(self, mu: ArrayLike, scale: ArrayLike) -> Array:
        y = np.random.normal(mu, scale)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def loglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.scale(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass  # TODO: old implementation was broken...

    def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.ones_like(mu)


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    glink: Link
    _links = [Logit, Log, Identity]  # Probit, Cauchy, LogC, CLogLog, LogLog

    def __init__(self, glink: Link = Logit()):
        super(Binomial, self).__init__(glink)

    def random_gen(self, mu: ArrayLike, scale=0.0) -> Array:
        y = np.random.binomial(1, mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def loglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return mu - mu ** 2


class Poisson(ExponentialFamily):
    glink: Link
    _links = [Identity, Log]  # Sqrt

    def __init__(self, glink: Link = Log()):
        super(Poisson, self).__init__(glink)

    def random_gen(self, mu: ArrayLike, scale: ArrayLike) -> Array:
        y = np.random.poisson(mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def loglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        logprob = jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return mu


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method, need work on this
    Assume alpha = 1/r = 1.
    """

    glink: Link
    alpha: float
    _links = [Identity, Log, NBlink, Power]  # CLogLog

    def __init__(
        self,
        glink: Link = Log(),
        alpha: float = 1.0,
    ):
        super(NegativeBinomial, self).__init__(glink)
        self.alpha = alpha

    def random_gen(self, mu: jnp.ndarray, sim_alpha: float) -> np.ndarray:
        r = 1 / sim_alpha
        p = round(mu) / (round(mu) + sim_alpha)
        y = np.random.negative_binomial(r, 1 - p)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.array([1.0])

    def loglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        r = 1 / self.alpha
        mu = self.glink.inverse(eta)
        p = mu / (mu + r)
        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = r * jnp.log1p(-p) + y * jnp.log(p)
        return jnp.sum(term1 + term2)

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return mu + self.alpha * mu ** 2

    def alpha_score(self, y: ArrayLike, mu: ArrayLike, alpha: ArrayLike) -> Array:
        """
        trigammma(x) = polygamma(1,x)
        """
        alpha_inv = 1 / alpha
        term1 = alpha_inv ** 2 * jnp.log1p(alpha * mu)
        term2 = (y - mu) / (mu * (alpha ** 2) + alpha)
        term3 = (digamma(alpha_inv) - digamma(y + alpha_inv)) * alpha_inv ** 2

        return jnp.sum(term1 + term2 + term3)

    def alpha_hess(self, y: ArrayLike, mu: ArrayLike, alpha: ArrayLike) -> Array:
        """
        trigammma(x) = polygamma(1,x)
        """
        alpha_inv = 1 / alpha
        term1 = -2 / (alpha ** 3) * jnp.log1p(alpha * mu)
        term2 = -mu / (mu * (alpha ** 3) + alpha ** 2)
        term3 = (y - mu) * (2 * alpha * mu + 1) / (alpha ** 2 * mu + alpha) ** 2
        term4 = 2 / (alpha ** 3) * (digamma(y + alpha_inv) - digamma(alpha_inv))
        term5 = (
            1 / (alpha ** 4) * (polygamma(1, y + alpha_inv) - polygamma(1, alpha_inv))
        )
        return jnp.sum(term1 + term2 + term3 + term4 + term5)

    def calc_dispersion(
        self, y: ArrayLike, mu: ArrayLike, alpha_old: ArrayLike, tol=1e-3, max_iter=1000
    ) -> Array:
        def body_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            score = self.alpha_score(y, mu, alpha_o)
            hess = self.alpha_hess(y, mu, alpha_o)
            alpha_n = alpha_o - score / hess
            diff = alpha_n - alpha_o

            return diff, num_iter + 1, alpha_n

        def cond_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_tuple = (10000.0, 0, alpha_old)
        diff, num_iters, alpha = lax.while_loop(cond_fun, body_fun, init_tuple)

        return alpha

    # def calc_dispersion(self, y: ArrayLike, mu: ArrayLike, alpha_old, tol=1e-3, max_iter=1000) -> Array:
    #     diff = 1000
    #     old = self.alpha
    #     idx = 1
    #
    #     # while jnp.logical_and(diff > tol, idx <= max_iter):
    #     while diff > tol:
    #         score = self.alpha_score(y, mu, old)
    #         hess = self.alpha_hess(y, mu, old)
    #         new = old - score / hess
    #         idx += 1
    #         diff = jnp.abs(new - old)
    #         old = new
    #
    #     return new

    def _set_alpha(self, alpha_n):
        self.alpha = alpha_n


# class Gamma(AbstractExponential):
#     """
#     wierd link function...need work on this
#     """
#
#     def __init__(
#         self,
#         glink: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#         glink_inv: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#         glink_der: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     ) -> None:
#         # super().__init__()
#         self.glink = glink if glink is not None else (lambda x: 1 / x)
#         self.glink_inv = glink_inv if glink_inv is not None else (lambda x: 1 / x)
#         self.glink_der = glink_der if glink_der is not None else (lambda x: -1 / x ** 2)
#
#     def random_gen(
#         self, alpha: ArrayLike, beta: ArrayLike, shape: tuple
#     ) -> np.ndarray:
#         y = np.random.gamma(alpha, beta, shape)
#         return y
#
#       def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
#         """
#         method of moment estimator for phi
#         """
#         mu = self.glink_inv(pred)
#         df = y.shape[0] - X.shape[1]
#         phi = jnp.sum(jnp.square(mu - y) / jnp.square(mu)) / df
#         return phi
#
#     def log_prob(self, y: score(self, y: ArrayLike, eta: ArrayLike) -> Array:
#         pass
#
#     def score(self, score(self, y: ArrayLike, eta: ArrayLike) -> Array:
#         pass
#
#     def variance(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
#         return mu ** 2
#
#     def init_mu(self, p: int, seed: Optional[int]) -> Array:
#         # need check with link function
#         key = jax.random.PRNGKey(seed)
#         key, key_init = random.split(key, 2)
#         return jax.random.normal(key, shape=(p, 1))
#
#     def tree_flatten(self):
#         pass
