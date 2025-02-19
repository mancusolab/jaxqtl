from abc import abstractmethod
from typing import ClassVar, List, Tuple, Type

import numpy as np

import equinox as eqx
import jax.debug
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jax import lax
from jax.scipy.special import gammaln, xlog1py, xlogy
from jaxtyping import Array, ArrayLike, ScalarLike

from .links import Identity, Link, Log, Logit, NBlink, Power


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
    _links: ClassVar[List[Type[Link]]]

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike) -> Array:
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        pass

    def score(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """
        For canonical link, this is X^t (y - mu)/phi, phi is the self.scale
        """
        return -X.T @ (y - mu) / self.scale(X, y, mu)

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
        pass

    def calc_weight(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        this is part of the Information matrix
        """
        mu_k = self.glink.inverse(eta)
        g_deriv_k = self.glink.deriv(mu_k)
        phi = self.scale(X, y, mu_k)
        weight_k = 1.0 / (phi * self.variance(mu_k, alpha) * g_deriv_k**2)

        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        return jnp.asarray(0.0)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
        tol: ScalarLike = 1e-3,
        max_iter: int = 1000,
        offset_eta: ScalarLike = 0.0,
    ) -> Array:
        return jnp.asarray(0.0)

    def _hlink(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.asarray(eta)

    def _hlink_score(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.ones_like(eta)

    def _hlink_hess(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return jnp.zeros_like(eta)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: Link = Identity()
    _links: ClassVar[List[Type[Link]]] = [Identity, Log, Power]

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
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
        alpha: ScalarLike = 0.0,
    ) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.scale(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return jnp.ones_like(mu)


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    glink: Link = Logit()
    _links: ClassVar[List[Type[Link]]] = [
        Logit,
        Log,
        Identity,
    ]  # Probit, Cauchy, LogC, CLogLog, LogLog

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
        y = np.random.binomial(1, mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return mu * (1 - mu)

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + 0.5) / 2.0)


class Poisson(ExponentialFamily):
    glink: Link = Log()
    _links: ClassVar[List[Type[Link]]] = [Identity, Log]  # Sqrt

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
        y = np.random.poisson(mu)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        logprob = jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return mu


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method
    Notation: alpha = 1/r = 1.
    Now only use Log link (not the canonical link of NB)
    """

    glink: Link = Log()
    _links: ClassVar[List[Type[Link]]] = [Identity, Log, NBlink, Power]  # CLogLog

    def random_gen(self, mu: jnp.ndarray, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> np.ndarray:
        r = 1 / alpha
        p = mu / (mu + r)
        y = np.random.negative_binomial(r, 1 - p)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike) -> Array:
        r = 1.0 / alpha
        mu = self.glink.inverse(eta)
        p = mu / (mu + r)
        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = xlog1py(r, -p) + xlogy(y, p)
        return -jnp.sum(term1 + term2)

    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return mu + alpha * (mu**2)

    def alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike
    ) -> Tuple[Array, Array]:
        """
        internally take exponential such as to take derivative wrt 1/alpha
        """

        def _ll(alpha):
            return self.negloglikelihood(X, y, eta, alpha)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)
        return _alpha_score(alpha), _alpha_hess(alpha)  # .reshape((1,))

    def log_alpha_score_and_hessian(
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

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)  # .reshape((1,))

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.1,
        step_size: ScalarLike = 0.1,
    ) -> Array:
        # TODO: update alpha such that it is lower bounded by 1e-6
        #   should have either parameter or smarter update on Manifold
        log_alpha = jnp.log(alpha)
        score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha)
        log_alpha_n = jnp.minimum(
            jnp.maximum(log_alpha - step_size * (score / hess), jnp.log(1e-8)),
            jnp.log(1e10),
        )

        return jnp.exp(log_alpha_n)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.1,
        step_size=0.1,
        tol=1e-3,
        max_iter=1000,
        offset_eta=0.0,
    ) -> Array:
        def body_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            log_alpha_o = jnp.log(alpha_o)
            score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha_o)
            log_alpha_n = jnp.minimum(
                jnp.maximum(log_alpha_o - step_size * (score / hess), jnp.log(1e-8)),
                jnp.log(1e10),
            )
            diff = jnp.exp(log_alpha_n) - jnp.exp(log_alpha_o)

            return diff.squeeze(), num_iter + 1, jnp.exp(log_alpha_n).squeeze()

        def cond_fun(val: Tuple):
            diff, num_iter, alpha_o = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_tuple = (10000.0, 0, alpha)
        diff, num_iters, alpha = lax.while_loop(cond_fun, body_fun, init_tuple)

        return alpha

    def _hlink(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        """
        Using log link in g function
        """
        return jnp.log1p(-1.0 / (alpha * jnp.exp(eta)))

    def _hlink_score(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return 1.0 / (alpha * jnp.exp(eta) + 1.0)

    def _hlink_hess(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return -alpha * jnp.exp(eta) / (alpha * jnp.exp(eta) + 1) ** 2
