from abc import ABC, abstractmethod
from typing import List, Tuple  # ,Optional

import numpy as np
from typing_extensions import Self

import jax.numpy as jnp
import jax.scipy.stats as jaxstats
from jax import Array
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax.typing import ArrayLike

from .links import Identity, Link, Log, Logit, NBlink, Power


@register_pytree_node_class
class ExponentialFamily(ABC):
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

    _links: List[Self]  # type: ignore

    def __init__(self, glink: Link, validate: bool = True):
        if validate:
            if not any([isinstance(glink, link) for link in self._links]):
                raise ValueError(f"Link {glink} is invalid for Family {self}")
        self.glink = glink

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def log_prob(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    @abstractmethod
    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike) -> Array:
        pass

    @abstractmethod
    def init_eta(self, y: ArrayLike) -> Array:
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
        phi = self.calc_phi(X, y, mu_k)
        V_mu = self.variance(mu_k)
        weight_k = 1 / (jnp.square(g_deriv_k) * V_mu * phi)
        return mu_k, g_deriv_k, weight_k

    def tree_flatten(self):
        children = (
            self.glink,
            False,
        )  # validation already occurred, we shouldn't need to redo it
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2), we can treat normal distribution as one-parameter EF
    """

    _links = [Identity, Log, Power]

    def __init__(self, glink: Link = Identity(), validate: bool = True):
        super(Gaussian, self).__init__(glink, validate)

    def random_gen(self, loc: ArrayLike, scale: ArrayLike) -> Array:
        y = np.random.normal(loc, scale)
        return y

    def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def log_prob(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.calc_phi(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass  # TODO: old implementation was broken...

    def variance(self, mu: ArrayLike) -> Array:
        return jnp.ones_like(mu)

    def init_eta(self, y: ArrayLike) -> Array:
        return jnp.zeros((len(y), 1))


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    _links = [Logit, Log, Identity]  # Probit, Cauchy, LogC, CLogLog, LogLog

    def __init__(self, glink: Link = Logit(), validate: bool = True):
        super(Binomial, self).__init__(glink, validate)

    def random_gen(self, p: ArrayLike) -> Array:
        y = np.random.binomial(1, p)
        return y

    def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def log_prob(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        logprob = jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, mu: ArrayLike) -> Array:
        return mu - mu ** 2

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)


class Poisson(ExponentialFamily):

    _links = [Identity, Log]  # Sqrt

    def __init__(self, glink: Link = Log(), validate: bool = True):
        super(Poisson, self).__init__(glink, validate)

    def random_gen(self, mu: ArrayLike) -> Array:
        y = np.random.poisson(mu)
        return y

    def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def log_prob(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        logprob = jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))
        return logprob

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, mu: ArrayLike) -> Array:
        return mu

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink(jnp.ones_like(y) * y.mean())
        # return self.glink(y + 0.5) # statsmodel use this


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method, need work on this
    Assume alpha = 1/r = 1.
    """

    _links = [Identity, Log, NBlink, Power]  # CLogLog

    def __init__(
        self, glink: Link = Log(), alpha: ArrayLike = 1.0, validate: bool = True
    ):
        self.alpha = alpha
        super(NegativeBinomial, self).__init__(glink, validate)

    def random_gen(self, mu: jnp.ndarray) -> np.ndarray:
        r = 1
        p = round(mu) / (round(mu) + 1)
        y = np.random.negative_binomial(r, 1 - p)
        return y

    def calc_phi(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def log_prob(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def score(self, y: ArrayLike, eta: ArrayLike) -> Array:
        pass

    def variance(self, mu: ArrayLike) -> Array:
        # estimate alpha
        # a = ((resid**2 / mu - 1) / mu).sum() / df_resid
        return mu + self.alpha * mu ** 2

    def init_eta(self, y: ArrayLike) -> Array:
        pass

    def tree_flatten(self):
        children = (
            self.glink,
            self.alpha,
            False,
        )  # validation already occurred, we shouldn't need to redo it
        aux = ()
        return children, aux


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
#     def variance(self, mu: ArrayLike) -> Array:
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
