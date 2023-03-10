from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as jaxstats
from jax.tree_util import register_pytree_node, register_pytree_node_class

from .links import Identity, Link, Log, Logit, NBlink


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

    def __init__(self, glink: Link):
        self.glink = glink
        return

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def calc_phi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def score(self) -> jnp.ndarray:
        pass

    @abstractmethod
    def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
        pass

    def calc_weight(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        """
        mu_k = self.glink.inverse(eta)
        g_deriv_k = self.glink.deriv(mu_k)
        phi = self.calc_phi(X, y, eta)
        V_mu = self.calc_Vmu(mu_k)
        weight_k = 1 / (jnp.square(g_deriv_k) * V_mu * phi)
        return mu_k, g_deriv_k, weight_k

    @abstractmethod
    def tree_flatten(self):
        pass

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2), we can treat normal distribution as one-parameter EF
    """

    def __init__(
        self,
        glink: Link = Identity(),
    ) -> None:
        super().__init__(glink)

    def random_gen(self, loc: float, scale: float) -> np.ndarray:
        y = np.random.normal(loc, scale)
        return y

    def calc_phi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        resid = jnp.sum(jnp.square(pred - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        logprob = jnp.sum(jaxstats.norm.logpdf(y, self.glink.inverse(eta), 1.0))
        return logprob

    def score(self):
        mu_grad = jnp.sum((self.y - self.mu) / self.scale)
        var_grad = jnp.sum(
            -0.5 * (1 / self.scale - jnp.square((self.y - self.mu) / self.scale))
        )
        return jnp.array([mu_grad, var_grad])

    def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([1.0])

    def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
        return jnp.zeros((p, 1))

    def __str__(self):
        return f"Normal with mean {self.mu}, sd {self.sd}, logP ={self._log_prob()}"

    def tree_flatten(self):
        pass


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    def __init__(
        self,
        glink: Link = Logit(),
    ) -> None:
        super().__init__(glink)

    def random_gen(self, p: np.ndarray) -> np.ndarray:
        y = np.random.binomial(1, p)
        return y

    def calc_phi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.array([1.0])

    def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        """
        this works if we're using sigmoid link
        -jnp.sum(nn.softplus(jnp.where(y, -eta, eta)))
        """
        pass

    def score(self):
        pass

    def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
        return mu - mu ** 2

    def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
        # need check with link function
        return jnp.zeros((p, 1))

    def tree_flatten(self):
        pass


class Poisson(ExponentialFamily):
    def __init__(
        self,
        glink: Link = Log(),
    ) -> None:
        super().__init__(glink)

    def random_gen(self, mu: np.ndarray) -> np.ndarray:
        y = np.random.poisson(mu)
        return y

    def calc_phi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.array([1.0])

    def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))

    def score(self):
        pass

    def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
        return mu

    def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
        # need check with link function
        return jnp.zeros((p, 1))

    def tree_flatten(self):
        pass


class NB(ExponentialFamily):
    """
    NB-2 method, need work on this
    Assume alpha = 1/r = 1.
    """

    def __init__(self, glink: Link = NBlink(alpha=1.0)):
        self.alpha = 1.0  # TODO: fix this glink.alpha
        super().__init__(glink)

    def random_gen(self, mu: jnp.ndarray) -> np.ndarray:
        r = 1
        p = round(mu) / (round(mu) + 1)
        y = np.random.negative_binomial(r, 1 - p)
        return y

    def calc_phi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.array([1.0])

    def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        pass

    def score(self):
        pass

    def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
        # estimate alpha
        # a = ((resid**2 / mu - 1) / mu).sum() / df_resid
        return mu + self.alpha * mu ** 2

    def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
        # need check with link function
        key = jax.random.PRNGKey(seed)
        key, key_init = jax.random.split(key, 2)
        return jax.random.normal(key, shape=(p, 1))

    def tree_flatten(self):
        pass


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
#         self, alpha: np.ndarray, beta: np.ndarray, shape: tuple
#     ) -> np.ndarray:
#         y = np.random.gamma(alpha, beta, shape)
#         return y
#
#     def calc_phi(
#         self,
#         X: jnp.ndarray,
#         y: jnp.ndarray,
#         pred: jnp.ndarray,
#     ) -> jnp.ndarray:
#         """
#         method of moment estimator for phi
#         """
#         mu = self.glink_inv(pred)
#         df = y.shape[0] - X.shape[1]
#         phi = jnp.sum(jnp.square(mu - y) / jnp.square(mu)) / df
#         return phi
#
#     def log_prob(self, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
#         pass
#
#     def score(self):
#         pass
#
#     def calc_Vmu(self, mu: jnp.ndarray) -> jnp.ndarray:
#         return mu ** 2
#
#     def init_mu(self, p: int, seed: Optional[int]) -> jnp.ndarray:
#         # need check with link function
#         key = jax.random.PRNGKey(seed)
#         key, key_init = random.split(key, 2)
#         return jax.random.normal(key, shape=(p, 1))
#
#     def tree_flatten(self):
#         pass


""" example usage...
class Normal(AbstractExponential):
    def __init__(self, scale: Union[jnp.ndarray, float] = 1.):
        self.scale = scale
"""
