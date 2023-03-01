from abc import ABC, abstractmethod

# typing.NamedTuple class is immutable (cannot change attribute values) [Chapter 7]
# from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from typing import Callable, Optional, Tuple

import jax.numpy as jnp

# import jax.scipy.stats as jaxstats
from jax.tree_util import register_pytree_node, register_pytree_node_class


@register_pytree_node_class
class AbstractExponential(ABC):
    """
    Define base class for exponential family distribution (One parameter EF for now).
    Provide all required link function relevant to generalized linear model (GLM).
    GLM: g(mu) = X @ b, where mu = E(Y|X)
    : hlink : h(X @ b) = theta, default is canonical link which returns identity function.
    allows user to provide other link function.
    : hlink_der : derivative of hlink function
    : glink : g(mu) = X @ b
    : glink_inv : inverse of glink, where g^-1(X @ b) = mu
    : glink_der : derivative of glink
    : log_prob : log joint density of all observations
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def _hlink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def _hlink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        # consider use jax.grad(func) -> function
        pass

    @abstractmethod
    def _glink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def _glink_inv(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def _glink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        # consider use jax.grad(func) -> function
        pass

    @abstractmethod
    def _calc_scale(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        # output a scalar for phi in EF
        pass

    # @abstractmethod
    # def _log_prob(self) -> jnp.ndarray:
    #     # output a scalar
    #     pass

    @abstractmethod
    def _score(self) -> jnp.ndarray:
        pass

    def calc_weight(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mu_k = self._glink_inv(pred)
        num = self._hlink_der(pred)
        g_deriv_k = self._glink_der(mu_k)
        phi = self._calc_scale(X, y, pred)
        weight_k = num / (g_deriv_k * phi)
        return mu_k, g_deriv_k, weight_k

    @abstractmethod
    def tree_flatten(self):
        pass

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Normal(AbstractExponential):
    def __init__(self) -> None:
        pass

    # def __init__(
    #         self, y: jnp.ndarray, mu: jnp.ndarray, sd: Union[jnp.ndarray, float] = 1.
    # ) -> None:
    #     super().__init__()
    #     self.mu = mu
    #     self.sd = sd
    #     self.scale = sd**2 # phi
    #     self.y = y
    #     self.nobs = len(y)

    def _hlink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return x

    def _hlink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.array([1.0])

    def _glink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return x

    def _glink_inv(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return x

    def _glink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.array([1.0])

    def _calc_scale(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        resid = jnp.sum(jnp.square(pred - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    # def _log_prob(self) -> jnp.ndarray:
    #     # output joint density as a scalar
    #     logprob = jaxstats.multivariate_normal.logpdf(
    #         self.y, jnp.zeros(self.nobs), jnp.diag(jnp.ones(self.nobs))
    #     )
    #     # logprob = jnp.sum(-0.5 * (jnp.log(self.sd) + jnp.log(2*jnp.pi)+ jnp.square(self.y - self.mu)))
    #     return logprob

    def _score(self):
        mu_grad = jnp.sum((self.y - self.mu) / self.scale)
        var_grad = jnp.sum(
            -0.5 * (1 / self.scale - jnp.square((self.y - self.mu) / self.scale))
        )
        return jnp.array([mu_grad, var_grad])

    def __str__(self):
        return f"Normal with mean {self.mu}, sd {self.sd}, logP ={self._log_prob()}"

    def tree_flatten(self):
        pass


class MVN(AbstractExponential):
    def __init__(self, mu, sd, x):
        super().__init__()
        self.mu = mu
        self.sd = sd  # sqrt(Var(Y|X))
        self.x = x

    def log_prob(self):
        logprob = -0.5 * (
            jnp.log(self.sd) + jnp.log(2 * jnp.pi) + jnp.square(self.x - self.mu)
        )
        return logprob

    def __str__(self):
        return f"Normal with mean {self.mu}, sd {self.sd}, logP(x={self.x})={self.log_prob()}"

    def tree_flatten(self):
        pass


class Binomial(AbstractExponential):
    def __init__(self) -> None:
        pass

    # def __init__(
    #         self, y: jnp.ndarray, mu: jnp.ndarray, sd: Union[jnp.ndarray, float] = 1.
    # ) -> None:
    #     super().__init__()
    #     self.mu = mu
    #     self.sd = sd
    #     self.scale = sd**2 # phi
    #     self.y = y
    #     self.nobs = len(y)

    def _hlink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return x

    def _hlink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.array([1.0])

    def _glink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.log(x / (1 - x))

    def _glink_inv(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.exp(-jnp.log1p(jnp.exp(-x)))

    def _glink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.exp(-jnp.log(x) - jnp.log(1 - x))

    def _calc_scale(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.array([1.0])

    def _log_prob(self) -> jnp.ndarray:
        # output joint density as a scalar
        pass

    def _score(self):
        pass

    # def __str__(self):
    #     return f"Binomial with mean {self.mu}, sd {self.sd}, logP ={self._log_prob()}"

    def tree_flatten(self):
        pass


class Poisson(AbstractExponential):
    def __init__(self) -> None:
        pass

    # def __init__(
    #         self, y: jnp.ndarray, mu: jnp.ndarray, sd: Union[jnp.ndarray, float] = 1.
    # ) -> None:
    #     super().__init__()
    #     self.mu = mu
    #     self.sd = sd
    #     self.scale = sd**2 # phi
    #     self.y = y
    #     self.nobs = len(y)

    def _hlink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return x

    def _hlink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.array([1.0])

    def _glink(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.log(x)

    def _glink_inv(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return jnp.exp(x)

    def _glink_der(
        self,
        x: jnp.ndarray,
        func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if func is not None:
            return func(x)
        else:
            return 1 / x

    def _calc_scale(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        pred: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.array([1.0])

    def _log_prob(self) -> jnp.ndarray:
        # output joint density as a scalar
        pass

    def _score(self):
        pass

    # def __str__(self):
    #     return f"Binomial with mean {self.mu}, sd {self.sd}, logP ={self._log_prob()}"

    def tree_flatten(self):
        pass


""" example usage...
class Normal(AbstractExponential):
    def __init__(self, scale: Union[jnp.ndarray, float] = 1.):
        self.scale = scale
"""
