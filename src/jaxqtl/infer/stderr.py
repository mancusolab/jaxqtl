from abc import abstractmethod

import equinox as eqx

from jax import Array
from jax.numpy import linalg as jnpla
from jaxtyping import ArrayLike, ScalarLike

from ..families.distribution import ExponentialFamily


class ErrVarEstimation(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        pass


class FisherInfoError(ErrVarEstimation):
    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        del eta, mu, alpha
        infor = (X * weight).T @ X
        asmpt_cov = jnpla.inv(infor)

        return asmpt_cov


class HuberError(ErrVarEstimation):
    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        """
        Huber white sandwich estimator using observed hessian
        """
        phi = 1.0  # pull scale from family?
        gprime = family.glink.deriv(mu)
        # calculate observed hessian
        W = 1 / phi * (family._hlink_score(eta, alpha) / gprime - family._hlink_hess(eta, alpha) * (y - mu))
        hess_inv = jnpla.inv(-(X * W).T @ X)

        score_no_x = (y - mu) / (family.variance(mu, alpha) * gprime) * phi
        Bs = (X * (score_no_x**2)).T @ X
        robust_cov = hess_inv @ Bs @ hess_inv

        return robust_cov
