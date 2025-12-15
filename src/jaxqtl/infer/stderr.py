from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax.numpy import linalg as jnpla
from jaxtyping import Array, ArrayLike, ScalarLike

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
        disp: ScalarLike = 1.0,
    ) -> Array:
        """calculate standard errors for SNP

        :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param eta: linear component eta
        :param mu: fitted mean
        :param weight: weight for each individual
        :param disp: dispersion parameter
        """
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
        disp: ScalarLike = 1.0,
    ) -> Array:
        weight = jnp.atleast_1d(weight)
        infor = (X * weight[:, jnp.newaxis]).T @ X
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
        disp: ScalarLike = 1.0,
    ) -> Array:
        """
        Huber white sandwich estimator using observed hessian
        """
        variance = family.variance(mu, disp)
        mu_eta = family.glink.inverse_deriv(eta)

        r = (y - mu) * mu_eta / variance
        Bs = (X * (r**2)[:, None]).T @ X
        W = jnp.atleast_1d(weight)
        Vh_inv = jnpla.inv(X.T @ (X * W[:, None]))

        robust_cov = Vh_inv @ Bs @ Vh_inv

        return robust_cov
