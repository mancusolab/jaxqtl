from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax.numpy import linalg as jnpla
from jaxtyping import Array, ArrayLike, ScalarLike

from ..distribution._expfam import ExponentialFamily


class AbstractVarianceEstimator(eqx.Module):
    r"""Base interface for LM/GLM coefficient covariance estimators.

    Concrete implementations are passed into [`jaxqtl.infer.GLM.fit`][] (and similar entrypoints) to compute the
    coefficient covariance matrix used for Wald-style standard errors and test statistics.
    """

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
        r"""Estimate the coefficient covariance matrix for a fitted GLM.

        **Arguments:**

        - `family`: GLM family implementing [`jaxqtl.distribution.ExponentialFamily`][].
        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `eta`: Linear predictor `$\eta$` with shape `(n,)`.
        - `mu`: Fitted mean with shape `(n,)`.
        - `weight`: IRLS weights with shape `(n,)` (or a scalar weight).
        - `disp`: Dispersion/scale parameter (family-specific).

        **Returns:**

        Coefficient covariance matrix with shape `(p, p)`.
        """
        pass


class FisherInfoError(AbstractVarianceEstimator):
    r"""Fisher information covariance estimator.

    Uses the IRLS working weights to form an observed-information approximation
    $\widehat{\mathrm{Cov}}(\hat\beta) \approx (X^\top W X)^{-1}$.
    """

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
        r"""Compute the Fisher information covariance estimate.

        **Arguments:**

        - `family`: GLM family implementing [`jaxqtl.distribution.ExponentialFamily`][] (unused).
        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)` (unused).
        - `eta`: Linear predictor `$\eta$` with shape `(n,)` (unused).
        - `mu`: Fitted mean with shape `(n,)` (unused).
        - `weight`: IRLS weights with shape `(n,)` (or a scalar weight).
        - `disp`: Dispersion/scale parameter (unused).

        **Returns:**

        Coefficient covariance matrix with shape `(p, p)`.
        """
        weight = jnp.atleast_1d(weight)
        infor = (X * weight[:, jnp.newaxis]).T @ X
        asmpt_cov = jnpla.inv(infor)

        return asmpt_cov


class HuberError(AbstractVarianceEstimator):
    r"""Huber-White sandwich covariance estimator.

    Computes a robust covariance estimate using an observed Hessian approximation and a score outer product, yielding
    a "sandwich" form that can be more stable under mean/variance misspecification.
    """

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
        r"""Compute the robust Huber-White covariance estimate.

        **Arguments:**

        - `family`: GLM family implementing [`jaxqtl.distribution.ExponentialFamily`][].
        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `eta`: Linear predictor `$\eta$` with shape `(n,)`.
        - `mu`: Fitted mean with shape `(n,)`.
        - `weight`: IRLS weights with shape `(n,)` (or a scalar weight).
        - `disp`: Dispersion/scale parameter (family-specific).

        **Returns:**

        Coefficient covariance matrix with shape `(p, p)`.
        """
        variance = family.variance(mu, disp)
        mu_eta = family.glink.inverse_deriv(eta)

        r = (y - mu) * mu_eta / variance
        Bs = (X * (r**2)[:, None]).T @ X
        W = jnp.atleast_1d(weight)
        Vh_inv = jnpla.inv(X.T @ (X * W[:, None]))

        robust_cov = Vh_inv @ Bs @ Vh_inv

        return robust_cov
