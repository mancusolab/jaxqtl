from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax.numpy.linalg import multi_dot
from jaxtyping import Array, ArrayLike

from ..infer.glm import AbstractLinearModel
from ..infer.stderr import ErrVarEstimation, FisherInfoError


class TestResult(NamedTuple):
    beta: Array
    se: Array
    p: Array
    z: Array
    num_iters: Array
    converged: Array
    disp: Array


class HypothesisTest(eqx.Module):
    model: AbstractLinearModel
    std_err: ErrVarEstimation = FisherInfoError()

    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        return self.test(X, G, y, offset)

    @abstractmethod
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


def _residualize_genotypes(
    X: ArrayLike,
    G: ArrayLike,
    resid_covar: ArrayLike,
    glm_wt: ArrayLike,
) -> Array:
    wgt = jnp.atleast_1d(glm_wt)
    x_W = X * wgt[:, jnp.newaxis]
    return G - multi_dot([X, resid_covar, x_W.T, G])


def _score_from_residuals(
    y_resid: ArrayLike,
    g_resid: ArrayLike,
    glm_wt: ArrayLike,
) -> tuple[Array, Array, Array, Array, Array]:
    y_resid = jnp.asarray(y_resid)
    g_resid = jnp.asarray(g_resid)

    wgt = jnp.atleast_1d(glm_wt)
    sqrt_wgt = jnp.sqrt(wgt)

    w_g_resid = g_resid * sqrt_wgt[:, jnp.newaxis]
    g_std = jnp.sqrt(jnp.sum(w_g_resid**2, axis=0))
    se = jnp.reciprocal(g_std)

    g_score = w_g_resid.T @ (sqrt_wgt * y_resid)
    zscore = g_score * se
    beta = zscore * se

    return beta, se, zscore, g_score, g_std
