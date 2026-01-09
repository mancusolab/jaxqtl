from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax.numpy.linalg import multi_dot
from jaxtyping import Array, ArrayLike

from ..infer.glm import AbstractLinearModel
from ..infer.stderr import AbstractVarianceEstimator, FisherInfoError


class TestResult(NamedTuple):
    r"""Container for per-variant association test results."""

    beta: Array
    se: Array
    p: Array
    z: Array
    num_iters: Array
    converged: Array
    disp: Array


class AbstractHypothesisTest(eqx.Module):
    r"""Abstract base class for per-variant association tests.

    Instances of this class are expected to take covariates, a genotype matrix, and an outcome vector,
    and return test statistics and p-values for each variant.
    """

    model: AbstractLinearModel
    std_err: AbstractVarianceEstimator = FisherInfoError()

    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        r"""Run the association test for variants in `G`.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A [`jaxqtl.hypothesis.base.TestResult`][] containing per-variant statistics.
        """
        return self.test(X, G, y, offset)

    @abstractmethod
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        r"""Implement the association test.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A [`jaxqtl.hypothesis.base.TestResult`][] containing per-variant statistics.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        r"""Return a short identifier for this test.

        **Arguments:**

        `None`

        **Returns:**

        A short string name for display and downstream metadata.
        """
        pass


def _residualize_genotypes(
    X: ArrayLike,
    G: ArrayLike,
    resid_covar: ArrayLike,
    glm_wt: ArrayLike,
) -> Array:
    r"""Residualize `G` against columns of `X` under a weighted inner product.

    This is used to compute genotype residuals after fitting the covariate-only model.

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `G`: Genotype matrix with shape `(n, m)`.
    - `resid_covar`: Covariance-like matrix used for residualization with shape `(p, p)`.
    - `glm_wt`: Per-sample weights with shape `(n,)` or a scalar weight.

    **Returns:**

    A residualized genotype matrix with shape `(n, m)`.
    """
    wgt = jnp.atleast_1d(glm_wt)
    x_W = X * wgt[:, jnp.newaxis]
    return G - multi_dot([X, resid_covar, x_W.T, G])


def _score_from_residuals(
    y_resid: ArrayLike,
    g_resid: ArrayLike,
    glm_wt: ArrayLike,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""Compute score statistics from residualized outcome and genotypes.

    **Arguments:**

    - `y_resid`: Residualized outcome with shape `(n,)`.
    - `g_resid`: Residualized genotype matrix with shape `(n, m)`.
    - `glm_wt`: Per-sample weights with shape `(n,)` or a scalar weight.

    **Returns:**

    A 5-tuple `(beta, se, z, score, std)` of per-variant arrays.
    """
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
