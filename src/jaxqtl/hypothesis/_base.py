# pattern: Functional Core

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from ..infer import (
    AbstractLinearModel,
    AbstractLinearSolve,
    AbstractVarianceEstimator,
    FisherInfoError,
)


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
        r"""Alias for [`jaxqtl.hypothesis.AbstractHypothesisTest.test`]."""
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

        A [`jaxqtl.hypothesis.TestResult`][] containing per-variant statistics.
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
    X: Array,
    G: Array,
    glm_wt: Array,
    solver: AbstractLinearSolve,
) -> Array:
    r"""Residualize `G` against columns of `X` under a weighted inner product.

    This computes $G_\perp = G - X\hat B$, where $\hat B$ is the weighted least-squares projection of each genotype
    onto `X`. The projection uses the model's solver rather than its coefficient covariance estimator because a
    sandwich covariance matrix does not define the projection geometry.

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `G`: Genotype matrix with shape `(n, m)`.
    - `glm_wt`: Per-sample weights with shape `(n,)` or a scalar weight.
    - `solver`: Linear solver used to project each genotype onto `X`.

    **Returns:**

    A residualized genotype matrix with shape `(n, m)`.
    """
    X = jnp.asarray(X)
    G = jnp.asarray(G)
    wgt = jnp.broadcast_to(jnp.asarray(glm_wt), (X.shape[0],))
    # Reuse the configured solver so residualization follows the model's numerical policy.
    coefficients = jax.vmap(lambda g: solver.wgt_lstsq(X, g, wgt), in_axes=1, out_axes=1)(G)
    return G - X @ coefficients


def _validate_score_variance_estimator(std_err: AbstractVarianceEstimator, test_name: str) -> None:
    r"""Require model-based Fisher information for score and SPA tests.

    Selecting a sandwich covariance estimator changes Wald coefficient uncertainty; it does not turn the existing
    score statistic or saddlepoint approximation into a misspecification-robust test.
    """
    if not isinstance(std_err, FisherInfoError):
        raise ValueError(
            f"{test_name} only supports FisherInfoError; "
            "alternative score-test variance estimators are not implemented."
        )


def _score_from_residuals(
    y_resid: Array,
    g_resid: Array,
    glm_wt: Array,
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
