# pattern: Functional Core

from typing import NamedTuple

import jax.numpy as jnp

from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from ..infer import AbstractLinearModel, AbstractVarianceEstimator, FisherInfoError
from ._base import (
    _residualize_genotypes,
    _score_from_residuals,
    _validate_score_variance_estimator,
    AbstractHypothesisTest,
    TestResult,
)


class ScoreState(NamedTuple):
    r"""Compact null-fit state shared across genotype blocks.

    Only residuals, weights, and scalar fit diagnostics are retained, so batched
    permutations do not copy covariates or unused fitted-model arrays.
    """

    resid: Array
    glm_wt: Array
    num_iters: Array
    converged: Array
    disp: Array
    negloglikelihood: Array


class ScoreTest(AbstractHypothesisTest[ScoreState]):
    r"""Score test for association between a variant and an outcome.

    For a null (covariate-only) fit, let $r_y$ be the working residuals and let $g$ be a variant genotype vector.
    After residualizing $g$ against covariates, the per-variant score statistic is
    $U = g^{\top} W r_y$, with variance $V = g^{\top} W g$, and the reported z-statistic is
    $z = U / \sqrt{V}$ with two-sided p-value $p = 2\Phi(-|z|)$ where $\Phi(\cdot)$ is the Normal CDF.

    This implementation requires [`jaxqtl.infer.FisherInfoError`][]. Sandwich covariance estimators apply to Wald
    coefficient inference and do not define a robust version of this score statistic.

    **Raises:**

    - `ValueError`: If `std_err` is not [`jaxqtl.infer.FisherInfoError`][].
    """

    model: AbstractLinearModel
    std_err: AbstractVarianceEstimator = FisherInfoError()

    def __check_init__(self) -> None:
        _validate_score_variance_estimator(self.std_err, self.__class__.__name__)

    def init(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> ScoreState:
        r"""Fit the null model and retain the state needed for score testing.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A compact `ScoreState` with residuals, weights, and scalar fit diagnostics.
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)

        fit = self.model.fit(X, y, offset, self.std_err)
        return ScoreState(
            resid=fit.resid,
            glm_wt=fit.glm_wt,
            num_iters=fit.num_iters,
            converged=fit.converged,
            disp=fit.disp,
            negloglikelihood=self.model.family.negloglikelihood(X, y, fit.eta, fit.disp),
        )

    def test(self, X: ArrayLike, G: ArrayLike, state: ScoreState) -> TestResult:
        r"""Score a genotype block using an initialized null model.

        **Arguments:**

        - `X`: Covariate matrix used by `init`, with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)`.
        - `state`: State returned by `init(X, y, offset)`.

        **Returns:**

        Per-variant statistics and scalar null-model diagnostics in a
        [`jaxqtl.hypothesis.TestResult`][].
        """
        X = jnp.asarray(X)
        G = jnp.asarray(G)
        g_resid = _residualize_genotypes(X, G, state.glm_wt, self.model.solver)
        beta, se, zscore, _, _ = _score_from_residuals(state.resid, g_resid, state.glm_wt)
        pval = 2 * norm.sf(jnp.fabs(zscore))

        return TestResult(
            beta=beta,
            se=se,
            p=pval,
            z=zscore,
            num_iters=state.num_iters,
            converged=state.converged,
            disp=state.disp,
            negloglikelihood=state.negloglikelihood,
        )

    @property
    def name(self) -> str:
        r"""Return the test name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the test.
        """
        return "score"
