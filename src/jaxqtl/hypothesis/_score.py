import jax.numpy as jnp

from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ._base import (
    _residualize_genotypes,
    _score_from_residuals,
    AbstractHypothesisTest,
    TestResult,
)


class ScoreTest(AbstractHypothesisTest):
    r"""Score test for association between a variant and an outcome.

    For a null (covariate-only) fit, let $r_y$ be the working residuals and let $g$ be a variant genotype vector.
    After residualizing $g$ against covariates, the per-variant score statistic is
    $U = g^{\top} W r_y$, with variance $V = g^{\top} W g$, and the reported z-statistic is
    $z = U / \sqrt{V}$ with two-sided p-value $p = 2\Phi(-|z|)$ where $\Phi(\cdot)$ is the Normal CDF.
    """

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        r"""Compute score-test statistics for each variant in `G`.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A [`jaxqtl.hypothesis.TestResult`][] containing per-variant score-test statistics.
        """
        glmstate_cov_only = self.model.fit(X, y, offset, self.std_err)
        y_resid = glmstate_cov_only.resid

        g_resid = _residualize_genotypes(X, G, glmstate_cov_only.resid_covar, glmstate_cov_only.glm_wt)
        beta, se, zscore, _, _ = _score_from_residuals(y_resid, g_resid, glmstate_cov_only.glm_wt)
        pval = 2 * norm.sf(jnp.fabs(zscore))

        return TestResult(
            beta=beta,
            se=se,
            p=pval,
            z=zscore,
            num_iters=glmstate_cov_only.num_iters,
            converged=glmstate_cov_only.converged,
            disp=glmstate_cov_only.disp,
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
