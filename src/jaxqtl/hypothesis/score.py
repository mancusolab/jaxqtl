import jax.numpy as jnp

from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from .base import _residualize_genotypes, _score_from_residuals, HypothesisTest, TestResult


class ScoreTest(HypothesisTest):
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
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
        return "score"
