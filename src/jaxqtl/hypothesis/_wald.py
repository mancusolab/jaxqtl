# pattern: Functional Core

import jax
import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import ArrayLike

from ..infer import LinearModel
from ._base import _residualize_genotypes, AbstractHypothesisTest, TestResult


class WaldTest(AbstractHypothesisTest):
    r"""Wald test for association between a variant and an outcome.

    For each variant, this fits a full model including the variant and reports
    $\hat\beta / \mathrm{se}(\hat\beta)$. [`jaxqtl.infer.LinearModel`][] uses a residualized Gaussian fast path and a
    Student's t reference distribution with the full model's residual degrees of freedom. Generalized linear models
    use a Normal reference distribution.
    """

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        r"""Compute Wald-test statistics for each variant in `G`.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A [`jaxqtl.hypothesis.TestResult`][] containing per-variant Wald-test statistics.

        **Raises:**

        - `ValueError`: For a linear model with no residual degrees of freedom after adding the tested variant.
        """
        X = jnp.asarray(X)
        G = jnp.asarray(G)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)
        if isinstance(self.model, LinearModel):
            model = self.model
            result = model.fit(X, y, offset, self.std_err)
            y_resid = result.resid
            G_resid = _residualize_genotypes(X, G, result.glm_wt, model.solver)
            df_resid = y.shape[0] - X.shape[1] - 1

            # Frisch-Waugh-Lovell preserves the genotype coefficient, but its inference must retain the full-model df.
            result = jax.vmap(
                lambda g_res: model.fit(g_res[:, jnp.newaxis], y_resid, std_err=self.std_err, df_resid=df_resid),
                in_axes=1,
            )(G_resid)
            negloglikelihood = jax.vmap(
                lambda g_res, eta, disp: model.family.negloglikelihood(g_res[:, jnp.newaxis], y_resid, eta, disp)
            )(G_resid.T, result.eta, result.disp)

            # The residualized fits have one coefficient each; the association API returns one scalar per variant.
            state = TestResult(
                beta=result.beta[:, 0],
                se=result.se[:, 0],
                p=result.p[:, 0],
                z=result.z[:, 0],
                num_iters=result.num_iters,
                converged=result.converged,
                disp=result.disp,
                negloglikelihood=negloglikelihood,
            )
        else:

            def _func(carry, snp):
                M = jnp.hstack((X, snp[:, jnp.newaxis]))
                glmstate = self.model.fit(M, y, offset, self.std_err)

                return carry, TestResult(
                    beta=glmstate.beta[-1],
                    se=glmstate.se[-1],
                    p=glmstate.p[-1],
                    z=glmstate.z[-1],
                    num_iters=glmstate.num_iters,
                    converged=glmstate.converged,
                    disp=glmstate.disp,
                    negloglikelihood=self.model.family.negloglikelihood(M, y, glmstate.eta, glmstate.disp),
                )

            _, state = lax.scan(_func, 0.0, G.T)

        return state

    @property
    def name(self) -> str:
        r"""Return the test name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the test.
        """
        return "wald"
