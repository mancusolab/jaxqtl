# pattern: Functional Core

from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from ..infer import AbstractLinearModel, AbstractVarianceEstimator, FisherInfoError, LinearModel
from ._base import _residualize_genotypes, AbstractHypothesisTest, TestResult


class GaussianWaldState(NamedTuple):
    r"""Covariate-only residuals and weights for Gaussian coefficient inference."""

    resid: Array
    glm_wt: Array


class GlmWaldState(NamedTuple):
    r"""Outcome and offset for fitting a full GLM for each variant."""

    y: Array
    offset: Array


WaldState = GaussianWaldState | GlmWaldState


class WaldTest(AbstractHypothesisTest[WaldState]):
    r"""Wald test for association between a variant and an outcome.

    For each variant, this fits a full model including the variant and reports
    $\hat\beta / \mathrm{se}(\hat\beta)$. [`jaxqtl.infer.LinearModel`][] uses a residualized Gaussian fast path and a
    Student's t reference distribution with the full model's residual degrees of freedom. Generalized linear models
    use a Normal reference distribution.
    """

    model: AbstractLinearModel
    std_err: AbstractVarianceEstimator = FisherInfoError()

    def init(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> WaldState:
        r"""Prepare an outcome for per-variant coefficient inference.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        Gaussian models return covariate-only residuals and weights. Generalized
        linear models retain the response and offset for their per-variant fits.
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)
        if isinstance(self.model, LinearModel):
            fit = self.model.fit(X, y, offset, self.std_err)
            return GaussianWaldState(resid=fit.resid, glm_wt=fit.glm_wt)
        return GlmWaldState(y=y, offset=offset)

    def test(self, X: ArrayLike, G: ArrayLike, state: WaldState) -> TestResult:
        r"""Compute Wald statistics using initialized outcome state.

        **Arguments:**

        - `X`: Covariate matrix used by `init`, with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `state`: State returned by this test's `init(X, y, offset)`.

        **Returns:**

        A [`jaxqtl.hypothesis.TestResult`][] with per-variant inference and fitted
        model diagnostics, including per-variant dispersion and likelihood.

        **Raises:**

        - `ValueError`: For a linear model with no residual degrees of freedom
          after adding the tested variant.
        """
        X = jnp.asarray(X)
        G = jnp.asarray(G)
        if isinstance(self.model, LinearModel):
            assert isinstance(state, GaussianWaldState)
            model = self.model
            y_resid = state.resid
            G_resid = _residualize_genotypes(X, G, state.glm_wt, model.solver)
            df_resid = X.shape[0] - X.shape[1] - 1

            # Frisch-Waugh-Lovell preserves the genotype coefficient, but its inference must retain the full-model df.
            result = jax.vmap(
                lambda g_res: model.fit(g_res[:, jnp.newaxis], y_resid, std_err=self.std_err, df_resid=df_resid),
                in_axes=1,
            )(G_resid)
            negloglikelihood = jax.vmap(
                lambda g_res, eta, disp: model.family.negloglikelihood(g_res[:, jnp.newaxis], y_resid, eta, disp)
            )(G_resid.T, result.eta, result.disp)

            # The residualized fits have one coefficient each; the association API returns one scalar per variant.
            return TestResult(
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
            assert isinstance(state, GlmWaldState)
            y, offset = state.y, state.offset

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

            _, result = lax.scan(_func, 0.0, G.T)
            return result

    @property
    def name(self) -> str:
        r"""Return the test name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the test.
        """
        return "wald"
