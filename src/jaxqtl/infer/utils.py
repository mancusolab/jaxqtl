from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from jax.numpy.linalg import multi_dot
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from ..families.distribution import Gaussian
from .glm import AbstractLinearModel
from .spa import CumulantGeneratingFunction, NegativeBinomialCGF, saddlepoint_pvalue
from .stderr import ErrVarEstimation, FisherInfoError


class TestResult(NamedTuple):
    beta: Array
    se: Array
    p: Array
    z: Array
    num_iters: Array
    converged: Array
    alpha: Array


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
        """hypothesis test for association between SNP and outcome

        :param X: covariate data matrix (nxp)
        :param G: genotype matrix
        :param y: outcome vector (nx1)
        :param offset: offset (nx1)
        :param se_estimator: estimator for standard error, default to fisher information
        :param max_iter: maximum iterations for fitting GLM, default to 1000
        :return: CisGLMState
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
        pass


class WaldTest(HypothesisTest):
    _is_linear: bool = eqx.field(static=True, init=False)

    def __post_init__(self):
        _is_linear = isinstance(self.model.family, Gaussian)

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        if self._is_linear:
            n, p = X.shape

            # fit model to covariates then compute residuals for y and G
            result = self.model.fit(X, y, offset, self.std_err)
            y_resid = result.resid
            x_W = X * result.glm_wtw
            G_resid = G - multi_dot([X, result.infor_inv, x_W.T, G])

            # fit residualized model, one snp at-a-time; but we can do this all in one go
            # using vmap
            result = jax.vmap(
                lambda g_res: self.model.fit(g_res[:, jnp.newaxis], y_resid, std_err=self.std_err), in_axes=1
            )(G_resid)

            state = TestResult(
                beta=result.beta,
                se=result.se,
                p=result.p,
                z=result.z,
                num_iters=result.num_iters,
                converged=result.converged,
                alpha=result.alpha,
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
                    alpha=glmstate.alpha,
                )

            _, state = lax.scan(_func, 0.0, G.T)

        return state


class ScoreTest(HypothesisTest):
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        # Note: linear model might start with bad init
        glmstate_cov_only = self.model.fit(X, y, offset, self.std_err)

        y_resid = glmstate_cov_only.resid
        wgt = glmstate_cov_only.glm_wt
        x_W = X * wgt[:, jnp.newaxis]
        sqrt_wgt = jnp.sqrt(wgt)

        g_resid = G - multi_dot([X, glmstate_cov_only.infor_inv, x_W.T, G])
        w_g_resid = g_resid * sqrt_wgt[:, jnp.newaxis]
        g_std = jnp.sqrt(jnp.sum(w_g_resid**2, axis=0))
        se = 1.0 / g_std  # this is SE for beta

        g_score = w_g_resid.T @ (sqrt_wgt * y_resid)

        # this is a little confusing bc we're multiplying by se, but its due to
        zscore = g_score * se
        beta = zscore * se

        pval = 2 * norm.sf(jnp.fabs(zscore))

        return TestResult(
            beta=beta,
            se=se,
            p=pval,
            z=zscore,
            num_iters=glmstate_cov_only.num_iters,
            converged=jnp.ones_like(pval) * glmstate_cov_only.converged,
            alpha=jnp.ones_like(pval) * glmstate_cov_only.alpha,
        )


class SpaTest(HypothesisTest):
    cgf: CumulantGeneratingFunction = NegativeBinomialCGF()

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        glmstate_cov_only = self.model.fit(X, y, offset, self.std_err)
        cgf_state = self.cgf.init(glmstate_cov_only)

        y_resid = glmstate_cov_only.resid
        wgt = glmstate_cov_only.glm_wt
        gprime = glmstate_cov_only.link_prime
        x_W = X * wgt[:, jnp.newaxis]
        sqrt_wgt = jnp.sqrt(wgt)

        g_resid = G - multi_dot([X, glmstate_cov_only.infor_inv, x_W.T, G])
        w_g_resid = g_resid * sqrt_wgt[:, jnp.newaxis]
        g_std = jnp.sqrt(jnp.sum(w_g_resid**2, axis=0))
        se = 1.0 / g_std

        g_score = w_g_resid.T @ (sqrt_wgt * y_resid)
        zscore = g_score / g_std
        beta = zscore / g_std

        spa_g_resid = g_resid * (wgt * gprime)[:, jnp.newaxis]

        def _pval(args, idx):
            pv = saddlepoint_pvalue(g_score[idx], spa_g_resid[:, idx], self.cgf, cgf_state)
            return args, pv

        _, gupval = lax.scan(_pval, 0.0, jnp.arange(G.shape[1]))

        return TestResult(
            beta=beta,
            se=se,
            p=gupval,
            z=zscore,
            num_iters=glmstate_cov_only.num_iters,
            converged=jnp.ones_like(gupval) * glmstate_cov_only.converged,
            alpha=jnp.ones_like(gupval) * glmstate_cov_only.alpha,
        )
