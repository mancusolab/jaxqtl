from abc import abstractmethod
from typing import NamedTuple, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp

from jax.numpy.linalg import multi_dot
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from ..families.distribution import ExponentialFamily
from .glm import GLM, GLMState
from .stderr import ErrVarEstimation, FisherInfoError


class CisGLMState(NamedTuple):
    beta: Array
    se: Array
    p: Array
    z: Array  # !! add z
    num_iters: Array
    converged: Array
    alpha: Array


def score_test_snp(G: ArrayLike, X: ArrayLike, glm_null_res: GLMState) -> Tuple[Array, Array, Array, Array]:
    """test for additional covariate g
    only require fit null model using fitted covariate only model + new vector g
    X is the full design matrix containing covariates and g
    calculate score in full model using the model fitted from null model

    :param G: genotype matrix
    :param X: covariate data matrix (nxp)
    :param glm_null_res: GLMState from null model (without adding SNP)
    :return: Score test statistics, p value, score, (expected) variance of score
    """
    y_resid = jnp.squeeze(glm_null_res.resid, -1)
    x_W = X * glm_null_res.glm_wt
    sqrt_wgt = jnp.sqrt(glm_null_res.glm_wt)

    g_resid = G - multi_dot([X, glm_null_res.infor_inv, x_W.T, G])
    w_g_resid = g_resid * sqrt_wgt
    g_var = jnp.sum(w_g_resid**2, axis=0)

    g_score = (g_resid * glm_null_res.glm_wt).T @ y_resid
    zscore = g_score / jnp.sqrt(g_var)

    pval = 2 * norm.sf(jnp.fabs(zscore))

    # # test spa test
    # cgf = gu.Poisson(glm_null_res.mu.flatten())  # pred_mean is predicted mean from GLM
    #
    # def _spa(_, jdx):
    #     pvalue_j = gu.saddlepoint_pvalue(g_score[jdx], g_resid[:, jdx], cgf)
    #     return None, pvalue_j
    #
    # _, spa_pval = lax.scan(_spa, None, jnp.arange(len(zscore)))  # p is num snps

    # import numpy as np; import jax; jax.debug.breakpoint()

    return zscore, pval, g_score, g_var


class HypothesisTest(eqx.Module):
    def __call__(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily,
        offset_eta: ArrayLike,
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 1000,
    ) -> CisGLMState:
        """hypothesis test for association between SNP and outcome

        :param X: covariate data matrix (nxp)
        :param G: genotype matrix
        :param y: outcome vector (nx1)
        :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
        :param offset_eta: offset (nx1)
        :param se_estimator: estimator for standard error, default to fisher information
        :param max_iter: maximum iterations for fitting GLM, default to 1000
        :return: CisGLMState
        """
        return self.test(X, G, y, family, offset_eta, se_estimator, max_iter)

    @abstractmethod
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily,
        offset_eta: ArrayLike,
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 1000,
    ) -> CisGLMState:
        pass


class WaldTest(HypothesisTest):
    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily,
        offset_eta: ArrayLike,
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 1000,
    ) -> CisGLMState:
        glm = GLM(family=family, max_iter=max_iter)

        def _func(carry, snp):
            M = jnp.hstack((X, snp[:, jnp.newaxis]))
            eta, alpha_n = glm.calc_eta_and_dispersion(M, y, offset_eta)
            glmstate = glm.fit(
                M,
                y,
                offset_eta=offset_eta,
                init=eta,
                alpha_init=alpha_n,
                se_estimator=se_estimator,
            )

            return carry, CisGLMState(
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
        family: ExponentialFamily,
        offset_eta: ArrayLike,
        se_estimator: ErrVarEstimation = FisherInfoError(),
        max_iter: int = 1000,
    ) -> CisGLMState:
        glm = GLM(family=family, max_iter=max_iter)

        eta, alpha_n = glm.calc_eta_and_dispersion(X, y, offset_eta)

        # Note: linear model might start with bad init
        glmstate_cov_only = glm.fit(
            X, y, offset_eta=offset_eta, init=eta, alpha_init=alpha_n, se_estimator=se_estimator
        )

        zscore, pval, score, score_var = score_test_snp(G, X, glmstate_cov_only)
        beta = score / score_var
        se = 1.0 / jnp.sqrt(score_var)

        return CisGLMState(
            beta=beta,
            se=se,
            p=pval,
            z=zscore,
            num_iters=glmstate_cov_only.num_iters,
            converged=jnp.ones_like(pval) * glmstate_cov_only.converged,
            alpha=jnp.ones_like(pval) * glmstate_cov_only.alpha,
        )
