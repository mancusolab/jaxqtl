from abc import abstractmethod
from typing import NamedTuple, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp

from jax.numpy.linalg import multi_dot
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from jaxqtl.infer.glm import GLM, GLMState


class CisGLMState(NamedTuple):
    beta: Array
    se: Array
    p: Array
    num_iters: Array
    converged: Array
    alpha: Array


def score_test_snp(G: ArrayLike, X: ArrayLike, glm_null_res: GLMState) -> Tuple[Array, Array, Array, Array]:
    """test for additional covariate g
    only require fit null model using fitted covariate only model + new vector g
    X is the full design matrix containing covariates and g
    calculate score in full model using the model fitted from null model
    """
    y_resid = jnp.squeeze(glm_null_res.resid, -1)
    x_W = X * glm_null_res.glm_wt
    sqrt_wgt = jnp.sqrt(glm_null_res.glm_wt)

    g_resid = G - multi_dot([X, glm_null_res.infor_inv, x_W.T, G])
    w_g_resid = g_resid * sqrt_wgt
    g_var = jnp.sum(w_g_resid**2, axis=0)

    g_score = (g_resid * glm_null_res.glm_wt).T @ y_resid
    Z = g_score / jnp.sqrt(g_var)

    pval = norm.cdf(-abs(Z)) * 2

    return Z, pval, g_score, g_var


class HypothesisTest(eqx.Module):
    def __call__(self, X, G, y, family, offset_eta, robust_se, max_iter):
        return self.test(X, G, y, family, offset_eta, robust_se, max_iter)

    @abstractmethod
    def test(self, X, G, y, family, offset_eta, robust_se, max_iter):
        pass


class WaldTest(HypothesisTest):
    def test(self, X, G, y, family, offset_eta, robust_se, max_iter):
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
                robust_se=robust_se,
            )

            return carry, CisGLMState(
                beta=glmstate.beta[-1],
                se=glmstate.se[-1],
                p=glmstate.p[-1],
                num_iters=glmstate.num_iters,
                converged=glmstate.converged,
                alpha=glmstate.alpha,
            )

        _, state = lax.scan(_func, 0.0, G.T)

        return state


class ScoreTest(HypothesisTest):
    def test(self, X, G, y, family, offset_eta, robust_se, max_iter):
        glm = GLM(family=family, max_iter=max_iter)

        eta, alpha_n = glm.calc_eta_and_dispersion(X, y, offset_eta)

        # Note: linear model might start with bad init
        glmstate_cov_only = glm.fit(X, y, offset_eta=offset_eta, init=eta, alpha_init=alpha_n)

        Z, pval, score, score_var = score_test_snp(G, X, glmstate_cov_only)
        beta = score / score_var
        se = 1.0 / jnp.sqrt(score_var)

        return CisGLMState(
            beta=beta,
            se=se,
            p=pval,
            num_iters=glmstate_cov_only.num_iters,
            converged=jnp.ones_like(pval) * glmstate_cov_only.converged,
            alpha=jnp.ones_like(pval) * glmstate_cov_only.alpha,
        )
