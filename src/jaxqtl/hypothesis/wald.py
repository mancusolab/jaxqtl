import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import ArrayLike

from ..families.distribution import Gaussian
from .base import _residualize_genotypes, HypothesisTest, TestResult


class WaldTest(HypothesisTest):
    _is_linear: bool = eqx.field(static=True, init=False)

    def __post_init__(self):
        self._is_linear = isinstance(self.model.family, Gaussian)

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        if self._is_linear:
            result = self.model.fit(X, y, offset, self.std_err)
            y_resid = result.resid
            G_resid = _residualize_genotypes(X, G, result.resid_covar, result.glm_wt)

            # fit residualized model, one snp at-a-time; but we can do this all in one go using vmap
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
                disp=result.disp,
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
                )

            _, state = lax.scan(_func, 0.0, G.T)

        return state

    @property
    def name(self) -> str:
        return "wald"
