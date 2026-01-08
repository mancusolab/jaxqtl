from abc import abstractmethod
from typing import Generic, Literal, NamedTuple, Protocol, TypeVar

import optimistix as optx

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jspec
import lineax as lx

from jax.scipy import stats
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, ScalarLike

from ..infer.glm import GLMState
from .base import _residualize_genotypes, _score_from_residuals, HypothesisTest, TestResult


class HasPredMean(Protocol):
    @property
    def pred_mean(self) -> Array:
        ...


CGFStateT = TypeVar("CGFStateT", bound=HasPredMean)


class CumulantGeneratingFunction(eqx.Module, Generic[CGFStateT]):
    """Abstract base for cumulant generating functions used by SPA."""

    @abstractmethod
    def init(self, glm_state: GLMState) -> CGFStateT:
        ...

    def __call__(self, t: Array, state: CGFStateT) -> Array:
        return self.cgf(t, state)

    @abstractmethod
    def cgf(self, t: Array, state: CGFStateT) -> Array:
        ...

    def get_t_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        return -jnp.inf, jnp.inf

    def get_score_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        return -jnp.inf, jnp.inf


class BasicCGFState(NamedTuple):
    pred_mean: Array


class PoissonCGF(CumulantGeneratingFunction[BasicCGFState]):
    def init(self, glm_state: GLMState) -> BasicCGFState:
        return BasicCGFState(glm_state.mu)

    def cgf(self, t: Array, state: BasicCGFState) -> Array:
        return state.pred_mean * jnp.expm1(t)

    def get_score_bounds(self, g_resid: Array, state: BasicCGFState) -> tuple:
        offset = jnp.sum(g_resid * state.pred_mean)
        ubound = jnp.sum(jnp.where(g_resid < 0, 0, g_resid)) - offset
        lbound = jnp.sum(jnp.where(g_resid > 0, 0, g_resid)) - offset
        return lbound, ubound


class NegBinCGFState(NamedTuple):
    pred_mean: Array
    r: Array


class NegativeBinomialCGF(CumulantGeneratingFunction[NegBinCGFState]):
    def init(self, glm_state: GLMState) -> NegBinCGFState:
        return NegBinCGFState(glm_state.mu, 1.0 / glm_state.disp)

    def cgf(self, t: Array, state: NegBinCGFState) -> Array:
        term = -(state.pred_mean / state.r) * jnp.expm1(t)
        return jspec.xlog1py(-state.r, term)

    def get_t_bounds(self, g_resid: Array, state: NegBinCGFState) -> tuple:
        u_max = jnp.log1p(state.r / state.pred_mean)

        rescaled = jnp.where(g_resid != 0, u_max / g_resid, 0.0)
        t_lower = jnp.max(jnp.where(g_resid < 0, rescaled, -jnp.inf))
        t_upper = jnp.min(jnp.where(g_resid > 0, rescaled, jnp.inf))

        return t_lower, t_upper

    def get_score_bounds(self, g_resid: Array, state: NegBinCGFState) -> tuple:
        offset = jnp.sum(g_resid * state.pred_mean)
        lbound = jnp.where(jnp.all(g_resid >= 0), -offset, -jnp.inf)
        ubound = jnp.where(jnp.all(g_resid <= 0), -offset, jnp.inf)

        return lbound, ubound


class GaussianCGFState(NamedTuple):
    pred_mean: Array
    variance: Array


class GaussianCGF(CumulantGeneratingFunction[GaussianCGFState]):
    def init(self, glm_state: GLMState) -> GaussianCGFState:
        return GaussianCGFState(glm_state.mu, jnp.reciprocal(glm_state.glm_wt))

    def cgf(self, t: Array, state: GaussianCGFState) -> Array:
        return t * (state.pred_mean + 0.5 * t * state.variance)


@eqx.filter_jit
def saddlepoint_pvalue(
    score: ScalarLike,
    g_resid: ArrayLike,
    cgf: CumulantGeneratingFunction[CGFStateT],
    state: CGFStateT,
    scale: ScalarLike = 1.0,
    two_sided_mode: Literal["rstar", "abs", "2min"] = "rstar",
    log_p: bool = False,
    cutoff: float = 1.96,
    max_iter: int = 100,
) -> Array:
    is_discrete = isinstance(cgf, (PoissonCGF, NegativeBinomialCGF))

    g_resid = jnp.asarray(g_resid, dtype=float)
    score = jnp.asarray(score, dtype=float)

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    t_bounds = cgf.get_t_bounds(g_resid, state)
    score_bounds = cgf.get_score_bounds(g_resid, state)

    offset = g_resid.T @ state.pred_mean

    @eqx.filter_value_and_grad
    def _closure(t):
        rescale = t * scale
        return jnp.sum(cgf(rescale * g_resid, state)) - rescale * offset

    def _fn(t, args):
        (current_score,) = args
        _val, deriv = _closure(t)
        return deriv - current_score

    _, (_, score_var) = jax.jvp(_closure, (0.0,), (1.0,))
    zscore = score / jnp.sqrt(score_var)
    log_p_normal_two_sided = jnp.log(2.0) + stats.norm.logsf(jnp.abs(zscore))

    _lbound, _ubound = score_bounds
    is_valid = (_lbound < score) & (score < _ubound)
    should_attempt_spa = (jnp.fabs(zscore) > cutoff) & is_valid

    def _spa(current_score):
        lower, upper = t_bounds
        sol = optx.root_find(
            _fn,
            solver,
            0.0,
            args=(current_score,),
            options={"lower": lower, "upper": upper},
            has_aux=False,
            max_steps=max_iter,
            throw=False,
            tags=frozenset({lx.positive_semidefinite_tag, lx.diagonal_tag}),
        )
        t_bar = sol.value

        (K_val, _K_p), (_, K_pp) = jax.jvp(_closure, (t_bar,), (1.0,))

        under_radical = 2 * (t_bar * current_score - K_val)
        w = jnp.sign(t_bar) * jnp.sqrt(under_radical)

        scale_factor = -jnp.expm1(-t_bar) if is_discrete else t_bar
        v = scale_factor * jnp.sqrt(K_pp)

        ratio = v / w
        r = w + jnp.log(ratio) / w
        r = jnp.where(ratio <= 0, jnp.nan, r)

        t_result_lower = stats.norm.logcdf(r)
        t_result_upper = stats.norm.logsf(r)
        t_result_symm = jnp.log(2.0) + jnp.where(r <= 0.0, t_result_lower, t_result_upper)

        w_is_valid = ~jnp.isnan(w)
        r_is_valid = ~jnp.isnan(r)
        solver_success = sol.result == optx.RESULTS.successful
        is_successful = w_is_valid & r_is_valid & solver_success

        return t_result_lower, t_result_upper, t_result_symm, is_successful

    def compute_spa_p_value(_):
        ascore = jnp.abs(score)
        _, log_upper_pos, log_tail_symm, is_successful = _spa(ascore)
        if two_sided_mode == "rstar":
            spa_result = log_tail_symm
        else:
            log_lower_neg, _, _, ok_neg = _spa(-ascore)
            is_successful = is_successful & ok_neg
            log_tails = jnp.array([log_upper_pos, log_lower_neg])
            if two_sided_mode == "abs":
                spa_result = logsumexp(log_tails)
            else:
                spa_result = jnp.log(2.0) + jnp.min(log_tails)

        return jnp.where(is_successful, spa_result, log_p_normal_two_sided)

    def compute_normal_p_value(_):
        return log_p_normal_two_sided

    final_log_p = lax.cond(
        should_attempt_spa,
        compute_spa_p_value,
        compute_normal_p_value,
        operand=None,
    )

    return jnp.exp(final_log_p) if not log_p else final_log_p


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
        wgt = jnp.atleast_1d(glmstate_cov_only.glm_wt)
        gprime = glmstate_cov_only.link_prime

        g_resid = _residualize_genotypes(X, G, glmstate_cov_only.resid_covar, wgt)
        beta, se, zscore, g_score, _ = _score_from_residuals(y_resid, g_resid, wgt)

        spa_g_resid = g_resid * (wgt * gprime)[:, jnp.newaxis]

        def _pval(args, idx):
            pv = saddlepoint_pvalue(g_score[idx], spa_g_resid[:, idx], self.cgf, cgf_state, two_sided_mode="abs")
            return args, pv

        _, gupval = lax.scan(_pval, 0.0, jnp.arange(G.shape[1]))

        return TestResult(
            beta=beta,
            se=se,
            p=gupval,
            z=zscore,
            num_iters=glmstate_cov_only.num_iters,
            converged=glmstate_cov_only.converged,
            disp=glmstate_cov_only.disp,
        )

    @property
    def name(self) -> str:
        return "score.spa"
