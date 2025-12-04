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

from .glm import GLMState


# set up a protocol to ensure that all downstream types can structurally match this object
# with any additional stuff for more complex cases...
class HasPredMean(Protocol):
    # annoying for type-checking reasons with mypy, but we need to define this as a property here to enforce read-only
    # type checking downstream, which is always the case with NamedTuple objects! >:(
    @property
    def pred_mean(self) -> Array:
        ...


CGFStateT = TypeVar("CGFStateT", bound=HasPredMean)


class CumulantGeneratingFunction(eqx.Module, Generic[CGFStateT]):
    """The abstract base class for instances of Exponential Family distributions."""

    @abstractmethod
    def init(self, glm_state: GLMState) -> CGFStateT:
        ...

    def __call__(self, t: Array, state: CGFStateT) -> Array:
        return self.cgf(t, state)

    @abstractmethod
    def cgf(self, t: Array, state: CGFStateT) -> Array:
        r"""Evaluates the cumulant generating function for each observation at `t`.

        Namely, $K_y(t) := \log \mathbb{E}[\exp(t \cdot y))]$
        where $y$ is the random variable distributed under [`giddyup.families.ExpFam`][] with
        mean corresponding to `pred_mean`.

        **Arguments:**

        - `t`: Number at which to evaluate `cgf`.

        **Returns:**

        The CGF evaluated under distribution assumptions at `t`.
        """
        ...

    def get_t_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        return -jnp.inf, jnp.inf

    def get_score_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        return -jnp.inf, jnp.inf


CumulantGeneratingFunction.__init__.__doc__ = """**Arguments:**

- `pred_mean`: The predicted mean for each observation.
"""


CumulantGeneratingFunction.__call__.__doc__ = """Alias for [giddyup.ExpFam.cgf][]"""


class BasicCGFState(NamedTuple):
    pred_mean: Array


class PoissonCGF(CumulantGeneratingFunction[BasicCGFState]):
    """The Poisson distribution family."""

    def init(self, glm_state: GLMState) -> BasicCGFState:
        return BasicCGFState(glm_state.mu)

    def cgf(self, t: Array, state: BasicCGFState) -> Array:
        r"""Evaluates the Poisson cumulant generating function for each observation at `t`.

        Namely, $K(t) := \mu \cdot (\exp(t) - 1)$ where
        $\mu$ corresponds to `pred_mean`.

        **Arguments:**

        - `t`: Number at which to evaluate `cgf`.

        **Returns:**

        The CGF evaluated under distribution assumptions at `t`.
        """
        return state.pred_mean * jnp.expm1(t)

    def get_score_bounds(self, g_resid: Array, state: BasicCGFState) -> tuple:
        offset = jnp.sum(g_resid * state.pred_mean)  # Fixed: element-wise multiplication then sum
        ubound = jnp.sum(jnp.where(g_resid < 0, 0, g_resid)) - offset
        lbound = jnp.sum(jnp.where(g_resid > 0, 0, g_resid)) - offset
        return lbound, ubound


class NegBinCGFState(NamedTuple):
    pred_mean: Array
    r: Array


class NegativeBinomialCGF(CumulantGeneratingFunction[NegBinCGFState]):
    """The NegativeBinomial distribution family."""

    def init(self, glm_state: GLMState) -> NegBinCGFState:
        return NegBinCGFState(glm_state.mu, 1.0 / glm_state.alpha)

    def cgf(self, t: Array, state: NegBinCGFState) -> Array:
        term = 1 - (state.pred_mean / state.r) * jnp.expm1(t)

        # Ensure the term is positive to avoid taking log of zero or negative numbers
        term = jnp.maximum(term, 1e-16)

        return jspec.xlogy(-state.r, term)

    def get_t_bounds(self, g_resid: Array, state: NegBinCGFState) -> tuple:
        """Compute t boundaries for saddlepoint approximation under Negative Binomial."""
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


# not sure why anyone would WANT to use this, but its here for completeness
class GaussianCGF(CumulantGeneratingFunction):
    """The Normal distribution family."""

    def init(self, glm_state: GLMState) -> GaussianCGFState:
        # variance is just 1 / wgt for Gaussian
        return GaussianCGFState(glm_state.mu, jnp.reciprocal(glm_state.glm_wt))

    def cgf(self, t: Array, state: GaussianCGFState) -> Array:
        r"""Evaluates the Normal cumulant generating function for each observation at `t`.

        Namely, $K(t) := t \cdot \mu + \frac{1}{2}\sigma^2 t^2$
        where $\mu$ corresponds to `pred_mean` and $\sigma^2$ to `variance`.

        **Arguments:**

        - `t`: Number at which to evaluate `cgf`.

        **Returns:**

        The CGF evaluated under distribution assumptions at `t`.
        """
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
    """
    Compute p-values of a score test using a saddlepoint approximation.
    This version ensures that either a two-sided SPA or a two-sided Normal approximation is used,
    and includes debug prints for comparison.
    """

    # Convert inputs to JAX arrays
    g_resid = jnp.asarray(g_resid, dtype=float)
    score = jnp.asarray(score, dtype=float)

    # Solver for root finding
    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    t_bounds = cgf.get_t_bounds(g_resid, state)
    score_bounds = cgf.get_score_bounds(g_resid, state)

    # Our score function
    @eqx.filter_value_and_grad
    def _closure(t):
        return jnp.sum(cgf((t * scale) * g_resid, state)) - t * scale * (g_resid.T @ state.pred_mean)

    # Wrapper around score function for root-finding
    def _fn(t, args):
        (current_score,) = args
        val, deriv = _closure(t)
        return deriv - current_score

    # This efficiently computes first and second order derivatives of our score function
    _, (_, score_var) = jax.jvp(_closure, (0.0,), (1.0,))

    # Pre-calculate the normal approximation result as a fallback
    zscore = score / jnp.sqrt(score_var)
    log_p_normal_two_sided = jnp.log(2.0) + stats.norm.logsf(jnp.abs(zscore))

    # If observed score is inside the theoretical bounds given the ExpFam model and score is big enough
    # (ie beyond null), then perform SPA correction
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

        (K_val, K_p), (_, K_pp) = jax.jvp(_closure, (t_bar,), (1.0,))

        # for numerical reasons we may have negatives, so push up to a tiny value
        under_radical = jnp.maximum(2 * (t_bar * current_score - K_val), 1e-16)
        w = jnp.sign(t_bar) * jnp.sqrt(under_radical)
        v = t_bar * jnp.sqrt(K_pp)

        # Lugannani-Rice formula
        # we can get diff 'bad' results depending on ratio being 0 (-inf) or negative (nan)
        # so just bottom out to 'nan' here
        ratio = v / w
        r = w + jnp.log(ratio) / w
        r = jnp.where(ratio <= 0, jnp.nan, r)

        # compute both tail probabilities and a two-sided assuming symmetry around appropriate tail
        t_result_lower = stats.norm.logcdf(r)
        t_result_upper = stats.norm.logsf(r)
        t_result_symm = jnp.log(2.0) + jnp.where(r <= 0.0, t_result_lower, t_result_upper)

        # validity checks
        w_is_valid = ~jnp.isnan(w)
        r_is_valid = ~jnp.isnan(r)
        solver_success = sol.result == optx.RESULTS.successful
        is_successful = w_is_valid & r_is_valid & solver_success

        return t_result_lower, t_result_upper, t_result_symm, is_successful

    def compute_spa_p_value(_):
        log_tail_lower, log_tail_upper, log_tail_symm, is_successful = _spa(score)
        if two_sided_mode == "rstar":
            spa_result = log_tail_symm
        else:
            log_lower_neg, log_upper_neg, _, ok_neg = _spa(-score)
            is_successful = is_successful & ok_neg
            log_tails = jnp.array([log_tail_upper, log_lower_neg])
            if two_sided_mode == "abs":
                spa_result = logsumexp(log_tails)
            elif two_sided_mode == "2min":
                spa_result = jnp.log(2.0) + jnp.min(log_tails)

        # if SPA failed for any reason, fall back to the normal result
        return jnp.where(is_successful, spa_result, log_p_normal_two_sided)

    def compute_normal_p_value(_):
        return log_p_normal_two_sided

    # primary branch check
    final_log_p = lax.cond(
        should_attempt_spa,
        compute_spa_p_value,  # True branch: attempt the SPA path
        compute_normal_p_value,  # False branch: use the Normal approximation path
        operand=None,
    )

    if not log_p:
        final_p = jnp.exp(final_log_p)
        return final_p
    else:
        return final_log_p
