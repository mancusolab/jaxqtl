from abc import abstractmethod
from typing import Generic, Literal, NamedTuple, Protocol, TypeVar

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jspec
import lineax as lx
import optimistix as optx

from jax.scipy import stats
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, ScalarLike

from ..infer.glm import GLMState
from .base import _residualize_genotypes, _score_from_residuals, AbstractHypothesisTest, TestResult


class HasPredMean(Protocol):
    r"""Protocol for SPA state objects exposing a predicted mean."""

    @property
    def pred_mean(self) -> Array:
        r"""Return the per-sample predicted mean.

        **Arguments:**

        `None`

        **Returns:**

        A 1D array of predicted means.
        """
        ...


CGFStateT = TypeVar("CGFStateT", bound=HasPredMean)


class CumulantGeneratingFunction(eqx.Module, Generic[CGFStateT]):
    r"""Abstract base for cumulant generating functions used by SPA."""

    @abstractmethod
    def init(self, glm_state: GLMState) -> CGFStateT:
        r"""Construct CGF state from a fitted model.

        **Arguments:**

        - `glm_state`: A fitted [`jaxqtl.infer.GLMState`][].

        **Returns:**

        A CGF state object used by [`jaxqtl.hypothesis.saddlepoint_pvalue`][].
        """
        ...

    def __call__(self, t: Array, state: CGFStateT) -> Array:
        r"""Evaluate the cumulant generating function.

        **Arguments:**

        - `t`: Evaluation point(s).
        - `state`: CGF state created by `init`.

        **Returns:**

        CGF value(s) evaluated at `t`.
        """
        return self.cgf(t, state)

    @abstractmethod
    def cgf(self, t: Array, state: CGFStateT) -> Array:
        r"""Evaluate the cumulant generating function.

        **Arguments:**

        - `t`: Evaluation point(s).
        - `state`: CGF state created by `init`.

        **Returns:**

        CGF value(s) evaluated at `t`.
        """
        ...

    def get_t_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        r"""Return bounds on the root-finding parameter `t`.

        **Arguments:**

        - `g_resid`: Residualized genotype vector with shape `(n,)`.
        - `state`: CGF state created by `init`.

        **Returns:**

        A `(lower, upper)` tuple of bounds for `t`.
        """
        return -jnp.inf, jnp.inf

    def get_score_bounds(self, g_resid: Array, state: CGFStateT) -> tuple:
        r"""Return bounds on the score statistic under the CGF model.

        **Arguments:**

        - `g_resid`: Residualized genotype vector with shape `(n,)`.
        - `state`: CGF state created by `init`.

        **Returns:**

        A `(lower, upper)` tuple of score bounds.
        """
        return -jnp.inf, jnp.inf


class BasicCGFState(NamedTuple):
    r"""CGF state containing only the predicted mean."""

    pred_mean: Array


class PoissonCGF(CumulantGeneratingFunction[BasicCGFState]):
    r"""CGF implementation for a Poisson mean model.

    For $Y \sim \mathrm{Poisson}(\mu)$, the cumulant generating function is
    $K(t) = \log \mathbb{E}[\exp(tY)] = \mu(\exp(t) - 1)$.
    In this implementation, $\mu$ is taken from the fitted model mean.
    """

    def init(self, glm_state: GLMState) -> BasicCGFState:
        r"""Initialize CGF state from a fitted model.

        **Arguments:**

        - `glm_state`: A fitted [`jaxqtl.infer.GLMState`][].

        **Returns:**

        A [`jaxqtl.hypothesis.spa.BasicCGFState`][].
        """
        return BasicCGFState(glm_state.mu)

    def cgf(self, t: Array, state: BasicCGFState) -> Array:
        r"""Evaluate the Poisson CGF.

        **Arguments:**

        - `t`: Evaluation point(s).
        - `state`: CGF state created by `init`.

        **Returns:**

        CGF value(s) evaluated at `t`.
        """
        return state.pred_mean * jnp.expm1(t)

    def get_score_bounds(self, g_resid: Array, state: BasicCGFState) -> tuple:
        r"""Compute score bounds for SPA under a Poisson model.

        **Arguments:**

        - `g_resid`: Residualized genotype vector with shape `(n,)`.
        - `state`: CGF state created by `init`.

        **Returns:**

        A `(lower, upper)` tuple of score bounds.
        """
        offset = jnp.sum(g_resid * state.pred_mean)
        ubound = jnp.sum(jnp.where(g_resid < 0, 0, g_resid)) - offset
        lbound = jnp.sum(jnp.where(g_resid > 0, 0, g_resid)) - offset
        return lbound, ubound


class NegBinCGFState(NamedTuple):
    r"""CGF state for Negative Binomial SPA."""

    pred_mean: Array
    r: Array


class NegativeBinomialCGF(CumulantGeneratingFunction[NegBinCGFState]):
    r"""CGF implementation for a Negative Binomial mean/dispersion model.

    For $Y \sim \mathrm{NegBin}(\mu, r)$ parameterized by mean $\mu$ and shape $r$,
    the cumulant generating function can be written as
    $K(t) = -r\log\left(1 - \frac{\mu}{r}(\exp(t) - 1)\right)$.
    This implementation uses $r = 1/\alpha$ where $\alpha$ is the fitted dispersion.
    """

    def init(self, glm_state: GLMState) -> NegBinCGFState:
        r"""Initialize CGF state from a fitted model.

        **Arguments:**

        - `glm_state`: A fitted [`jaxqtl.infer.GLMState`][].

        **Returns:**

        A [`jaxqtl.hypothesis.spa.NegBinCGFState`][].
        """
        return NegBinCGFState(glm_state.mu, 1.0 / glm_state.disp)

    def cgf(self, t: Array, state: NegBinCGFState) -> Array:
        r"""Evaluate the Negative Binomial CGF.

        **Arguments:**

        - `t`: Evaluation point(s).
        - `state`: CGF state created by `init`.

        **Returns:**

        CGF value(s) evaluated at `t`.
        """
        term = -(state.pred_mean / state.r) * jnp.expm1(t)
        return jspec.xlog1py(-state.r, term)

    def get_t_bounds(self, g_resid: Array, state: NegBinCGFState) -> tuple:
        r"""Compute bounds on `t` for Negative Binomial SPA.

        **Arguments:**

        - `g_resid`: Residualized genotype vector with shape `(n,)`.
        - `state`: CGF state created by `init`.

        **Returns:**

        A `(lower, upper)` tuple of bounds for `t`.
        """
        u_max = jnp.log1p(state.r / state.pred_mean)

        rescaled = jnp.where(g_resid != 0, u_max / g_resid, 0.0)
        t_lower = jnp.max(jnp.where(g_resid < 0, rescaled, -jnp.inf))
        t_upper = jnp.min(jnp.where(g_resid > 0, rescaled, jnp.inf))

        return t_lower, t_upper

    def get_score_bounds(self, g_resid: Array, state: NegBinCGFState) -> tuple:
        r"""Compute score bounds for Negative Binomial SPA.

        **Arguments:**

        - `g_resid`: Residualized genotype vector with shape `(n,)`.
        - `state`: CGF state created by `init`.

        **Returns:**

        A `(lower, upper)` tuple of score bounds.
        """
        offset = jnp.sum(g_resid * state.pred_mean)
        lbound = jnp.where(jnp.all(g_resid >= 0), -offset, -jnp.inf)
        ubound = jnp.where(jnp.all(g_resid <= 0), -offset, jnp.inf)

        return lbound, ubound


class GaussianCGFState(NamedTuple):
    r"""CGF state for Gaussian SPA."""

    pred_mean: Array
    variance: Array


class GaussianCGF(CumulantGeneratingFunction[GaussianCGFState]):
    r"""CGF implementation for a Gaussian mean/variance model.

    For $Y \sim \mathcal{N}(\mu, \sigma^2)$, the cumulant generating function is
    $K(t) = \mu t + \frac{1}{2}\sigma^2 t^2$.
    """

    def init(self, glm_state: GLMState) -> GaussianCGFState:
        r"""Initialize CGF state from a fitted model.

        **Arguments:**

        - `glm_state`: A fitted [`jaxqtl.infer.GLMState`][].

        **Returns:**

        A [`jaxqtl.hypothesis.spa.GaussianCGFState`][].
        """
        return GaussianCGFState(glm_state.mu, jnp.reciprocal(glm_state.glm_wt))

    def cgf(self, t: Array, state: GaussianCGFState) -> Array:
        r"""Evaluate the Gaussian CGF.

        **Arguments:**

        - `t`: Evaluation point(s).
        - `state`: CGF state created by `init`.

        **Returns:**

        CGF value(s) evaluated at `t`.
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
    r"""Compute an SPA-corrected p-value for a score statistic.

    **Arguments:**

    - `score`: Observed score statistic.
    - `g_resid`: Residualized genotype vector with shape `(n,)`.
    - `cgf`: A [`jaxqtl.hypothesis.CumulantGeneratingFunction`][] implementation.
    - `state`: CGF state created by `cgf.init`.
    - `scale`: Scalar multiplier applied to the SPA root-finding parameter.
    - `two_sided_mode`: Strategy for two-sided p-values (`"rstar"`, `"abs"`, or `"2min"`).
    - `log_p`: Whether to return the log p-value.
    - `cutoff`: Threshold on the normal z-score above which SPA is attempted.
    - `max_iter`: Maximum number of root-finding steps.

    **Returns:**

    A scalar p-value (or log p-value) for the observed score statistic.
    """
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


class SpaTest(AbstractHypothesisTest):
    r"""Saddlepoint approximation (SPA) score test.

    This starts from a score statistic $S$ for each variant under the null model, then computes a
    saddlepoint approximation based on the cumulant generating function $K(t)$ implied by the fitted mean model.
    A root $\hat t$ is found such that $K'(\hat t) = S$, then the Barndorff-Nielsen approximation is used via
    $r^* = w + \log(v/w)/w$ with $w = \mathrm{sign}(\hat t)\sqrt{2(\hat t S - K(\hat t))}$ and
    $v = \hat t\sqrt{K''(\hat t)}$, yielding a two-sided p-value via a Normal tail approximation.

    !!! info

        For discrete families a continuity correction term is included, such that
        $v = (1 - \exp(-\hat t))\sqrt{K''(\hat t)}$.
    """

    cgf: CumulantGeneratingFunction = NegativeBinomialCGF()

    def test(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
    ) -> TestResult:
        r"""Compute SPA-corrected p-values for each variant in `G`.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `G`: Genotype matrix with shape `(n, m)` (variants in columns).
        - `y`: Outcome vector with shape `(n,)`.
        - `offset`: Offset vector with shape `(n,)`, or a scalar offset.

        **Returns:**

        A [`jaxqtl.hypothesis.TestResult`][] containing per-variant SPA p-values.
        """
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
        r"""Return the test name.

        **Arguments:**

        `None`

        **Returns:**

        A short string identifier for the test.
        """
        return "score.spa"
