# pattern: Functional Core

import math

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax import grad, lax
from jax.numpy import linalg as jnla
from jax.scipy import stats as jaxstats
from jax.scipy.special import polygamma
from jaxtyping import Array, ScalarLike

from ..distribution import ExponentialFamily, NegativeBinomial
from ..distribution._expfam import _NB2_DISPERSION_MAX, _NB2_DISPERSION_MIN, _nb2_log_alpha_derivatives
from ._solve import AbstractLinearSolve


_MAX_STEP_TRIALS = 25


class SolveResult(NamedTuple):
    r"""Container for IRLS solver outputs.

    **Attributes:**

    - `beta`: Fitted coefficient vector with shape `(p,)`.
    - `num_iters`: Number of solver iterations.
    - `converged`: Boolean convergence indicator.
    - `disp`: Fitted family dispersion. [`jaxqtl.infer.lstsq`][] returns 1.
    """

    beta: Array
    num_iters: int
    converged: Array
    disp: Array


def irls(
    X: Array,
    y: Array,
    offset: Array,
    eta: Array,
    family: ExponentialFamily,
    solver: AbstractLinearSolve,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    disp_init: ScalarLike = 0.0,
    gtol: float = 1e-3,
) -> SolveResult:
    r"""Solve a GLM with iteratively reweighted least squares (IRLS).

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `y`: Outcome vector with shape `(n,)`.
    - `offset`: Offset vector with shape `(n,)`, or a scalar offset.
    - `eta`: Initial linear predictor $\eta$ with shape `(n,)`, excluding the separately supplied `offset`. IRLS adds
      the offset when constructing the complete initial predictor.
    - `family`: GLM family implementing [`jaxqtl.distribution.ExponentialFamily`][].
    - `solver`: Linear solver implementing [`jaxqtl.infer.AbstractLinearSolve`][].
    - `max_iter`: Maximum IRLS iterations.
    - `tol`: Absolute tolerance on the change in total negative log likelihood. A small change triggers the
      gradient check; it does not establish convergence on its own.
    - `step_size`: Initial step size for each IRLS update. Rejected updates are
      retried with successively halved step sizes.
    - `disp_init`: Initial dispersion estimate.
    - `gtol`: Tolerance on the per-observation gradient infinity norm. Coefficient gradients are divided by
      each design column's root-mean-square magnitude. NB2 also requires a small projected alpha-space gradient,
      respecting its dispersion bounds. Defaults to `1e-3`.

    **Returns:**

    A [`jaxqtl.infer.SolveResult`][] containing fitted coefficients, dispersion, and convergence metadata.

    **Failure Modes:**

    `converged` is false on exhausted backtracking, an iteration limit, or an unchanged nonstationary state.
    """
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and greater than 0")
    if not math.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be finite and greater than 0")

    return _irls(X, y, offset, eta, family, solver, max_iter, tol, step_size, disp_init, gtol)


@eqx.filter_jit
def _irls(
    X: Array,
    y: Array,
    offset: Array,
    eta: Array,
    family: ExponentialFamily,
    solver: AbstractLinearSolve,
    max_iter: int,
    tol: float,
    step_size: float,
    disp_init: ScalarLike,
    gtol: float,
) -> SolveResult:
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    eta = jnp.asarray(eta)
    disp_init = jnp.asarray(disp_init)
    n, _ = X.shape
    column_scale = jnp.sqrt(jnp.mean(X**2, axis=0))
    gradient_scale = n * jnp.where(column_scale > 0, column_scale, 1.0)

    def stationary(eta, disp):
        # Evaluate at the accepted (eta, alpha), not at the old alpha used to
        # construct its dispersion update. No information-matrix solve is needed.
        mu, link_deriv, weight = family.calc_weight(eta, disp)
        beta_gradient = -(X.T @ (weight * link_deriv * (y - mu)))
        beta_error = jnp.max(jnp.abs(beta_gradient) / gradient_scale)
        converged = jnp.isfinite(beta_error) & (beta_error <= gtol)
        if isinstance(family, NegativeBinomial):
            log_alpha_gradient, _, _, _ = _nb2_log_alpha_derivatives(y, family.glink.log_inverse(eta), disp)
            alpha_gradient = log_alpha_gradient / disp
            # Log-alpha gradients vanish spuriously near zero. Test the alpha
            # gradient, retaining only feasible descent directions at a bound.
            projected = jnp.where(disp <= _NB2_DISPERSION_MIN, jnp.minimum(alpha_gradient, 0.0), alpha_gradient)
            projected = jnp.where(disp >= _NB2_DISPERSION_MAX, jnp.maximum(projected, 0.0), projected)
            converged = converged & jnp.isfinite(alpha_gradient) & (jnp.abs(projected) / n <= gtol)
        return converged

    def body_fun(val: tuple):
        likelihood_o, converged, num_iter, beta_o, eta_o, disp_o, failed_o = val

        mu_k, g_deriv_k, weight = family.calc_weight(eta_o, disp_o)
        working_residual = g_deriv_k * (y - mu_k)
        # eta_o - offset = X @ beta_o, so solve only for the increment. The
        # residual, factorization, and predictor direction are shared by trials.
        beta_direction = solver.wgt_lstsq(X, working_residual, weight)
        eta_direction = X @ beta_direction

        def halving_cond(trial: tuple):
            trial_step, num_trials, accepted, likelihood, beta, eta, disp = trial
            return jnp.logical_and(~accepted, num_trials < _MAX_STEP_TRIALS)

        def halving_body(trial: tuple):
            trial_step, num_trials, accepted, likelihood, beta, eta, disp = trial
            beta_trial = beta_o + trial_step * beta_direction
            eta_trial = eta_o + trial_step * eta_direction
            disp_trial = family.update_dispersion(X, y, eta_trial, disp_o, trial_step)
            likelihood_trial = family.negloglikelihood(X, y, eta_trial, disp_trial)

            finite_trial = (
                jnp.isfinite(likelihood_trial)
                & jnp.all(jnp.isfinite(beta_trial))
                & jnp.all(jnp.isfinite(eta_trial))
                & jnp.all(jnp.isfinite(disp_trial))
            )
            accept_trial = finite_trial & (likelihood_trial <= likelihood_o)

            likelihood = jnp.where(accept_trial, likelihood_trial, likelihood)
            beta = jnp.where(accept_trial, beta_trial, beta)
            eta = jnp.where(accept_trial, eta_trial, eta)
            disp = jnp.where(accept_trial, disp_trial, disp)

            return trial_step / 2.0, num_trials + 1, accept_trial, likelihood, beta, eta, disp

        trial_init = (
            jnp.asarray(step_size),
            jnp.asarray(0),
            jnp.asarray(False),
            likelihood_o,
            beta_o,
            eta_o,
            disp_o,
        )
        _, _, accepted, likelihood_n, beta_n, eta_n, disp_n = lax.while_loop(
            halving_cond,
            halving_body,
            trial_init,
        )

        diff = likelihood_n - likelihood_o
        converged = lax.cond(
            accepted & (jnp.abs(diff) <= tol),
            lambda: stationary(eta_n, disp_n),
            lambda: jnp.asarray(False),
        )
        unchanged = jnp.all(eta_n == eta_o) & jnp.all(disp_n == disp_o)
        failed_n = ~accepted | (unchanged & ~converged)

        return likelihood_n, converged, num_iter + 1, beta_n, eta_n, disp_n, failed_n

    def cond_fun(val: tuple):
        likelihood_o, converged, num_iter, beta, eta, disp, failed = val
        return ~converged & (num_iter < max_iter) & ~failed

    init_beta = solver.lstsq(X, eta)
    init_eta = X @ init_beta + offset
    init_likelihood = family.negloglikelihood(X, y, init_eta, disp_init)
    init_tuple = (init_likelihood, jnp.asarray(False), 0, init_beta, init_eta, disp_init, jnp.asarray(False))

    likelihood_n, converged, num_iters, beta, eta, disp, failed = lax.while_loop(cond_fun, body_fun, init_tuple)

    return SolveResult(beta, num_iters, converged, disp)


@eqx.filter_jit
def lstsq(
    X: Array,
    y: Array,
    solver: AbstractLinearSolve,
) -> SolveResult:
    r"""Solve an unweighted least-squares problem.

    This is used as a fast path for Gaussian models.

    **Arguments:**

    - `X`: Covariate matrix with shape `(n, p)`.
    - `y`: Outcome vector with shape `(n,)`.
    - `solver`: Linear solver implementing [`jaxqtl.infer.AbstractLinearSolve`][].

    **Returns:**

    A [`jaxqtl.infer.SolveResult`][] with `disp` set to 1 and `num_iters` set to 1.
    """
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    beta = solver.lstsq(X, y)
    alpha = jnp.array(1)
    converged = jnp.array(True)
    num_iters = 1

    return SolveResult(beta, num_iters, converged, alpha)


class BetaParams(NamedTuple):
    r"""Container for fitted Beta approximation parameters.

    **Attributes:**

    - `k`: First fitted Beta shape parameter.
    - `n`: Second fitted Beta shape parameter.
    - `converged`: Boolean indicator that the parameter iteration met its tolerance.
    """

    k: Array
    n: Array
    converged: Array


@eqx.filter_jit
def infer_beta_params(
    p_perm: Array,
    init: Array,
    step_size=0.1,
    tol=1e-3,
    max_iter=500,
) -> BetaParams:
    r"""Fit a Beta approximation to a collection of permutation p-values.

    Given permutation p-values $p_1, \dots, p_R$, this estimates parameters `k` and `n` such that
    $p \sim \mathrm{Beta}(k, n)$. The implementation uses a natural-gradient/Newton-style iteration on the
    positive parameter manifold.

    **Arguments:**

    - `p_perm`: Permutation p-values with shape `(R,)`.
    - `init`: Initial parameter vector `(k, n)`.
    - `step_size`: Update step size.
    - `tol`: Convergence tolerance on the change in objective value.
    - `max_iter`: Maximum number of iterations.

    **Returns:**

    A [`jaxqtl.infer.BetaParams`][] with fitted parameters and a convergence indicator.
    """

    p_perm = jnp.asarray(p_perm)
    init = jnp.asarray(init)
    step_size = jnp.asarray(step_size)
    tol = jnp.asarray(tol)

    def loglik(params: Array, p: Array) -> Array:
        return jnp.sum(jaxstats.beta.logpdf(p, params[0], params[1]))

    def info_and_christoffel(params: Array, p: Array) -> tuple[Array, Array]:
        # Compute the Fisher information matrix for Beta(k, n) and the Christoffel symbols of the second kind.
        # These are used to take a second-order natural-gradient step while keeping parameters on R_+^2.
        k, n = params

        # reuse terms
        pg_1k = polygamma(1, k)
        pg_1n = polygamma(1, n)
        pg_1kn = polygamma(1, k + n)

        pg_2k = polygamma(2, k)
        pg_2n = polygamma(2, n)
        pg_2kn = polygamma(2, k + n)

        # fisher information matrix
        i_kn = -pg_1kn
        i_k = pg_1k + i_kn
        i_n = pg_1n + i_kn

        # Fisher information matrix; positive definite for valid parameter values.
        info_mat = -len(p) * jnp.array([[i_k, i_kn], [i_kn, i_n]])

        # first sub-matrix of the unscaled 2nd-order Christoffell symbol
        i_kkn = pg_1n * pg_2kn
        i_k = -pg_1n * pg_2k + i_kkn + pg_1kn * pg_2k
        i_knn = i_kkn - pg_1kn * pg_2n

        # second sub-matrix of the unscaled 2nd-order Christoffell symbol
        i_nnk = pg_1k * pg_2kn
        i_nkk = i_nnk - pg_1kn * pg_2k
        i_n = -pg_1k * pg_2n + i_nnk + pg_1kn * pg_2n

        # scale for the 2nd-order Christoffel symbol
        scale = -pg_1k * pg_1n + (pg_1k + pg_1n) * pg_1kn

        # combine into single tensor
        sec_gamma = 0.5 * jnp.array([[[i_k, i_kkn], [i_kkn, i_knn]], [[i_nkk, i_nnk], [i_nnk, i_n]]]) / scale

        return info_mat, sec_gamma

    score_fn = grad(loglik)

    def body_fun(val: tuple):
        old_lik, diff, num_iter, old_param = val
        # first order approx to RGD => NGD
        # direction = NatGrad
        info_mat, gamma = info_and_christoffel(old_param, p_perm)
        direction = jnla.solve(info_mat, score_fn(old_param, p_perm))

        # take second order approx to RGD
        adjustment = jnp.einsum("cab,a,b->c", gamma, direction, direction)
        new_param = old_param - step_size * direction - 0.5 * step_size**2 * adjustment
        new_param = jnp.clip(new_param, 1e-8, jnp.inf)

        new_lik = loglik(new_param, p_perm)
        diff = new_lik - old_lik

        return new_lik, diff, num_iter + 1, new_param

    def cond_fun(val: tuple):
        old_lik, diff, num_iter, old_param = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_lik = loglik(init, p_perm)
    init_tuple = (init_lik, jnp.asarray(jnp.inf), 0, init)
    lik, diff, num_iters, params = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter).astype(float)

    return BetaParams(params[0], params[1], converged)
