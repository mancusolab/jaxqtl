# pattern: Functional Core

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax import grad, lax
from jax.numpy import linalg as jnla
from jax.scipy import stats as jaxstats
from jax.scipy.special import polygamma
from jaxtyping import Array, ScalarLike

from ..distribution import ExponentialFamily
from ._solve import AbstractLinearSolve


class SolveResult(NamedTuple):
    r"""Container for IRLS solver outputs.

    Stores fitted coefficients, dispersion, and basic convergence metadata from [`jaxqtl.infer.irls`][].
    """

    beta: Array
    num_iters: int
    converged: Array
    disp: Array


@eqx.filter_jit
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
    - `tol`: Convergence tolerance on the change in objective value.
    - `step_size`: Step size applied to the IRLS update.
    - `disp_init`: Initial dispersion estimate.

    **Returns:**

    A [`jaxqtl.infer.SolveResult`][] containing fitted coefficients, dispersion, and convergence metadata.
    """
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    eta = jnp.asarray(eta)
    disp_init = jnp.asarray(disp_init)
    n, p = X.shape

    def body_fun(val: tuple):
        likelihood_o, diff, num_iter, beta_o, eta_o, disp_o = val

        mu_k, g_deriv_k, weight = family.calc_weight(eta_o, disp_o)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset

        beta = solver.wgt_lstsq(X, r, weight)

        eta_n = X @ beta + offset

        alpha_n = family.update_dispersion(X, y, eta_n, disp_o, step_size)
        likelihood_n = family.negloglikelihood(X, y, eta_n, alpha_n)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, alpha_n

    def cond_fun(val: tuple):
        likelihood_o, diff, num_iter, beta, eta, disp = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros(p)
    init_tuple = (10000.0, 10000.0, 0, init_beta, eta + offset, disp_init)

    likelihood_n, diff, num_iters, beta, eta, disp = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter)

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

    Stores the fitted Beta shape parameters from [`jaxqtl.infer.infer_beta_params`][] along with a convergence
    indicator.
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
