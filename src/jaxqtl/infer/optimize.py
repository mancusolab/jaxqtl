from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax import grad, lax
from jax.numpy import linalg as jnla
from jax.scipy import stats as jaxstats
from jax.scipy.special import polygamma
from jaxtyping import Array, ArrayLike, ScalarLike

from ..families.distribution import ExponentialFamily
from .solve import LinearSolve


class IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    disp: Array


@eqx.filter_jit
def irls(
    X: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    eta: ArrayLike,
    family: ExponentialFamily,
    solver: LinearSolve,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    disp_init: ScalarLike = 0.0,
) -> IRLSState:
    """IRLS to solve GLM

    :param X: covariate data matrix (nxp)
    :param y: outcome vector (nx1)
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param solver: linear equation solver
    :param eta: linear component eta
    :param max_iter: maximum iterations for fitting GLM, default to 1000
    :param tol: tolerance for stopping, default to 0.001
    :param step_size: step size to update the parameter at each step, default to 1.0
    :param offset: offset (nx1)
    :param disp_init: initial value for dispersion parameter
    :return: IRLSState
    """
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

    return IRLSState(beta, num_iters, converged, disp)


@eqx.filter_jit
def lstsq(
    X: ArrayLike,
    y: ArrayLike,
    solver: LinearSolve,
) -> IRLSState:
    beta = solver.lstsq(X, y)
    alpha = jnp.array(1)
    converged = jnp.array(True)
    num_iters = 1

    return IRLSState(beta, num_iters, converged, alpha)


class BetaParams(NamedTuple):
    k: float
    n: float
    converged: bool


@eqx.filter_jit
def infer_beta_params(
    p_perm: ArrayLike,
    init: ArrayLike,
    step_size=0.1,
    tol=1e-3,
    max_iter=500,
) -> BetaParams:
    """Infer shape and scale parameter for beta distribution
    given p values from R permutations (strongest signals),
    use newton's method to estimate beta distribution parameters:
    p ~ Beta(k, n)

    :param p_perm: permutation minimum p values
    :param init: initial value for shape and scale
    :param step_size: step size to update parameters at each step, default to 0.1
    :param tol: tolerance for stopping, default to 0.001
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :return:
    """

    def loglik(params, p: ArrayLike) -> Array:
        return jnp.sum(jaxstats.beta.logpdf(p, params[0], params[1]))

    def info_and_christoffel(params: ArrayLike, p: ArrayLike) -> tuple[Array, Array]:
        """
        We compute the FIM under the Beta(k, n) distribution as well as the Christoffel symbols of the 2nd kind.
        We use the christoffel symbols to perform a 2nd-order natural gradient approach which keeps us on the positive
        manifold. Fancy-shmancy way to do this rather than staying in log space, but here we leverage the underlying
        geometry of the distribution rather than log<->exp for R<->R+.
        """
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

        new_lik = loglik(new_param, p_perm)
        diff = old_lik - new_lik

        return new_lik, diff, num_iter + 1, new_param

    def cond_fun(val: tuple):
        old_lik, diff, num_iter, old_param = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_tuple = (10000.0, 1000.0, 0, init)
    lik, diff, num_iters, params = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter).astype(float)

    return BetaParams(params[0], params[1], converged)
