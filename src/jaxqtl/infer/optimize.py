from typing import NamedTuple, Tuple

import equinox as eqx

from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

from ..families.distribution import ExponentialFamily
from .solve import LinearSolve


class IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    alpha: Array


# # @partial(jax.jit, static_argnames=["max_iter", "tol"])
@eqx.filter_jit
def irls(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    solver: LinearSolve,
    eta: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    offset_eta: ArrayLike = 0.0,
    alpha_init: ArrayLike = 0.0,
) -> IRLSState:
    def body_fun(val: Tuple):
        diff, num_iter, beta_o, eta_o, alpha_o = val

        beta = solver(X, y, eta_o, family, step_size, offset_eta, alpha_o)
        eta_n = X @ beta + offset_eta
        alpha_n = family.calc_dispersion(X, y, eta_n, alpha_o, step_size)
        # TODO: check if update once change anything
        # alpha_n = family.update_dispersion(X, y, eta_n, alpha_o, step_size)

        likelihood_o = family.negloglikelihood(X, y, eta_o, alpha_o)
        likelihood_n = family.negloglikelihood(X, y, eta_n, alpha_n)
        diff = likelihood_n - likelihood_o

        return diff, num_iter + 1, beta, eta_n, alpha_n

    def cond_fun(val: Tuple):
        diff, num_iter, beta, eta, alpha = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros((X.shape[1], 1))
    init_tuple = (10000.0, 0, init_beta, eta + offset_eta, alpha_init)

    diff, num_iters, beta, eta, alpha = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter)

    return IRLSState(beta, num_iters, converged, alpha)
