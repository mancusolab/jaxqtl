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


# @partial(jax.jit, static_argnames=["max_iter", "tol"])
@eqx.filter_jit
def irls(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: ExponentialFamily,
    solver: LinearSolve,
    eta: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-3,
    stepsize: float = 1.0,
    offset_eta: ArrayLike = 0.0,
) -> IRLSState:
    def body_fun(val: Tuple):
        diff, num_iter, beta_o, eta_o = val
        beta = solver(X, y, eta_o, family, stepsize, offset_eta)
        # jax.debug.breakpoint()
        eta_n = X @ beta + offset_eta
        likelihood_o = family.loglikelihood(X, y, eta_o)
        likelihood_n = family.loglikelihood(X, y, eta_n)
        diff = likelihood_n - likelihood_o

        return diff, num_iter + 1, beta, eta_n

    def cond_fun(val: Tuple):
        diff, num_iter, beta, eta = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros((X.shape[1], 1))
    init_tuple = (10000.0, 0, init_beta, eta + offset_eta)

    diff, num_iters, beta, eta = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter)

    return IRLSState(beta, num_iters, converged)


# For Debug
# def irls(
#     X: jnp.ndarray,
#     y: jnp.ndarray,
#     family: ExponentialFamily,
#     solver: LinearSolve,
#     eta: jnp.ndarray,
#     max_iter: int = 1000,
#     tol: float = 1e-3,
# ) -> IRLSState:
#
#     diff, num_iters, eta_o = (10000.0, 0, eta)
#     converged = False
#
#     while jnp.fabs(diff) > tol and num_iters <= max_iter:
#         beta = solver(X, y, eta_o, family)
#         eta_n = X @ beta
#         likelihood_o = family.likelihood(X, y, eta_o)
#         likelihood_n = family.likelihood(X, y, eta_n)
#         diff = likelihood_n - likelihood_o
#
#         if jnp.fabs(diff) < tol:
#             converged = True
#             break
#         num_iters += 1
#         eta_o = eta_n
#
#     return IRLSState(beta, num_iters, converged)
