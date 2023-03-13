from typing import NamedTuple, Tuple

# import jax
from jax import lax, numpy as jnp

from ..families.distribution import ExponentialFamily
from .solve import LinearSolve


class IRLSState(NamedTuple):
    beta: jnp.ndarray
    num_iters: int
    converged: bool


# @jax.jit
def irls(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: ExponentialFamily,
    solver: LinearSolve,
    eta: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:
    def body_fun(val: Tuple):
        diff, num_iter, beta_o, eta_o = val
        beta = solver(X, y, eta_o, family)
        eta_n = X @ beta
        likelihood_o = family.log_prob(X, y, eta_o)
        likelihood_n = family.log_prob(X, y, eta_n)
        diff = likelihood_n - likelihood_o

        return diff, num_iter + 1, beta, eta_n

    def cond_fun(val: Tuple):
        diff, num_iter, beta, eta = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros((X.shape[1], 1))
    init_tuple = (10000.0, 0, init_beta, eta)

    diff, num_iters, beta, eta = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = num_iters < max_iter and jnp.fabs(diff) < tol

    return IRLSState(beta, num_iters, converged)
