from typing import NamedTuple, Tuple

import equinox as eqx

from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

from ..families.distribution import ExponentialFamily
from .solve import LinearSolve


class IRLSState(NamedTuple):
    beta: Array
    infor_se: Array
    num_iters: int
    converged: Array


# @partial(jax.jit, static_argnames=["max_iter", "tol"])
@eqx.filter_jit
def irls(
    X: ArrayLike,
    g: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    solver: LinearSolve,
    eta: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-3,
    stepsize: float = 1.0,
    offset_eta: ArrayLike = 0.0,
) -> IRLSState:
    def body_fun(val: Tuple):
        diff, num_iter, cov_beta_o, g_beta_o, eta_o, infor_se = val
        cov_beta, g_beta, infor_se = solver(
            X, g, y, eta_o, family, stepsize, offset_eta
        )
        # jax.debug.breakpoint()
        eta_n = X @ cov_beta + g * g_beta + offset_eta
        likelihood_o = family.loglikelihood(X, y, eta_o)  # fine with Poisson
        likelihood_n = family.loglikelihood(X, y, eta_n)
        diff = likelihood_n - likelihood_o

        return diff, num_iter + 1, cov_beta, g_beta, eta_n, infor_se

    def cond_fun(val: Tuple):
        diff, num_iter, cov_beta, g_beta, eta, infor_se = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros((X.shape[1], 1))
    init_tuple = (
        10000.0,
        0,
        init_beta,
        jnp.zeros((1, 1)),
        eta + offset_eta,
        jnp.zeros((X.shape[1] + 1,)),
    )

    diff, num_iters, cov_beta, g_beta, eta, infor_se = lax.while_loop(
        cond_fun, body_fun, init_tuple
    )
    converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter)

    return IRLSState(
        jnp.append(cov_beta, g_beta)[:, jnp.newaxis], infor_se, num_iters, converged
    )
