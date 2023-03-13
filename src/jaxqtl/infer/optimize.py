# from functools import partial
from typing import NamedTuple, Tuple

import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla

# import jax.scipy.stats as jaxstats
from jax import lax, numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily

from .solve import LinearSolve


class IRLSState(NamedTuple):
    beta: jnp.ndarray
    num_iters: int
    converged: bool


def OLS(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    Q, R = jnpla.qr(X)
    return jspla.solve_triangular(R, Q.T @ y)


# @jit
def irls(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: ExponentialFamily,
    solver: LinearSolve,
    eta: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:

    # @partial(jit, static_argnums=(3,4))
    # truth = jnp.array([[-1.498017,  0.463852,  0.010495,  0.378555]]).reshape((4,1))
    def body_fun(val: Tuple):
        old_eta, beta = val
        beta = solver(X, y, old_eta, family)
        return old_eta, beta

    def cond_fun(val: Tuple):
        old_eta, beta = val
        new_pdf = family.log_prob(X, y, X @ beta)
        old_pdf = family.log_prob(X, y, old_eta)
        delta = jnp.abs(old_pdf - new_pdf)
        return delta > tol  # and count <= max_iter

    init_beta = jnp.zeros((X.shape[1], 1))
    init_tuple = (eta, init_beta)
    old_eta, beta = lax.while_loop(cond_fun, body_fun, init_tuple)

    num_iters = max_iter
    converged = True

    # converged = False
    # old_pdf = family.log_prob(X, y, eta)
    # for idx in range(max_iter):
    #     beta = solver(X, y, eta, family)
    #     new_pdf = family.log_prob(X, y, X @ beta)
    #     delta = jnp.abs(new_pdf - old_pdf)
    #     if delta <= tol:
    #         converged = True
    #         num_iters = idx + 1  # start count at 0
    #         break
    #     else:
    #         num_iters = max_iter
    #
    #     eta = X @ beta
    #     old_pdf = new_pdf

    return IRLSState(beta, num_iters, converged)
