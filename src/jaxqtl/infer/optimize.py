from typing import NamedTuple

import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax import numpy as jnp

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
    seed: int,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:

    converged = False

    eta = family.init_eta(y)
    old_pdf = family.log_prob(X, y, eta)

    for idx in range(max_iter):
        beta = solver(X, y, eta, family)
        new_pdf = family.log_prob(X, y, X @ beta)
        delta = jnp.abs(new_pdf - old_pdf)
        if delta <= tol:
            converged = True
            num_iters = idx + 1  # start count at 0
            break
        else:
            num_iters = max_iter

        eta = X @ beta
        old_pdf = new_pdf

    return IRLSState(beta, num_iters, converged)
