from typing import NamedTuple

import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
from jax import numpy as jnp

from .distribution import AbstractExponential
from .solve import AbstractLinearSolve


class IRLSState(NamedTuple):
    beta: jnp.ndarray
    num_iters: int
    converged: bool


def OLS(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    Q, R = jnpla.qr(X)
    return jspla.solve_triangular(R, Q.T @ y)


def irls(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: AbstractExponential,
    solver: AbstractLinearSolve,
    seed: int,
    init: str = "default",
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:

    converged = False
    pfeatures = X.shape[1]
    if init == "OLS":
        old_beta = OLS(X, y)
    elif init == "default":
        old_beta = family.init_mu(pfeatures, seed)
    else:
        print("init method not found.")

    for idx in range(max_iter):
        new_beta = solver(X, y, old_beta, family)
        norm = jnpla.norm(new_beta - old_beta)  # alternative check the log likelihood
        if norm <= tol:
            converged = True
            num_iters = idx + 1  # start count at 0
            break

        old_beta = new_beta

    return IRLSState(new_beta, num_iters, converged)
