from typing import NamedTuple

from jax import numpy as jnp
from jax.numpy import linalg as jnpla

from .distribution import AbstractExponential
from .solve import AbstractLinearSolve


class IRLSState(NamedTuple):
    beta: jnp.ndarray
    num_iters: int
    converged: bool


def irls(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: AbstractExponential,
    solver: AbstractLinearSolve,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:

    converged = False
    old_beta = solver(X, y, family)

    for idx in range(max_iter):
        new_beta = solver(X, y, family)
        norm = jnpla.norm(new_beta - old_beta)  # alternative check the log likelihood
        if norm <= tol:
            converged = True
            break

        old_beta = new_beta

    return IRLSState(new_beta, idx, converged)
