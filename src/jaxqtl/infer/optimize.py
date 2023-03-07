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
    seed: int,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> IRLSState:

    converged = False
    pfeatures = X.shape[1]
    old_beta = family.init_mu(pfeatures, seed)

    for idx in range(max_iter):
        new_beta = solver(X, y, old_beta, family)
        norm = jnpla.norm(new_beta - old_beta)  # alternative check the log likelihood
        if norm <= tol:
            converged = True
            num_iters = idx + 1  # start count at 0
            break

        old_beta = new_beta

    return IRLSState(new_beta, num_iters, converged)
