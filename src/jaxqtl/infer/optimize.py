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

    mu = family.init_mu(y)
    eta = family.glink(mu)
    phi = family.calc_phi(X, y, mu)
    old_pdf = family.log_prob(y, mu, phi)

    for idx in range(max_iter):
        beta = solver(X, y, eta, family)
        eta = X @ beta
        mu = family.glink.inverse(eta)
        phi = family.calc_phi(X, y, mu)
        new_pdf = family.log_prob(y, mu, phi)
        delta = jnp.abs(new_pdf - old_pdf)  # alternative check the log likelihood
        if delta <= tol:
            converged = True
            num_iters = idx + 1  # start count at 0
            break
        else:
            old_pdf = new_pdf
            num_iters = idx + 1

    return IRLSState(beta, num_iters, converged)
