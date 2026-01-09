from typing import NamedTuple

import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import Array, ArrayLike

from .families.distribution import ExponentialFamily, NegativeBinomial


class SimulatedData(NamedTuple):
    r"""Container for simulated covariates, genotype, offset, linear predictor, and outcome."""

    X: Array
    g: Array
    y: Array
    offset: Array

    beta: Array
    gamma: float
    eta: Array
    mu: Array

    family: ExponentialFamily


def simulate_pheno(
    key: Array,
    n: int,
    family: ExponentialFamily,
    beta: ArrayLike,
    gamma: float,
    maf: float = 0.3,
    offset: ArrayLike | None = None,
    dispersion: float | None = None,
) -> SimulatedData:
    r"""Simulate phenotype data under a GLM with a supplied [`jaxqtl.families.ExponentialFamily`][].

    This simulates covariates $X$, a single genotype vector $g \in \{0,1,2\}$, and an optional offset, then forms
    a linear predictor $\eta$ and mean $\mu = g^{-1}(\eta)$ using the family link.

    **Arguments:**

    - `key`: JAX PRNG key used to sample `X`, `g`, and `y`.
    - `n`: Number of samples.
    - `family`: GLM family; `family.glink.inverse` is used to obtain `mu` and `family.sample` is used to draw `y`.
    - `beta`: Covariate effect sizes with shape `(p,)`.
    - `gamma`: Additive genotype effect size.
    - `maf`: Minor allele frequency used for genotype simulation.
    - `offset`: Optional offset with shape `(n,)`. If `None`, zeros are used.
    - `dispersion`: Optional dispersion/scale parameter. If `None`, defaults to `0.1` for
      [`jaxqtl.families.NegativeBinomial`][] and `1.0` otherwise.

    **Returns:**

    A [`jaxqtl.sim.SimulatedData`][] containing simulated `X`, `g`, `y`, `offset`, and derived quantities.
    """
    beta = jnp.asarray(beta)
    p = int(beta.shape[0])

    if beta.shape[0] != p:
        raise ValueError(f"`beta` length {beta.shape[0]} does not match `p`={p}")

    key, x_key, g_key, eps_key = rdm.split(key, 4)

    X = rdm.normal(x_key, shape=(n, p))
    # simulate genotype as Binomial(2, maf) scaled to {0,1,2}
    g = rdm.binomial(g_key, n=2, p=maf, shape=(n,))

    if offset is None:
        offset = jnp.zeros((n,))
    elif offset.shape[0] != n:
        raise ValueError(f"`offset` length {offset.shape[0]} does not match `n`={n}")

    eta = offset + X @ beta + gamma * g
    mu = family.glink.inverse(eta)

    if dispersion is None:
        dispersion = 0.1 if isinstance(family, NegativeBinomial) else 1.0

    y = family.sample(eps_key, eta, dispersion)

    return SimulatedData(X=X, g=g, y=y, offset=offset, beta=beta, gamma=gamma, eta=eta, mu=mu, family=family)
