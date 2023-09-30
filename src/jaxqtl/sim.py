from typing import NamedTuple

import numpy as np

import jax.numpy as jnp
from jax import Array

from .families.distribution import ExponentialFamily, Gaussian


class SimState(NamedTuple):
    X: Array
    y: Array
    beta: Array


class SimData:
    def __init__(self, nobs: int, family: ExponentialFamily = Gaussian()) -> None:
        self.nobs = nobs
        self.pfeatures = 4
        self.family = family

    def gen_data(self, seed: int = 1, scale: float = 1.0, alpha: float = 0.0):
        n = self.nobs
        p = self.pfeatures
        X_shape = (n, p)
        beta_shape = (p, 1)

        np.random.seed(seed)
        X = np.zeros(X_shape)
        maf = 0.3
        # h2g = 0.1
        # M = 100

        X[:, 0] = np.ones((n,))  # intercept
        X[:, 1] = np.random.normal(40, 4, (n,))  # center, standardize age
        X[:, 2] = np.random.binomial(2, maf, (n,))  # genotype (0,1,2)
        X[:, 3] = np.random.binomial(1, 0.5, (n,))  # sex (0, 1)

        beta = np.random.normal(0, 1, beta_shape)
        beta[1] = 0.01  # np.random.normal(0, h2g/M) # causal eQTL effect

        eta = X @ beta
        mu = self.family.glink.inverse(eta)

        y = self.family.random_gen(mu, scale=scale, alpha=alpha)

        return SimState(jnp.array(X), jnp.array(y), jnp.array(beta))
