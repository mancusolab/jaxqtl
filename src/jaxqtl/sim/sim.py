from typing import NamedTuple

import numpy as np

from src.jaxqtl.infer.families.distribution import Binomial, Gaussian, Poisson


class SimState(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray


class SimData:
    def __init__(self, family: str, nobs: int) -> None:
        self.nobs = nobs
        self.pfeatures = 6

        if family == "Gaussian":
            self.family = Gaussian()
        elif family == "Binomial":
            self.family = Binomial()
        elif family == "Poisson":
            self.family = Poisson()
        else:
            print("no family found")

    def gen_data(self):
        n = self.nobs
        p = self.pfeatures
        X_shape = (n, p)
        beta_shape = (p, 1)
        y_shape = (n, 1)

        X = np.zeros(X_shape)
        maf = 0.3
        # h2g = 0.1
        # M = 100

        X[:, 0] = np.ones((n,))
        X[:, 1] = np.random.binomial(3, maf, (n,))  # genotype (0,1,2)
        X[:, 2] = np.random.binomial(1, 0.5, (n,))  # sex (0, 1)
        X[:, 3] = np.random.normal(0, 1, (n,))  # center, standardize age
        X[:, 4] = np.random.normal(0, 1, (n,))  # pseudo PC1
        X[:, 5] = np.random.normal(0, 1, (n,))  # pseudo PC2

        beta = np.random.normal(0, 1, beta_shape)
        beta[1] = 0.01  # np.random.normal(0, h2g/M) # causal eQTL effect

        eta = X @ beta
        mu = self.family.glink.inverse(eta)
        # sigma = 1  # sigma

        # TODO: need to call this function with diff parameters
        # y = self.family.random_gen(mu, sigma, y_shape)
        y = self.family.random_gen(mu, y_shape)
        # _, _, W = self.family.calc_weight(X, y, eta)
        # se = np.sqrt(np.diag(np.linalg.inv((X * W).T @ X) * sigma ** 2))

        return SimState(X, y, beta)
