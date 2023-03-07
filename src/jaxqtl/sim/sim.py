from typing import NamedTuple

import numpy as np
from numpy import random

from src.jaxqtl.infer.distribution import NB, Binomial, Gamma, Normal, Poisson


class SimState(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    se: np.ndarray


class SimData:
    def __init__(self, family: str, nobs: int) -> None:
        self.nobs = nobs
        self.pfeatures = 2

        if family == "Gaussian":
            self.family = Normal()
        elif family == "Binomial":
            self.family = Binomial()
        elif family == "Poisson":
            self.family = Poisson()
        elif family == "Gamma":
            self.family = Gamma()
        elif family == "NB":
            self.family = NB()
        else:
            print("no family found")

    def gen_data(self):
        X_shape = (self.nobs, self.pfeatures)
        beta_shape = (self.pfeatures, 1)
        y_shape = (self.nobs, 1)

        X = np.zeros(X_shape)
        maf = 0.3
        X[:, 0] = random.binomial(3, maf, (self.nobs,))  # genotype (0,1,2)
        X[:, 1] = random.binomial(1, 0.5, (self.nobs,))  # sex (0, 1)

        beta = random.normal(0, 1, beta_shape)

        eta = X @ beta
        mu = self.family.glink_inv(eta)
        sigma = 1  # sigma

        # TODO: need to call this function with diff parameters
        # y = self.family.random_gen(mu, sigma, y_shape)
        y = self.family.random_gen(mu, y_shape)
        _, _, W = self.family.calc_weight(X, y, eta)
        se = np.sqrt(np.diag(np.linalg.inv((X * W).T @ X) * sigma ** 2))

        return SimState(X, y, beta, se)
