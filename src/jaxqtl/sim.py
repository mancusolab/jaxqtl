from typing import NamedTuple

import numpy as np
import pandas as pd
import qtl.norm
from numpy import ndarray

import jax.numpy as jnp
from jax import Array

from .families.distribution import (
    ExponentialFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from .infer.glm import GLM
from .infer.utils import score_test_snp


class SimState(NamedTuple):
    X: Array
    y: Array
    beta: Array


class SimResState(NamedTuple):
    pval_nb_wald: ndarray
    pval_pois_wald: ndarray
    pval_nb_wald_robust: ndarray
    pval_pois_wald_robust: ndarray
    pval_lm: ndarray
    pval_nb_score: ndarray
    pval_pois_score: ndarray


class SimData:
    def __init__(self, nobs: int, family: ExponentialFamily = Gaussian()) -> None:
        self.nobs = nobs
        self.pfeatures = 4  # intercept, sex, age, genotype of one SNP
        self.family = family

    def gen_data(
        self,
        scale: float = 1.0,
        alpha: float = 0.0,
        maf: float = 0.3,
        model: str = "alt",
        true_beta: float = 0.0,
        seed: int = 1,
    ) -> SimState:
        n = self.nobs
        p = self.pfeatures
        X_shape = (n, p)
        beta_shape = (p, 1)

        X = np.zeros(X_shape)

        np.random.seed(seed)

        X[:, 0] = np.ones((n,))  # intercept
        age = np.random.normal(40, 4, (n,))
        X[:, 1] = (age - age.mean()) / age.std()  # center, standardize age
        sex = np.random.binomial(1, 0.5, (n,))
        X[:, 2] = (sex - sex.mean()) / sex.std()  # sex (0, 1)

        # simulate genotype for one SNP
        X[:, 3] = np.random.binomial(2, maf, (n,))  # genotype (0,1,2)

        # generate true betas
        beta = np.random.normal(0, 1, beta_shape)

        if model == "null":
            beta[-1] = 0.0
        else:
            beta[-1] = true_beta

        eta = X @ beta
        mu = self.family.glink.inverse(eta)

        y = self.family.random_gen(mu, scale=scale, alpha=alpha)

        return SimState(jnp.array(X), jnp.array(y), jnp.array(beta))


def run_sim(
    seed: int = 1,
    scale: float = 1.0,
    alpha: float = 0.01,
    maf: float = 0.3,
    n: int = 1000,
    model: str = "alt",
    num_sim: int = 1000,
    true_beta: float = 0.0,
    sim_family: ExponentialFamily = NegativeBinomial(),
) -> SimResState:
    np.random.seed(seed)
    sim = SimData(n, sim_family)

    pval_nb_wald = np.array([])
    pval_pois_wald = np.array([])

    pval_nb_wald_robust = np.array([])
    pval_pois_wald_robust = np.array([])

    pval_lm = np.array([])

    pval_nb_score = np.array([])
    pval_pois_score = np.array([])

    for i in range(num_sim):
        X, y, beta = sim.gen_data(
            alpha=alpha, maf=maf, model=model, scale=scale, true_beta=true_beta, seed=i
        )

        # fit poisson or negative binomial
        jaxqtl_pois = GLM(family=Poisson())
        init_pois = jaxqtl_pois.family.init_eta(y)
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, robust_se=False)

        nb_fam = NegativeBinomial()
        alpha_init = len(y) / jnp.sum(
            (y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2
        )
        alpha_n = nb_fam.calc_dispersion(X, y, glm_state_pois.eta, alpha=alpha_init)

        jaxqtl_nb = GLM(family=nb_fam)
        init_nb = nb_fam.init_eta(y)
        glm_state_nb = jaxqtl_nb.fit(
            X, y, init=init_nb, alpha_init=alpha_n, robust_se=False
        )

        pval_nb_wald = np.append(pval_nb_wald, glm_state_nb.p[-1])
        pval_pois_wald = np.append(pval_pois_wald, glm_state_pois.p[-1])

        # robust poisson and NB
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, robust_se=True)
        glm_state_nb = jaxqtl_nb.fit(
            X, y, init=init_nb, alpha_init=alpha_n, robust_se=True
        )

        pval_nb_wald_robust = np.append(pval_nb_wald_robust, glm_state_nb.p[-1])
        pval_pois_wald_robust = np.append(pval_pois_wald_robust, glm_state_pois.p[-1])

        # fit lm
        norm_df = qtl.norm.inverse_normal_transform(pd.DataFrame(y).T)
        y_norm = np.array(norm_df.T)

        jaxqtl_lm = GLM(family=Gaussian())
        init_lm = jaxqtl_lm.family.init_eta(y_norm)
        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm)
        pval_lm = np.append(pval_lm, glm_state.p[-1])

        # score test for poisson and NB
        X_cov = X[:, 0:-1]
        glm_null_pois = jaxqtl_pois.fit(X_cov, y, init=init_pois)
        _, pval = score_test_snp(
            G=X[:, -1].reshape((n, 1)), X=X_cov, glm_null_res=glm_null_pois
        )

        pval_pois_score = np.append(pval_pois_score, pval)

        alpha_init = len(y) / jnp.sum(
            (y / nb_fam.glink.inverse(glm_null_pois.eta) - 1) ** 2
        )
        alpha_n = nb_fam.calc_dispersion(X_cov, y, glm_null_pois.eta, alpha=alpha_init)

        glm_state_nb = jaxqtl_nb.fit(X_cov, y, init=init_nb, alpha_init=alpha_n)
        _, pval = score_test_snp(
            G=X[:, -1].reshape((n, 1)), X=X_cov, glm_null_res=glm_state_nb
        )

        pval_nb_score = np.append(pval_nb_score, pval)

    return SimResState(
        pval_nb_wald=pval_nb_wald,
        pval_pois_wald=pval_pois_wald,
        pval_nb_wald_robust=pval_nb_wald_robust,
        pval_pois_wald_robust=pval_pois_wald_robust,
        pval_nb_score=pval_nb_score,
        pval_pois_score=pval_pois_score,
        pval_lm=pval_lm,
    )
