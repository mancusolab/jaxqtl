import numpy as np
import pandas as pd

from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve
from jaxqtl.infer.stderr import FisherInfoError
from jaxqtl.sim import sim_data


config.update("jax_enable_x64", True)

step_size = 1.0
max_iter = 100
true_beta = 0.1


def test_sim_poisson():
    seed = 1
    n = 1000
    V_a = 0.1

    X, y, beta, _, _ = sim_data(
        nobs=n, family=Poisson(), method="bulk", alpha=0.0, maf=0.3, V_a=V_a, seed=seed, libsize=1
    )

    # no intercept
    mod = smPoisson(np.array(y), np.array(X))
    sm_state = mod.fit()

    jaxqtl_poisson_cho = GLM(
        family=Poisson(),
        max_iter=max_iter,
        solver=CholeskySolve(),
        step_size=step_size,
    )
    init_pois = jaxqtl_poisson_cho.family.init_eta(y)
    glm_state = jaxqtl_poisson_cho.fit(X, y, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)


# def test_sim_NB():
seed = 1
n = 982
true_alpha = 10
beta0 = -10
V_a = 0.1
# libsize = 1  # not using lib size

libsize = jnp.array(pd.read_csv("../example/data/CD4_ET.libsize.tsv", sep="\t").iloc[:, 0]).reshape((-1, 1))

log_offset = jnp.log(libsize)

X, y, beta, _, _ = sim_data(
    nobs=n,
    family=NegativeBinomial(),
    method="bulk",
    alpha=true_alpha,
    maf=0.1,
    V_a=V_a,
    seed=seed,
    beta0=beta0,
    libsize=libsize,
)

mod = smNB(np.array(y), np.array(X), offset=log_offset.ravel())
sm_state = mod.fit(maxiter=100)

jaxqtl_nb = GLM(
    family=NegativeBinomial(),
    max_iter=max_iter,
    solver=CholeskySolve(),
)

init_nb, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X, y, log_offset)
alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

glm_state = jaxqtl_nb.fit(X, y, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset, se_estimator=FisherInfoError())

# print(f"jaxqtl alpha: {glm_state.alpha}")
# print(f"jaxqtl beta: {glm_state.beta}")
# print(f"jaxqtl pval: {glm_state.p}")
# print(f"statsmodel params: {sm_state.params}")
# assert_array_eq(glm_state.alpha, sm_state.params[-1], rtol=1e-3)
# assert_array_eq(glm_state.alpha, true_alpha, rtol=1e-3)
