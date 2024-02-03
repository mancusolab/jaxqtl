import numpy as np
import pandas as pd

from pandas_plink import read_plink
from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CGSolve, CholeskySolve
from jaxqtl.sim import run_sim, sim_data


config.update("jax_enable_x64", True)

step_size = 1.0
max_iter = 100
true_beta = 0.1


def test_sim_poisson():
    seed = 1
    n = 1000

    X, y, beta, _, _ = sim_data(
        nobs=n, family=Poisson(), method="bulk", alpha=0.0, maf=0.3, eqtl_beta=true_beta, seed=seed, libsize=1
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


def test_sim_NB():
    seed = 1
    n = 1000
    true_alpha = 1
    beta0 = 1

    X, y, beta, _, _ = sim_data(
        nobs=n,
        family=NegativeBinomial(),
        method="bulk",
        alpha=true_alpha,
        maf=0.1,
        eqtl_beta=true_beta,
        seed=seed,
        beta0=beta0,
        libsize=1,
    )

    mod = smNB(np.array(y), np.array(X))
    sm_state = mod.fit(maxiter=100)

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=max_iter,
        solver=CGSolve(),
        step_size=step_size,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois)

    nb_fam = NegativeBinomial()
    alpha_init = len(y) / jnp.sum((y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2)
    alpha_n = nb_fam.estimate_dispersion(X, y, glm_state_pois.eta, alpha=alpha_init)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=max_iter,
        solver=CholeskySolve(),
        step_size=step_size,
    )
    init_nb = jaxqtl_nb.family.init_eta(y)
    glm_state = jaxqtl_nb.fit(X, y, init=init_nb, alpha_init=alpha_n)

    print(f"jaxqtl alpha: {glm_state.alpha}")
    print(f"jaxqtl beta: {glm_state.beta}")
    print(f"jaxqtl pval: {glm_state.p}")
    print(f"statsmodel params: {sm_state.params}")
    assert_array_eq(glm_state.alpha, sm_state.params[-1], rtol=1e-3)
    assert_array_eq(glm_state.alpha, true_alpha, rtol=1e-3)


# def test_sim():
#     """
#     test sim for single cell data
#     """
n = 982
num_cells = 100
family = Poisson()
chr = 1

bim, fam, bed = read_plink(f"../example/data/sim_chr{chr}", verbose=False)
G = bed.compute()  # MxN array

NK_covar = pd.read_csv("../example/data/NK_covar_libsize.tsv", sep="\t")

covar = jnp.array(NK_covar[['sex', 'age']])
covar = covar / jnp.std(covar, axis=0)  # gives higher counts
# libsize = jnp.array(NK_covar['libsize']).reshape((-1, 1))
libsize = jnp.ones((n, 1))

res = run_sim(
    nobs=n,
    num_cells=num_cells,
    num_sim=3,
    beta0=1.0,
    family=family,
    sample_covar_arr=covar,
    covar_var=0.1,
    m_causal=10,
    # eqtl_beta=0,
    libsize=libsize,
    method="bulk",
    G=G,
    out_path="../example/data/test_sim",
)

print(res)
