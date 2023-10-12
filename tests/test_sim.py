import numpy as np
from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

import jax.numpy as jnp
from jax.config import config

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)

step_size = 1.0
max_iter = 100


def test_sim_poisson():
    seed = 1
    n = 1000
    family = Poisson()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data(alpha=0.0, maf=0.3, model="alt", true_beta=0.1, seed=seed)

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
    family = NegativeBinomial()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data(
        alpha=true_alpha, maf=0.1, model="alt", true_beta=0.1, seed=seed, beta0=beta0
    )

    mod = smNB(np.array(y), np.array(X))
    sm_state = mod.fit(maxiter=100)

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=max_iter,
        solver=CholeskySolve(),
        step_size=step_size,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois)

    nb_fam = family
    alpha_init = len(y) / jnp.sum(
        (y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2
    )
    alpha_n = nb_fam.calc_dispersion(X, y, glm_state_pois.eta, alpha=alpha_init)

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
