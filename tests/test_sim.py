import numpy as np
from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

from jax.config import config

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve, QRSolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)

stepsize = 1.0
maxiter = 100


def test_sim_poisson():
    np.random.seed(1)

    n = 1000
    family = Poisson()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data()

    # no intercept
    mod = smPoisson(np.array(y), np.array(X))
    sm_state = mod.fit()

    jaxqtl_poisson_cho = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_poisson_cho.family.init_eta(y)
    glm_state = jaxqtl_poisson_cho.fit(X, y, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)


def test_sim_NB():
    np.random.seed(1)

    n = 1000
    true_alpha = 0.05
    family = NegativeBinomial()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data(alpha=true_alpha)

    mod = smNB(np.array(y), np.array(X))
    sm_state = mod.fit()

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois)

    nb_fam = family
    alpha_n = nb_fam.calc_dispersion(X, y, glm_state_pois.eta)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=maxiter,
        solver=QRSolve(),
        step_size=stepsize,
    )
    init_nb = jaxqtl_nb.family.init_eta(y)
    glm_state = jaxqtl_nb.fit(X, y, init=init_nb, alpha_init=alpha_n)

    print(f"jaxqtl alpha: {glm_state.alpha}")
    assert_array_eq(glm_state.beta, sm_state.params[:-1])
    assert_array_eq(glm_state.se, sm_state.bse[:-1], rtol=1e-2)
    assert_array_eq(glm_state.alpha, sm_state.params[-1])
