import numpy as np
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)
from util import assert_betas_eq

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)


def test_sim_poisson():
    np.random.seed(1)

    n = 1000
    family = Poisson()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data()

    # no intercept
    sm_state = smPoisson(y, X).fit(disp=0)

    jaxqtl_poisson = GLM(
        X=X,
        y=y,
        family=Poisson(),
        solver=CholeskySolve(),
        append=False,
        maxiter=100,
    )
    glm_state = jaxqtl_poisson.fit()
    assert_betas_eq(glm_state, sm_state)
