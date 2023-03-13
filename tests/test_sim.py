import numpy as np
import numpy.testing as nptest
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)
from statsmodels.genmod.generalized_linear_model import GLMResults

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.infer.solve import CholeskySolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)


def _assert_betas_eq(state: GLMState, sm_state: GLMResults):
    nptest.assert_allclose(state.beta, sm_state.params)


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
    _assert_betas_eq(glm_state, sm_state)


# test NB
# res = smNB(y, X).fit()
# print(res.summary())
#
# test_NB = GLM(
#     X=X, y=y, family="NB", solver=solver, append=False, init="default", maxiter=100,
#     link="Log"
# )
# test_NB.fit()
# print(test_NB)
