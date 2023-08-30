import numpy as np
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)
from utils import assert_array_eq, assert_betas_eq

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)

np.random.seed(2)

n = 1000
family = Poisson()

sim = SimData(n, family)
X, y, beta = sim.gen_data()

# no intercept
sm_state = smPoisson(y, X).fit(disp=0)

np.random.seed(1)

n = 1000
family = Poisson()

sim = SimData(n, family)
X, y, beta = sim.gen_data()


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
        max_iter=100,
    )
    glm_state = jaxqtl_poisson.fit()
    assert_betas_eq(glm_state, sm_state)


def test_sim_poisson_scoretest():
    np.random.seed(1)

    n = 1000
    family = Poisson()

    sim = SimData(n, family)
    X, y, beta = sim.gen_data()

    X = X[:, [0, 2, 3, 1]]  # last p = 0.125

    mod_full = GLM(
        X=X,
        y=y,
        family=Poisson(),
        solver=CholeskySolve(),
        append=False,
        max_iter=100,
    ).fit()
    print(mod_full.p[-1])

    mod_null = GLM(
        X=X[:, 0:3],
        y=y,
        family=Poisson(),
        solver=CholeskySolve(),
        append=False,
        max_iter=100,
    ).fit()

    pval_score = GLM.score_test_snp(Poisson(), X, y, mod_null, 1.0)

    assert_array_eq(pval_score, mod_full.p[-1])
