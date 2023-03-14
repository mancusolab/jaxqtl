import numpy as np
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve  # , QRSolve, CGSolve
from jaxqtl.sim import SimData

config.update("jax_enable_x64", True)

np.random.seed(1)

n = 10000
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

fitstatsmodel = smPoisson(y, X).fit(disp=0)

print(sm_state.summary())
print(glm_state)

"""
time it in ipython:

%timeit -n10 -r10 fitjaxqtl = GLM(X=X,y=y,family=Poisson(),solver=CGSolve(),append=False,maxiter=100).fit()
%timeit -n10 -r10 fitjaxqtl = GLM(X=X,y=y,family=Poisson(),solver=CholeskySolve(),append=False,maxiter=100).fit()
%timeit -n10 -r10 fitjaxqtl = GLM(X=X,y=y,family=Poisson(),solver=QRSolve(),append=False,maxiter=100).fit()
%timeit -n10 -r10 fitstatsmodel= smPoisson(y, X).fit(disp=0)
"""

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
