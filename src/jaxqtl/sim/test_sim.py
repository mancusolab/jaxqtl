import numpy as np
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)

from jax.config import config

from src.jaxqtl.infer.families.distribution import Poisson
from src.jaxqtl.infer.glm import GLM
from src.jaxqtl.infer.solve import CholeskySolve
from src.jaxqtl.sim.sim import SimData

config.update("jax_enable_x64", True)


np.random.seed(1)

n = 1000  # TODO: not converge for seed=1, n=1000
solver = "qr"
family = "Poisson"

sim = SimData(family, n)
X, y, beta = sim.gen_data()

print(beta)
print(f"y: {np.quantile(y, [0, 0.25, 0.5, 0.75, 1])}")
print(f"mean y: {np.mean(y)}")

# no intercept
smPoisson_res = smPoisson(y, X).fit(disp=0)
print(smPoisson_res.summary())

# TODO: this has numerical issue when features have too large values (eg. PC has sd=2)
jaxqtl_poisson = GLM(
    X=X,
    y=y,
    family=Poisson(),
    solver=CholeskySolve(),
    append=False,
    init="default",
    maxiter=100,
)
jaxqtl_poisson.fit()
print(jaxqtl_poisson)


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
