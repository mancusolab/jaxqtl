import numpy as np
from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
    Poisson as smPoisson,
)

from src.jaxqtl.infer.glm import GLM
from src.jaxqtl.sim.sim import SimData

np.random.seed(1)

n = 100  # TODO: not converge for seed=1, n=1000
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
    X=X, y=y, family=family, solver=solver, append=False, init="default", maxiter=100
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
