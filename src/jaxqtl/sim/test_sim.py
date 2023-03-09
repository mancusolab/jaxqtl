import numpy as np
from statsmodels.discrete.discrete_model import Poisson as smPoisson

from src.jaxqtl.infer.glm import GLM
from src.jaxqtl.sim.sim import SimData

np.random.seed(3)

n = 1000
solver = "qr"

sim = SimData("Poisson", n)
X, y, beta = sim.gen_data()
print(beta)

# no intercept
poisson_res = smPoisson(y, X).fit(disp=0)
print(poisson_res.summary())

# TODO: this has numerical issue when features have too large values
test_poisson = GLM(
    X=X, y=y, family="Poisson", solver=solver, append=False, init="default", maxiter=100
)
test_poisson.fit()
print(test_poisson)
