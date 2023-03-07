import numpy as np
from statsmodels.discrete.discrete_model import Poisson

from src.jaxqtl.infer.glm import GLM
from src.jaxqtl.sim.sim import SimData

np.random.seed(10)

n = 100
solver = "CG"

sim = SimData("Poisson", n)
X, y, beta, se = sim.gen_data()
print(beta)
print(se)

# no intercept
poisson_res = Poisson(y, X).fit(disp=0)
print(poisson_res.summary())

test_poisson = GLM(
    X=X, y=y, family="Poisson", solver=solver, append=False, init="default"
)
test_poisson.fit()
print(test_poisson)
