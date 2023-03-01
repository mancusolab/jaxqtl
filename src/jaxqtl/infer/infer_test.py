import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

import src.jaxqtl.infer.distribution as family
from src.jaxqtl.infer.solve import GLM

# load toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X

# -------------------------------------------------#

# # test linear regression function
# TODO!!: need fix residual variance for calculating SE
test_irls = GLM(
    X=spector_data.exog, y=spector_data.endog, family=family.Normal(), append=False
)
test_irls.fit()

mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

# -------------------------------------------------#

# # test IRLS: logistic regression
test_irls = GLM(
    X=spector_data.exog, y=spector_data.endog, family=family.Binomial(), append=False
)
test_irls.fit()

mod = sm.Logit(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

# # test poisson regression
test_poisson = GLM(
    X=spector_data.exog, y=spector_data.endog, family=family.Poisson(), append=False
)
test_poisson.fit()

res = Poisson(spector_data.endog, spector_data.exog).fit(disp=0)
print(res.summary())

# -------------------------------------------------#

# # test distribution.py
# from src.jaxqtl.infer.distribution import Normal
# import jax.numpy as jnp
# import jax
#
# y = jnp.asarray(spector_data.endog)
# normal1 = Normal(y, jnp.array([0.]), jnp.array([1]))
# print(normal1)
