import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)

from src.jaxqtl.infer.families.distribution import Binomial, Poisson
from src.jaxqtl.infer.glm import GLM

# load toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X
solver = "qr"  # qr, cholesky, CG
# -------------------------------------------------#

# test linear regression function
mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

test_irls = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    append=False,
    init="default",
    maxiter=100,
)
test_irls.fit()
print(test_irls)

# -------------------------------------------------#

# test IRLS: logistic regression
mod = sm.Logit(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

test_logit = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family=Binomial(),
    append=False,
    init="default",
    maxiter=100,
)
test_logit.fit()
print(test_logit)

# test poisson regression
res = smPoisson(spector_data.endog, spector_data.exog).fit(disp=0)
print(res.summary())

test_poisson = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family=Poisson(),
    append=False,
    init="default",
    maxiter=100,
)
test_poisson.fit()
print(test_poisson)

# -------------------------------------------------#

# data = sm.datasets.scotland.load()
# data.exog = sm.add_constant(data.exog)
# gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
# gamma_results = gamma_model.fit()
# print(gamma_results.summary())

# test_Gamma = GLM(
#     X=data.exog,
#     y=data.endog,
#     family="Gamma",
#     seed=123,
#     solver=solver,
#     append=False,
# )
# test_Gamma.fit()
# print(test_Gamma)

# data = sm.datasets.scotland.load()
# data.exog = sm.add_constant(data.exog)
# NB_model = NegativeBinomial(spector_data.endog, spector_data.exog)
# NB_results = NB_model.fit(maxiter=100)
# print(NB_results.summary())

# test_NB = GLM(
#     X=data.exog,
#     y=data.endog,
#     family="NB",
#     seed=123,
#     solver=solver,
#     append=False,
#     init="default",
# )
# test_NB.fit()
# print(test_NB)
