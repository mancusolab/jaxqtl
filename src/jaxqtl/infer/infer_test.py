import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

from src.jaxqtl.infer.glm import GLM

# load toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X

# -------------------------------------------------#

# test linear regression function
test_irls = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family="Gaussian",
    solver="qr",
    append=False,
)
test_irls.fit()
print(test_irls)

mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

# -------------------------------------------------#

# test IRLS: logistic regression
test_logit = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family="Binomial",
    solver="qr",
    append=False,
)
test_logit.fit()
print(test_logit)

mod = sm.Logit(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

# # test poisson regression
test_poisson = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family="Poisson",
    solver="qr",
    append=False,
)
test_poisson.fit()
print(test_poisson)

res = Poisson(spector_data.endog, spector_data.exog).fit(disp=0)
print(res.summary())

# -------------------------------------------------#
