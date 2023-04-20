import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from utils import assert_betas_eq

from jax.config import config

from jaxqtl.families.distribution import Binomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve

config.update("jax_enable_x64", True)

# io toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X
solver = CholeskySolve

stepsize = 1.0
maxiter = 100


def test_linear_regression():

    # test linear regression function
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    test_irls = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    )

    glm_state = test_irls.fit()
    assert_betas_eq(glm_state, sm_state)

    test_irls = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        solver=CholeskySolve(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    )
    glm_state = test_irls.fit()
    assert_betas_eq(glm_state, sm_state)


def test_logistic():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    test_logit = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        family=Binomial(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    )
    glm_state = test_logit.fit()
    assert_betas_eq(glm_state, sm_state)


def test_poisson():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit(disp=0)

    test_poisson = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        family=Poisson(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    )
    glm_state = test_poisson.fit()
    assert_betas_eq(glm_state, sm_state)


# -------------------------------------------------#

# data = sm.datasets.scotland.io()
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

# data = sm.datasets.scotland.io()
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
