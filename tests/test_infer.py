import numpy.testing as nptest
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from statsmodels.genmod.generalized_linear_model import GLMResults

from jax.config import config

from jaxqtl.families.distribution import Binomial, Poisson
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.infer.solve import CholeskySolve

config.update("jax_enable_x64", True)

# load toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X
solver = CholeskySolve


def _assert_betas_eq(state: GLMState, sm_state: GLMResults):
    nptest.assert_allclose(state.beta, sm_state.params)


def test_linear_regression():

    # test linear regression function
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    test_irls = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        append=False,
        init="default",
        maxiter=100,
    )
    glm_state = test_irls.fit()
    _assert_betas_eq(glm_state, sm_state)

    test_irls = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        solver=CholeskySolve(),
        append=False,
        init="default",
        maxiter=100,
    )
    glm_state = test_irls.fit()
    _assert_betas_eq(glm_state, sm_state)


def test_logistic():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    test_logit = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        family=Binomial(),
        append=False,
        init="default",
        maxiter=100,
    )
    glm_state = test_logit.fit()
    _assert_betas_eq(glm_state, sm_state)


def test_poisson():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit(disp=0)

    test_poisson = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        family=Poisson(),
        append=False,
        init="default",
        maxiter=100,
    )
    glm_state = test_poisson.fit()
    _assert_betas_eq(glm_state, sm_state)


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
