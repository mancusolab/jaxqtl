import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from utils import assert_array_eq, assert_betas_eq

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax.config import config

from jaxqtl.families.distribution import Binomial, Gaussian, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CholeskySolve

config.update("jax_enable_x64", True)

# io toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X
solver = CholeskySolve

stepsize = 1.0
maxiter = 100

test_resid_family = Gaussian()  # Poisson reg result is closer


def test_resid_reg():
    X = spector_data.exog.copy()
    y = jnp.array(spector_data.endog)[:, jnp.newaxis]

    truth = GLM(
        X=X,
        y=y,
        append=False,
        family=test_resid_family,
        maxiter=maxiter,
        stepsize=stepsize,
    ).fit()

    covar = X.drop("PSI", axis=1)

    glmstate_null = GLM(
        X=covar,
        y=y,
        family=test_resid_family,
        append=False,
        maxiter=100,
    ).fit()

    PSI = jnp.array(X["PSI"])
    w_half_X = jnp.sqrt(glmstate_null.glm_wt) * jnp.array(covar)
    projection_covar = w_half_X @ jnpla.inv(w_half_X.T @ w_half_X) @ w_half_X.T  # nxn

    # covar_on_g = jnpla.lstsq(jnp.array(covar), PSI[:, jnp.newaxis])[0]
    X["PSI_resid"] = (
        PSI[:, jnp.newaxis]
        - projection_covar @ PSI[:, jnp.newaxis]
        # PSI[:, jnp.newaxis] - jnp.array(covar) @ covar_on_g
    )

    glmstate = GLM(
        X=X["PSI_resid"],
        y=y,
        family=test_resid_family,
        append=False,
        maxiter=1000,
    ).fit(glmstate_null.eta)

    print(f"betas: truth={truth.beta[-1]}, est={glmstate.beta[-1]}")
    print(f"SE: truth={truth.se[-1]}, est={glmstate.se[-1]}")
    print(
        f"Z: truth={truth.beta[-1]/truth.se[-1]}, est={glmstate.beta[-1]/glmstate.se[-1]}"
    )
    assert_array_eq(glmstate.beta[-1], truth.beta[-1])
    assert_array_eq(glmstate.se[-1], truth.se[-1])


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
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


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
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson():
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
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_1D_X():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog["PSI"])
    sm_state = mod.fit(disp=0)

    test_poisson = GLM(
        X=spector_data.exog["PSI"],
        y=spector_data.endog,
        family=Poisson(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    )
    glm_state = test_poisson.fit()
    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


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
