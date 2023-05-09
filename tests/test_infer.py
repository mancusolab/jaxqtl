import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from utils import assert_array_eq, assert_betas_eq

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax.config import config

from jaxqtl.families.distribution import Binomial, Poisson  # , Gaussian
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CGSolve, CholeskySolve

config.update("jax_enable_x64", True)

# io toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X
solver = CholeskySolve

stepsize = 1.0
maxiter = 100

test_resid_family = Binomial()  # Poisson reg result is closer

test_irls = GLM(
    X=spector_data.exog,
    y=spector_data.endog,
    family=Poisson(),
    append=False,
    maxiter=maxiter,
    stepsize=stepsize,
)

glm_state = test_irls.fit()


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
    w_X = glmstate_null.glm_wt * jnp.array(covar)
    projection_covar = (
        jnp.array(covar) @ jnpla.inv(w_half_X.T @ w_half_X) @ w_X.T
    )  # nxn
    # projection_covar = w_half_X @ jnpla.inv(w_half_X.T @ w_half_X) @ w_half_X.T  # nxn

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

    mod_null = sm.GLM(spector_data.endog, covar, family=sm.families.Binomial()).fit()
    mod_null_eta = mod_null.get_prediction(covar, which="linear").predicted
    mod_null_mu = mod_null.get_prediction(covar, which="mean").predicted
    glm_wt = mod_null_mu * (1 - mod_null_mu)
    w_half_X = jnp.sqrt(glm_wt[:, jnp.newaxis]) * jnp.array(covar)
    w_X = glm_wt[:, jnp.newaxis] * jnp.array(covar)
    projection_covar = (
        jnp.array(covar) @ jnpla.inv(w_half_X.T @ w_half_X) @ w_X.T
    )  # nxn
    # projection_covar = w_half_X @ jnpla.inv(w_half_X.T @ w_half_X) @ w_half_X.T  # nxn

    X["PSI_resid"] = PSI[:, jnp.newaxis] - projection_covar @ PSI[:, jnp.newaxis]
    mod_G = sm.GLM(
        spector_data.endog,
        X["PSI_resid"],
        family=sm.families.Binomial(),
        offset=mod_null_eta,
    ).fit()

    print(f"statsmodel betas: est={mod_G.params[-1]}")
    print(f"statsmodel SE: est={mod_G.bse[-1]}")

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
        solver=CGSolve(),
        stepsize=stepsize,
    )
    glm_state = test_logit.fit()
    assert_betas_eq(glm_state, sm_state, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


def test_poisson():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

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


def test_CGsolve():
    dat = jnp.array(pd.read_csv("./example/local/ENSG00000178607_onesnp.tsv", sep="\t"))
    y = dat[:, -1][:, jnp.newaxis]
    X = dat[:, 0:-1]

    sm_state = smPoisson(np.array(y), np.array(X)).fit(disp=0)

    glm_state = GLM(
        X=X,
        y=y,
        family=Poisson(),
        solver=CGSolve(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    ).fit()
    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson_scoretest():
    mod_full = GLM(
        X=spector_data.exog,
        y=spector_data.endog,
        family=Poisson(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    ).fit()
    print(mod_full.p[-1])

    mod_null = GLM(
        X=spector_data.exog.drop("GPA", axis=1),
        y=spector_data.endog,
        family=Poisson(),
        append=False,
        maxiter=maxiter,
        stepsize=stepsize,
    ).fit()

    pval_score = GLM.score_test_add_g(
        Poisson(),
        jnp.array(spector_data.exog),
        jnp.array(spector_data.endog)[:, jnp.newaxis],
        mod_null,
        1.0,
    )

    # the discrepancy might be caused by small sample size n=30
    assert_array_eq(pval_score, mod_full.p[-1])


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
