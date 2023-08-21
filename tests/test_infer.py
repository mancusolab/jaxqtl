import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from utils import assert_array_eq, assert_betas_eq

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax.config import config

from jaxqtl.families.distribution import Binomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CGSolve, CholeskySolve, QRSolve

config.update("jax_enable_x64", True)

# io toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X

y_arr = jnp.array(spector_data.endog)[:, jnp.newaxis]
X_arr = jnp.array(spector_data.exog)

maxiter = 100
stepsize = 1.0


def test_linear_regression_cho():

    # test linear regression function
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_cho = GLM(
        solver=CholeskySolve(),
        maxiter=maxiter,
        stepsize=stepsize,
    )

    init_lm = jaxqtl_cho.family.init_eta(jnp.array(spector_data.endog)[:, jnp.newaxis])

    glm_state = jaxqtl_cho.fit(X_arr, y_arr, init=init_lm)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_binomial_cg():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_bin_cg = GLM(
        family=Binomial(),
        maxiter=maxiter,
        solver=CGSolve(),
        stepsize=stepsize,
    )
    init_logistic = jaxqtl_bin_cg.family.init_eta(y_arr)
    glm_state = jaxqtl_bin_cg.fit(X_arr, y_arr, init=init_logistic)

    assert_betas_eq(glm_state, sm_state, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


def test_binomial_cho():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_bin_cho = GLM(
        family=Binomial(),
        maxiter=maxiter,
        solver=CholeskySolve(),
        stepsize=stepsize,
    )
    init_logistic = jaxqtl_bin_cho.family.init_eta(y_arr)
    glm_state = jaxqtl_bin_cho.fit(X_arr, y_arr, init=init_logistic)

    assert_betas_eq(glm_state, sm_state, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


def test_poisson_qr():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_qr = GLM(
        family=Poisson(),
        maxiter=maxiter,
        solver=QRSolve(),
        stepsize=stepsize,
    )
    init_pois = jaxqtl_poisson_qr.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_qr.fit(X_arr, y_arr, init=init_pois)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson_cho():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_cho = GLM(
        family=Poisson(),
        maxiter=maxiter,
        solver=CholeskySolve(),
        stepsize=stepsize,
    )
    init_pois = jaxqtl_poisson_cho.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_cho.fit(X_arr, y_arr, init=init_pois)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson_cg():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_cg = GLM(
        family=Poisson(),
        maxiter=maxiter,
        solver=CGSolve(),
        stepsize=stepsize,
    )
    init_pois = jaxqtl_poisson_cg.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_cg.fit(X_arr, y_arr, init=init_pois)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_CGsolve_realdata():
    """
    # AssertionError: get diff result
    """
    dat = jnp.array(
        pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    )
    y = dat[:, -2][:, jnp.newaxis]
    X = dat[:, 0:-2]

    sm_state = smPoisson(np.array(y), np.array(X)).fit()

    jaxqtl_poisson_cg = GLM(
        family=Poisson(), maxiter=maxiter, solver=CGSolve(), stepsize=stepsize
    )
    init_pois = jaxqtl_poisson_cg.family.init_eta(y)
    glm_state = jaxqtl_poisson_cg.fit(X, y, init=init_pois)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_1D_X():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog["PSI"])
    sm_state = mod.fit()

    X_arr = jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis]
    y_arr = jnp.array(spector_data.endog)[:, jnp.newaxis]

    jaxqtl_pois = GLM(family=Poisson(), maxiter=maxiter, stepsize=stepsize)
    init_pois = jaxqtl_pois.family.init_eta(y_arr)
    glm_state = jaxqtl_pois.fit(X_arr, y_arr, init=init_pois)

    assert_betas_eq(glm_state, sm_state)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_sandwich():
    """
    Compare sandwitch estimator from stats model to jaxqtl
    """
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])[:, jnp.newaxis]
    library_size = jnp.array(dat["log_offset"])[:, jnp.newaxis]

    sm_mod = sm.GLM(
        np.array(y),
        np.array(M),
        family=sm.families.Poisson(),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit()
    white_cov = statsmodels.stats.sandwich_covariance.cov_white_simple(
        sm_mod, use_correction=False
    )

    jaxqtl_pois = GLM(family=Poisson(), maxiter=100, solver=CholeskySolve())
    init_pois = jaxqtl_pois.family.init_eta(y)

    glmstate = jaxqtl_pois.fit(
        M, y, init=init_pois, offset_eta=library_size, robust_se=True
    )

    assert_array_eq(glmstate.se ** 2, jnp.diag(white_cov))


def test_poisson_scoretest():
    offset = pd.read_csv("./example/data/spector_offset.tsv", sep="\t")
    R_res = pd.read_csv("./example/data/spector_scoretest_pois_Rres.tsv", sep="\t")
    jaxqtl_pois = GLM(family=Poisson(), maxiter=maxiter, stepsize=stepsize)
    init_pois = jaxqtl_pois.family.init_eta(y_arr)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))
    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )
    Z_GPA, pval_GPA = jaxqtl_pois.score_test_add_g(
        jnp.array(spector_data.exog["GPA"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
        jnp.log(jnp.array(offset)),
    )
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )
    Z_TUCE, pval_TUCE = jaxqtl_pois.score_test_add_g(
        jnp.array(spector_data.exog["TUCE"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
        jnp.log(jnp.array(offset)),
    )
    print(f"Add TUCE variable: pval={pval_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )
    Z_PSI, pval_PSI = jaxqtl_pois.score_test_add_g(
        jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
        jnp.log(jnp.array(offset)),
    )
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[1], pval_TUCE[1], pval_PSI[1]]).T[0]  # fix shape
    Z_vec = jnp.array([Z_GPA[1], Z_TUCE[1], Z_PSI[1]]).T[0]  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]))
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]))


def test_bin_scoretest():
    R_res = pd.read_csv("./example/data/spector_scoretest_bin_Rres.tsv", sep="\t")
    jaxqtl_bin = GLM(family=Binomial(), maxiter=maxiter, stepsize=stepsize)
    init_bin = jaxqtl_bin.family.init_eta(y_arr)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))
    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_GPA, pval_GPA = jaxqtl_bin.score_test_add_g(
        jnp.array(spector_data.exog["GPA"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
    )
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_TUCE, pval_TUCE = jaxqtl_bin.score_test_add_g(
        jnp.array(spector_data.exog["TUCE"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
    )
    print(f"Add TUCE variable: pval={Z_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_PSI, pval_PSI = jaxqtl_bin.score_test_add_g(
        jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis],
        X_covar,
        y_arr,
        mod_null,
    )
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[1], pval_TUCE[1], pval_PSI[1]]).T[0]  # fix shape
    Z_vec = jnp.array([Z_GPA[1], Z_TUCE[1], Z_PSI[1]]).T[0]  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]))
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]))


def test_resid_reg():
    """
    project out covariates first;
    results are wrong
    check ref:
    https://timothy-barry.github.io/posts/2020-07-07-generalized-linear-models/
    https://github.com/rgcgithub/regenie/blob/master/src/Step2_Models.cpp
    """
    test_resid_family = Binomial()  # Poisson reg result is closer

    X = spector_data.exog.copy()
    y = jnp.array(spector_data.endog)[:, jnp.newaxis]

    jaxqtl_pois = GLM(
        family=test_resid_family,
        maxiter=maxiter,
        stepsize=stepsize,
        solver=CholeskySolve(),
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    truth = jaxqtl_pois.fit(jnp.array(X), y, init=init_pois)

    covar = X.drop("PSI", axis=1)
    covar_X_arr = jnp.array(covar)

    glmstate_null = jaxqtl_pois.fit(covar_X_arr, y, init=init_pois)

    PSI = jnp.array(X["PSI"])
    w_X = jnp.array(glmstate_null.glm_wt) * covar_X_arr

    projection_covar = covar_X_arr @ jnpla.inv(w_X.T @ covar_X_arr) @ w_X.T  # nxn

    X["PSI_resid"] = PSI[:, jnp.newaxis] - projection_covar @ PSI[:, jnp.newaxis]

    glmstate = jaxqtl_pois.fit(
        jnp.array(X["PSI_resid"])[:, jnp.newaxis],
        y,
        offset_eta=glmstate_null.eta,
        init=init_pois,
    )

    print(f"betas: truth={truth.beta[-1]}, est={glmstate.beta}")
    print(f"SE: truth={truth.se[-1]}, est={glmstate.se}")
    print(f"Z: truth={truth.beta[-1]/truth.se[-1]}, est={glmstate.beta/glmstate.se}")

    # repeat with statsmodel
    mod_null = sm.GLM(spector_data.endog, covar, family=sm.families.Binomial()).fit()
    mod_null_eta = mod_null.get_prediction(covar, which="linear").predicted
    mod_null_mu = mod_null.get_prediction(covar, which="mean").predicted
    glm_wt = mod_null_mu * (1 - mod_null_mu)
    w_X = glm_wt[:, jnp.newaxis] * covar_X_arr

    projection_covar = covar_X_arr @ jnpla.inv(w_X.T @ covar_X_arr) @ w_X.T  # nxn

    X["PSI_resid"] = PSI[:, jnp.newaxis] - projection_covar @ PSI[:, jnp.newaxis]
    mod_G = sm.GLM(
        spector_data.endog,
        X["PSI_resid"],
        family=sm.families.Binomial(),
        offset=mod_null_eta,
    ).fit()

    print(f"statsmodel betas: est={mod_G.params[-1]}")
    print(f"statsmodel SE: est={mod_G.bse[-1]}")

    assert_array_eq(glmstate.beta, truth.beta[-1])
    assert_array_eq(glmstate.se, truth.se[-1])
