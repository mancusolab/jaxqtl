import numpy as np
import pandas as pd
import pytest
import statsmodels
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import Binomial, Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM, LinearModel
from jaxqtl.infer.solve import CGSolve, CholeskySolve, QRSolve
from jaxqtl.infer.stderr import HuberError
from jaxqtl.infer.utils import ScoreTest


config.update("jax_enable_x64", True)

# io toy example data from statsmodels to compare
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)  # X

y_arr = jnp.array(spector_data.endog)
X_arr = jnp.array(spector_data.exog)

maxiter = 100
stepsize = 1.0

jaxqtl_lm = GLM(family=Gaussian(), max_iter=maxiter, step_size=stepsize)
X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))


def score_test_snp(a, b, c):
    pass


@pytest.mark.parametrize("solver", (CGSolve(), CholeskySolve(), QRSolve()))
def test_linear_regression(solver):
    # test linear regression function
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_cho = LinearModel(solver=solver)
    glm_state = jaxqtl_cho.fit(X_arr, y_arr)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


@pytest.mark.parametrize("solver", (CGSolve(), CholeskySolve(), QRSolve()))
def test_binomial(solver):
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_bin_cg = GLM(
        family=Binomial(),
        max_iter=maxiter,
        solver=solver,
        step_size=stepsize,
    )
    glm_state = jaxqtl_bin_cg.fit(X_arr, y_arr)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


@pytest.mark.parametrize("solver", (CGSolve(), CholeskySolve(), QRSolve()))
def test_poisson(solver):
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_qr = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=solver,
        step_size=stepsize,
    )
    glm_state = jaxqtl_poisson_qr.fit(X_arr, y_arr)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_CGsolve_realdata():
    """
    # AssertionError: get diff result
    """
    dat = jnp.array(pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t"))
    y = dat[:, -2]
    X = dat[:, 0:-2]

    sm_state = smPoisson(np.array(y), np.array(X)).fit()

    jaxqtl_poisson_cg = GLM(family=Poisson(), max_iter=maxiter, solver=CGSolve(), step_size=stepsize)
    glm_state = jaxqtl_poisson_cg.fit(X, y)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-3)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-3)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-3)


def test_1D_X():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog["PSI"])
    sm_state = mod.fit()

    X_arr = jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis]
    y_arr = jnp.array(spector_data.endog)

    jaxqtl_pois = GLM(family=Poisson(), max_iter=maxiter, step_size=stepsize)
    glm_state = jaxqtl_pois.fit(X_arr, y_arr)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_robust_SE_Poisson():
    """
    Compare sandwitch estimator from stats model to jaxqtl
    """
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])
    library_size = jnp.array(dat["log_offset"])

    # test poisson robust
    sm_mod = sm.GLM(
        np.array(y),
        np.array(M),
        family=sm.families.Poisson(),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit()
    white_cov = statsmodels.stats.sandwich_covariance.cov_white_simple(sm_mod, use_correction=False)

    jaxqtl_pois = GLM(family=Poisson(), solver=CholeskySolve())

    glmstate = jaxqtl_pois.fit(
        M,
        y,
        offset=library_size,
        std_err=HuberError(),
    )

    assert_array_eq(glmstate.se**2, jnp.diag(white_cov))


def test_robust_SE_lm():
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])

    # test lm robust
    sm_mod = sm.GLM(np.array(y), np.array(M), family=sm.families.Gaussian()).fit()
    white_cov = statsmodels.stats.sandwich_covariance.cov_white_simple(sm_mod, use_correction=False)

    jaxqtl_lm = GLM(family=Gaussian(), solver=CholeskySolve())

    glmstate = jaxqtl_lm.fit(
        M,
        y,
        std_err=HuberError(),
    )

    assert_array_eq(glmstate.se**2, jnp.diag(white_cov))


def test_NB():
    # test negative binomial
    # Note: closer to R values
    R_res = pd.read_csv("./example/data/ENSG00000178607_rs74787440_Rres.tsv", sep="\t")

    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])
    library_size = jnp.array(dat["log_offset"])

    sm_mod = smNB(
        np.array(y),
        np.array(M),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit(maxiter=100)
    sm_alpha = sm_mod.params[-1]  # alpha estimate

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        solver=CholeskySolve(),
        max_iter=maxiter,
        step_size=stepsize,
    )
    glm_state = jaxqtl_nb.fit(
        M,
        y,
        offset=library_size,
    )

    assert_array_eq(glm_state.disp, sm_alpha, rtol=1e-2)
    assert_array_eq(glm_state.beta / glm_state.se, R_res["Z"], rtol=1e-2)


def test_NB_robust():
    """
    Inversion of information matrix might cause the discrepancy
    """
    # test negative binomial
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])
    library_size = jnp.array(dat["log_offset"])

    sm_mod = smNB(
        np.array(y),
        np.array(M),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit(maxiter=100)
    white_cov = statsmodels.stats.sandwich_covariance.cov_white_simple(sm_mod, use_correction=False)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        solver=CholeskySolve(),
        max_iter=maxiter,
        step_size=stepsize,
    )

    glm_state_robust = jaxqtl_nb.fit(M, y, offset=library_size, std_err=HuberError())

    assert_array_eq(glm_state_robust.se**2, jnp.diag(white_cov)[:-1], rtol=1e-3)


def test_lm_scoretest():
    jaxqtl_lm = LinearModel(family=Gaussian())

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))

    # statsmodel result
    sm_glm = sm.GLM(np.array(y_arr), np.array(X_covar), family=sm.families.Gaussian())
    sm_res = sm_glm.fit()
    chi2, sm_p, _ = sm_res.score_test(params_constrained=sm_res.params, exog_extra=spector_data.exog["GPA"])

    def _get_var(name):
        return jnp.asarray(spector_data.exog[name])[:, None]

    score_test = ScoreTest(jaxqtl_lm)
    _, _, pval_GPA, Z_GPA, _, _, _ = score_test(X_covar, _get_var("GPA"), y_arr, 0)
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")
    assert_array_eq(pval_GPA, jnp.array(sm_p), rtol=1e-3)  # check result with statsmodel

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    _, _, pval_TUCE, Z_TUCE, _, _, _ = score_test(X_covar, _get_var("TUCE"), y_arr, 0)
    print(f"Add TUCE variable: pval={pval_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))

    _, _, pval_PSI, Z_PSI, _, _, _ = score_test(X_covar, _get_var("PSI"), y_arr, 0)
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")


def test_poisson_scoretest():
    offset = pd.read_csv("./example/data/spector_offset.tsv", sep="\t")
    R_res = pd.read_csv("./example/data/spector_scoretest_pois_Rres.tsv", sep="\t")

    jaxqtl_pois = GLM(family=Poisson(), max_iter=maxiter, step_size=stepsize)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))
    offset = np.log(np.array(offset).squeeze())

    # statsmodel result
    sm_glm = sm.GLM(
        np.array(y_arr),
        np.array(X_covar),
        family=sm.families.Poisson(),
        offset=np.log(np.array(offset).squeeze()),
    )
    sm_res = sm_glm.fit()

    # print(sm_res.summary())
    chi2, sm_p, _ = sm_res.score_test(params_constrained=sm_res.params, exog_extra=spector_data.exog["GPA"])

    def _get_var(name):
        return jnp.asarray(spector_data.exog[name])[:, None]

    offset = jnp.asarray(offset)
    score_test = ScoreTest(jaxqtl_pois)
    _, _, pval_GPA, Z_GPA, _, _, _ = score_test(X_covar, _get_var("GPA"), y_arr, offset)
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")
    assert_array_eq(pval_GPA, jnp.array(sm_p), rtol=1e-3)  # check result with statsmodel

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    _, _, pval_TUCE, Z_TUCE, _, _, _ = score_test(X_covar, _get_var("TUCE"), y_arr, offset)
    print(f"Add TUCE variable: pval={pval_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    _, _, pval_PSI, Z_PSI, _, _, _ = score_test(X_covar, _get_var("PSI"), y_arr, offset)
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[0], pval_TUCE[0], pval_PSI[0]])  # fix shape
    Z_vec = jnp.array([Z_GPA[0], Z_TUCE[0], Z_PSI[0]])  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]), rtol=1e-3)
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]), rtol=1e-3)


def test_bin_scoretest():
    R_res = pd.read_csv("./example/data/spector_scoretest_bin_Rres.tsv", sep="\t")
    jaxqtl_bin = GLM(family=Binomial(), max_iter=maxiter, step_size=stepsize)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))

    sm_glm = sm.GLM(np.array(y_arr), np.array(X_covar), family=sm.families.Binomial())
    sm_res = sm_glm.fit()

    # print(sm_res.summary())
    chi2, sm_p, _ = sm_res.score_test(params_constrained=sm_res.params, exog_extra=spector_data.exog["GPA"])

    def _get_var(name):
        return jnp.asarray(spector_data.exog[name])[:, None]

    score_test = ScoreTest(jaxqtl_bin)
    _, _, pval_GPA, Z_GPA, _, _, _ = score_test(X_covar, _get_var("GPA"), y_arr, 0)
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")
    assert_array_eq(pval_GPA, jnp.array(sm_p), rtol=1e-3)  # check result with statsmodel

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    _, _, pval_TUCE, Z_TUCE, _, _, _ = score_test(X_covar, _get_var("TUCE"), y_arr, 0)
    print(f"Add TUCE variable: pval={Z_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    _, _, pval_PSI, Z_PSI, _, _, _ = score_test(X_covar, _get_var("PSI"), y_arr, 0)
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[0], pval_TUCE[0], pval_PSI[0]])  # fix shape
    Z_vec = jnp.array([Z_GPA[0], Z_TUCE[0], Z_PSI[0]])  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]), rtol=1e-3)
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]), rtol=1e-3)


def test_nb_scoretest():
    # note: this is different b/c glm.scoretest() in R computes residuals like lm
    Rres = pd.read_csv("./example/data/ENSG00000178607_rs74787440.nb.scoretest.tsv", sep="\t")
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])
    library_size = jnp.array(dat["log_offset"])

    # print(sm_res.summary())
    M_cov = M[:, 0:-1]

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    score_test = ScoreTest(jaxqtl_nb)

    _, _, pval, Z, _, _, _ = score_test(M_cov, M[:, -1][:, jnp.newaxis], y, library_size)
    assert_array_eq(Z, Rres["Z"])
