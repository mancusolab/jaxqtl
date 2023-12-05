import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (
    NegativeBinomial as smNB,
    Poisson as smPoisson,
)
from utils import assert_array_eq

import jax.numpy as jnp
from jax import config

from jaxqtl.families.distribution import Binomial, NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.solve import CGSolve, CholeskySolve, QRSolve
from jaxqtl.infer.utils import score_test_snp

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
        max_iter=maxiter,
        step_size=stepsize,
    )

    init_lm = jaxqtl_cho.family.init_eta(jnp.array(spector_data.endog)[:, jnp.newaxis])

    glm_state = jaxqtl_cho.fit(X_arr, y_arr, init=init_lm)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_binomial_cg():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_bin_cg = GLM(
        family=Binomial(),
        max_iter=maxiter,
        solver=CGSolve(),
        step_size=stepsize,
    )
    init_logistic = jaxqtl_bin_cg.family.init_eta(y_arr)
    glm_state = jaxqtl_bin_cg.fit(X_arr, y_arr, init=init_logistic)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


def test_binomial_cho():
    # test logistic regression
    mod = sm.Logit(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_bin_cho = GLM(
        family=Binomial(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_logistic = jaxqtl_bin_cho.family.init_eta(y_arr)
    glm_state = jaxqtl_bin_cho.fit(X_arr, y_arr, init=init_logistic)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-4)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-4)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-4)


def test_poisson_qr():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_qr = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=QRSolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_poisson_qr.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_qr.fit(X_arr, y_arr, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson_cho():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_cho = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_poisson_cho.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_cho.fit(X_arr, y_arr, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_poisson_cg():
    # test logistic regression
    mod = smPoisson(spector_data.endog, spector_data.exog)
    sm_state = mod.fit()

    jaxqtl_poisson_cg = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CGSolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_poisson_cg.family.init_eta(y_arr)
    glm_state = jaxqtl_poisson_cg.fit(X_arr, y_arr, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
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
        family=Poisson(), max_iter=maxiter, solver=CGSolve(), step_size=stepsize
    )
    init_pois = jaxqtl_poisson_cg.family.init_eta(y)
    glm_state = jaxqtl_poisson_cg.fit(X, y, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_1D_X():
    # test poisson regression
    mod = smPoisson(spector_data.endog, spector_data.exog["PSI"])
    sm_state = mod.fit()

    X_arr = jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis]
    y_arr = jnp.array(spector_data.endog)[:, jnp.newaxis]

    jaxqtl_pois = GLM(family=Poisson(), max_iter=maxiter, step_size=stepsize)
    init_pois = jaxqtl_pois.family.init_eta(y_arr)
    glm_state = jaxqtl_pois.fit(X_arr, y_arr, init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)


def test_robust_SE():
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

    jaxqtl_pois = GLM(family=Poisson(), max_iter=100, solver=CholeskySolve())
    init_pois = jaxqtl_pois.family.init_eta(y)

    glmstate = jaxqtl_pois.fit(
        M, y, init=init_pois, offset_eta=library_size, robust_se=True
    )

    assert_array_eq(glmstate.se**2, jnp.diag(white_cov))


def test_NB():
    # test negative binomial
    # Note: closer to R values
    R_res = pd.read_csv("./example/data/ENSG00000178607_rs74787440_Rres.tsv", sep="\t")

    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])[:, jnp.newaxis]
    library_size = jnp.array(dat["log_offset"])[:, jnp.newaxis]

    sm_mod = smNB(
        np.array(y),
        np.array(M),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit(maxiter=100)
    sm_alpha = sm_mod.params[-1]  # alpha estimate

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(M, y, init=init_pois, offset_eta=library_size)

    nb_fam = NegativeBinomial()

    alpha_init = len(y) / jnp.sum(
        (y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2
    )
    alpha_n = nb_fam.estimate_dispersion(M, y, glm_state_pois.eta, alpha=alpha_init)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    glm_state = jaxqtl_nb.fit(
        M,
        y,
        init=glm_state_pois.eta,
        offset_eta=library_size,
        alpha_init=alpha_n.squeeze(),
    )

    assert_array_eq(glm_state.alpha, sm_alpha, rtol=1e-2)
    assert_array_eq(glm_state.beta / glm_state.se, R_res["Z"], rtol=1e-2)
    # assert_array_eq(glm_state.beta, sm_mod.params[:-1])
    # assert_array_eq(glm_state.se, sm_mod.bse[:-1])


def test_NB_robust():
    """
    Inversion of information matrix might cause the discrepancy
    """
    # test negative binomial
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])[:, jnp.newaxis]
    library_size = jnp.array(dat["log_offset"])[:, jnp.newaxis]

    sm_mod = smNB(
        np.array(y),
        np.array(M),
        offset=np.array(library_size).reshape((len(library_size),)),
    ).fit(maxiter=100)
    white_cov = statsmodels.stats.sandwich_covariance.cov_white_simple(
        sm_mod, use_correction=False
    )

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(M, y, init=init_pois, offset_eta=library_size)

    nb_fam = NegativeBinomial()

    alpha_init = len(y) / jnp.sum(
        (y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2
    )
    alpha_n = nb_fam.estimate_dispersion(M, y, glm_state_pois.eta, alpha=alpha_init)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )

    glm_state_robust = jaxqtl_nb.fit(
        M,
        y,
        init=glm_state_pois.eta,
        offset_eta=library_size,
        alpha_init=alpha_n.squeeze(),
        robust_se=True,
    )

    assert_array_eq(glm_state_robust.se**2, jnp.diag(white_cov)[:-1], rtol=1e-3)


def test_poisson_scoretest():
    offset = pd.read_csv("./example/data/spector_offset.tsv", sep="\t")
    R_res = pd.read_csv("./example/data/spector_scoretest_pois_Rres.tsv", sep="\t")

    jaxqtl_pois = GLM(family=Poisson(), max_iter=maxiter, step_size=stepsize)
    init_pois = jaxqtl_pois.family.init_eta(y_arr)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))

    # statsmodel result
    sm_glm = sm.GLM(
        np.array(y_arr),
        np.array(X_covar),
        family=sm.families.Poisson(),
        offset=np.log(np.array(offset).squeeze()),
    )
    sm_res = sm_glm.fit()

    # print(sm_res.summary())
    chi2, sm_p, _ = sm_res.score_test(
        params_constrained=sm_res.params, exog_extra=spector_data.exog["GPA"]
    )

    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )
    Z_GPA, pval_GPA, _, _ = score_test_snp(
        jnp.array(spector_data.exog["GPA"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")
    assert_array_eq(
        pval_GPA, jnp.array(sm_p), rtol=1e-3
    )  # check result with statsmodel

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )

    Z_TUCE, pval_TUCE, _, _ = score_test_snp(
        jnp.array(spector_data.exog["TUCE"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add TUCE variable: pval={pval_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    mod_null = jaxqtl_pois.fit(
        X_covar, y_arr, init=init_pois, offset_eta=jnp.log(jnp.array(offset))
    )

    Z_PSI, pval_PSI, _, _ = score_test_snp(
        jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[0], pval_TUCE[0], pval_PSI[0]])  # fix shape
    Z_vec = jnp.array([Z_GPA[0], Z_TUCE[0], Z_PSI[0]])  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]), rtol=1e-3)
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]), rtol=1e-3)


def test_bin_scoretest():
    R_res = pd.read_csv("./example/data/spector_scoretest_bin_Rres.tsv", sep="\t")
    jaxqtl_bin = GLM(family=Binomial(), max_iter=maxiter, step_size=stepsize)
    init_bin = jaxqtl_bin.family.init_eta(y_arr)

    X_covar = jnp.array(spector_data.exog.drop("GPA", axis=1))

    sm_glm = sm.GLM(np.array(y_arr), np.array(X_covar), family=sm.families.Binomial())
    sm_res = sm_glm.fit()

    # print(sm_res.summary())
    chi2, sm_p, _ = sm_res.score_test(
        params_constrained=sm_res.params, exog_extra=spector_data.exog["GPA"]
    )

    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_GPA, pval_GPA, _, _ = score_test_snp(
        jnp.array(spector_data.exog["GPA"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add GPA variable: pval={pval_GPA}, Z={Z_GPA}")
    assert_array_eq(
        pval_GPA, jnp.array(sm_p), rtol=1e-3
    )  # check result with statsmodel

    X_covar = jnp.array(spector_data.exog.drop("TUCE", axis=1))
    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_TUCE, pval_TUCE, _, _ = score_test_snp(
        jnp.array(spector_data.exog["TUCE"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add TUCE variable: pval={Z_TUCE}, Z={Z_TUCE}")

    X_covar = jnp.array(spector_data.exog.drop("PSI", axis=1))
    mod_null = jaxqtl_bin.fit(X_covar, y_arr, init=init_bin)
    Z_PSI, pval_PSI, _, _ = score_test_snp(
        jnp.array(spector_data.exog["PSI"])[:, jnp.newaxis], X_covar, mod_null
    )
    print(f"Add PSI variable: pval={pval_PSI}, Z={Z_PSI}")

    pval_vec = jnp.array([pval_GPA[0], pval_TUCE[0], pval_PSI[0]])  # fix shape
    Z_vec = jnp.array([Z_GPA[0], Z_TUCE[0], Z_PSI[0]])  # fix shape
    assert_array_eq(pval_vec, jnp.array(R_res["pval"]), rtol=1e-3)
    assert_array_eq(Z_vec, jnp.array(R_res["Z"]), rtol=1e-3)


def test_nb_scoretest():
    Rres = pd.read_csv(
        "./example/data/ENSG00000178607_rs74787440.nb.scoretest.tsv", sep="\t"
    )
    dat = pd.read_csv("./example/data/ENSG00000178607_rs74787440.gz", sep="\t")
    M = jnp.array(dat.iloc[:, 0:12])
    y = jnp.array(dat["y"])[:, jnp.newaxis]
    library_size = jnp.array(dat["log_offset"])[:, jnp.newaxis]

    # print(sm_res.summary())
    M_cov = M[:, 0:-1]

    jaxqtl_pois = GLM(
        family=Poisson(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    init_pois = jaxqtl_pois.family.init_eta(y)
    glm_state_pois = jaxqtl_pois.fit(M_cov, y, init=init_pois, offset_eta=library_size)

    nb_fam = NegativeBinomial()
    alpha_init = len(y) / jnp.sum(
        (y / nb_fam.glink.inverse(glm_state_pois.eta) - 1) ** 2
    )
    alpha_n = nb_fam.estimate_dispersion(M_cov, y, glm_state_pois.eta, alpha=alpha_init)

    jaxqtl_nb = GLM(
        family=NegativeBinomial(),
        max_iter=maxiter,
        solver=CholeskySolve(),
        step_size=stepsize,
    )
    glm_state = jaxqtl_nb.fit(
        M_cov, y, init=glm_state_pois.eta, offset_eta=library_size, alpha_init=alpha_n
    )

    Z, pval, _, _ = score_test_snp(M[:, -1][:, jnp.newaxis], M_cov, glm_state)

    assert_array_eq(Z, Rres["Z"])
