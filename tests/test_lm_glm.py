import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw

import jax.numpy as jnp

from jax import config

from jaxqtl.distribution._expfam import Binomial, Gaussian, Poisson
from jaxqtl.distribution._links import IdentityLink, LogitLink, LogLink, PowerLink
from jaxqtl.infer._glm import GeneralizedLinearModel, LinearModel
from jaxqtl.infer._solve import CGSolve, CholeskySolve, QRSolve
from jaxqtl.infer._stderr import HuberError


config.update("jax_enable_x64", True)


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve(), CGSolve()))
def test_linear_model_matches_statsmodels(solver):
    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X, prepend=True)
    beta = rng.normal(size=(p + 1,))
    y = X @ beta + rng.normal(size=(n,))

    sm_state = sm.OLS(y, X).fit()

    model = LinearModel(solver=solver)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    np.testing.assert_allclose(np.asarray(glm_state.beta), sm_state.params, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(glm_state.p), sm_state.pvalues, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
@pytest.mark.parametrize(
    ("link", "sm_link"),
    [
        (LogLink(), sm.families.links.Log()),
    ],
)
def test_glm_poisson_matches_statsmodels_glm_link(solver, link, sm_link):
    spector = sm.datasets.spector.load()
    spector.exog = sm.add_constant(spector.exog, prepend=True)

    y = np.asarray(spector.endog)
    X = np.asarray(spector.exog)

    sm_state = sm.GLM(y, X, family=sm.families.Poisson(link=sm_link)).fit(disp=0)

    model = GeneralizedLinearModel(family=Poisson(glink=link), solver=solver, max_iter=200, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    np.testing.assert_allclose(np.asarray(glm_state.beta), sm_state.params, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(glm_state.p), sm_state.pvalues, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
def test_glm_poisson_identity_link_runs(solver):
    rng = np.random.default_rng(0)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X, prepend=True)
    beta = rng.normal(size=(p + 1,)) * 0.1
    beta[0] = 2.0  # keep eta positive
    eta = X @ beta
    y = rng.poisson(lam=np.clip(eta, 1e-3, np.inf))

    model = GeneralizedLinearModel(family=Poisson(glink=IdentityLink()), solver=solver, max_iter=300, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    assert jnp.all(jnp.isfinite(glm_state.beta))
    assert jnp.all(jnp.isfinite(glm_state.se))
    assert jnp.all(jnp.isfinite(glm_state.p))


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
@pytest.mark.parametrize(
    ("link", "sm_link"),
    [
        (LogitLink(), sm.families.links.Logit()),
    ],
)
def test_glm_binomial_matches_statsmodels_glm_link(solver, link, sm_link):
    spector = sm.datasets.spector.load()
    spector.exog = sm.add_constant(spector.exog, prepend=True)

    y = np.asarray(spector.endog)
    X = np.asarray(spector.exog)

    sm_state = sm.GLM(y, X, family=sm.families.Binomial(link=sm_link)).fit(disp=0)

    model = GeneralizedLinearModel(family=Binomial(glink=link), solver=solver, max_iter=200, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    np.testing.assert_allclose(np.asarray(glm_state.beta), sm_state.params, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(glm_state.p), sm_state.pvalues, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
@pytest.mark.parametrize("link", (LogLink(), IdentityLink()))
def test_glm_binomial_noncanonical_links_run(solver, link):
    rng = np.random.default_rng(0)
    n, p = 400, 3
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X, prepend=True)
    beta = rng.normal(size=(p + 1,)) * 0.2

    if isinstance(link, LogLink):
        # Ensure eta <= 0 so mu = exp(eta) in (0, 1].
        beta[0] = -1.0
        eta = X @ beta
        mu = np.clip(np.exp(np.minimum(eta, 0.0)), 1e-6, 1 - 1e-6)
    else:
        # Identity link: eta is the mean; keep within (0,1).
        beta[0] = 0.5
        eta = X @ beta
        mu = np.clip(eta, 1e-3, 1 - 1e-3)

    y = rng.binomial(n=1, p=mu)

    model = GeneralizedLinearModel(family=Binomial(glink=link), solver=solver, max_iter=300, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    assert jnp.all(jnp.isfinite(glm_state.beta))
    assert jnp.all(jnp.isfinite(glm_state.se))
    assert jnp.all((glm_state.p >= 0) & (glm_state.p <= 1))


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
@pytest.mark.parametrize(
    ("link", "sm_link"),
    [
        (IdentityLink(), sm.families.links.Identity()),
        (LogLink(), sm.families.links.Log()),
    ],
)
def test_glm_gaussian_matches_statsmodels_glm_link(solver, link, sm_link):
    rng = np.random.default_rng(0)
    n, p = 250, 3
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X, prepend=True)
    beta = rng.normal(size=(p + 1,))

    if isinstance(link, LogLink):
        eta = X @ beta
        mu = np.exp(eta)
        y = np.clip(mu + rng.normal(scale=0.1, size=(n,)), 1e-6, np.inf)
    else:
        y = X @ beta + rng.normal(size=(n,))

    sm_state = sm.GLM(y, X, family=sm.families.Gaussian(link=sm_link)).fit()

    model = GeneralizedLinearModel(family=Gaussian(glink=link), solver=solver, max_iter=200, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    np.testing.assert_allclose(np.asarray(glm_state.beta), sm_state.params, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(glm_state.p), sm_state.pvalues, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("solver", (CholeskySolve(), QRSolve()))
@pytest.mark.parametrize("link", (PowerLink(0.5), PowerLink(2.0)))
def test_glm_gaussian_power_link_runs(solver, link):
    rng = np.random.default_rng(0)
    n = 300
    X = np.ones((n, 1))
    mu = np.clip(np.exp(0.1 * rng.normal(size=(n,))) + 1.0, 1e-3, np.inf)
    y = np.clip(mu + rng.normal(scale=0.05, size=(n,)), 1e-3, np.inf)

    model = GeneralizedLinearModel(family=Gaussian(glink=link), solver=solver, max_iter=200, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y))

    assert jnp.all(jnp.isfinite(glm_state.beta))
    assert jnp.all(jnp.isfinite(glm_state.se))
    assert jnp.all(jnp.isfinite(glm_state.p))


@pytest.mark.parametrize("family", (Gaussian(), Poisson()))
def test_huber_error_matches_statsmodels_white_cov(family):
    rng = np.random.default_rng(0)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X, prepend=True)
    beta = rng.normal(size=(p + 1,))
    offset = rng.normal(size=(n,)) * 0.1

    if isinstance(family, Gaussian):
        mu = X @ beta + offset
        y = mu + rng.normal(size=(n,))
        sm_model = sm.GLM(y, X, family=sm.families.Gaussian(), offset=offset).fit()
    else:
        eta = X @ beta + offset
        mu = np.exp(eta)
        y = rng.poisson(mu)
        sm_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

    white_cov = sw.cov_white_simple(sm_model, use_correction=False)

    model = GeneralizedLinearModel(family=family, solver=CholeskySolve(), max_iter=200, step_size=1.0)
    glm_state = model.fit(jnp.asarray(X), jnp.asarray(y), offset=jnp.asarray(offset), std_err=HuberError())

    np.testing.assert_allclose(np.asarray(glm_state.se**2), np.diag(white_cov), rtol=1e-2, atol=1e-2)
