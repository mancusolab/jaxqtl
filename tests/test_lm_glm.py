# pattern: Functional Core

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw

import equinox as eqx
import jax.numpy as jnp

from jax import config

from jaxqtl.distribution._expfam import Binomial, Gaussian, NegativeBinomial, Poisson
from jaxqtl.distribution._links import IdentityLink, LogitLink, LogLink, PowerLink
from jaxqtl.infer._glm import _NBInit, GeneralizedLinearModel, LinearModel
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


@pytest.mark.parametrize("offset_kind", ("scalar", "vector"))
def test_linear_model_offset_state_and_huber_covariance_match_statsmodels(offset_kind):
    rng = np.random.default_rng(15)
    n, p = 120, 3
    X = sm.add_constant(rng.normal(size=(n, p)), prepend=True)
    beta = rng.normal(size=p + 1)
    offset = 0.4 if offset_kind == "scalar" else rng.normal(scale=0.2, size=n)
    offset_vector = np.broadcast_to(offset, (n,))
    error_scale = 0.5 + 0.3 * np.abs(X[:, 1])
    y = X @ beta + offset_vector + rng.normal(scale=error_scale)

    sm_state = sm.OLS(y - offset_vector, X).fit()
    expected_covariance = sw.cov_white_simple(sm_state, use_correction=False)
    model_state = LinearModel().fit(X, y, offset=offset, std_err=HuberError())

    expected_eta = X @ sm_state.params + offset_vector
    expected_residual = y - expected_eta
    expected_dispersion = np.sum(expected_residual**2) / (n - X.shape[1])
    np.testing.assert_allclose(np.asarray(model_state.beta), sm_state.params, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(model_state.eta), expected_eta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(model_state.mu), expected_eta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(model_state.resid), expected_residual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(model_state.disp), expected_dispersion, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(model_state.resid_covar), expected_covariance, rtol=1e-5, atol=1e-5)


def test_linear_model_residual_df_override_controls_inference():
    rng = np.random.default_rng(12)
    n = 40
    g = rng.normal(size=(n, 1))
    y = 0.7 * g[:, 0] + rng.normal(size=n)

    model = LinearModel()
    default = model.fit(g, y)
    adjusted = model.fit(g, y, df_resid=n - 5)

    expected_dispersion = np.sum(np.asarray(adjusted.resid) ** 2) / (n - 5)
    np.testing.assert_allclose(np.asarray(adjusted.disp), expected_dispersion)
    assert adjusted.se[0] > default.se[0]
    assert adjusted.p[0] > default.p[0]


@pytest.mark.parametrize("df_resid", (0, -1))
def test_linear_model_rejects_nonpositive_residual_df(df_resid):
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        LinearModel().fit(jnp.ones((3, 1)), jnp.arange(3.0), df_resid=df_resid)


def test_linear_model_rejects_nonidentity_link():
    with pytest.raises(ValueError, match="IdentityLink"):
        LinearModel(family=Gaussian(glink=LogLink()))


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


@pytest.mark.parametrize("offset_kind", ("scalar", "vector"))
def test_negative_binomial_initializer_returns_predictor_without_offset(offset_kind):
    rng = np.random.default_rng(16)
    n = 120
    X = sm.add_constant(rng.normal(size=(n, 2)), prepend=True)
    beta = np.array([0.4, 0.2, -0.15])
    offset = 0.3 if offset_kind == "scalar" else np.linspace(-0.2, 0.4, n)
    complete_eta = X @ beta + offset
    mu = np.exp(complete_eta)
    alpha = 0.4
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu))
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    family = NegativeBinomial()
    solver = CholeskySolve()
    max_iter = 100
    tol = 1e-4
    step_size = 1.0

    initializer = _NBInit(family, solver)
    initializer_eta, initializer_dispersion = initializer.init(
        X, y, offset, max_iter=max_iter, tol=tol, step_size=step_size
    )
    jitted_eta, jitted_dispersion = eqx.filter_jit(initializer.init)(
        X, y, offset, max_iter=max_iter, tol=tol, step_size=step_size
    )
    poisson_state = GeneralizedLinearModel(
        family=Poisson(), solver=solver, max_iter=max_iter, tol=tol, step_size=step_size
    ).fit(X, y, offset)

    poisson_eta = poisson_state.eta
    moment_inverse_dispersion = n / jnp.sum((y / family.glink.inverse(poisson_eta) - 1) ** 2)
    expected_dispersion = family.estimate_dispersion(
        X, y, poisson_eta, disp=1.0 / moment_inverse_dispersion, max_iter=max_iter
    )
    expected_dispersion = jnp.nan_to_num(expected_dispersion, nan=0.1)

    np.testing.assert_allclose(initializer_dispersion, expected_dispersion)
    np.testing.assert_allclose(initializer_eta + offset, poisson_eta)
    np.testing.assert_allclose(jitted_eta, initializer_eta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jitted_dispersion, initializer_dispersion, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jitted_eta + offset, poisson_eta)


def test_negative_binomial_fit_with_offset_matches_jit():
    rng = np.random.default_rng(18)
    n = 160
    X = sm.add_constant(rng.normal(size=(n, 2)), prepend=True)
    beta = np.array([0.35, 0.2, -0.15])
    offset = np.linspace(-0.35, 0.3, n) + 0.1 * np.sin(np.linspace(0.0, 3.0 * np.pi, n))
    eta = X @ beta + offset
    mu = np.exp(eta)
    alpha = 0.35
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu))
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    model = GeneralizedLinearModel(family=NegativeBinomial(), max_iter=200, tol=1e-4)
    eager = model.fit(X, y, offset)
    jitted = eqx.filter_jit(model.fit)(X, y, offset)

    numerical_fields = (
        "beta",
        "se",
        "z",
        "p",
        "eta",
        "mu",
        "glm_wt",
        "link_prime",
        "resid_covar",
        "resid",
        "disp",
    )
    for state in (eager, jitted):
        assert bool(np.asarray(state.converged))
        assert np.isfinite(np.asarray(state.disp))
        assert np.isfinite(np.asarray(state.num_iters))
        assert np.asarray(state.disp) > 0.0
        for field in ("beta", "se", "z", "p"):
            assert getattr(state, field).shape == (X.shape[1],)
        for field in ("eta", "mu", "glm_wt", "link_prime", "resid"):
            assert getattr(state, field).shape == (n,)
        assert state.resid_covar.shape == (X.shape[1], X.shape[1])
        assert np.asarray(state.disp).shape == ()
        assert np.asarray(state.num_iters).shape == ()
        assert np.asarray(state.converged).shape == ()
        for field in numerical_fields:
            assert np.all(np.isfinite(np.asarray(getattr(state, field))))

        np.testing.assert_allclose(np.asarray(state.eta), np.asarray(X @ state.beta + offset))
        np.testing.assert_allclose(np.asarray(state.mu), np.exp(np.asarray(state.eta)))
        np.testing.assert_allclose(
            np.asarray(state.resid),
            (np.asarray(y) - np.asarray(state.mu)) / np.asarray(state.mu),
        )

    for field in numerical_fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, field)),
            np.asarray(getattr(eager, field)),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_array_equal(np.asarray(jitted.num_iters), np.asarray(eager.num_iters))
    np.testing.assert_array_equal(np.asarray(jitted.converged), np.asarray(eager.converged))


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
