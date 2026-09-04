# pattern: Functional Core

import pytest

import jax
import jax.numpy as jnp
import jax.random as rdm

from jax import config
from jax.scipy.special import gammaln, xlogy
from jax.scipy.stats import nbinom

from jaxqtl.distribution._expfam import (
    _nb2_centered_lgamma_ratio,
    _nb2_exprel,
    _nb2_log_alpha_derivatives,
    _nb2_mean_terms,
    _nb2_riemannian_direction,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from jaxqtl.distribution._links import IdentityLink, InverseLink, LogitLink, LogLink, NBLink, PowerLink


config.update("jax_enable_x64", True)


def test_poisson_negloglikelihood_accepts_fractional_response():
    y = jnp.asarray([0.0, 1.5, 3.25])
    mu = jnp.asarray([1.0, 2.0, 4.0])

    actual = Poisson().negloglikelihood(jnp.empty((y.size, 0)), y, jnp.log(mu), 1.0)
    expected = jnp.sum(mu - xlogy(y, mu) + gammaln(y + 1.0))

    assert jnp.isfinite(actual)
    assert float(actual) == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("link", "mu"),
    [
        (IdentityLink(), jnp.linspace(-2.0, 2.0, 101)),
        (LogLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (InverseLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (LogitLink(), jnp.linspace(0.05, 0.95, 101)),
        (PowerLink(0.5), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (PowerLink(2.0), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (NBLink(0.3), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
    ],
)
def test_link_roundtrip_inverse(link, mu):
    eta = link(mu)
    mu_back = link.inverse(eta)
    assert jnp.all(jnp.isfinite(eta))
    assert jnp.all(jnp.isfinite(mu_back))
    assert jnp.allclose(mu_back, mu, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("link", "eta"),
    [
        (IdentityLink(), jnp.linspace(0.1, 2.0, 21)),
        (InverseLink(), jnp.linspace(0.1, 2.0, 21)),
        (LogitLink(), jnp.linspace(-2.0, 2.0, 21)),
        (PowerLink(0.5), jnp.linspace(0.1, 2.0, 21)),
        (NBLink(0.3), jnp.linspace(-2.0, -0.1, 21)),
    ],
)
def test_link_log_inverse_matches_log_of_inverse(link, eta):
    expected = jnp.log(link.inverse(eta))

    actual = link.log_inverse(eta)

    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_log_link_log_inverse_is_stable_for_extreme_predictors():
    eta = jnp.asarray([-1000.0, 1000.0])

    actual = jax.jit(LogLink().log_inverse)(eta)
    gradient = jax.grad(lambda value: LogLink().log_inverse(value))(1000.0)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.array_equal(actual, eta)
    assert gradient == pytest.approx(1.0)


def test_negative_binomial_negloglikelihood_is_finite_for_extreme_log_predictors():
    y = jnp.asarray([0.0, 2.0])
    eta = jnp.asarray([-1000.0, 1000.0])

    actual = NegativeBinomial().negloglikelihood(jnp.empty((y.size, 0)), y, eta, 0.5)

    assert jnp.isfinite(actual)


def test_negative_binomial_negloglikelihood_matches_jax_in_moderate_regime():
    y = jnp.asarray([0.0, 1.0, 3.0, 20.0])
    mu = jnp.asarray([0.2, 1.5, 4.0, 25.0])
    alpha = 0.4
    size = 1.0 / alpha

    actual = NegativeBinomial().negloglikelihood(jnp.empty((y.size, 0)), y, jnp.log(mu), alpha)
    expected = -jnp.sum(nbinom.logpmf(y, size, size / (size + mu)))

    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_negative_binomial_poisson_limit_is_infinite_not_nan_for_extreme_log_predictor():
    y = jnp.asarray([0.0, 2.0])
    eta = jnp.asarray([-1000.0, 1000.0])

    actual = NegativeBinomial().negloglikelihood(jnp.empty((y.size, 0)), y, eta, 0.0)

    assert jnp.isposinf(actual)


def test_negative_binomial_poisson_limit_has_finite_first_and_second_derivatives():
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    mu = jnp.asarray([0.1, 1.0, 10.0, 100.0])
    eta = jnp.log(mu)
    X = jnp.empty((y.size, 0))
    family = NegativeBinomial()

    def objective(alpha):
        return family.negloglikelihood(X, y, eta, alpha)

    actual = objective(0.0)
    gradient = jax.grad(objective)(0.0)
    hessian = jax.grad(jax.grad(objective))(0.0)
    expected = Poisson().negloglikelihood(X, y, eta, 1.0)
    expected_gradient = jnp.sum(y * mu - mu**2 / 2.0 - y * (y - 1.0) / 2.0)
    expected_hessian = jnp.sum(y * (y - 1.0) * (2.0 * y - 1.0) / 6.0 - y * mu**2 + 2.0 * mu**3 / 3.0)

    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
    assert float(gradient) == pytest.approx(float(expected_gradient), rel=1e-12, abs=1e-12)
    assert float(hessian) == pytest.approx(float(expected_hessian), rel=1e-12, abs=1e-12)


def test_negative_binomial_tiny_dispersion_is_stable_under_jit_and_vmap():
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    eta = jnp.log(jnp.asarray([0.1, 1.0, 10.0, 100.0]))
    X = jnp.empty((y.size, 0))
    family = NegativeBinomial()

    def value_gradient_hessian(alpha):
        objective = lambda value: family.negloglikelihood(X, y, eta, value)
        return objective(alpha), jax.grad(objective)(alpha), jax.grad(jax.grad(objective))(alpha)

    alpha = jnp.asarray([0.0, 1e-16, 1e-12, 1e-9, 1e-6, 1e-3, 1.0])
    eager = jax.vmap(value_gradient_hessian)(alpha)
    compiled = jax.jit(jax.vmap(value_gradient_hessian))(alpha)

    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        assert jnp.all(jnp.isfinite(eager_value))
        assert jnp.allclose(compiled_value, eager_value, rtol=1e-10, atol=1e-10)


def test_nb2_centered_lgamma_ratio_graph_uses_scalar_cond():
    y = jnp.asarray([0.0, 1.5, 10.0])
    alpha = jnp.asarray(0.2)

    jaxpr = jax.make_jaxpr(_nb2_centered_lgamma_ratio)(y, alpha).jaxpr

    # Guard against eager evaluation of every numerical regime via ``jnp.where``.
    assert any(eqn.primitive.name == "cond" for eqn in jaxpr.eqns)


def test_nb2_mean_terms_graph_avoids_redundant_log1p():
    y = jnp.asarray([0.0, 1.5, 10.0])
    log_mu = jnp.log(jnp.asarray([0.1, 1.0, 10.0]))
    alpha = jnp.asarray(0.2)

    jaxpr = jax.make_jaxpr(_nb2_mean_terms)(y, log_mu, alpha).jaxpr

    # A top-level ``log1p`` means the series path has resumed duplicate work.
    assert all(eqn.primitive.name != "log1p" for eqn in jaxpr.eqns)


@pytest.mark.parametrize("alpha", [1e-6, 5e-4, 0.3])
def test_nb2_centered_log_alpha_hessian_matches_public_likelihood_autodiff(alpha):
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    eta = jnp.log(jnp.asarray([0.1, 1.0, 10.0, 100.0]))
    X = jnp.empty((y.size, 0))
    family = NegativeBinomial()
    log_alpha = jnp.log(jnp.asarray(alpha))

    def objective(value):
        return family.negloglikelihood(X, y, eta, jnp.exp(value))

    expected_score = jax.grad(objective)(log_alpha)
    expected_hessian = jax.hessian(objective)(log_alpha)
    score, centered_hessian, a2, a3 = jax.jit(_nb2_log_alpha_derivatives)(y, eta, jnp.asarray(alpha))
    z = jax.nn.sigmoid(log_alpha + eta)

    assert float(score) == pytest.approx(float(expected_score), rel=1e-9, abs=1e-9)
    assert float(score + centered_hessian) == pytest.approx(float(expected_hessian), rel=1e-9, abs=1e-9)
    assert float(a2) == pytest.approx(float(jnp.sum(z**2)), rel=1e-12, abs=1e-12)
    assert float(a3) == pytest.approx(float(jnp.sum(z**3)), rel=1e-12, abs=1e-12)


def test_nb2_centered_log_alpha_hessian_is_stable_at_poisson_boundary():
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    mu = jnp.asarray([0.1, 1.0, 10.0, 100.0])
    alpha = jnp.asarray(1e-12)
    _, centered_hessian, _, _ = _nb2_log_alpha_derivatives(y, jnp.log(mu), alpha)
    expected_alpha_hessian = jnp.sum(y * (y - 1.0) * (2.0 * y - 1.0) / 6.0 - y * mu**2 + 2.0 * mu**3 / 3.0)

    assert jnp.isfinite(centered_hessian)
    assert float(centered_hessian / alpha**2) == pytest.approx(float(expected_alpha_hessian), rel=1e-8)


def test_nb2_exprel_is_stable_under_jit_and_second_order_autodiff():
    values = jnp.asarray([-1.0, -1e-12, 0.0, 1e-12, 1.0])
    safe_values = jnp.where(values == 0.0, jnp.ones_like(values), values)
    expected = jnp.where(values == 0.0, 1.0, jnp.expm1(values) / safe_values)
    actual = jax.jit(jax.vmap(_nb2_exprel))(values)
    first_derivative = jax.grad(_nb2_exprel)(0.0)
    second_derivative = jax.grad(jax.grad(_nb2_exprel))(0.0)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert float(first_derivative) == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert float(second_derivative) == pytest.approx(1.0 / 3.0, rel=1e-12, abs=1e-12)


def test_nb2_riemannian_hessian_matches_explicit_log_alpha_formula():
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    eta = jnp.log(jnp.asarray([0.1, 1.0, 10.0, 100.0]))
    alpha = jnp.asarray(0.3)
    X = jnp.empty((y.size, 0))
    family = NegativeBinomial()
    score, centered_hessian, a2, a3 = _nb2_log_alpha_derivatives(y, eta, alpha)
    connection_complement = a3 / a2

    def objective(log_alpha):
        return family.negloglikelihood(X, y, eta, jnp.exp(log_alpha))

    ordinary_hessian = jax.hessian(objective)(jnp.log(alpha))
    expected = ordinary_hessian - (1.0 - connection_complement) * score
    actual = centered_hessian + connection_complement * score

    assert float(actual) == pytest.approx(float(expected), rel=1e-9, abs=1e-9)


def test_nb2_riemannian_direction_uses_positive_hessian_and_metric_fallback():
    score = jnp.asarray(2.0)
    a2 = jnp.asarray(4.0)
    a3 = jnp.asarray(1.0)
    positive_direction, connection_complement = _nb2_riemannian_direction(score, jnp.asarray(3.0), a2, a3)
    fallback_direction, fallback_connection = _nb2_riemannian_direction(score, jnp.asarray(-2.0), a2, a3)

    assert float(connection_complement) == pytest.approx(0.25)
    assert float(positive_direction) == pytest.approx(-2.0 / 3.5)
    assert float(fallback_connection) == pytest.approx(0.25)
    assert float(fallback_direction) == pytest.approx(-1.0)
    assert float(score * fallback_direction) <= 0.0


def test_nb2_riemannian_direction_is_noop_for_degenerate_metric():
    direction, connection_complement = _nb2_riemannian_direction(
        jnp.asarray(1.0), jnp.asarray(-1.0), jnp.asarray(0.0), jnp.asarray(0.0)
    )

    assert float(direction) == 0.0
    assert float(connection_complement) == 0.0


def test_negative_binomial_dispersion_update_uses_variance_manifold_retraction():
    y = jnp.asarray([0.0, 1.5, 10.0, 100.0])
    eta = jnp.log(jnp.asarray([0.1, 1.0, 10.0, 100.0]))
    X = jnp.empty((y.size, 0))
    alpha = jnp.asarray(0.3)
    step_size = jnp.asarray(0.1)
    family = NegativeBinomial()
    score, centered_hessian, a2, a3 = _nb2_log_alpha_derivatives(y, eta, alpha)
    direction, connection_complement = _nb2_riemannian_direction(score, centered_hessian, a2, a3)
    scaled_direction = step_size * direction
    expected = alpha * (1.0 + scaled_direction * _nb2_exprel(connection_complement * scaled_direction))

    actual = family.update_dispersion(X, y, eta, alpha, step_size)

    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_negative_binomial_dispersion_update_is_finite_and_bounded_for_extreme_inputs():
    y = jnp.asarray([0.0, 0.5, 2.25, 100.0])
    eta = jnp.asarray([-20.0, -2.0, 2.0, 20.0])
    X = jnp.empty((y.size, 0))
    dispersions = jnp.asarray([1e-12, 1e-9, 1e-4, 0.2, 10.0, 1e9, 1e12])
    family = NegativeBinomial()

    actual = jax.jit(jax.vmap(lambda alpha: family.update_dispersion(X, y, eta, alpha, 1.0)))(dispersions)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.all(actual >= 1e-9)
    assert jnp.all(actual <= 1e9)


@pytest.mark.parametrize(
    ("y", "disp"),
    [
        (jnp.asarray([0.0, -0.5, 2.0]), 0.2),
        (jnp.asarray([0.0, 0.5, 2.0]), -0.2),
    ],
)
def test_negative_binomial_rejects_negative_response_or_dispersion(y, disp):
    eta = jnp.zeros_like(y)

    actual = NegativeBinomial().negloglikelihood(jnp.empty((y.size, 0)), y, eta, disp)

    assert jnp.isposinf(actual)


@pytest.mark.parametrize(
    ("link", "mu"),
    [
        (IdentityLink(), jnp.linspace(-2.0, 2.0, 101)),
        (LogLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (InverseLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (LogitLink(), jnp.linspace(0.05, 0.95, 101)),
        (PowerLink(1.5), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (NBLink(0.3), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
    ],
)
def test_link_derivative_matches_autodiff(link, mu):
    mu = jnp.asarray(mu)

    def _forward(x):
        return link(x)

    deriv_ad = jax.vmap(jax.grad(lambda x: _forward(x).astype(float)))(mu.astype(float))
    deriv_impl = link.deriv(mu)
    assert jnp.all(jnp.isfinite(deriv_ad))
    assert jnp.all(jnp.isfinite(deriv_impl))
    assert jnp.allclose(deriv_impl, deriv_ad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("family", "disp"),
    [
        (Gaussian(), 1.0),
        (Poisson(), 1.0),
        (Binomial(), 1.0),
        (NegativeBinomial(), 0.2),
        (Gamma(), 0.7),
    ],
)
def test_family_calc_weight_shapes_and_finiteness(family, disp):
    eta = jnp.linspace(-1.0, 1.0, 128)
    mu, link_prime, weight = family.calc_weight(eta, disp)
    assert mu.shape == eta.shape
    assert link_prime.shape == eta.shape
    assert jnp.shape(weight) in [(), eta.shape]
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(jnp.isfinite(link_prime))
    assert jnp.all(jnp.isfinite(jnp.atleast_1d(weight)))


@pytest.mark.parametrize(
    ("family", "disp"),
    [
        (Gaussian(), 1.3),
        (Poisson(), 1.0),
        (Binomial(), 1.0),
        (NegativeBinomial(), 0.4),
        (Gamma(), 0.6),
    ],
)
def test_family_sample_shapes_and_mean_approx(family, disp):
    # Monte Carlo check that E[y] ≈ mu using a fixed PRNG stream.
    if isinstance(family, Gaussian):
        mu = jnp.linspace(-1.0, 1.0, 64)
    elif isinstance(family, Binomial):
        mu = jnp.linspace(0.1, 0.9, 64)
    else:
        mu = jnp.exp(jnp.linspace(-1.0, 1.0, 64))

    eta = family.glink(mu)

    base_key = rdm.key(0)
    keys = rdm.split(base_key, 1000)
    ys = jax.vmap(lambda k: family.sample(k, eta, disp), in_axes=0)(keys)

    assert ys.shape == (keys.shape[0],) + eta.shape
    assert jnp.all(jnp.isfinite(ys))

    mean_hat = jnp.mean(ys, axis=0)

    # Deterministic, but allow generous tolerance for discrete outcomes.
    assert jnp.allclose(mean_hat, mu, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    ("family_ctor", "bad_link"),
    [
        (Gaussian, LogitLink()),
        (Poisson, LogitLink()),
        (Binomial, InverseLink()),
        (NegativeBinomial, InverseLink()),
        (Gamma, LogitLink()),
    ],
)
def test_invalid_links_raise_value_error(family_ctor, bad_link):
    with pytest.raises(ValueError):
        family_ctor(glink=bad_link)
