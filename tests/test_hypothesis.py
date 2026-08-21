# pattern: Functional Core

import math

import numpy as np
import pytest

import equinox as eqx
import jax.numpy as jnp

from jaxqtl.distribution import Gaussian, LogLink
from jaxqtl.hypothesis import ScoreTest, SpaTest, WaldTest
from jaxqtl.infer import (
    CGSolve,
    CholeskySolve,
    FisherInfoError,
    GeneralizedLinearModel,
    HuberError,
    LinearModel,
    QRSolve,
)


@pytest.mark.parametrize(
    "solver", (CholeskySolve(), QRSolve(), CGSolve()), ids=("cholesky", "qr", "conjugate-gradient")
)
@pytest.mark.parametrize("std_err", (FisherInfoError(), HuberError()), ids=("fisher", "huber-white"))
def test_gaussian_wald_matches_explicit_full_model(solver, std_err):
    rng = np.random.default_rng(3)
    n, p, m = 80, 3, 4
    X = np.column_stack((np.ones(n), rng.normal(size=(n, p - 1))))
    G = rng.normal(size=(n, m))
    y = X @ np.array([0.5, -0.4, 0.3]) + G[:, 0] * 0.7 + rng.normal(size=n)

    model = LinearModel(solver=solver)
    result = WaldTest(model=model, std_err=std_err).test(X, G, y, 0.0)
    expected = [model.fit(jnp.column_stack((X, G[:, index])), y, 0.0, std_err) for index in range(m)]

    assert result.beta.shape == (m,)
    assert result.se.shape == (m,)
    assert result.z.shape == (m,)
    assert result.p.shape == (m,)
    np.testing.assert_allclose(np.asarray(result.beta), [fit.beta[-1] for fit in expected], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(result.se), [fit.se[-1] for fit in expected], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(result.z), [fit.z[-1] for fit in expected], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(result.p), [fit.p[-1] for fit in expected], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("offset_kind", ("scalar", "vector"))
@pytest.mark.parametrize("std_err", (FisherInfoError(), HuberError()), ids=("fisher", "huber-white"))
def test_gaussian_wald_with_offset_matches_full_model_and_jit(offset_kind, std_err):
    rng = np.random.default_rng(13)
    n, m = 80, 3
    X = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    G = rng.normal(size=(n, m))
    offset = 0.25 if offset_kind == "scalar" else 0.2 * G[:, 1] + np.linspace(-0.1, 0.2, n)
    error_scale = 0.5 + 0.25 * np.abs(X[:, 1])
    y = X @ np.array([0.5, -0.3, 0.2]) + G @ np.array([0.4, -0.2, 0.1]) + offset
    y += rng.normal(scale=error_scale)

    model = LinearModel()
    test = WaldTest(model=model, std_err=std_err)
    eager = test.test(X, G, y, offset)
    jitted = eqx.filter_jit(test.test)(X, G, y, offset)
    expected = [model.fit(jnp.column_stack((X, G[:, index])), y, offset, std_err) for index in range(m)]

    for field in ("beta", "se", "z", "p"):
        actual = np.asarray(getattr(eager, field))
        oracle = np.asarray([getattr(fit, field)[-1] for fit in expected])
        assert actual.shape == (m,)
        np.testing.assert_allclose(actual, oracle, rtol=2e-4, atol=2e-5)

    for field in ("beta", "se", "z", "p", "disp"):
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, field)),
            np.asarray(getattr(eager, field)),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_array_equal(np.asarray(jitted.num_iters), np.asarray(eager.num_iters))
    np.testing.assert_array_equal(np.asarray(jitted.converged), np.asarray(eager.converged))


def test_gaussian_glm_uses_general_wald_path():
    rng = np.random.default_rng(4)
    n, m = 60, 2
    X = np.column_stack((np.ones(n), rng.normal(size=n)))
    G = rng.normal(scale=0.1, size=(n, m))
    eta = X @ np.array([0.7, 0.1]) + G[:, 0] * 0.2
    y = np.exp(eta) + rng.normal(scale=0.05, size=n)
    model = GeneralizedLinearModel(family=Gaussian(glink=LogLink()), max_iter=200)

    result = WaldTest(model=model).test(X, G, y, 0.0)
    expected = [model.fit(jnp.column_stack((X, G[:, index])), y) for index in range(m)]

    assert result.beta.shape == (m,)
    np.testing.assert_allclose(np.asarray(result.beta), [fit.beta[-1] for fit in expected], rtol=1e-5, atol=1e-5)


def test_gaussian_score_with_offset_matches_closed_form_and_jit():
    rng = np.random.default_rng(17)
    n, m = 90, 3
    X = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    G = rng.normal(size=(n, m))
    offset = 0.3 * G[:, 1] + np.linspace(-0.2, 0.2, n)
    y = X @ np.array([0.4, -0.25, 0.15]) + G @ np.array([0.35, -0.1, 0.2]) + offset
    y += rng.normal(scale=0.7, size=n)

    test = ScoreTest(model=LinearModel(), std_err=FisherInfoError())
    eager = test.test(X, G, y, offset)
    jitted = eqx.filter_jit(test.test)(X, G, y, offset)

    # Build the efficient-score oracle independently of the hypothesis-test helpers.
    adjusted_y = y - offset
    null_beta = np.linalg.lstsq(X, adjusted_y, rcond=None)[0]
    null_residual = adjusted_y - X @ null_beta
    genotype_coefficients = np.linalg.lstsq(X, G, rcond=None)[0]
    residualized_genotypes = G - X @ genotype_coefficients
    dispersion = np.sum(null_residual**2) / (n - X.shape[1])
    score = residualized_genotypes.T @ null_residual / dispersion
    information = np.sum(residualized_genotypes**2, axis=0) / dispersion
    expected_se = 1.0 / np.sqrt(information)
    expected_beta = score / information
    expected_z = score / np.sqrt(information)
    expected_p = np.asarray([math.erfc(abs(value) / math.sqrt(2.0)) for value in expected_z])

    expected = {
        "beta": expected_beta,
        "se": expected_se,
        "z": expected_z,
        "p": expected_p,
    }
    for field, oracle in expected.items():
        actual = np.asarray(getattr(eager, field))
        assert actual.shape == (m,)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, oracle, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(eager.disp), dispersion, rtol=2e-4, atol=2e-5)
    assert np.asarray(eager.disp).shape == ()
    assert np.asarray(eager.num_iters).shape == ()
    assert np.asarray(eager.converged).shape == ()
    assert np.all(np.isfinite(np.asarray(eager.disp)))
    assert np.all(np.isfinite(np.asarray(eager.num_iters)))
    assert bool(np.asarray(eager.converged))

    for field in ("beta", "se", "z", "p", "disp"):
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, field)),
            np.asarray(getattr(eager, field)),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_array_equal(np.asarray(jitted.num_iters), np.asarray(eager.num_iters))
    np.testing.assert_array_equal(np.asarray(jitted.converged), np.asarray(eager.converged))


@pytest.mark.parametrize("test_type", (ScoreTest, SpaTest))
def test_score_based_tests_reject_huber_error(test_type):
    with pytest.raises(ValueError, match="only supports FisherInfoError"):
        test_type(model=LinearModel(), std_err=HuberError())
