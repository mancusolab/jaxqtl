# pattern: Functional Core

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
def test_gaussian_wald_fisher_matches_explicit_full_model_with_offset(offset_kind):
    rng = np.random.default_rng(13)
    n, m = 50, 3
    X = np.column_stack((np.ones(n), rng.normal(size=n)))
    G = rng.normal(size=(n, m))
    offset = 0.2 if offset_kind == "scalar" else rng.normal(scale=0.2, size=n)
    y = X @ np.array([0.5, -0.3]) + G[:, 0] * 0.4 + offset + rng.normal(size=n)

    model = LinearModel()
    result = WaldTest(model=model).test(X, G, y, offset)
    expected = [model.fit(jnp.column_stack((X, G[:, index])), y, offset) for index in range(m)]

    np.testing.assert_allclose(np.asarray(result.beta), [fit.beta[-1] for fit in expected], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.se), [fit.se[-1] for fit in expected], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.p), [fit.p[-1] for fit in expected], rtol=1e-5, atol=1e-5)


def test_gaussian_wald_jit_matches_eager_execution():
    rng = np.random.default_rng(14)
    n, m = 40, 2
    X = np.column_stack((np.ones(n), rng.normal(size=n)))
    G = rng.normal(size=(n, m))
    y = X @ np.array([0.2, 0.5]) + G[:, 0] * 0.3 + rng.normal(size=n)
    test = WaldTest(model=LinearModel(), std_err=HuberError())

    eager = test.test(X, G, y, 0.0)
    jitted = eqx.filter_jit(test.test)(X, G, y, 0.0)

    np.testing.assert_allclose(np.asarray(jitted.beta), np.asarray(eager.beta), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jitted.se), np.asarray(eager.se), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jitted.p), np.asarray(eager.p), rtol=1e-5, atol=1e-5)


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


@pytest.mark.parametrize("test_type", (ScoreTest, SpaTest))
def test_score_based_tests_reject_huber_error(test_type):
    with pytest.raises(ValueError, match="only supports FisherInfoError"):
        test_type(model=LinearModel(), std_err=HuberError())
