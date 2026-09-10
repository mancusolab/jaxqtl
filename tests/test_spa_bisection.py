# pattern: Functional Core
"""Regression checks for SPA roots in sparse and unbounded-domain models."""

import math

import pytest

from scipy.optimize import brentq
from scipy.special import ndtr

import jax
import jax.numpy as jnp

from jaxqtl.hypothesis._spa import (
    BasicCGFState,
    GaussianCGF,
    GaussianCGFState,
    NegativeBinomialCGF,
    NegBinCGFState,
    PoissonCGF,
    saddlepoint_pvalue,
)


@pytest.mark.parametrize("alpha", [1e-9, 0.0001911442384])
@pytest.mark.parametrize("negative_weight", [-0.02, -0.005])
@pytest.mark.parametrize("x64", [False, True])
def test_sparse_nb_spa_matches_independently_bracketed_roots(alpha, negative_weight, x64):
    with jax.enable_x64(x64):
        mu = [0.0127, 0.05]
        g = [1.0, negative_weight]
        score = 0.9873
        r = 1 / alpha
        center = sum(a * b for a, b in zip(mu, g))

        def cumulants(t):
            k, kp, kpp = -t * center, -center, 0.0
            for m, a in zip(mu, g):
                e = math.exp(t * a)
                d = 1 - m / r * math.expm1(t * a)
                k -= r * math.log1p(-m / r * math.expm1(t * a))
                kp += a * m * e / d
                kpp += a * a * m * e * (1 + m / r) / d**2
            return k, kp, kpp

        bounds = [0.9 * math.log1p(r / mu[1]) / negative_weight, 0.9 * math.log1p(r / mu[0])]
        tails = []
        for target in [score, -score]:
            t = brentq(lambda t: cumulants(t)[1] - target, *bounds)
            k, _, kpp = cumulants(t)
            w = math.copysign(math.sqrt(2 * (t * target - k)), t)
            log_factor = math.log(-math.expm1(-t)) if t > 0 else -t + math.log1p(-math.exp(t))
            corrected = w + (log_factor + 0.5 * math.log(kpp) - math.log(abs(w))) / w
            tails.append(ndtr(-corrected if target > 0 else corrected))
        observed = saddlepoint_pvalue(
            score,
            jnp.array(g),
            NegativeBinomialCGF(),
            NegBinCGFState(jnp.array(mu), jnp.array(r)),
            two_sided_mode="abs",
        )
        assert float(observed) == pytest.approx(sum(tails), rel=1e-6 if x64 else 2e-5)


def test_failed_spa_does_not_silently_return_normal_tail():
    with jax.enable_x64(True):
        p = saddlepoint_pvalue(
            0.9873,
            jnp.array([1.0, -0.02]),
            NegativeBinomialCGF(),
            NegBinCGFState(jnp.array([0.0127, 0.05]), jnp.array(1e9)),
            two_sided_mode="abs",
            max_iter=1,
        )
        assert jnp.isnan(p)


@pytest.mark.parametrize("x64", [False, True])
def test_unbounded_gaussian_spa_vmap_and_gradient(x64):
    with jax.enable_x64(x64):
        state = GaussianCGFState(jnp.zeros(2), jnp.full(2, 0.5))

        def pvalue(score):
            return saddlepoint_pvalue(score, jnp.array([1.0, -1.0]), GaussianCGF(), state)

        scores = jnp.array([0.5, 3.0, -4.0, 10.0])
        actual = jax.jit(jax.vmap(pvalue))(scores)
        expected = 2 * jax.scipy.stats.norm.sf(jnp.abs(scores))
        assert jnp.allclose(actual, expected, rtol=2e-5, atol=0)
        derivative = jax.jit(jax.grad(pvalue))(jnp.array(3.0))
        assert derivative == pytest.approx(float(-2 * jax.scipy.stats.norm.pdf(3.0)), rel=2e-5)


def test_sparse_poisson_spa_unbounded_domain():
    with jax.enable_x64(True):
        p = saddlepoint_pvalue(
            0.9873,
            jnp.array([1.0, -0.02]),
            PoissonCGF(),
            BasicCGFState(jnp.array([0.0127, 0.05])),
            two_sided_mode="abs",
        )
        assert float(p) == pytest.approx(0.01297173849793784, rel=1e-6)
