# pattern: Functional Core

import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxqtl.distribution import NegativeBinomial, Poisson
from jaxqtl.hypothesis import GaussianCGF, NegativeBinomialCGF, PoissonCGF, ScoreTest, SpaTest, WaldTest
from jaxqtl.infer import FisherInfoError, GeneralizedLinearModel, HuberError, LinearModel


def _inputs(counts, vector_offset):
    keys = jr.split(jr.key(412), 4)
    n = 64
    X = jnp.column_stack((jnp.ones(n), jr.normal(keys[0], (n,))))
    G = jr.binomial(keys[1], n=2, p=0.3, shape=(n, 3))
    offset = jnp.linspace(-0.2, 0.2, n) if vector_offset else jnp.asarray(0.15)
    eta = X @ jnp.array([0.8, 0.2]) + 0.4 * G[:, 0] + offset
    y = jr.poisson(keys[2], jnp.exp(eta)).astype(X.dtype) if counts else eta + jr.normal(keys[3], (n,))
    return X, G, y, offset


_TESTS = (
    (ScoreTest(model=LinearModel()), False),
    (ScoreTest(model=GeneralizedLinearModel(family=NegativeBinomial())), True),
    (SpaTest(model=LinearModel(), cgf=GaussianCGF()), False),
    (SpaTest(model=GeneralizedLinearModel(family=Poisson()), cgf=PoissonCGF()), True),
    (SpaTest(model=GeneralizedLinearModel(family=NegativeBinomial()), cgf=NegativeBinomialCGF()), True),
    (WaldTest(model=LinearModel(), std_err=FisherInfoError()), False),
    (WaldTest(model=LinearModel(), std_err=HuberError()), False),
    (WaldTest(model=GeneralizedLinearModel(family=Poisson())), True),
)


@pytest.mark.parametrize(
    "test,counts",
    _TESTS,
    ids=("score-lm", "score-nb", "spa-lm", "spa-poisson", "spa-nb", "wald-lm", "wald-huber", "wald-poisson"),
)
@pytest.mark.parametrize("vector_offset", (False, True), ids=("scalar-offset", "vector-offset"))
def test_initialized_test_reuses_state_for_new_genotypes(test, counts, vector_offset):
    X, G, y, offset = _inputs(counts, vector_offset)
    state = eqx.filter_jit(test.init)(X, y, offset)
    initialized = eqx.filter_jit(test.test)
    # Compare compiled paths: float32 SPA already differs between eager/JIT in the frozen implementation.
    whole = eqx.filter_jit(test)(X, G, y, offset)
    actual = initialized(X, G, state)
    for expected_value, actual_value in zip(whole, actual, strict=True):
        assert jnp.allclose(actual_value, expected_value, rtol=2e-4, atol=2e-5, equal_nan=True)

    # The phenotype fit is reusable when both SNP values and the number of SNPs change.
    changed = G[:, :2] + 0.2 * X[:, 1, None]
    expected_changed = eqx.filter_jit(test)(X, changed, y, offset)
    actual_changed = initialized(X, changed, state)
    for expected_value, actual_value in zip(expected_changed, actual_changed, strict=True):
        assert jnp.allclose(actual_value, expected_value, rtol=2e-4, atol=2e-5, equal_nan=True)


def test_score_initialization_is_compact_for_permutation_batches():
    X, _, y, offset = _inputs(True, True)
    test = ScoreTest(model=GeneralizedLinearModel(family=NegativeBinomial()))
    state = eqx.filter_jit(test.init)(X, y, offset)
    leaves = jax.tree.leaves(state)
    assert len(leaves) == 6
    assert sum(leaf.size for leaf in leaves) <= 2 * len(y) + 4
    assert all(leaf.ndim <= 1 for leaf in leaves)
    assert state.resid.shape == y.shape
    assert state.glm_wt.shape == y.shape


@pytest.mark.parametrize(
    "test,counts",
    _TESTS,
    ids=("score-lm", "score-nb", "spa-lm", "spa-poisson", "spa-nb", "wald-lm", "wald-huber", "wald-poisson"),
)
def test_initialization_states_batch_without_covariate_copies(test, counts):
    X, G, y, offset = _inputs(counts, True)
    batch = eqx.filter_jit(eqx.filter_vmap(test.init, in_axes=(None, 0, 0)))(
        X, jnp.stack((y, y[::-1])), jnp.stack((offset, offset[::-1]))
    )
    for leaf in jax.tree.leaves(eqx.filter(batch, eqx.is_array)):
        assert leaf.shape[0] == 2
        assert leaf.ndim <= 2
    results = eqx.filter_jit(eqx.filter_vmap(test.test, in_axes=(None, None, eqx.if_array(0))))(X, G, batch)
    assert results.z.shape == (2, G.shape[1])
