# pattern: Functional Core

import pytest

import equinox as eqx
import jax.numpy as jnp

from jaxqtl.hypothesis import ACAT, BetaPermutation, ScoreTest
from jaxqtl.infer import LinearModel


def _result():
    X = jnp.ones((12, 1))
    G = jnp.arange(24.0).reshape(12, 2) % 5
    y = jnp.arange(12.0) % 3
    return ScoreTest(LinearModel())(X, G, y, jnp.asarray(0.0))


def test_acat_reduction_masks_padding_and_uses_whole_window_weights():
    method = ACAT()
    result = _result()
    state = method.init(result.p.dtype, num_variants=3)
    update = eqx.filter_jit(method.update)
    state = update(state, jnp.array([0.02, 0.3]), jnp.array([True, True]))
    state = update(state, jnp.array([0.4, 0.0]), jnp.array([True, False]))
    pvalue, auxiliary = eqx.filter_jit(method.finalize)(state, None)
    expected_statistic = jnp.mean(jnp.tan((0.5 - jnp.array([0.02, 0.3, 0.4])) * jnp.pi))
    expected = 0.5 - jnp.arctan(expected_statistic) / jnp.pi
    assert jnp.allclose(pvalue, expected, atol=1e-6)
    assert auxiliary is None


@pytest.mark.parametrize("all_nan", [False, True])
def test_permutation_reduction_keeps_nan_policy_and_excludes_padding(all_nan):
    method = BetaPermutation()
    result = _result()
    state = method.init(result.z.dtype, num_variants=3)
    state = eqx.filter_jit(method.update)(state, jnp.array([jnp.nan, jnp.nan]), jnp.array([True, True]))
    state = eqx.filter_jit(method.update)(
        state,
        jnp.array([jnp.nan if all_nan else -4.0, 100.0]),
        jnp.array([True, False]),
    )
    assert jnp.isnan(state) if all_nan else state == 4.0


def test_beta_finalize_uses_explicit_lead_statistic_instead_of_window_maximum():
    from jaxqtl.hypothesis._aggregate import BetaCalibration, PermutationReference
    from jaxqtl.infer import BetaParams

    class FixedCalibration(BetaPermutation):
        def fit_calibration(self, z_stats_perm, dof):
            return BetaCalibration(
                BetaParams(jnp.array(1.0), jnp.array(1.0), jnp.array(True)), jnp.array(0.0), jnp.array(True)
            )

    method = FixedCalibration()
    result = _result()._replace(z=jnp.array([4.0, 1.0]), p=jnp.array([0.2, 0.01]))
    state = method.update(method.init(result.z.dtype, num_variants=2), result.z, jnp.array([True, True]))
    reference = PermutationReference(jnp.array([1.0, 2.0, 3.0]), 10)
    actual, calibration = eqx.filter_jit(method.finalize)(result.z[1], reference)
    expected = eqx.filter_jit(method.adjust)(result.z[1], calibration)
    assert jnp.allclose(actual, expected)
    assert actual > method.adjust(state, calibration)


def test_beta_finalize_rejects_variantwise_input_but_adjust_accepts_it():
    from jaxqtl.hypothesis import PermutationReference
    from jaxqtl.hypothesis._aggregate import BetaCalibration
    from jaxqtl.infer import BetaParams

    method = BetaPermutation()
    calibration = BetaCalibration(
        BetaParams(jnp.array(1.0), jnp.array(1.0), jnp.array(True)),
        jnp.array(0.0),
        jnp.array(True),
    )
    z = jnp.array([1.0, 2.0])
    assert eqx.filter_jit(method.adjust)(z, calibration).shape == (2,)
    with pytest.raises(ValueError, match="scalar"):
        eqx.filter_jit(method.finalize)(z, PermutationReference(jnp.arange(1.0, 65.0), 10))
