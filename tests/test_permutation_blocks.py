# pattern: Functional Core
"""Permutation ordering, block reduction, and calibration parity."""

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxqtl.distribution import NegativeBinomial
from jaxqtl.hypothesis import ACAT, BetaPermutation, ScoreTest
from jaxqtl.infer import GeneralizedLinearModel, LinearModel
from jaxqtl.map import _scan, cis as cis_map
from jaxqtl.map.cis import _run_cis_scan, map_cis_single, select_lead_variant


def _permutation_maxima(X, G, y, offset, test, permutations, key, **options):
    return _scan.AssociationScan(test, permutations, **options).permutation_maxima(X, G, y, offset, key)


def _permutation_scan(X, G, y, offset, test, permutations, key, **options):
    return _run_cis_scan(_scan.AssociationScan(test, permutations, **options), X, G, y, offset, key)[:2]


def _inputs(m=11, scalar_offset=False):
    rng = np.random.default_rng(981)
    n = 48
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
    offset = np.linspace(-0.3, 0.3, n)
    mu = np.exp(X @ np.array([0.7, 0.15]) + offset)
    y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    if scalar_offset:
        offset = np.array(0.1)
    return tuple(jnp.asarray(a) for a in (X, G, y, offset))


@pytest.mark.parametrize("block_size,batch_size", [(1, 1), (4, 3), (11, 4), (16, 8)])
@pytest.mark.parametrize("scalar_offset", [False, True])
def test_permutation_maxima_match_legacy_sequence(block_size, batch_size, scalar_offset):
    X, G, y, offset = _inputs(scalar_offset=scalar_offset)
    test = ScoreTest(model=LinearModel())
    perms = BetaPermutation(max_perm_direct=7)
    key = jax.random.key(3)
    expected = _scan.full_permutation_maxima(X, G, y, offset, test, perms, key)
    actual = _permutation_maxima(X, G, y, offset, test, perms, key, block_size=block_size, batch_size=batch_size)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    assert actual.shape == (7,)


def test_nb_permutation_maxima_match_legacy():
    with jax.enable_x64(True):
        X, G, y, offset = _inputs(m=5)
        test = ScoreTest(model=GeneralizedLinearModel(family=NegativeBinomial(), gtol=1e-6))
        perms = BetaPermutation(max_perm_direct=5)
        key = jax.random.key(8)
        expected = _scan.full_permutation_maxima(X, G, y, offset, test, perms, key)
        actual = _permutation_maxima(X, G, y, offset, test, perms, key, block_size=3, batch_size=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_permutation_maxima_ignore_padding_but_preserve_all_invalid(all_invalid):
    X, G, y, offset = _inputs(m=5)
    host = np.asarray(G).copy()
    host[:, :4] = 0.0
    if all_invalid:
        host[:] = 0.0
    G = jnp.asarray(host)
    test = ScoreTest(model=LinearModel())
    perms = BetaPermutation(max_perm_direct=5)
    key = jax.random.key(3)
    expected = _scan.full_permutation_maxima(X, G, y, offset, test, perms, key)
    actual = _permutation_maxima(X, G, y, offset, test, perms, key, block_size=4, batch_size=3)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("use_tdist", [False])
def test_gene_pvalue_matches_explicit_snp_adjustment_and_reference_finalization(use_tdist):
    with jax.enable_x64(True):
        X, G, y, offset = _inputs()
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=64, use_tdist=use_tdist)
        key = jax.random.key(12)
        result = eqx.filter_jit(test)(X, G, y, offset)
        expected = map_cis_single(X, G, y, offset, test, perms, key)[1]
        maxima = _scan.full_permutation_maxima(X, G, y, offset, test, perms, key)
        from jaxqtl.hypothesis import PermutationReference

        lead = select_lead_variant(result.p, key)
        assert expected[0].shape == ()
        snp_adjusted = eqx.filter_jit(perms.adjust)(result.z, expected[1])
        assert snp_adjusted.shape == result.z.shape
        np.testing.assert_allclose(expected[0], snp_adjusted[lead], rtol=2e-5, atol=1e-7)
        actual = eqx.filter_jit(perms.finalize)(
            result.z[lead], PermutationReference(maxima, X.shape[0] - X.shape[1] - 1)
        )
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-7)


def test_each_permutation_is_fitted_once_across_snp_blocks(monkeypatch):
    jax.clear_caches()
    calls = []
    original = ScoreTest.init

    def counted(self, X, y, offset):
        jax.debug.callback(lambda _: calls.append(1), y[0])
        return original(self, X, y, offset)

    monkeypatch.setattr(ScoreTest, "init", counted)
    X, G, y, offset = _inputs(m=11)
    result = _permutation_maxima(
        X,
        G,
        y,
        offset,
        ScoreTest(model=LinearModel()),
        BetaPermutation(max_perm_direct=7),
        jax.random.key(2),
        block_size=3,
        batch_size=4,
    )
    result.block_until_ready()
    jax.effects_barrier()
    assert len(calls) == 7


def test_blocked_permutation_result_matches_whole_window_and_skips_acat(monkeypatch):
    with jax.enable_x64(True):
        X, G, y, offset = _inputs()
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=64)
        key = jax.random.key(12)
        expected = cis_map.map_cis_single(X, G, y, offset, test, perms, key)

        def unexpected(*args, **kwargs):
            pytest.fail("permutation tests must not compute ACAT")

        monkeypatch.setattr(ACAT, "finalize", unexpected)
        actual = _permutation_scan(X, G, y, offset, test, perms, key, block_size=4, batch_size=7)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-7)


def test_map_cis_routes_standard_score_permutations_to_blocks(monkeypatch):
    from types import SimpleNamespace
    from typing import cast

    import polars as pl

    from jaxqtl.map.data import CisData, ReadyDataState

    X, G, y, offset = _inputs(m=5)
    info = pl.DataFrame({"snp": [f"rs{i}" for i in range(5)], "pos": list(range(5)), "a1": ["A"] * 5, "a0": ["C"] * 5})
    gene = CisData(X, G, y, offset, "gene1", "22", 100, 110, info, 1, 200)
    data = SimpleNamespace(iter_cis=lambda *args, **kwargs: iter([gene]))
    test = ScoreTest(model=LinearModel())
    perms = BetaPermutation(max_perm_direct=64)
    routed = []
    original = _scan.AssociationScan.permutation_maxima

    def counted(*args, **kwargs):
        routed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan.AssociationScan, "permutation_maxima", counted)
    output = pl.concat(list(cis_map.map_cis(cast(ReadyDataState, data), test, perms, verbose=False, seed=12)))
    assert routed == [True]
    assert output["adj_method"].item() == "BETA"
    assert output["num_var"].item() == 5
    assert output["result_valid"].item()


def test_permutation_kernels_reuse_compilations_for_new_window_widths(caplog):
    import logging

    test = ScoreTest(model=LinearModel())
    permutations = BetaPermutation(max_perm_direct=7)

    def run(m):
        X, G, y, offset = _inputs(m=m)
        return _permutation_maxima(X, G, y, offset, test, permutations, jax.random.key(7), block_size=4, batch_size=3)

    # Input construction runs outside the compile-log scope.
    inputs = [_inputs(m=m) for m in [3, 5, 9, 17]]
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        X, G, y, offset = inputs[0]
        _permutation_maxima(
            X, G, y, offset, test, permutations, jax.random.key(7), block_size=4, batch_size=3
        ).block_until_ready()
        caplog.clear()
        for X, G, y, offset in inputs[1:]:
            values = _permutation_maxima(
                X, G, y, offset, test, permutations, jax.random.key(7), block_size=4, batch_size=3
            )
            values.block_until_ready()
    compilations = [r.getMessage() for r in caplog.records if "Compiling " in r.getMessage()]
    assert compilations == []


@pytest.mark.parametrize(
    "setting,value", [("block_size", 0), ("batch_size", 0), ("batch_size", -1), ("batch_size", True)]
)
def test_permutation_block_configuration_rejects_invalid_sizes(setting, value):
    X, G, y, offset = _inputs()
    with pytest.raises(ValueError, match="positive integer"):
        _permutation_maxima(
            X,
            G,
            y,
            offset,
            ScoreTest(model=LinearModel()),
            BetaPermutation(max_perm_direct=7),
            jax.random.key(3),
            **{setting: value},
        )


def test_permutation_states_are_bounded_by_batch_and_donor_count():
    X, G, y, offset = _inputs()
    _, states = _scan._initialize_permutations(ScoreTest(model=LinearModel()), X, y, offset, jax.random.key(1), 3)
    leaves = jax.tree.leaves(states)
    assert sum(array.size for array in leaves) <= 2 * 3 * X.shape[0] + 4 * 3
    for array in leaves:
        assert array.shape[0] == 3
        assert array.size <= 3 * X.shape[0]


def test_mixed_precision_permutation_scores_preserve_dtype():
    with jax.enable_x64(True):
        X, G, y, offset = _inputs(m=5)
        X, y, offset = (a.astype(jnp.float32) for a in (X, y, offset))
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=7)
        key = jax.random.key(3)
        expected = _scan.full_permutation_maxima(X, G, y, offset, test, perms, key)
        actual = _permutation_maxima(X, G, y, offset, test, perms, key, block_size=4, batch_size=3)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_student_t_calibration_blocked_path_matches_legacy():
    with jax.enable_x64(True):
        rng = np.random.default_rng(9)
        n = 100
        X = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]))
        G = jnp.asarray(rng.normal(size=(n, 3)))
        y = jnp.asarray(rng.normal(size=n))
        offset = jnp.zeros(n)
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=128, use_tdist=True)
        key = jax.random.key(1)
        expected = cis_map.map_cis_single(X, G, y, offset, test, perms, key)
        actual = _permutation_scan(X, G, y, offset, test, perms, key, block_size=2, batch_size=13)
        np.testing.assert_allclose(actual[0].z, expected[0].z, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=1e-5, atol=1e-7)
        assert bool(actual[1][1][2]) == bool(expected[1][1][2])
        assert bool(actual[1][1][0].converged) == bool(expected[1][1][0].converged)


def test_student_t_calibration_preserves_existing_failure():
    with jax.enable_x64(True):
        X, G, y, offset = _inputs()
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=64, use_tdist=True)
        key = jax.random.key(12)
        with pytest.raises(eqx.EquinoxRuntimeError, match="maximum number of steps"):
            cis_map.map_cis_single(X, G, y, offset, test, perms, key)
        with pytest.raises(eqx.EquinoxRuntimeError, match="maximum number of steps"):
            _permutation_scan(X, G, y, offset, test, perms, key, block_size=4, batch_size=7)


def test_explicit_float32_inputs_preserve_calibrated_dtype_under_x64():
    with jax.enable_x64(True):
        rng = np.random.default_rng(9)
        n = 100
        X = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]), dtype=jnp.float32)
        G = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float32)
        y = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
        offset = jnp.zeros(n, dtype=jnp.float32)
        test = ScoreTest(model=LinearModel())
        perms = BetaPermutation(max_perm_direct=128, use_tdist=True)
        key = jax.random.key(1)
        expected = cis_map.map_cis_single(X, G, y, offset, test, perms, key)
        actual = _permutation_scan(X, G, y, offset, test, perms, key, block_size=2, batch_size=13)
        assert actual[1][0].dtype == expected[1][0].dtype
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=2e-4, atol=1e-5)
