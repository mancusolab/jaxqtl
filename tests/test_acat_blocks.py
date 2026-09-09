# pattern: Functional Core
"""Numerical and compilation contracts for fixed-block score + ACAT scans."""

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxqtl.distribution import NegativeBinomial
from jaxqtl.hypothesis import ACAT, ScoreTest
from jaxqtl.infer import GeneralizedLinearModel, LinearModel
from jaxqtl.map import _scan, cis as cis_map


def _acat_scan(X, G, y, offset, test, *, block_size=2048):
    return _scan.AssociationScan(test, ACAT(), block_size=block_size).run(X, G, y, offset, jax.random.key(1))


def _inputs(n=64, m=11):
    rng = np.random.default_rng(44)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    G = rng.binomial(2, 0.25, size=(n, m)).astype(float)
    offset = np.linspace(-0.2, 0.2, n)
    mu = np.exp(X @ np.array([0.8, -0.15]) + offset)
    y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    return tuple(jnp.asarray(a) for a in (X, G, y, offset))


@pytest.mark.parametrize("shape", [(11,), (3, 11)])
def test_full_host_blocks_use_views_without_padding_copies(shape, monkeypatch):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    transferred = []
    device_put = jax.device_put

    def track_transfer(block):
        transferred.append(block)
        return device_put(block)

    monkeypatch.setattr(jax, "device_put", track_transfer)
    blocks = list(_scan._HostBuffers(values).blocks(4))
    for (start, stop, block), source in zip(blocks[:2], transferred[:2], strict=True):
        assert np.shares_memory(source, values)
        np.testing.assert_array_equal(block, values[..., start:stop])


@pytest.mark.parametrize("shape", [(11,), (3, 11)])
def test_partial_host_block_is_reused_without_mutating_the_source(shape):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    original = values.copy()
    buffers = _scan._HostBuffers(values)
    first = list(buffers.blocks(4))[-1]
    second = list(buffers.blocks(4))[-1]
    assert first[2] is second[2]
    assert first[:2] == second[:2] == (8, 11)
    np.testing.assert_array_equal(first[2][..., :3], original[..., 8:11])
    np.testing.assert_array_equal(first[2][..., 3], original[..., 8])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("negative_binomial", [False, True])
@pytest.mark.parametrize("scalar_offset", [False, True])
def test_score_from_null_matches_existing_test(negative_binomial, scalar_offset):
    X, G, y, offset = _inputs()
    if scalar_offset:
        offset = jnp.array(0.1)
    model = GeneralizedLinearModel(family=NegativeBinomial(), gtol=1e-6) if negative_binomial else LinearModel()
    test = ScoreTest(model=model)
    expected = eqx.filter_jit(test)(X, G, y, offset)
    state = eqx.filter_jit(test.init)(X, y, offset)
    actual = eqx.filter_jit(test.test)(X, G, state)
    for field in expected._fields:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("block_size", [1, 4, 11, 16])
def test_blocked_acat_matches_whole_window(block_size):
    X, G, y, offset = _inputs()
    test = ScoreTest(model=LinearModel())
    expected, (expected_acat, _) = cis_map.map_cis_single(X, G, y, offset, test, ACAT(), jax.random.key(1))
    actual, (actual_acat, aux) = _acat_scan(X, G, y, offset, test, block_size=block_size)
    for field in expected._fields:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(actual_acat, expected_acat, rtol=2e-4, atol=1e-5)
    assert aux is None
    assert actual.p.shape == (G.shape[1],)


def test_blocked_score_kernels_reuse_traces_across_window_sizes(monkeypatch):
    jax.clear_caches()
    counts = {"fit": 0, "score": 0}
    fit, score = ScoreTest.init, ScoreTest.test

    def counted_fit(self, *args):
        counts["fit"] += 1
        return fit(self, *args)

    def counted_score(self, *args):
        counts["score"] += 1
        return score(self, *args)

    monkeypatch.setattr(ScoreTest, "init", counted_fit)
    monkeypatch.setattr(ScoreTest, "test", counted_score)
    test = ScoreTest(model=LinearModel())
    for m in [3, 4, 9, 17]:
        X, G, y, offset = _inputs(m=m)
        result, (p, _) = _acat_scan(X, G, y, offset, test, block_size=4)
        assert result.p.shape == (m,)
        assert np.isfinite(float(p))
    assert counts == {"fit": 1, "score": 1}


@pytest.mark.parametrize("pvalues", [[0.01, 0.2, 0.8], [1e-20, 0.4], [0.0, 0.2], [1.0, 0.2], [0.01, float("nan")]])
def test_masked_acat_ignores_padding_and_uses_global_weights(pvalues):
    from jaxqtl.hypothesis import _aggregate

    p = jnp.asarray(pvalues)
    # Padding deliberately contains NaN and both endpoints: none may affect ACAT.
    padded = jnp.concatenate([p, jnp.array([float("nan"), 0.0, 1.0])])
    valid = jnp.arange(len(padded)) < len(p)
    components = eqx.filter_jit(_aggregate._acat_components)(padded, valid, jnp.asarray(1.0 / len(p)))
    actual = eqx.filter_jit(_aggregate._acat_pvalue)(*components)
    clean_components = eqx.filter_jit(_aggregate._acat_components)(
        p, jnp.ones_like(p, dtype=bool), jnp.asarray(1.0 / len(p))
    )
    expected = eqx.filter_jit(_aggregate._acat_pvalue)(*clean_components)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7, equal_nan=True)


def test_acat_endpoints_across_blocks_still_raise():
    from jaxqtl.hypothesis import _aggregate

    a = _aggregate._acat_components(jnp.array([0.0, 0.2]), jnp.array([True, True]), jnp.array(0.25))
    b = _aggregate._acat_components(jnp.array([0.3, 1.0]), jnp.array([True, True]), jnp.array(0.25))
    with pytest.raises(Exception, match="both 0 and 1"):
        eqx.filter_jit(_aggregate._acat_pvalue)(a[0] + b[0], a[1] | b[1], a[2] | b[2])


@pytest.mark.parametrize("block_size", [0, -1, 2.5, True])
def test_block_size_validation(block_size):
    X, G, y, offset = _inputs()
    with pytest.raises(ValueError, match="positive integer"):
        _acat_scan(X, G, y, offset, ScoreTest(model=LinearModel()), block_size=block_size)


def test_empty_window_is_rejected():
    X, G, y, offset = _inputs(m=0)
    with pytest.raises(ValueError, match="at least one variant"):
        _acat_scan(X, G, y, offset, ScoreTest(model=LinearModel()))


@pytest.mark.parametrize("all_constant", [False, True])
def test_real_constant_variants_are_not_treated_as_padding(all_constant):
    X, G, y, offset = _inputs()
    host = np.asarray(G).copy()
    host[:, 0] = 0.0
    if all_constant:
        host[:] = 0.0
    G = jnp.asarray(host)
    test = ScoreTest(model=LinearModel())
    expected, (expected_acat, _) = cis_map.map_cis_single(X, G, y, offset, test, ACAT(), jax.random.key(1))
    actual, (p, _) = _acat_scan(X, G, y, offset, test, block_size=4)
    np.testing.assert_allclose(actual.p, expected.p, rtol=2e-4, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(p, expected_acat, rtol=2e-4, atol=1e-5, equal_nan=True)


def test_map_cis_acat_retains_kernels_and_real_snp_metadata(monkeypatch):
    from types import SimpleNamespace
    from typing import cast

    import polars as pl

    from jaxqtl.map.data import CisData, ReadyDataState

    X, G, y, offset = _inputs(m=5)
    X = jnp.ones((64, 1))
    y = jnp.tile(jnp.array([1.0, 4.0, 2.0, 1.0]), 16)
    offset = jnp.zeros(64)
    # An exact tie crosses block boundaries; selection must use real SNP order.
    host = np.asarray(G).copy()
    host[:, 0] = np.tile([0.0, 1.0, 2.0, 1.0], 16)
    host[:, 4] = host[:, 0]
    host[:, 1:4] = 0.0
    G = jnp.asarray(host)
    info = pl.DataFrame(
        {"snp": [f"rs{i}" for i in range(5)], "pos": list(range(100, 105)), "a1": ["A"] * 5, "a0": ["C"] * 5}
    )
    genes = [CisData(X, G, y, offset, f"gene{i}", "22", 100, 110, info, 1, 200) for i in range(51)]
    data = SimpleNamespace(iter_cis=lambda *a, **k: iter(genes))
    test = ScoreTest(model=LinearModel())
    expected_test, expected_acat = cis_map.map_cis_single(X, G, y, offset, test, ACAT(), jax.random.key(1))

    assert expected_test.p[0] == expected_test.p[4] == np.nanmin(expected_test.p)
    monkeypatch.setattr(ACAT, "block_size", 4)

    # The specialized score+ACAT path must bypass the full-window JIT and clearing.
    def unexpected(*args, **kwargs):
        pytest.fail("blocked ACAT must not call the full-window kernel or clear caches")

    monkeypatch.setattr(_scan, "full_scan", unexpected)
    monkeypatch.setattr(cis_map.jax, "clear_caches", unexpected)
    output = pl.concat(list(cis_map.map_cis(cast(ReadyDataState, data), test, ACAT(), verbose=False, seed=1)))
    assert output.height == 51
    assert output["num_var"].to_list() == [5] * 51
    assert output["adj_method"].to_list() == ["ACAT"] * 51
    key = jax.random.key(1)
    for i, gene in enumerate(genes):
        key, _, select_key = jax.random.split(key, 3)
        expected = cis_map._process_cis_result(gene, expected_test, expected_acat, select_key, gene_test=ACAT())
        row = output.row(i, named=True)
        assert row["snp"] == expected["snp"]
        assert row["model_converged"] == expected["model_converged"]
        np.testing.assert_allclose(row["pvalue_adj"], expected["pvalue_adj"], rtol=2e-4, atol=1e-5)


def test_blocked_acat_preserves_mixed_precision_score_dtype():
    with jax.enable_x64(True):
        X, G, y, offset = _inputs()
        X, y, offset = (a.astype(jnp.float32) for a in (X, y, offset))
        test = ScoreTest(model=LinearModel())
        expected, (expected_p, _) = cis_map.map_cis_single(X, G, y, offset, test, ACAT(), jax.random.key(1))
        actual, (actual_p, _) = _acat_scan(X, G, y, offset, test, block_size=4)
        for field in ("beta", "se", "p", "z"):
            assert getattr(actual, field).dtype == getattr(expected, field).dtype
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=1e-6, atol=1e-10)
        assert actual_p.dtype == expected_p.dtype
        np.testing.assert_allclose(actual_p, expected_p, rtol=1e-6, atol=1e-10)


def test_full_acat_map_reuses_compilations_across_window_sizes(caplog):
    import logging

    from types import SimpleNamespace
    from typing import cast

    import polars as pl

    from jaxqtl.map.data import CisData, ReadyDataState

    genes = []
    for m in (3, 5, 9, 17):
        X, G, y, offset = _inputs(m=m)
        info = pl.DataFrame(
            {"snp": [f"rs{i}" for i in range(m)], "pos": list(range(m)), "a1": ["A"] * m, "a0": ["C"] * m}
        )
        genes.append(CisData(X, G, y, offset, f"gene{m}", "22", 100, 110, info, 1, 200))
    test = ScoreTest(model=LinearModel())

    def run(gene):
        data = SimpleNamespace(iter_cis=lambda *a, **k: iter([gene]))
        return list(cis_map.map_cis(cast(ReadyDataState, data), test, ACAT(), verbose=False))

    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        run(genes[0])
        caplog.clear()
        for gene in genes[1:]:
            assert run(gene)[0].height == 1
    compilations = [r.getMessage() for r in caplog.records if "Compiling " in r.getMessage()]
    assert compilations == []


def test_padding_does_not_introduce_nans_with_debug_checks():
    X, G, y, offset = _inputs(m=3)
    with jax.debug_nans(True):
        result, (p, _) = _acat_scan(X, G, y, offset, ScoreTest(model=LinearModel()), block_size=4)
    assert np.isfinite(result.p).all()
    assert np.isfinite(p)


def test_legacy_result_processing_keeps_genotypes_on_device(monkeypatch):
    import polars as pl

    from jaxqtl.map.data import CisData

    X, G, y, offset = _inputs(m=3)
    info = pl.DataFrame({"snp": ["a", "b", "c"], "pos": [1, 2, 3], "a1": ["A"] * 3, "a0": ["C"] * 3})
    gene = CisData(X, G, y, offset, "gene", "22", 100, 110, info, 1, 200)
    result, aggregate = cis_map.map_cis_single(
        X, G, y, offset, ScoreTest(model=LinearModel()), ACAT(), jax.random.key(1)
    )
    get_snp_info = CisData.get_snp_info

    def check_device_genotypes(self, index):
        assert self.G is G
        return get_snp_info(self, index)

    monkeypatch.setattr(CisData, "get_snp_info", check_device_genotypes)
    output = cis_map._process_cis_result(gene, result, aggregate, jax.random.key(1), gene_test=ACAT())
    assert output["result_valid"]
