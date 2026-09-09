# pattern: Functional Core
"""Class-based execution preserves statistics and compilation boundaries."""

import logging

import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxqtl.distribution import NegativeBinomial, Poisson
from jaxqtl.hypothesis import ACAT, BetaPermutation, GaussianCGF, ScoreTest, SpaTest, WaldTest
from jaxqtl.hypothesis._aggregate import BetaCalibration
from jaxqtl.infer import BetaParams, GeneralizedLinearModel, HuberError, LinearModel
from jaxqtl.map import _scan


def _inputs(m=7) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.key(12), 3)
    X = jnp.column_stack((jnp.ones(40), jax.random.normal(keys[0], (40,))))
    G = jax.random.normal(keys[1], (40, m))
    y = 0.2 * X[:, 1] + jax.random.normal(keys[2], (40,))
    return X, G, y, jnp.linspace(-0.1, 0.1, 40)


def _wald_inputs(m=7, *, counts=False, vector_offset=True) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    X, G, y, offset = _inputs(m)
    if not vector_offset:
        offset = jnp.asarray(0.15)
    if counts:
        gamma_key, count_key = jax.random.split(jax.random.key(413))
        mu = jnp.exp(1.0 + 0.2 * X[:, 1] + 0.15 * G[:, 0] + offset)
        rate = jax.random.gamma(gamma_key, 2.0, shape=y.shape) * mu / 2.0
        y = jax.random.poisson(count_key, rate).astype(X.dtype)
    return X, G, y, offset


def _nominal_gene(m, dtype=jnp.float32):
    import polars as pl

    from jaxqtl.map.data import CisData

    X, _, y, offset = _inputs(m)
    genotype = 2.0 * jax.random.uniform(jax.random.key(414), (len(y), m))
    genotype = jnp.clip(genotype + 0.7 * jnp.linspace(-1.0, 1.0, m), 0.0, 2.0).astype(dtype)
    info = pl.DataFrame(
        {
            "chrom": ["22"] * m,
            "snp": [f"rs{i}" for i in range(m)],
            "pos": list(range(101, 101 + m)),
            "a1": ["A"] * m,
            "a0": ["C"] * m,
        },
        schema={"chrom": pl.String, "snp": pl.String, "pos": pl.Int64, "a1": pl.String, "a0": pl.String},
    )
    return CisData(X, jax.device_get(genotype), y, offset, "gene1", "22", 100, 110, info, 1, 3000)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.int32])
@pytest.mark.parametrize("width", [0, 2051], ids=["empty", "full-and-partial-blocks"])
def test_cis_allele_metadata_preserves_values_dtypes_and_empty_windows(dtype, width):
    import polars as pl

    with jax.enable_x64(True):
        gene = _nominal_gene(width, dtype)
        counts = jnp.sum(gene.G, axis=0)
        expected_af = counts / (2.0 * len(gene.y))
        expected_mac = jnp.where(expected_af <= 0.5, counts, 2 * len(gene.y) - counts)
        actual = gene.get_cis_info()
        assert actual.height == width
        assert actual.schema["af"] == pl.Series(jax.device_get(expected_af)).dtype
        assert actual.schema["ma_count"] == pl.Int64
        assert jnp.allclose(actual["af"].to_jax(), expected_af, rtol=2e-6, atol=1e-7)
        assert actual["ma_count"].to_list() == expected_mac.astype(jnp.int64).tolist()
        assert actual["tss_distance"].to_list() == list(range(width))


def test_nominal_mapping_reuses_compilation_and_preserves_allele_metadata(caplog):
    from types import SimpleNamespace
    from typing import cast

    import polars as pl

    from jaxqtl.map.cis import map_cis
    from jaxqtl.map.data import ReadyDataState

    genes = [_nominal_gene(m) for m in (3, 7, 2051)]
    data = [
        cast(ReadyDataState, SimpleNamespace(iter_cis=lambda *args, gene=gene, **kwargs: iter([gene])))
        for gene in genes
    ]
    test = WaldTest(LinearModel())
    aggregation = ACAT()
    outputs = []
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        list(map_cis(data[0], test, aggregation, mode="nominal", verbose=False))
        caplog.clear()
        for dataset in data[1:]:
            outputs.append(pl.concat(list(map_cis(dataset, test, aggregation, mode="nominal", verbose=False))))
    assert not [record.getMessage() for record in caplog.records if "Compiling " in record.getMessage()]
    for gene, output in zip(genes[1:], outputs, strict=True):
        counts = jnp.sum(gene.G, axis=0)
        expected_af = counts / (2.0 * len(gene.y))
        expected_mac = jnp.where(expected_af <= 0.5, counts, 2 * len(gene.y) - counts)
        assert jnp.allclose(output["af"].to_jax(), expected_af, rtol=2e-6, atol=1e-7)
        assert output["ma_count"].to_list() == expected_mac.astype(jnp.int32).tolist()


@pytest.mark.parametrize(
    "test", [ScoreTest(LinearModel()), SpaTest(LinearModel(), cgf=GaussianCGF()), WaldTest(LinearModel())]
)
def test_scan_uses_concrete_initialization_and_matches_direct_acat(test):
    X, G, y, offset = _inputs()
    scan = _scan.AssociationScan(test, ACAT(), block_size=4)
    observed = eqx.filter_jit(test)(X, G, y, offset)
    expected = eqx.filter_jit(ACAT())(X, G, y, offset, observed, test, jax.random.key(1))
    result, aggregate = scan.run(X, G, y, offset, jax.random.key(1))
    for actual, reference in zip(
        jax.tree.leaves((result, aggregate)), jax.tree.leaves((observed, expected)), strict=True
    ):
        assert jnp.allclose(actual, reference, rtol=2e-4, atol=1e-5, equal_nan=True)
    assert scan.blocked
    assert scan.cache_clear_interval is None


@pytest.mark.parametrize(
    "test,counts",
    [
        (SpaTest(LinearModel(), cgf=GaussianCGF()), False),
        (WaldTest(LinearModel()), False),
        (WaldTest(GeneralizedLinearModel(family=Poisson())), True),
        (WaldTest(GeneralizedLinearModel(family=NegativeBinomial())), True),
    ],
    ids=["spa-lm", "wald-lm", "wald-poisson", "wald-nb"],
)
def test_block_scan_reuses_compilation_for_new_window_sizes(caplog, test, counts):
    inputs = [_wald_inputs(m, counts=counts) for m in (3, 7, 11)]
    scan = _scan.AssociationScan(test, ACAT(), block_size=4)
    key = jax.random.key(1)
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        jax.block_until_ready(scan.run(*inputs[0], key=key))
        caplog.clear()
        for args in inputs[1:]:
            jax.block_until_ready(scan.run(*args, key=key))
    assert not [record.getMessage() for record in caplog.records if "Compiling " in record.getMessage()]


@pytest.mark.parametrize(
    "test,counts",
    [
        (WaldTest(LinearModel()), False),
        (WaldTest(LinearModel(), std_err=HuberError()), False),
        (WaldTest(GeneralizedLinearModel(family=Poisson())), True),
        (WaldTest(GeneralizedLinearModel(family=NegativeBinomial())), True),
    ],
    ids=["lm-fisher", "lm-huber", "poisson", "nb"],
)
@pytest.mark.parametrize("vector_offset", [False, True], ids=["scalar-offset", "vector-offset"])
def test_wald_blocks_preserve_every_variant_statistic(test, counts, vector_offset):
    inputs = _wald_inputs(counts=counts, vector_offset=vector_offset)
    expected = eqx.filter_jit(test)(*inputs)
    scan = _scan.AssociationScan(test, ACAT(), block_size=4)
    actual, _ = scan.observed(*inputs)
    assert actual.z.shape == (7,)
    assert jnp.all(jnp.isfinite(actual.p))
    for name, result, reference in zip(actual._fields, actual, expected, strict=True):
        if name in ("num_iters", "converged"):
            assert jnp.array_equal(result, reference), name
        else:
            assert jnp.allclose(result, reference, rtol=2e-4, atol=2e-5), name


@pytest.mark.parametrize(
    "test,counts",
    [
        (WaldTest(LinearModel()), False),
        (WaldTest(GeneralizedLinearModel(family=Poisson())), True),
        (WaldTest(GeneralizedLinearModel(family=NegativeBinomial())), True),
    ],
    ids=["lm", "poisson", "nb"],
)
def test_wald_permutation_maxima_preserve_rng_order_across_blocks_and_batches(test, counts):
    inputs = _wald_inputs(counts=counts)
    permutations = BetaPermutation(max_perm_direct=5)
    key = jax.random.key(8)
    expected = eqx.filter_jit(permutations._run_permutations)(*inputs, test, key)
    scan = _scan.AssociationScan(test, permutations, block_size=4, batch_size=3)
    actual = scan.permutation_maxima(*inputs, key)
    assert actual.shape == (5,)
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_wald_glm_permutation_kernels_reuse_compilation_for_new_window_sizes(caplog):
    inputs = [_wald_inputs(m, counts=True) for m in (3, 7, 11)]
    test = WaldTest(GeneralizedLinearModel(family=NegativeBinomial()))
    scan = _scan.AssociationScan(test, BetaPermutation(max_perm_direct=5), block_size=4, batch_size=3)
    key = jax.random.key(8)
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        # The first gene warms both the full batch of three and final batch of two permutations.
        jax.block_until_ready(scan.permutation_maxima(*inputs[0], key))
        caplog.clear()
        for args in inputs[1:]:
            jax.block_until_ready(scan.permutation_maxima(*args, key))
    assert not [record.getMessage() for record in caplog.records if "Compiling " in record.getMessage()]


def test_wald_permutation_maxima_exclude_an_extreme_padded_snp():
    X, G, y, offset = _inputs(3)
    test = WaldTest(LinearModel())
    _, states = _scan._initialize_permutations(test, X, y, offset, jax.random.key(8), 1)
    state = jax.tree.map(lambda value: value[0], states)
    # A padded SNP that predicts the permuted residual almost perfectly must not dominate the real SNPs.
    padded = jnp.column_stack((G, state.resid))
    all_stats = eqx.filter_jit(test.test)(X, padded, state).z
    assert jnp.abs(all_stats[-1]) > 100 * jnp.max(jnp.abs(all_stats[:-1]))
    actual = _scan._permutation_block_maxima(test, X, padded, states, jnp.asarray(3))
    assert jnp.allclose(actual, jnp.max(jnp.abs(all_stats[:-1])), rtol=2e-5, atol=1e-5)


def test_wald_beta_calibrated_scan_matches_full_window():
    with jax.enable_x64(True):
        inputs = _wald_inputs()
        test = WaldTest(LinearModel())
        permutations = BetaPermutation(max_perm_direct=64)
        key = jax.random.key(12)
        expected = _scan.full_scan(*inputs, test, permutations, key)
        scan = _scan.AssociationScan(test, permutations, block_size=4, batch_size=7)
        actual = scan.run(*inputs, key)
        assert scan.blocked
        assert jnp.all(jnp.isfinite(actual[1][0]))
        for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert jnp.allclose(result, reference, rtol=2e-5, atol=1e-7)


def test_spa_permutations_preserve_underlying_score_maxima():
    X, G, y, offset = _inputs()
    test = SpaTest(LinearModel(), cgf=GaussianCGF())
    permutations = BetaPermutation(max_perm_direct=5)
    key = jax.random.key(8)
    expected = eqx.filter_jit(permutations._run_permutations)(X, G, y, offset, test, key)
    scan = _scan.AssociationScan(test, permutations, block_size=4, batch_size=3)
    actual = scan.permutation_maxima(X, G, y, offset, key)
    assert jnp.allclose(actual, expected, rtol=2e-5, atol=1e-5)


def test_lead_selection_preserves_finite_filtering_and_cross_block_ties():
    p = jnp.full(2050, jnp.nan).at[2].set(0.01).at[2049].set(0.01).at[100].set(jnp.inf)
    key = jax.random.key(4)
    expected = int(jax.random.choice(key, jnp.array([2, 2049]), replace=False))
    assert _scan.select_lead_variant(p, key) == expected
    assert _scan.select_lead_variant(jnp.array([jnp.nan, jnp.inf]), key) is None


@pytest.mark.parametrize("mode", ["cis", "nominal"])
def test_wald_mapping_uses_fixed_blocks_without_clearing_caches(monkeypatch, mode):
    from types import SimpleNamespace
    from typing import cast

    import polars as pl

    from jaxqtl.map import cis as cis_map
    from jaxqtl.map.data import CisData, ReadyDataState

    genes = []
    for i in range(41):
        X, G, y, offset = _inputs(3 if i % 2 else 5)
        m = G.shape[1]
        info = pl.DataFrame(
            {
                "snp": [f"rs{j}" for j in range(m)],
                "chrom": ["22"] * m,
                "pos": list(range(m)),
                "a1": ["A"] * m,
                "a0": ["C"] * m,
            }
        )
        genes.append(CisData(X, G, y, offset, f"gene{i}", "22", 100, 110, info, 1, 200))
    widths, cleared, ingress_options = [], [], []
    original = _scan._test_block

    def block(test, X, G, *args):
        widths.append(G.shape[1])
        return original(test, X, G, *args)

    def unexpected(*args, **kwargs):
        pytest.fail("Wald mapping must use fixed genotype blocks")

    def iter_cis(*args, **kwargs):
        ingress_options.append(kwargs)
        return iter(genes)

    monkeypatch.setattr(_scan, "full_scan", unexpected)
    monkeypatch.setattr(_scan, "_test_block", block)
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(len(widths)))
    data = cast(ReadyDataState, SimpleNamespace(iter_cis=iter_cis))
    output = pl.concat(list(cis_map.map_cis(data, WaldTest(LinearModel()), ACAT(), mode=mode, verbose=False)))
    assert output.height == (41 if mode == "cis" else sum(gene.num_snps for gene in genes))
    assert len(widths) >= 41
    assert len(set(widths)) == 1
    assert ingress_options == [{"host_genotypes": True}]
    assert not cleared


def test_acat_finalize_override_agrees_between_full_and_blocked_execution():
    class AdjustedACAT(ACAT):
        def finalize(self, state):
            return 0.5 * super().finalize(state)

    X, G, y, offset = _inputs()
    test = ScoreTest(LinearModel())
    aggregation = AdjustedACAT()
    key = jax.random.key(1)
    _, expected = _scan.full_scan(X, G, y, offset, test, aggregation, key)
    _, actual = _scan.AssociationScan(test, aggregation, block_size=4).run(X, G, y, offset, key)
    assert jnp.allclose(actual[0], expected[0], rtol=2e-5, atol=1e-6)


def test_acat_update_uses_the_configured_instance_method():
    class ScaledContributions(ACAT):
        def update(self, state, block):
            return super().update(state, block._replace(statistic=2.0 * block.statistic))

    X, G, y, offset = _inputs()
    aggregation = ScaledContributions()
    scan = _scan.AssociationScan(ScoreTest(LinearModel()), aggregation, block_size=4)
    result, actual = scan.observed(X, G, y, offset, reduction=aggregation)
    components = aggregation.components(result.p, jnp.ones_like(result.p, dtype=bool), jnp.asarray(1.0 / len(result.p)))
    expected = aggregation.update(aggregation.init(result.p.dtype), components)
    assert jnp.allclose(actual.statistic, expected.statistic, rtol=2e-5, atol=1e-6)


def test_permutation_calibration_override_agrees_between_full_and_blocked_execution():
    class FixedCalibration(BetaPermutation):
        def fit_calibration(self, z_stats_perm, dof):
            return BetaCalibration(
                BetaParams(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(True)),
                jnp.asarray(0.25, dtype=z_stats_perm.dtype),
                jnp.asarray(True),
            )

    X, G, y, offset = _inputs()
    test = ScoreTest(LinearModel())
    aggregation = FixedCalibration(max_perm_direct=64)
    key = jax.random.key(1)
    _, expected = _scan.full_scan(X, G, y, offset, test, aggregation, key)
    _, actual = _scan.AssociationScan(test, aggregation, block_size=4, batch_size=32).run(X, G, y, offset, key)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.allclose(observed, reference, rtol=2e-5, atol=1e-6)


def test_permutation_adjustment_override_agrees_between_full_and_blocked_arrays():
    class AdjustedPermutation(BetaPermutation):
        def adjust(self, z, calibration):
            return 0.25 + 0.5 * super().adjust(z, calibration)

    aggregation = AdjustedPermutation()
    calibration = BetaCalibration(
        BetaParams(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(True)),
        jnp.asarray(0.25),
        jnp.asarray(True),
    )
    z = jnp.linspace(-3.0, 3.0, 7)
    expected = eqx.filter_jit(aggregation.adjust)(z, calibration)
    actual = _scan.AssociationScan(ScoreTest(LinearModel()), aggregation, block_size=4).adjust(z, calibration)
    assert jnp.allclose(actual, expected, rtol=2e-5, atol=1e-6)


def test_map_cis_single_preserves_public_keyword_arguments():
    from jaxqtl.map.cis import map_cis_single

    X, G, y, offset = _inputs()
    test, aggregation, key = ScoreTest(LinearModel()), ACAT(), jax.random.key(1)
    expected = map_cis_single(X, G, y, offset, test, aggregation, key)
    actual = map_cis_single(X=X, G=G, y=y, offset=offset, snp_test=test, gene_test=aggregation, key=key)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.array_equal(observed, reference, equal_nan=True)


def test_permutation_scoring_finishes_each_block_before_requesting_the_next(monkeypatch):
    X, G, y, offset = _inputs(11)
    scan = _scan.AssociationScan(
        ScoreTest(LinearModel()), BetaPermutation(max_perm_direct=5), block_size=4, batch_size=3
    )
    pending, completed = [], []
    original_blocks = _scan._HostBuffers.blocks
    original_merge = _scan._merge_maxima
    original_ready = jax.block_until_ready

    def guarded_blocks(buffers, block_size):
        blocks = original_blocks(buffers, block_size)
        while True:
            assert not pending, "The preceding SNP block must finish before another is transferred"
            try:
                block = next(blocks)
            except StopIteration:
                return
            yield block

    def track_merge(previous, values):
        result = original_merge(previous, values)
        pending.append(result)
        return result

    def track_ready(value):
        result = original_ready(value)
        if pending and value is pending[0]:
            completed.append(pending.pop())
        return result

    monkeypatch.setattr(_scan._HostBuffers, "blocks", guarded_blocks)
    monkeypatch.setattr(_scan, "_merge_maxima", track_merge)
    monkeypatch.setattr(jax, "block_until_ready", track_ready)
    actual = scan.permutation_maxima(X, G, y, offset, jax.random.key(1))
    assert actual.shape == (5,)
    assert len(completed) == 6  # Three SNP blocks in each of two permutation batches.
    assert not pending
