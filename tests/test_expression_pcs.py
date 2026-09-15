# pattern: Imperative Shell

import polars as pl
import pytest

import jax.numpy as jnp
import jax.random as jr

from jaxqtl.io._pheno import _prob_pca, ExpressionData


def _expression(n, p):
    left_key, right_key = jr.split(jr.key(10))
    left, _ = jnp.linalg.qr(jr.normal(left_key, (n, 5)))
    right, _ = jnp.linalg.qr(jr.normal(right_key, (p, 5)))
    values = (left * jnp.array([20.0, 10.0, 5.0, 1.0, 0.5])) @ right.T
    return values - values.mean(axis=0)


@pytest.mark.parametrize("shape", [(30, 50), (50, 30)])
@pytest.mark.parametrize("seed", [0, 1])
def test_prob_pca_returns_ordered_sample_principal_directions(shape, seed):
    values = _expression(*shape)
    pcs, fitted_singular_values = _prob_pca(jr.key(seed), values, 3)
    expected, singular_values, _ = jnp.linalg.svd(values, full_matrices=False)

    assert pcs.shape == (shape[0], 3)
    assert jnp.allclose(pcs.T @ pcs, jnp.eye(3), atol=2e-5)
    energy = jnp.sum((pcs.T @ values) ** 2, axis=1)
    assert jnp.all(jnp.diff(energy) <= 0)
    assert jnp.allclose(energy, singular_values[:3] ** 2, rtol=2e-4)
    assert jnp.allclose(fitted_singular_values, singular_values[:3], rtol=2e-4)
    # Singular-vector signs are arbitrary; each ordered direction must still agree.
    assert jnp.allclose(jnp.abs(expected[:, :3].T @ pcs), jnp.eye(3), atol=2e-3)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("transform", [None, "log1p", "lognorm"])
def test_compute_pcs_preserves_individuals_and_exports_k_ordered_components(k, transform):
    expression = jnp.exp(_expression(30, 12) / 4.0 + 2.0) - 1.0
    iids = [f"donor_{i}" for i in reversed(range(30))]
    genes = [f"gene_{i}" for i in range(12)]
    pheno = pl.DataFrame({"iid": iids, **dict(zip(genes, expression.T.tolist(), strict=True))})
    # Stored totals can include genes excluded from PCA, and arrive in a different order.
    library_sizes = jnp.linspace(100.0, 10000.0, 30)
    libsize = pl.DataFrame({"iid": iids, "libsize": library_sizes.tolist()}).reverse()
    data = ExpressionData(pheno, pl.DataFrame(), libsize)

    result, explained_variance_ratio = data.compute_pcs(k, jr.key(1), transform=transform)

    assert result.shape == (30, k + 1)
    assert result["iid"].to_list() == iids
    pcs = result.select(pl.exclude("iid")).to_jax()
    if transform == "lognorm":
        transformed = jnp.log1p(expression * jnp.median(library_sizes) / library_sizes[:, None])
    elif transform == "log1p":
        transformed = jnp.log1p(expression)
    else:
        transformed = expression
    standardized = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0)
    expected, singular_values, _ = jnp.linalg.svd(standardized, full_matrices=False)
    assert jnp.allclose(jnp.abs(expected[:, :k].T @ pcs), jnp.eye(k), atol=2e-3)
    assert result.columns == ["iid", *(f"ExprPC{i}" for i in range(1, k + 1))]
    assert explained_variance_ratio.shape == (k,)
    assert jnp.allclose(explained_variance_ratio, singular_values[:k] ** 2 / jnp.sum(singular_values**2), rtol=2e-4)
    assert jnp.all(jnp.diff(explained_variance_ratio) <= 0)
    assert 0 < explained_variance_ratio.sum() < 1


@pytest.mark.parametrize("size", [0.0, -1.0, float("nan"), float("inf"), None])
def test_lognorm_rejects_invalid_library_sizes(size):
    data = ExpressionData(
        pl.DataFrame({"iid": ["a", "b", "c"], "gene": [1.0, 2.0, 3.0]}),
        pl.DataFrame(),
        pl.DataFrame({"iid": ["a", "b", "c"], "libsize": [size, 10.0, 20.0]}),
    )
    with pytest.raises(ValueError, match="library sizes"):
        data.compute_pcs(1, jr.key(1), transform="lognorm")


def test_lognorm_rejects_missing_library_size():
    data = ExpressionData(
        pl.DataFrame({"iid": ["a", "b", "c"], "gene": [1.0, 2.0, 3.0]}),
        pl.DataFrame(),
        pl.DataFrame({"iid": ["b", "c"], "libsize": [10.0, 20.0]}),
    )
    with pytest.raises(ValueError, match="library sizes"):
        data.compute_pcs(1, jr.key(1), transform="lognorm")


def _small_data(values, iids=None):
    values = jnp.asarray(values)
    pheno = pl.DataFrame(
        {
            "iid": iids if iids is not None else [f"s{i}" for i in range(values.shape[0])],
            **{f"g{i}": col.tolist() for i, col in enumerate(values.T)},
        }
    )
    return ExpressionData(pheno, pl.DataFrame(), pl.DataFrame())


def test_compute_pcs_removes_constant_genes_before_standardizing():
    data = _small_data([[1.0, 2.0, 7.0], [4.0, 1.0, 7.0], [2.0, 5.0, 7.0], [8.0, 3.0, 7.0]])
    pcs, ratios = data.compute_pcs(2, jr.key(1))
    assert jnp.all(jnp.isfinite(pcs.select(pl.exclude("iid")).to_jax()))
    assert ratios.sum() == pytest.approx(1.0, abs=1e-5)


def test_compute_pcs_filters_genes_constant_after_library_normalization():
    data = _small_data([[2.0, 2.0], [4.0, 3.0], [8.0, 4.0], [16.0, 3.0]])
    data.libsize = pl.DataFrame({"iid": ["s0", "s1", "s2", "s3"], "libsize": [1.0, 2.0, 4.0, 8.0]})
    with pytest.raises(ValueError, match="variable genes"):
        data.compute_pcs(2, jr.key(1), transform="lognorm")


@pytest.mark.parametrize("n", [3, 7, 30])
def test_log_transform_of_constant_expression_is_rejected(n):
    with pytest.raises(ValueError, match="variable genes"):
        _small_data([[7.0]] * n).compute_pcs(1, jr.key(1), transform="log1p")


@pytest.mark.parametrize(
    "values,k",
    [
        ([[1.0, 2.0]], 1),
        ([[1.0, 2.0], [2.0, 3.0]], 2),
        ([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], 2),
        ([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], 1),
    ],
)
def test_compute_pcs_rejects_unusable_dimensions(values, k):
    with pytest.raises(ValueError, match="samples|variable genes|num_pcs"):
        _small_data(values).compute_pcs(k, jr.key(1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_compute_pcs_rejects_nonfinite_expression(value):
    with pytest.raises(ValueError, match="finite"):
        _small_data([[1.0], [value], [3.0]]).compute_pcs(1, jr.key(1))


@pytest.mark.parametrize("transform", ["log1p", "lognorm"])
def test_compute_pcs_rejects_negative_counts_for_log_transforms(transform):
    with pytest.raises(ValueError, match="nonnegative"):
        _small_data([[1.0], [-0.5], [3.0]]).compute_pcs(1, jr.key(1), transform=transform)


@pytest.mark.parametrize("iids", [["a", "a", "b"], ["a", None, "b"]])
def test_compute_pcs_rejects_invalid_sample_ids(iids):
    with pytest.raises(ValueError, match="sample IDs"):
        _small_data([[1.0], [2.0], [3.0]], iids).compute_pcs(1, jr.key(1))
