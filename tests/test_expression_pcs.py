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
    pcs = _prob_pca(jr.key(seed), values, 3)
    expected, singular_values, _ = jnp.linalg.svd(values, full_matrices=False)

    assert pcs.shape == (shape[0], 3)
    assert jnp.allclose(pcs.T @ pcs, jnp.eye(3), atol=2e-5)
    energy = jnp.sum((pcs.T @ values) ** 2, axis=1)
    assert jnp.all(jnp.diff(energy) <= 0)
    assert jnp.allclose(energy, singular_values[:3] ** 2, rtol=2e-4)
    # Singular-vector signs are arbitrary; each ordered direction must still agree.
    assert jnp.allclose(jnp.abs(expected[:, :3].T @ pcs), jnp.eye(3), atol=2e-3)


@pytest.mark.parametrize("k", [1, 3])
def test_compute_pcs_preserves_individuals_and_exports_k_ordered_components(k):
    expression = jnp.exp(_expression(30, 12) / 4.0 + 2.0) - 1.0
    iids = [f"donor_{i}" for i in reversed(range(30))]
    genes = [f"gene_{i}" for i in range(12)]
    pheno = pl.DataFrame({"iid": iids, **dict(zip(genes, expression.T.tolist(), strict=True))})
    data = ExpressionData(pheno, pl.DataFrame(), pl.DataFrame())

    result = data.compute_pcs(k, jr.key(1), transform="log1p")

    assert result.shape == (30, k + 1)
    assert result.columns == ["iid", *(f"ExprPC{i}" for i in range(k))]
    assert result["iid"].to_list() == iids
    pcs = result.select(pl.exclude("iid")).to_jax()
    transformed = jnp.log1p(expression)
    standardized = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0)
    expected, _, _ = jnp.linalg.svd(standardized, full_matrices=False)
    assert jnp.allclose(jnp.abs(expected[:, :k].T @ pcs), jnp.eye(k), atol=2e-3)
