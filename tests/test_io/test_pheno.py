# pattern: Imperative Shell

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import qtl.norm

import jax
import jax.numpy as jnp

from jaxqtl.io._normalization import edger_cpm, inverse_normal_transform
from jaxqtl.io._pheno import bed_transform_y, ExpressionData


def _write_bed(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "phenotypes.bed"
    path.write_text(body)
    return path


def test_expression_data_filters_genes_by_chromosome_and_preserves_libsize() -> None:
    expression = ExpressionData(
        pheno=pl.DataFrame(
            {
                "iid": ["sample1", "sample2"],
                "gene1": [1.0, 2.0],
                "gene2": [3.0, 4.0],
                "gene3": [5.0, 6.0],
            }
        ),
        pheno_meta=pl.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr1"],
                "start": [10, 20, 30],
                "end": [11, 21, 31],
                "phenotype_id": ["gene1", "gene2", "gene3"],
            }
        ),
        libsize=pl.DataFrame({"iid": ["sample1", "sample2"], "libsize": [9.0, 12.0]}),
    )

    filtered = expression.filter_genes_by_chromosomes({"chr2"})

    assert filtered.pheno.columns == ["iid", "gene2"]
    assert filtered.pheno_meta.get_column("phenotype_id").to_list() == ["gene2"]
    assert filtered.pheno_meta.get_column("chrom").to_list() == ["chr2"]
    assert filtered.libsize.equals(expression.libsize)


def test_bed_transform_y_log1p_filters_zero_rows_and_transforms_columns(tmp_path: Path) -> None:
    path = _write_bed(
        tmp_path,
        "#chr\tstart\tend\tgene\ts1\ts2\n"
        "1\t100\t200\tgene1\t0\t1\n"
        "1\t100\t200\tgene2\t0\t0\n"
        "1\t300\t400\tgene3\t2\t5\n",
    )

    out = bed_transform_y(path)
    assert out.columns == ["#chr", "start", "end", "gene", "s1", "s2"]
    assert out.height == 2
    assert out["#chr"].to_list() == ["1", "1"]

    np.testing.assert_allclose(out["s1"].to_numpy(), np.log1p(np.array([0.0, 2.0])))
    np.testing.assert_allclose(out["s2"].to_numpy(), np.log1p(np.array([1.0, 5.0])))


def test_jax_normalization_matches_qtl_norm_under_jit() -> None:
    counts = np.array(
        [
            [0.0, 20.0, 30.0],
            [100.0, 120.0, 80.0],
            [5.0, 7.0, 11.0],
            [60.0, 45.0, 90.0],
            [25.0, 35.0, 20.0],
            [40.0, 50.0, 65.0],
        ]
    )
    expected_cpm = qtl.norm.edger_cpm(pd.DataFrame(counts), normalized_lib_sizes=True)
    expected = np.asarray(qtl.norm.inverse_normal_transform(expected_cpm))

    actual_cpm = jax.jit(edger_cpm)(jnp.asarray(counts))
    actual = jax.jit(inverse_normal_transform)(actual_cpm)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_bed_transform_y_tmm_matches_qtl_norm(tmp_path: Path) -> None:
    path = _write_bed(
        tmp_path,
        "#Chr\tstart\tend\tgene\ts1\ts2\nX\t10\t20\tgene1\t1\t2\nX\t30\t40\tgene2\t0\t0\nX\t50\t60\tgene3\t3\t4\n",
    )
    out = bed_transform_y(path, method="tmm")

    assert out.height == 2
    expected_cpm = qtl.norm.edger_cpm(
        pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])),
        normalized_lib_sizes=True,
    )
    expected = np.asarray(qtl.norm.inverse_normal_transform(expected_cpm))
    np.testing.assert_allclose(out[["s1", "s2"]].to_numpy(), expected, rtol=1e-5, atol=1e-5)


def test_bed_transform_y_unsupported_mode_raises_error(tmp_path: Path) -> None:
    path = _write_bed(
        tmp_path,
        "#chr\tstart\tend\tgene\ts1\n1\t100\t200\tgene1\t1\n",
    )

    with pytest.raises(ValueError, match="Unsupported mode"):
        bed_transform_y(path, method="not-real")
