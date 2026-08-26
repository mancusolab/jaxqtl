# pattern: Imperative Shell

from pathlib import Path

import numpy as np
import polars as pl
import pytest

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


def test_bed_transform_y_tmm_applies_external_transforms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _write_bed(
        tmp_path,
        "#Chr\tstart\tend\tgene\ts1\ts2\nX\t10\t20\tgene1\t1\t2\nX\t30\t40\tgene2\t0\t0\nX\t50\t60\tgene3\t3\t4\n",
    )
    captured: dict[str, np.ndarray] = {}

    def fake_edger_cpm(counts, normalized_lib_sizes=True):
        captured["counts"] = np.asarray(counts)
        return captured["counts"] * 2

    def fake_inverse_normal_transform(counts):
        counts_array = np.asarray(counts)
        # make output deterministic and row order-preserving
        return np.arange(counts_array.size, dtype=float).reshape(counts_array.shape)

    monkeypatch.setattr("jaxqtl.io._pheno.qtl.norm.edger_cpm", fake_edger_cpm)
    monkeypatch.setattr("jaxqtl.io._pheno.qtl.norm.inverse_normal_transform", fake_inverse_normal_transform)

    out = bed_transform_y(path, method="tmm")

    assert out.height == 2
    assert captured["counts"].shape == (2, 2)
    np.testing.assert_allclose(captured["counts"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(out["s1"].to_numpy(), np.array([0.0, 2.0]))
    np.testing.assert_allclose(out["s2"].to_numpy(), np.array([1.0, 3.0]))


def test_bed_transform_y_unsupported_mode_raises_error(tmp_path: Path) -> None:
    path = _write_bed(
        tmp_path,
        "#chr\tstart\tend\tgene\ts1\n1\t100\t200\tgene1\t1\n",
    )

    with pytest.raises(ValueError, match="Unsupported mode"):
        bed_transform_y(path, method="not-real")
