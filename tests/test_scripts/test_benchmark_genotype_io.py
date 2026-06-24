# pattern: Mixed (unavoidable)
# Reason: These tests verify script-level Parquet comparison behavior using real
# temporary files, while script imports require adding the scripts directory.

import sys

from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_genotype_io


def _qtl_frame(**overrides: object) -> pl.DataFrame:
    values: dict[str, list[object]] = {
        "phenotype_id": ["gene1"],
        "chrom": ["22"],
        "snp": ["rs1"],
        "pos": [101],
        "a0": ["A"],
        "a1": ["G"],
        "tss_distance": [10],
        "af": [0.25],
        "ma_count": [4],
        "beta": [0.35],
        "se": [0.12],
        "pvalue": [0.01],
        "pvalue_adj": [0.02],
        "num_var": [8],
        "adj_method": ["bonferroni"],
        "model_converged": [True],
    }
    for column, value in overrides.items():
        values[column] = [value]
    return pl.DataFrame(values)


def test_compare_outputs_uses_allele_aware_mode_for_parquet_results(tmp_path: Path) -> None:
    suffix = ".nominal.score.parquet.gz"
    current_prefix = tmp_path / "current" / "jaxqtl"
    baseline_prefix = tmp_path / "baseline" / "jaxqtl"
    current_prefix.parent.mkdir()
    baseline_prefix.parent.mkdir()
    current_path = Path(str(current_prefix) + suffix)
    baseline_path = Path(str(baseline_prefix) + suffix)
    _qtl_frame(a0="A", a1="G", beta=0.35, af=0.25).write_parquet(current_path)
    _qtl_frame(a0="G", a1="A", beta=-0.35, af=0.75).write_parquet(baseline_path)

    comparisons = benchmark_genotype_io._compare_outputs(
        current_prefix=current_prefix,
        baseline_prefix=baseline_prefix,
        suffixes=(suffix,),
        rtol=1e-8,
        atol=1e-12,
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["suffix"] == suffix
    assert comparison["current"] == str(current_path)
    assert comparison["baseline"] == str(baseline_path)
    assert comparison["equal"] is True
    assert comparison["comparison_mode"] == "qtl_allele_aware"
    assert "shape_left" in comparison
    assert "shape_right" in comparison
    assert "columns_left" in comparison
    assert "columns_right" in comparison
    assert comparison["columns"]


def test_compare_outputs_preserves_missing_output_failure(tmp_path: Path) -> None:
    suffix = ".nominal.score.parquet.gz"
    current_prefix = tmp_path / "current" / "jaxqtl"
    baseline_prefix = tmp_path / "baseline" / "jaxqtl"

    comparisons = benchmark_genotype_io._compare_outputs(
        current_prefix=current_prefix,
        baseline_prefix=baseline_prefix,
        suffixes=(suffix,),
        rtol=1e-8,
        atol=1e-12,
    )

    assert comparisons == [
        {
            "suffix": suffix,
            "equal": False,
            "reason": "missing output",
            "current": str(Path(str(current_prefix) + suffix)),
            "baseline": str(Path(str(baseline_prefix) + suffix)),
        }
    ]
