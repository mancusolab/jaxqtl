# pattern: Mixed (unavoidable)
# Reason: The tests are pure in-memory comparator checks, but this repository does
# not make top-level scripts importable from pytest collection without adding the
# scripts directory to sys.path.

import sys

from pathlib import Path

import polars as pl
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import benchmark_core


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
        "beta": [0.75],
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


def _compare_qtl_frames(left: pl.DataFrame, right: pl.DataFrame):
    compare_qtl_frames = getattr(benchmark_core, "compare_qtl_frames")
    return compare_qtl_frames(left, right, rtol=1e-8, atol=1e-12)


def test_strict_comparison_passes_for_identical_frames() -> None:
    frame = pl.DataFrame(
        {
            "snp": ["rs1", "rs2"],
            "beta": [0.1, -0.2],
            "pvalue": [0.05, 0.01],
        }
    )

    comparison = benchmark_core.compare_frames(frame, frame.clone(), rtol=1e-8, atol=1e-12)

    assert comparison.equal


def test_strict_comparison_fails_for_non_allele_numeric_mismatch() -> None:
    left = _qtl_frame(se=0.12)
    right = _qtl_frame(se=0.15)

    comparison = benchmark_core.compare_frames(left, right, rtol=1e-8, atol=1e-12)

    assert not comparison.equal
    se_result = next(result for result in comparison.column_results if result.column == "se")
    assert not se_result.equal


def test_qtl_comparison_passes_for_unchanged_alleles_and_matching_numeric_columns() -> None:
    left = _qtl_frame()
    right = _qtl_frame()

    comparison = _compare_qtl_frames(left, right)

    assert comparison.equal


def test_qtl_comparison_passes_for_unchanged_alleles_and_direct_beta() -> None:
    left = _qtl_frame(beta=-0.35)
    right = _qtl_frame(beta=-0.35)

    comparison = _compare_qtl_frames(left, right)

    assert comparison.equal


def test_qtl_comparison_fails_for_unchanged_alleles_and_negated_beta() -> None:
    left = _qtl_frame(beta=0.35)
    right = _qtl_frame(beta=-0.35)

    comparison = _compare_qtl_frames(left, right)

    assert not comparison.equal


def test_qtl_comparison_passes_for_swapped_alleles_and_negated_beta() -> None:
    left = _qtl_frame(a0="A", a1="G", beta=0.35, af=0.25)
    right = _qtl_frame(a0="G", a1="A", beta=-0.35, af=0.75)

    comparison = _compare_qtl_frames(left, right)

    assert comparison.equal


def test_qtl_comparison_passes_for_swapped_float32_af_complements() -> None:
    left = _qtl_frame(a0="A", a1="G", beta=0.35)
    right = _qtl_frame(a0="G", a1="A", beta=-0.35)
    left = left.with_columns(pl.Series("af", [0.029999999329447746], dtype=pl.Float32))
    right = right.with_columns(pl.Series("af", [0.9699999690055847], dtype=pl.Float32))

    comparison = _compare_qtl_frames(left, right)

    assert comparison.equal


def test_qtl_comparison_accepts_rare_variant_beta_when_signed_wald_statistic_matches() -> None:
    left = _qtl_frame(
        a0="G",
        a1="A",
        af=0.029999999329447746,
        ma_count=6,
        beta=0.8715675273572471,
        se=667867.9794536964,
        pvalue=0.9999989587608702,
    )
    right = _qtl_frame(
        a0="A",
        a1="G",
        af=0.9699999690055847,
        ma_count=6,
        beta=-0.8786884813160236,
        se=667867.9793869716,
        pvalue=0.9999989502536509,
    )

    comparison = _compare_qtl_frames(left, right)

    assert comparison.equal


def test_qtl_comparison_fails_for_swapped_alleles_and_direct_beta() -> None:
    left = _qtl_frame(a0="A", a1="G", beta=0.35, af=0.25)
    right = _qtl_frame(a0="G", a1="A", beta=0.35, af=0.75)

    comparison = _compare_qtl_frames(left, right)

    assert not comparison.equal


@pytest.mark.parametrize(
    "column, value",
    [("snp", "rs2"), ("phenotype_id", "gene2"), ("pos", 202)],
)
def test_qtl_comparison_fails_for_identity_column_mismatches(column: str, value: object) -> None:
    left = _qtl_frame()
    right = _qtl_frame(**{column: value})

    comparison = _compare_qtl_frames(left, right)

    assert not comparison.equal


def test_qtl_comparison_fails_when_output_columns_differ() -> None:
    left = _qtl_frame()
    right = _qtl_frame().rename({"pvalue_adj": "qvalue"})

    comparison = _compare_qtl_frames(left, right)

    assert not comparison.equal
    assert comparison.reason == "column mismatch"
