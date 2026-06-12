# pattern: Functional Core

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ColumnComparison:
    column: str
    kind: str
    equal: bool
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    mismatch_count: int | None = None


@dataclass(frozen=True)
class FrameComparison:
    equal: bool
    shape_left: tuple[int, int]
    shape_right: tuple[int, int]
    columns_left: tuple[str, ...]
    columns_right: tuple[str, ...]
    column_results: tuple[ColumnComparison, ...]
    reason: str | None = None


def compare_frames(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> FrameComparison:
    """Compare two result frames while treating numeric and categorical columns appropriately."""
    shape_left = left.shape
    shape_right = right.shape
    columns_left = tuple(left.columns)
    columns_right = tuple(right.columns)
    if shape_left != shape_right:
        return FrameComparison(
            equal=False,
            shape_left=shape_left,
            shape_right=shape_right,
            columns_left=columns_left,
            columns_right=columns_right,
            column_results=(),
            reason="shape mismatch",
        )
    if columns_left != columns_right:
        return FrameComparison(
            equal=False,
            shape_left=shape_left,
            shape_right=shape_right,
            columns_left=columns_left,
            columns_right=columns_right,
            column_results=(),
            reason="column mismatch",
        )

    column_results = tuple(
        _compare_column(left.get_column(column), right.get_column(column), rtol=rtol, atol=atol)
        for column in left.columns
    )
    return FrameComparison(
        equal=all(result.equal for result in column_results),
        shape_left=shape_left,
        shape_right=shape_right,
        columns_left=columns_left,
        columns_right=columns_right,
        column_results=column_results,
    )


def comparison_to_dict(comparison: FrameComparison) -> dict[str, Any]:
    return {
        "equal": comparison.equal,
        "shape_left": comparison.shape_left,
        "shape_right": comparison.shape_right,
        "columns_left": comparison.columns_left,
        "columns_right": comparison.columns_right,
        "reason": comparison.reason,
        "columns": [
            {
                "column": result.column,
                "kind": result.kind,
                "equal": result.equal,
                "max_abs_diff": result.max_abs_diff,
                "max_rel_diff": result.max_rel_diff,
                "mismatch_count": result.mismatch_count,
            }
            for result in comparison.column_results
        ],
    }


def _compare_column(left: pl.Series, right: pl.Series, *, rtol: float, atol: float) -> ColumnComparison:
    if left.dtype.is_numeric() and right.dtype.is_numeric():
        return _compare_numeric_column(left, right, rtol=rtol, atol=atol)
    return _compare_exact_column(left, right)


def _compare_numeric_column(left: pl.Series, right: pl.Series, *, rtol: float, atol: float) -> ColumnComparison:
    left_array = left.to_numpy()
    right_array = right.to_numpy()
    close = np.isclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True)
    abs_diff = np.abs(left_array - right_array)
    denom = np.maximum(np.abs(right_array), np.finfo(float).tiny)
    rel_diff = abs_diff / denom
    finite_abs = abs_diff[np.isfinite(abs_diff)]
    finite_rel = rel_diff[np.isfinite(rel_diff)]
    return ColumnComparison(
        column=left.name,
        kind="numeric",
        equal=bool(np.all(close)),
        max_abs_diff=float(np.max(finite_abs)) if finite_abs.size else 0.0,
        max_rel_diff=float(np.max(finite_rel)) if finite_rel.size else 0.0,
        mismatch_count=int(np.size(close) - np.count_nonzero(close)),
    )


def _compare_exact_column(left: pl.Series, right: pl.Series) -> ColumnComparison:
    left_values = left.to_list()
    right_values = right.to_list()
    mismatches = sum(a != b for a, b in zip(left_values, right_values, strict=True))
    return ColumnComparison(
        column=left.name,
        kind="exact",
        equal=mismatches == 0,
        mismatch_count=mismatches,
    )
