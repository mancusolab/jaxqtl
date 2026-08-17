# pattern: Functional Core

import math

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import polars as pl

from scipy import sparse


MAX_EXACT_FLOAT64_INT = 2**53

SparseCopyEvent: TypeAlias = Literal[
    "npz_materialized",
    "sparse_family_normalized",
    "csc_to_csr",
    "canonicalized",
]


@dataclass(frozen=True, slots=True)
class SparseSingleCellData:
    r"""Canonical sparse counts and matrix-ordered single-cell metadata.

    **Arguments:**

    counts
        Integer-valued CSR counts with cells on rows and genes on columns.
    cell_metadata
        Cell metadata sorted by ``matrix_index``.
    gene_metadata
        Gene metadata sorted by ``matrix_index`` with canonical chromosomes.
    cell_type_column
        Name of the selected cell-type metadata column.
    cell_ids
        Cell identifiers in count-matrix row order.
    gene_ids
        Gene identifiers in count-matrix column order.
    donor_ids
        Donor identifiers in count-matrix row order.
    cell_types
        Cell-type values in count-matrix row order.
    gene_chromosomes
        Canonical chromosome labels in count-matrix column order.
    copy_events
        Semantic normalization/materialization events. These events do not
        promise object or buffer identity.
    """

    counts: sparse.csr_array
    cell_metadata: pl.DataFrame
    gene_metadata: pl.DataFrame
    cell_type_column: str
    cell_ids: np.ndarray
    gene_ids: np.ndarray
    donor_ids: np.ndarray
    cell_types: np.ndarray
    gene_chromosomes: np.ndarray
    copy_events: tuple[SparseCopyEvent, ...]


def _validate_stored_counts(values: np.ndarray, *, context: str) -> None:
    dtype = values.dtype
    if np.issubdtype(dtype, np.bool_):
        raise TypeError("sparse counts cannot use boolean storage")
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        raise TypeError(f"sparse counts must use real integer or floating storage, got {dtype}")
    if np.issubdtype(dtype, np.floating):
        if not np.isfinite(values).all():
            raise ValueError(f"{context} count values must be finite")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{context} count values must be integer-valued")
    elif np.issubdtype(dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError(f"{context} count values cannot be negative")
    if np.any(values > MAX_EXACT_FLOAT64_INT):
        raise ValueError(f"{context} count values must be at most 2**53 for exact float64 conversion")


def _checked_group_sum(values: np.ndarray, *, source_dtype: np.dtype, coordinate: tuple[int, int]) -> int:
    if np.issubdtype(source_dtype, np.integer):
        total = sum(int(value) for value in values)
        dtype_info = np.iinfo(source_dtype)
        if total < dtype_info.min or total > dtype_info.max:
            raise ValueError(f"duplicate count sum at {coordinate} is outside source dtype {source_dtype}")
    else:
        float_total = float(math.fsum(float(value) for value in values))
        if not math.isfinite(float_total) or not float_total.is_integer():
            raise ValueError(f"duplicate count sum at {coordinate} must be a finite exact integer")
        source_value = np.asarray(float_total, dtype=source_dtype).item()
        if float(source_value) != float_total:
            raise ValueError(
                f"duplicate count sum at {coordinate} is not exactly representable in source dtype {source_dtype}"
            )
        total = int(float_total)
    if not 0 <= total <= MAX_EXACT_FLOAT64_INT:
        raise ValueError(f"duplicate count sum at {coordinate} exceeds the 2**53 exact float64 integer bound")
    return total


def _consolidate_compressed(counts, *, is_csr: bool, output_dtype: np.dtype) -> sparse.csr_array:
    major_size = counts.shape[0] if is_csr else counts.shape[1]
    source_dtype = counts.data.dtype
    consolidated_indices: list[int] = []
    consolidated_values: list[int] = []
    consolidated_indptr = [0]

    for major_index in range(major_size):
        start = int(counts.indptr[major_index])
        stop = int(counts.indptr[major_index + 1])
        segment_indices = np.asarray(counts.indices[start:stop])
        segment_values = np.asarray(counts.data[start:stop])
        order = np.argsort(segment_indices, kind="stable")
        sorted_indices = segment_indices[order]
        sorted_values = segment_values[order]

        group_start = 0
        while group_start < len(sorted_indices):
            minor_index = int(sorted_indices[group_start])
            group_stop = group_start + 1
            while group_stop < len(sorted_indices) and sorted_indices[group_stop] == minor_index:
                group_stop += 1
            coordinate = (major_index, minor_index) if is_csr else (minor_index, major_index)
            value = _checked_group_sum(
                sorted_values[group_start:group_stop],
                source_dtype=source_dtype,
                coordinate=coordinate,
            )
            consolidated_indices.append(minor_index)
            consolidated_values.append(value)
            group_start = group_stop
        consolidated_indptr.append(len(consolidated_indices))

    compressed_parts = (
        np.asarray(consolidated_values, dtype=output_dtype),
        np.asarray(consolidated_indices, dtype=np.int64),
        np.asarray(consolidated_indptr, dtype=np.int64),
    )
    if is_csr:
        return sparse.csr_array(compressed_parts, shape=counts.shape)
    return sparse.csc_array(compressed_parts, shape=counts.shape).tocsr()


def _canonical_counts(counts) -> tuple[sparse.csr_array, tuple[SparseCopyEvent, ...]]:
    is_csr = sparse.isspmatrix_csr(counts) or isinstance(counts, sparse.csr_array)
    is_csc = sparse.isspmatrix_csc(counts) or isinstance(counts, sparse.csc_array)
    if not (is_csr or is_csc):
        raise TypeError("counts must be a SciPy CSR or CSC sparse array or matrix")
    if counts.ndim != 2:
        raise ValueError("counts must be two-dimensional with rows as cells and columns as genes")
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("counts cannot have an empty cell or gene axis")

    values = np.asarray(counts.data)
    _validate_stored_counts(values, context="stored")
    output_dtype = np.dtype(np.int64) if np.issubdtype(values.dtype, np.floating) else values.dtype

    events: list[SparseCopyEvent] = []
    if sparse.isspmatrix(counts):
        events.append("sparse_family_normalized")
    if is_csc:
        events.append("csc_to_csr")
    needs_canonicalization = not counts.has_canonical_format or np.issubdtype(values.dtype, np.floating)
    if needs_canonicalization:
        events.append("canonicalized")

    if counts.has_canonical_format:
        compressed_parts = (
            values.astype(output_dtype, copy=True),
            np.asarray(counts.indices).copy(),
            np.asarray(counts.indptr).copy(),
        )
        if is_csr:
            canonical = sparse.csr_array(compressed_parts, shape=counts.shape)
        else:
            canonical = sparse.csc_array(compressed_parts, shape=counts.shape).tocsr()
    else:
        canonical = _consolidate_compressed(counts, is_csr=is_csr, output_dtype=output_dtype)

    _validate_stored_counts(np.asarray(canonical.data), context="canonicalized")
    if not np.issubdtype(canonical.dtype, np.integer):
        raise ValueError("canonicalized sparse counts must retain integer storage")
    if not canonical.has_canonical_format:
        raise ValueError("canonicalized sparse counts must have sorted indices without duplicates")
    return canonical, tuple(events)


def _ordered_metadata(
    frame: pl.DataFrame,
    *,
    frame_name: str,
    expected_length: int,
    required_columns: tuple[str, ...],
) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{frame_name} metadata must be an already-materialized Polars DataFrame")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} metadata is missing required columns: {', '.join(missing)}")
    if frame.height != expected_length:
        raise ValueError(
            f"{frame_name} metadata axis length {frame.height} does not match count axis length {expected_length}"
        )

    indices = frame.get_column("matrix_index")
    if not indices.dtype.is_integer():
        raise TypeError(f"{frame_name} metadata matrix_index must use an integer dtype")
    if indices.null_count() != 0:
        raise ValueError(f"{frame_name} metadata matrix_index cannot contain null values")
    index_values = np.asarray(indices.to_numpy())
    if np.unique(index_values).size != expected_length:
        raise ValueError(f"{frame_name} metadata matrix_index values must be unique; duplicate values were found")
    if np.any(index_values < 0) or np.any(index_values >= expected_length):
        expected_range = f"0..{expected_length - 1}"
        raise ValueError(
            f"{frame_name} metadata matrix_index values are out of range; expected exact coverage of {expected_range}"
        )
    ordered = frame.sort("matrix_index")
    expected = np.arange(expected_length, dtype=np.int64)
    if not np.array_equal(np.asarray(ordered["matrix_index"].to_numpy(), dtype=np.int64), expected):
        raise ValueError(f"{frame_name} metadata matrix_index values are gapped; expected exact axis coverage")
    return ordered


def _validated_string_values(
    frame: pl.DataFrame,
    column: str,
    *,
    unique: bool,
) -> tuple[str, ...]:
    series = frame.get_column(column)
    if series.null_count() != 0:
        raise ValueError(f"{column} values cannot contain null values")
    values = tuple(series.to_list())
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{column} values must be strings")
    if any(not value.strip() for value in values):
        raise ValueError(f"{column} values must be nonempty strings")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{column} values must be unique")
    return values


def _canonical_chromosome(value: object) -> str:
    if value is None:
        raise ValueError("chrom values cannot contain null values")
    label = str(value).strip().upper()
    if not label:
        raise ValueError("chrom values cannot be empty")
    if label.startswith("CHR"):
        label = label[3:]
    if label == "M":
        label = "MT"
    supported = {str(chromosome) for chromosome in range(1, 23)} | {"X", "Y", "MT"}
    if label not in supported:
        raise ValueError(f"unsupported chromosome label {value!r}; expected 1-22, X, Y, or MT")
    return label


def _readonly_strings(values: tuple[str, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.str_)
    array.flags.writeable = False
    return array


def normalize_sparse_single_cell(
    counts,
    cell_metadata: pl.DataFrame,
    gene_metadata: pl.DataFrame,
    *,
    cell_type_column: str,
) -> SparseSingleCellData:
    r"""Validate and normalize an already-materialized sparse single-cell dataset.

    **Arguments:**

    counts
        SciPy CSR or CSC sparse counts with cells on rows and genes on columns.
    cell_metadata
        Materialized Polars cell metadata keyed by ``matrix_index``.
    gene_metadata
        Materialized Polars gene metadata keyed by ``matrix_index``.
    cell_type_column
        Required cell metadata column containing nonempty cell-type labels.

    **Returns:**

    A canonical integer CSR count matrix and metadata frozen into matrix order.

    **Raises:**

    TypeError
        If sparse families, dtypes, or metadata container types are unsupported.
    ValueError
        If counts or metadata violate the single-cell ingress contract.
    """
    if not isinstance(cell_type_column, str) or not cell_type_column.strip():
        raise ValueError("cell_type_column must be a nonempty column name")
    if cell_type_column in {"matrix_index", "cell_id", "donor_id"}:
        raise ValueError("cell_type_column must be distinct from matrix_index, cell_id, and donor_id")

    canonical_counts, copy_events = _canonical_counts(counts)
    cells = _ordered_metadata(
        cell_metadata,
        frame_name="cell",
        expected_length=canonical_counts.shape[0],
        required_columns=("matrix_index", "cell_id", "donor_id", cell_type_column),
    )
    genes = _ordered_metadata(
        gene_metadata,
        frame_name="gene",
        expected_length=canonical_counts.shape[1],
        required_columns=("matrix_index", "gene_id", "chrom"),
    )

    cell_ids = _validated_string_values(cells, "cell_id", unique=True)
    gene_ids = _validated_string_values(genes, "gene_id", unique=True)
    donor_ids = _validated_string_values(cells, "donor_id", unique=False)
    cell_types = _validated_string_values(cells, cell_type_column, unique=False)
    if genes["chrom"].null_count() != 0:
        raise ValueError("chrom values cannot contain null values")
    chromosomes = tuple(_canonical_chromosome(value) for value in genes["chrom"].to_list())
    genes = genes.with_columns(pl.Series("chrom", chromosomes, dtype=pl.String))

    return SparseSingleCellData(
        counts=canonical_counts,
        cell_metadata=cells,
        gene_metadata=genes,
        cell_type_column=cell_type_column,
        cell_ids=_readonly_strings(cell_ids),
        gene_ids=_readonly_strings(gene_ids),
        donor_ids=_readonly_strings(donor_ids),
        cell_types=_readonly_strings(cell_types),
        gene_chromosomes=_readonly_strings(chromosomes),
        copy_events=copy_events,
    )
