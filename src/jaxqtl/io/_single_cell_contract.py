# pattern: Functional Core

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True, init=False)
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
    _cell_metadata: pl.DataFrame = field(repr=False)
    _gene_metadata: pl.DataFrame = field(repr=False)
    cell_type_column: str
    cell_ids: np.ndarray
    gene_ids: np.ndarray
    donor_ids: np.ndarray
    cell_types: np.ndarray
    gene_chromosomes: np.ndarray
    copy_events: tuple[SparseCopyEvent, ...]

    def __init__(
        self,
        counts: sparse.csr_array,
        cell_metadata: pl.DataFrame,
        gene_metadata: pl.DataFrame,
        cell_type_column: str,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        donor_ids: np.ndarray,
        cell_types: np.ndarray,
        gene_chromosomes: np.ndarray,
        copy_events: tuple[SparseCopyEvent, ...],
    ) -> None:
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "_cell_metadata", cell_metadata.clone())
        object.__setattr__(self, "_gene_metadata", gene_metadata.clone())
        object.__setattr__(self, "cell_type_column", cell_type_column)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "donor_ids", donor_ids)
        object.__setattr__(self, "cell_types", cell_types)
        object.__setattr__(self, "gene_chromosomes", gene_chromosomes)
        object.__setattr__(self, "copy_events", copy_events)

    @property
    def cell_metadata(self) -> pl.DataFrame:
        r"""Return a defensive clone of matrix-ordered cell metadata."""
        return self._cell_metadata.clone()

    @property
    def gene_metadata(self) -> pl.DataFrame:
        r"""Return a defensive clone of matrix-ordered gene metadata."""
        return self._gene_metadata.clone()


@dataclass(frozen=True, slots=True, init=False)
class SelectedSingleCellData:
    r"""Canonical cells selected for state-factor construction.

    **Arguments:**

    counts
        Integer-valued CSR counts for retained cells. Row positions are newly
        dense after selection.
    cell_metadata
        Retained cell metadata in source-matrix order. Its ``matrix_index``
        column remains the immutable source-row provenance.
    gene_metadata
        Gene metadata in count-matrix column order.
    cell_type_column
        Name of the cell-type metadata column.
    selected_cell_type
        Explicit or inferred cell type, or ``None`` for an opted-in mixed set.
    allow_mixed_cell_types
        Whether the caller explicitly retained multiple cell types.
    original_matrix_indices
        Source count-matrix row indices for retained cells.
    cell_ids
        Cell identifiers in selected count-matrix row order.
    cell_types
        Cell-type values in selected count-matrix row order.
    gene_ids
        Gene identifiers in count-matrix column order.
    gene_chromosomes
        Canonical chromosome labels in count-matrix column order.
    donor_ids
        Unique donor identifiers ordered by first retained cell.
    donor_index
        Dense zero-based cell-to-donor indices in selected matrix row order.
    donor_counts
        Number of retained cells for each entry in ``donor_ids``.
    copy_events
        Source ingress normalization/materialization events.
    """

    counts: sparse.csr_array
    _cell_metadata: pl.DataFrame = field(repr=False)
    _gene_metadata: pl.DataFrame = field(repr=False)
    cell_type_column: str
    selected_cell_type: str | None
    allow_mixed_cell_types: bool
    original_matrix_indices: np.ndarray
    cell_ids: np.ndarray
    cell_types: np.ndarray
    gene_ids: np.ndarray
    gene_chromosomes: np.ndarray
    donor_ids: np.ndarray
    donor_index: np.ndarray
    donor_counts: np.ndarray
    copy_events: tuple[SparseCopyEvent, ...]

    def __init__(
        self,
        counts: sparse.csr_array,
        cell_metadata: pl.DataFrame,
        gene_metadata: pl.DataFrame,
        cell_type_column: str,
        selected_cell_type: str | None,
        allow_mixed_cell_types: bool,
        original_matrix_indices: np.ndarray,
        cell_ids: np.ndarray,
        cell_types: np.ndarray,
        gene_ids: np.ndarray,
        gene_chromosomes: np.ndarray,
        donor_ids: np.ndarray,
        donor_index: np.ndarray,
        donor_counts: np.ndarray,
        copy_events: tuple[SparseCopyEvent, ...],
    ) -> None:
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "_cell_metadata", cell_metadata.clone())
        object.__setattr__(self, "_gene_metadata", gene_metadata.clone())
        object.__setattr__(self, "cell_type_column", cell_type_column)
        object.__setattr__(self, "selected_cell_type", selected_cell_type)
        object.__setattr__(self, "allow_mixed_cell_types", allow_mixed_cell_types)
        object.__setattr__(self, "original_matrix_indices", original_matrix_indices)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "cell_types", cell_types)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "gene_chromosomes", gene_chromosomes)
        object.__setattr__(self, "donor_ids", donor_ids)
        object.__setattr__(self, "donor_index", donor_index)
        object.__setattr__(self, "donor_counts", donor_counts)
        object.__setattr__(self, "copy_events", copy_events)

    @property
    def cell_metadata(self) -> pl.DataFrame:
        r"""Return a defensive clone retaining original row provenance."""
        return self._cell_metadata.clone()

    @property
    def gene_metadata(self) -> pl.DataFrame:
        r"""Return a defensive clone of matrix-ordered gene metadata."""
        return self._gene_metadata.clone()


def _validate_stored_counts(values: np.ndarray, *, context: str) -> None:
    dtype = values.dtype
    if np.issubdtype(dtype, np.bool_):
        raise TypeError("sparse counts cannot use boolean storage")
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        raise TypeError(f"sparse counts must use real integer or floating storage, got {dtype}")
    if np.issubdtype(dtype, np.floating):
        if not np.isfinite(values).all():
            raise ValueError(f"{context} count values must be finite")
        if np.any(values < 0):
            raise ValueError(f"{context} count values cannot be negative")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{context} count values must be integer-valued")
    elif np.issubdtype(dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError(f"{context} count values cannot be negative")
    if np.any(values > MAX_EXACT_FLOAT64_INT):
        raise ValueError(f"{context} count values must be at most 2**53 for exact float64 conversion")


def _checked_group_sum(values: np.ndarray, *, source_dtype: np.dtype, coordinate: tuple[int, int]) -> int:
    total = sum(int(value) for value in values)
    if not 0 <= total <= MAX_EXACT_FLOAT64_INT:
        raise ValueError(f"duplicate count sum at {coordinate} exceeds the 2**53 exact float64 integer bound")

    if np.issubdtype(source_dtype, np.integer):
        dtype_info = np.iinfo(source_dtype)
        if total < dtype_info.min or total > dtype_info.max:
            raise ValueError(f"duplicate count sum at {coordinate} is outside source dtype {source_dtype}")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            source_value = np.asarray(total, dtype=source_dtype).item()
        if not np.isfinite(source_value) or int(source_value) != total:
            raise ValueError(
                f"duplicate count sum at {coordinate} is not exactly representable in source dtype {source_dtype}"
            )
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
    canonical.data.flags.writeable = False
    canonical.indices.flags.writeable = False
    canonical.indptr.flags.writeable = False
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


def _readonly_array(values, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _freeze_csr_buffers(counts: sparse.csr_array) -> sparse.csr_array:
    counts.data.flags.writeable = False
    counts.indices.flags.writeable = False
    counts.indptr.flags.writeable = False
    return counts


def _validate_selection_source(data: SparseSingleCellData) -> pl.DataFrame:
    if not isinstance(data, SparseSingleCellData):
        raise TypeError("data must be a normalized SparseSingleCellData instance")
    if not isinstance(data.counts, sparse.csr_array):
        raise TypeError("data counts must be a canonical SciPy CSR sparse array")

    cells = data.cell_metadata
    n_cells = data.counts.shape[0]
    if cells.height != n_cells:
        raise ValueError(f"cell metadata row count {cells.height} does not match count matrix row count {n_cells}")
    required_columns = ("matrix_index", "cell_id", "donor_id", data.cell_type_column)
    missing = [column for column in required_columns if column not in cells.columns]
    if missing:
        raise ValueError(f"cell metadata is missing required columns: {', '.join(missing)}")

    vector_lengths = {
        "cell_ids": data.cell_ids.size,
        "donor_ids": data.donor_ids.size,
        "cell_types": data.cell_types.size,
    }
    invalid_lengths = [name for name, length in vector_lengths.items() if length != n_cells]
    if invalid_lengths:
        raise ValueError(
            "single-cell source vector lengths do not match count matrix rows: " + ", ".join(invalid_lengths)
        )

    matrix_indices = np.asarray(cells["matrix_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(matrix_indices, np.arange(n_cells, dtype=np.int64)):
        raise ValueError("cell metadata must remain in canonical source matrix order before selection")
    expected_vectors = (
        ("cell_ids", data.cell_ids, cells["cell_id"].to_list()),
        ("donor_ids", data.donor_ids, cells["donor_id"].to_list()),
        ("cell_types", data.cell_types, cells[data.cell_type_column].to_list()),
    )
    for name, array, metadata_values in expected_vectors:
        if array.tolist() != metadata_values:
            raise ValueError(f"{name} do not match matrix-ordered cell metadata")
    return cells


def select_single_cell_data(
    data: SparseSingleCellData,
    *,
    cell_type: str | None = None,
    allow_mixed_cell_types: bool = False,
) -> SelectedSingleCellData:
    r"""Select cells and freeze donor indexing for state-factor construction.

    **Arguments:**

    data
        Canonical sparse counts and matrix-ordered metadata from single-cell
        ingress normalization.
    cell_type
        Cell type to retain. It may be omitted only for a single-type source or
        when ``allow_mixed_cell_types`` is true.
    allow_mixed_cell_types
        Explicitly retain all source cell types. This cannot be combined with
        an explicit ``cell_type``.

    **Returns:**

    A canonical immutable selected-cell contract with sparse counts, source-row
    provenance, and dense first-retained-order donor indexing.

    **Raises:**

    TypeError
        If the input or option types violate the public contract.
    ValueError
        If selection is ambiguous, contradictory, unknown, empty, or the source
        cell axis is inconsistent.
    """
    if not isinstance(allow_mixed_cell_types, bool):
        raise TypeError("allow_mixed_cell_types must be a boolean")
    if cell_type is not None and not isinstance(cell_type, str):
        raise TypeError("cell_type must be a string or None")
    if isinstance(cell_type, str) and not cell_type.strip():
        raise ValueError("cell_type must be a nonempty string when provided")
    if cell_type is not None and allow_mixed_cell_types:
        raise ValueError("cannot provide cell_type together with allow_mixed_cell_types=True")

    cells = _validate_selection_source(data)
    observed_types = tuple(dict.fromkeys(data.cell_types.tolist()))
    retained_mixed_cell_types = False
    if cell_type is None:
        if len(observed_types) == 1:
            selected_cell_type = observed_types[0]
            selected_mask = np.ones(data.counts.shape[0], dtype=np.bool_)
        elif len(observed_types) > 1 and allow_mixed_cell_types:
            selected_cell_type = None
            retained_mixed_cell_types = True
            selected_mask = np.ones(data.counts.shape[0], dtype=np.bool_)
        elif len(observed_types) > 1:
            raise ValueError(
                "source contains multiple cell types; provide cell_type or set allow_mixed_cell_types=True"
            )
        else:
            selected_cell_type = None
            selected_mask = np.zeros(data.counts.shape[0], dtype=np.bool_)
    else:
        if cell_type not in observed_types:
            raise ValueError(f"unknown cell_type {cell_type!r}; available values are {observed_types}")
        selected_cell_type = cell_type
        selected_mask = data.cell_types == cell_type

    retained = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    if retained.size == 0:
        raise ValueError("cell selection retained zero cells")

    selected_counts = _freeze_csr_buffers(sparse.csr_array(data.counts[retained, :]))
    selected_cells = cells.filter(pl.Series("selected", selected_mask))
    original_matrix_indices = _readonly_array(
        selected_cells["matrix_index"].to_numpy(),
        dtype=np.int64,
    )
    selected_cell_ids = _readonly_strings(tuple(data.cell_ids[retained].tolist()))
    selected_cell_types = _readonly_strings(tuple(data.cell_types[retained].tolist()))
    retained_cell_donors = data.donor_ids[retained].tolist()

    donor_lookup: dict[str, int] = {}
    ordered_donors: list[str] = []
    donor_index = np.empty(retained.size, dtype=np.int64)
    for cell_index, donor_id in enumerate(retained_cell_donors):
        if donor_id not in donor_lookup:
            donor_lookup[donor_id] = len(ordered_donors)
            ordered_donors.append(donor_id)
        donor_index[cell_index] = donor_lookup[donor_id]
    donor_counts = np.bincount(donor_index, minlength=len(ordered_donors)).astype(np.int64, copy=False)

    return SelectedSingleCellData(
        counts=selected_counts,
        cell_metadata=selected_cells,
        gene_metadata=data.gene_metadata,
        cell_type_column=data.cell_type_column,
        selected_cell_type=selected_cell_type,
        allow_mixed_cell_types=retained_mixed_cell_types,
        original_matrix_indices=original_matrix_indices,
        cell_ids=selected_cell_ids,
        cell_types=selected_cell_types,
        gene_ids=_readonly_strings(tuple(data.gene_ids.tolist())),
        gene_chromosomes=_readonly_strings(tuple(data.gene_chromosomes.tolist())),
        donor_ids=_readonly_strings(tuple(ordered_donors)),
        donor_index=_readonly_array(donor_index, dtype=np.int64),
        donor_counts=_readonly_array(donor_counts, dtype=np.int64),
        copy_events=data.copy_events,
    )


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
