import importlib

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scipy import sparse

import jaxqtl.io as qtl_io


MAX_EXACT_FLOAT64_INT = 2**53


def _metadata(
    *,
    cell_indices: tuple[int, ...] = (2, 0, 1),
    gene_indices: tuple[int, ...] = (1, 2, 0),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cells_by_index = {
        0: ("cell-0", "donor-0", "B"),
        1: ("cell-1", "donor-0", "T"),
        2: ("cell-2", "donor-1", "B"),
    }
    genes_by_index = {
        0: ("gene-0", "chr1"),
        1: ("gene-1", "x"),
        2: ("gene-2", "ChrM"),
    }
    cells = pl.DataFrame(
        {
            "matrix_index": list(cell_indices),
            "cell_id": [cells_by_index[index][0] for index in cell_indices],
            "donor_id": [cells_by_index[index][1] for index in cell_indices],
            "cell_type": [cells_by_index[index][2] for index in cell_indices],
        }
    )
    genes = pl.DataFrame(
        {
            "matrix_index": list(gene_indices),
            "gene_id": [genes_by_index[index][0] for index in gene_indices],
            "chrom": [genes_by_index[index][1] for index in gene_indices],
        }
    )
    return cells, genes


def _normalizer():
    module = importlib.import_module("jaxqtl.io._single_cell_contract")
    return module.normalize_sparse_single_cell


def _normalize(
    counts,
    cells: pl.DataFrame | None = None,
    genes: pl.DataFrame | None = None,
):
    default_cells, default_genes = _metadata()
    return _normalizer()(
        counts,
        default_cells if cells is None else cells,
        default_genes if genes is None else genes,
        cell_type_column="cell_type",
    )


def _duplicate_csr(data: list[int | float], *, dtype) -> sparse.csr_array:
    values = np.asarray(data, dtype=dtype)
    return sparse.csr_array(
        (values, np.zeros(len(values), dtype=np.int32), np.array([0, len(values), len(values), len(values)])),
        shape=(3, 3),
    )


def _write_single_cell_inputs(
    tmp_path: Path,
    counts,
    *,
    cells: pl.DataFrame | None = None,
    genes: pl.DataFrame | None = None,
) -> tuple[Path, Path, Path]:
    default_cells, default_genes = _metadata()
    counts_path = tmp_path / "counts.npz"
    cells_path = tmp_path / "cells.parquet"
    genes_path = tmp_path / "genes.parquet"
    sparse.save_npz(counts_path, counts)
    (default_cells if cells is None else cells).write_parquet(cells_path)
    (default_genes if genes is None else genes).write_parquet(genes_path)
    return counts_path, cells_path, genes_path


def test_sparse_single_cell_contract_is_available_from_io_package() -> None:
    assert hasattr(qtl_io, "SparseSingleCellData"), "SparseSingleCellData is not available from jaxqtl.io"


@pytest.mark.parametrize(
    ("counts", "expected_events"),
    [
        (sparse.csr_array(np.diag(np.asarray([1, 2, 3], dtype=np.int32))), ()),
        (sparse.csc_array(np.diag(np.asarray([1, 2, 3], dtype=np.int32))), ("csc_to_csr",)),
        (sparse.csr_matrix(np.diag(np.asarray([1, 2, 3], dtype=np.int32))), ("sparse_family_normalized",)),
        (
            sparse.csc_matrix(np.diag(np.asarray([1, 2, 3], dtype=np.int32))),
            ("sparse_family_normalized", "csc_to_csr"),
        ),
    ],
)
def test_normalizes_supported_sparse_families_and_orders_metadata(counts, expected_events: tuple[str, ...]) -> None:
    result = _normalize(counts)

    assert isinstance(result, qtl_io.SparseSingleCellData)
    assert isinstance(result.counts, sparse.csr_array)
    assert result.counts.shape == (3, 3)
    assert result.counts.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(result.counts.toarray(), np.diag([1, 2, 3]))
    assert result.cell_metadata["matrix_index"].to_list() == [0, 1, 2]
    assert result.gene_metadata["matrix_index"].to_list() == [0, 1, 2]
    assert result.cell_ids.tolist() == ["cell-0", "cell-1", "cell-2"]
    assert result.gene_ids.tolist() == ["gene-0", "gene-1", "gene-2"]
    assert result.donor_ids.tolist() == ["donor-0", "donor-0", "donor-1"]
    assert result.cell_types.tolist() == ["B", "T", "B"]
    assert result.gene_chromosomes.tolist() == ["1", "X", "MT"]
    assert result.copy_events == expected_events


def test_result_contract_is_frozen() -> None:
    result = _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)))

    with pytest.raises(FrozenInstanceError):
        result.copy_events = ()


def test_integral_float_storage_becomes_integer_and_records_canonicalization() -> None:
    counts = sparse.csr_array(np.diag([1.0, 2.0, 3.0]).astype(np.float64))

    result = _normalize(counts)

    assert np.issubdtype(result.counts.dtype, np.integer)
    np.testing.assert_array_equal(result.counts.toarray(), np.diag([1, 2, 3]))
    assert result.copy_events == ("canonicalized",)


def test_duplicate_coordinates_are_accumulated_without_source_dtype_overflow() -> None:
    counts = _duplicate_csr([2, 3], dtype=np.int16)
    assert not counts.has_canonical_format

    result = _normalize(counts)

    assert result.counts.has_canonical_format
    assert result.counts[0, 0] == 5
    assert result.counts.dtype == np.dtype(np.int16)
    assert result.copy_events == ("canonicalized",)


@pytest.mark.parametrize(
    ("values", "accepted", "expected"),
    [
        ([MAX_EXACT_FLOAT64_INT - 2, 1], True, MAX_EXACT_FLOAT64_INT - 1),
        ([MAX_EXACT_FLOAT64_INT - 1, 1], True, MAX_EXACT_FLOAT64_INT),
        ([MAX_EXACT_FLOAT64_INT, 1], False, None),
    ],
)
def test_duplicate_sums_respect_float64_exact_integer_bound(values, accepted: bool, expected: int | None) -> None:
    counts = _duplicate_csr(values, dtype=np.uint64)

    if accepted:
        result = _normalize(counts)
        assert result.counts[0, 0] == expected
    else:
        with pytest.raises(ValueError, match=r"duplicate.*2\*\*53|duplicate.*exact float64"):
            _normalize(counts)


def test_duplicate_sum_rejects_source_integer_dtype_overflow() -> None:
    counts = _duplicate_csr([100, 50], dtype=np.int8)

    with pytest.raises(ValueError, match="duplicate.*int8"):
        _normalize(counts)


def test_duplicate_float_sum_must_remain_exact_in_source_dtype() -> None:
    counts = _duplicate_csr([2**24, 1], dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate.*float32"):
        _normalize(counts)


@pytest.mark.parametrize(
    "counts",
    [
        sparse.bsr_array(np.eye(3, dtype=np.int8)),
        sparse.coo_array(np.eye(3, dtype=np.int8)),
        sparse.dia_array(np.eye(3, dtype=np.int8)),
        np.eye(3, dtype=np.int8),
    ],
)
def test_rejects_unsupported_or_dense_count_containers(counts) -> None:
    with pytest.raises(TypeError, match="CSR or CSC"):
        _normalize(counts)


@pytest.mark.parametrize(
    ("values", "dtype", "message"),
    [
        ([True, False, True], np.bool_, "boolean"),
        ([1, -1, 2], np.int64, "negative"),
        ([1.0, -1.0, 2.0], np.float64, "negative"),
        ([1.0, 1.5, 2.0], np.float64, "integer-valued"),
        ([1.0, np.nan, 2.0], np.float64, "finite"),
        ([1.0, np.inf, 2.0], np.float64, "finite"),
        ([MAX_EXACT_FLOAT64_INT - 1, MAX_EXACT_FLOAT64_INT, 0], np.uint64, None),
        ([MAX_EXACT_FLOAT64_INT + 1, 0, 0], np.uint64, "2\\*\\*53"),
    ],
)
def test_validates_stored_count_values(values, dtype, message: str | None) -> None:
    counts = sparse.csr_array(np.diag(np.asarray(values, dtype=dtype)))

    if message is None:
        result = _normalize(counts)
        assert result.counts[0, 0] == MAX_EXACT_FLOAT64_INT - 1
        assert result.counts[1, 1] == MAX_EXACT_FLOAT64_INT
    else:
        with pytest.raises((TypeError, ValueError), match=message):
            _normalize(counts)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (sparse.csr_array(np.array([1, 2, 3], dtype=np.int8)), "two-dimensional"),
        (sparse.csr_array((0, 3), dtype=np.int8), "empty.*axis"),
        (sparse.csr_array((3, 0), dtype=np.int8), "empty.*axis"),
    ],
)
def test_rejects_invalid_count_shapes(counts, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize(counts)


@pytest.mark.parametrize(
    ("frame_name", "frame", "message"),
    [
        ("cells", pl.DataFrame({"matrix_index": [0, 1], "cell_id": ["c0", "c1"]}), "missing required columns"),
        (
            "genes",
            pl.DataFrame({"matrix_index": [0, 1, 2], "gene_id": ["g0", "g1", "g2"]}),
            "missing required columns",
        ),
    ],
)
def test_rejects_missing_metadata_columns(frame_name: str, frame: pl.DataFrame, message: str) -> None:
    cells, genes = _metadata()
    if frame_name == "cells":
        cells = frame
    else:
        genes = frame

    with pytest.raises(ValueError, match=message):
        _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)), cells, genes)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([0, 1], "axis length"),
        ([0, 1, 1], "duplicate"),
        ([0, 2, 3], "out of range"),
        ([0, 2, 2], "duplicate"),
        ([0, 1, -1], "out of range"),
        ([0, 1, None], "null"),
        ([0.0, 1.0, 2.0], "integer dtype"),
    ],
)
def test_rejects_invalid_cell_matrix_indices(indices, message: str) -> None:
    cells, genes = _metadata(cell_indices=(0, 1, 2))
    if len(indices) != cells.height:
        cells = cells.head(len(indices))
    cells = cells.with_columns(pl.Series("matrix_index", indices))

    with pytest.raises((TypeError, ValueError), match=message):
        _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)), cells, genes)


def test_rejects_gapped_gene_matrix_indices() -> None:
    cells, genes = _metadata(gene_indices=(0, 1, 2))
    genes = pl.concat(
        [
            genes.filter(pl.col("matrix_index") != 1),
            pl.DataFrame({"matrix_index": [3], "gene_id": ["gene-3"], "chrom": ["3"]}),
        ]
    )

    with pytest.raises(ValueError, match="out of range|gapped"):
        _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)), cells, genes)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("cell_id", ["cell-0", "cell-0", "cell-2"], "cell_id.*unique"),
        ("cell_id", ["cell-0", None, "cell-2"], "cell_id.*null"),
        ("cell_id", ["cell-0", " ", "cell-2"], "cell_id.*nonempty"),
        ("donor_id", ["donor-0", None, "donor-1"], "donor_id.*null"),
        ("donor_id", ["donor-0", "", "donor-1"], "donor_id.*nonempty"),
        ("cell_type", ["B", None, "T"], "cell_type.*null"),
        ("cell_type", ["B", "", "T"], "cell_type.*nonempty"),
    ],
)
def test_rejects_invalid_cell_identifiers(column: str, values: list[str | None], message: str) -> None:
    cells, genes = _metadata(cell_indices=(0, 1, 2))
    cells = cells.with_columns(pl.Series(column, values))

    with pytest.raises(ValueError, match=message):
        _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)), cells, genes)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("gene_id", ["gene-0", "gene-0", "gene-2"], "gene_id.*unique"),
        ("gene_id", ["gene-0", None, "gene-2"], "gene_id.*null"),
        ("gene_id", ["gene-0", "", "gene-2"], "gene_id.*nonempty"),
        ("chrom", ["1", None, "3"], "chrom.*null"),
        ("chrom", ["1", "", "3"], "chrom.*empty"),
        ("chrom", ["1", "chr23", "3"], "unsupported chromosome"),
    ],
)
def test_rejects_invalid_gene_identifiers(column: str, values: list[str | None], message: str) -> None:
    cells, genes = _metadata(gene_indices=(0, 1, 2))
    genes = genes.with_columns(pl.Series(column, values))

    with pytest.raises(ValueError, match=message):
        _normalize(sparse.csr_array(np.eye(3, dtype=np.int8)), cells, genes)


def test_rejects_axis_length_mismatch_without_transposing() -> None:
    cells, genes = _metadata()
    transposed_shape = sparse.csr_array(np.ones((2, 3), dtype=np.int8))

    with pytest.raises(ValueError, match="cell metadata.*axis length"):
        _normalize(transposed_shape, cells, genes)


def test_rejects_empty_or_reserved_cell_type_column_name() -> None:
    cells, genes = _metadata()
    counts = sparse.csr_array(np.eye(3, dtype=np.int8))

    with pytest.raises(ValueError, match="cell_type_column.*nonempty"):
        _normalizer()(counts, cells, genes, cell_type_column="")
    with pytest.raises(ValueError, match="cell_type_column.*distinct"):
        _normalizer()(counts, cells, genes, cell_type_column="donor_id")


def test_sparse_single_cell_loader_is_available_from_io_package() -> None:
    assert callable(getattr(qtl_io, "load_sparse_single_cell", None)), (
        "load_sparse_single_cell is not available from jaxqtl.io"
    )


def test_loads_projected_npz_and_parquet_inputs_once_in_matrix_order(tmp_path: Path) -> None:
    counts = sparse.csr_array(
        np.asarray(
            [
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 5],
            ],
            dtype=np.int32,
        )
    )
    cells, genes = _metadata()
    cells = cells.with_columns(pl.lit("not projected").alias("unused_cell_column"))
    genes = genes.with_columns(pl.lit(999).alias("unused_gene_column"))
    counts_path, cells_path, genes_path = _write_single_cell_inputs(
        tmp_path,
        counts,
        cells=cells,
        genes=genes,
    )

    result = qtl_io.load_sparse_single_cell(
        counts_path,
        cells_path,
        genes_path,
        cell_type_column="cell_type",
    )

    assert isinstance(result.counts, sparse.csr_array)
    np.testing.assert_array_equal(result.counts.toarray(), counts.toarray())
    assert result.cell_metadata.columns == ["matrix_index", "cell_id", "donor_id", "cell_type"]
    assert result.gene_metadata.columns == ["matrix_index", "gene_id", "chrom"]
    assert result.cell_ids.tolist() == ["cell-0", "cell-1", "cell-2"]
    assert result.gene_ids.tolist() == ["gene-0", "gene-1", "gene-2"]
    assert result.gene_chromosomes.tolist() == ["1", "X", "MT"]
    assert result.copy_events == ("npz_materialized",)


def test_loads_csc_npz_and_reports_conversion_events(tmp_path: Path) -> None:
    counts = sparse.csc_array(np.diag(np.asarray([1, 2, 3], dtype=np.int16)))
    counts_path, cells_path, genes_path = _write_single_cell_inputs(tmp_path, counts)

    result = qtl_io.load_sparse_single_cell(
        counts_path,
        cells_path,
        genes_path,
        cell_type_column="cell_type",
    )

    assert isinstance(result.counts, sparse.csr_array)
    assert result.counts.dtype == np.dtype(np.int16)
    assert result.copy_events == ("npz_materialized", "csc_to_csr")


def test_loader_reports_missing_count_file_with_path_context(tmp_path: Path) -> None:
    cells, genes = _metadata()
    cells_path = tmp_path / "cells.parquet"
    genes_path = tmp_path / "genes.parquet"
    cells.write_parquet(cells_path)
    genes.write_parquet(genes_path)
    missing_counts = tmp_path / "missing-counts.npz"

    with pytest.raises(ValueError, match=r"count NPZ.*missing-counts\.npz"):
        qtl_io.load_sparse_single_cell(
            missing_counts,
            cells_path,
            genes_path,
            cell_type_column="cell_type",
        )


@pytest.mark.parametrize(
    ("missing_metadata", "message"),
    [
        ("cell", r"cell metadata Parquet.*missing-cells\.parquet"),
        ("gene", r"gene metadata Parquet.*missing-genes\.parquet"),
    ],
)
def test_loader_reports_missing_metadata_file_with_path_context(
    tmp_path: Path,
    missing_metadata: str,
    message: str,
) -> None:
    counts_path, cells_path, genes_path = _write_single_cell_inputs(
        tmp_path,
        sparse.csr_array(np.eye(3, dtype=np.int8)),
    )
    if missing_metadata == "cell":
        cells_path = tmp_path / "missing-cells.parquet"
    else:
        genes_path = tmp_path / "missing-genes.parquet"

    with pytest.raises(ValueError, match=message):
        qtl_io.load_sparse_single_cell(
            counts_path,
            cells_path,
            genes_path,
            cell_type_column="cell_type",
        )


@pytest.mark.parametrize(
    ("metadata", "dropped_column", "message"),
    [
        ("cell", "donor_id", "cell metadata.*donor_id"),
        ("gene", "chrom", "gene metadata.*chrom"),
    ],
)
def test_loader_reports_projected_parquet_schema_failures(
    tmp_path: Path,
    metadata: str,
    dropped_column: str,
    message: str,
) -> None:
    counts = sparse.csr_array(np.eye(3, dtype=np.int8))
    cells, genes = _metadata()
    if metadata == "cell":
        cells = cells.drop(dropped_column)
    else:
        genes = genes.drop(dropped_column)
    counts_path, cells_path, genes_path = _write_single_cell_inputs(
        tmp_path,
        counts,
        cells=cells,
        genes=genes,
    )

    with pytest.raises(ValueError, match=message):
        qtl_io.load_sparse_single_cell(
            counts_path,
            cells_path,
            genes_path,
            cell_type_column="cell_type",
        )


def test_loader_rejects_unsupported_sparse_npz_family(tmp_path: Path) -> None:
    counts = sparse.coo_array(np.eye(3, dtype=np.int8))
    counts_path, cells_path, genes_path = _write_single_cell_inputs(tmp_path, counts)

    with pytest.raises(TypeError, match="CSR or CSC"):
        qtl_io.load_sparse_single_cell(
            counts_path,
            cells_path,
            genes_path,
            cell_type_column="cell_type",
        )


def test_loader_exposes_transposed_non_square_orientation_error(tmp_path: Path) -> None:
    counts = sparse.csr_array(np.asarray([[1, 0], [0, 2], [3, 0]], dtype=np.int16))
    cells = pl.DataFrame(
        {
            "matrix_index": [0, 1, 2],
            "cell_id": ["cell-0", "cell-1", "cell-2"],
            "donor_id": ["donor-0", "donor-0", "donor-1"],
            "cell_type": ["B", "B", "T"],
        }
    )
    genes = pl.DataFrame(
        {
            "matrix_index": [0, 1],
            "gene_id": ["gene-0", "gene-1"],
            "chrom": ["1", "2"],
        }
    )
    transposed = counts.T.tocsr()
    counts_path, cells_path, genes_path = _write_single_cell_inputs(
        tmp_path,
        transposed,
        cells=cells,
        genes=genes,
    )

    with pytest.raises(ValueError, match="cell metadata axis length"):
        qtl_io.load_sparse_single_cell(
            counts_path,
            cells_path,
            genes_path,
            cell_type_column="cell_type",
        )


def test_loader_rejects_empty_cell_type_column_before_scanning(tmp_path: Path) -> None:
    counts_path, cells_path, genes_path = _write_single_cell_inputs(
        tmp_path,
        sparse.csr_array(np.eye(3, dtype=np.int8)),
    )

    with pytest.raises(ValueError, match="cell_type_column.*nonempty"):
        qtl_io.load_sparse_single_cell(
            counts_path,
            cells_path,
            genes_path,
            cell_type_column="",
        )
