# pattern: Imperative Shell

from dataclasses import replace
from os import PathLike

import polars as pl

from scipy import sparse

from ._single_cell_contract import normalize_sparse_single_cell, SparseCopyEvent, SparseSingleCellData


def _collect_projected_metadata(
    query: pl.LazyFrame,
    *,
    path: str | PathLike[str],
    metadata_name: str,
    required_columns: tuple[str, ...],
) -> pl.DataFrame:
    try:
        return query.collect()
    except (OSError, pl.exceptions.PolarsError) as exc:
        columns = ", ".join(required_columns)
        raise ValueError(
            f"failed to load {metadata_name} metadata Parquet {str(path)!r}; required columns: {columns}: {exc}"
        ) from exc


def load_sparse_single_cell(
    counts_path: str | PathLike[str],
    cells_path: str | PathLike[str],
    genes_path: str | PathLike[str],
    *,
    cell_type_column: str,
) -> SparseSingleCellData:
    r"""Load and reconcile sparse single-cell counts and Parquet metadata.

    **Arguments:**

    counts_path
        Path to a SciPy CSR or CSC ``.npz`` count matrix. Rows are cells and
        columns are genes.
    cells_path
        Path to Parquet cell metadata keyed by ``matrix_index``.
    genes_path
        Path to Parquet gene metadata keyed by ``matrix_index``.
    cell_type_column
        Cell metadata column containing the cell-type label.

    **Returns:**

    Canonical CSR counts plus metadata frozen into matrix-axis order.

    **Raises:**

    ValueError
        If a file cannot be loaded or its contents violate the ingress
        contract.
    TypeError
        If the loaded sparse object is not CSR or CSC.

    !!! note

        Axis coverage detects an accidental transpose for non-square matrices.
        A square ``.npz`` matrix does not contain enough information to diagnose
        that orientation mistake on its own.
    """
    if not isinstance(cell_type_column, str) or not cell_type_column.strip():
        raise ValueError("cell_type_column must be a nonempty column name")
    if cell_type_column in {"matrix_index", "cell_id", "donor_id"}:
        raise ValueError("cell_type_column must be distinct from matrix_index, cell_id, and donor_id")

    try:
        counts = sparse.load_npz(counts_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load count NPZ {str(counts_path)!r}: {exc}") from exc

    cell_columns = ("matrix_index", "cell_id", "donor_id", cell_type_column)
    gene_columns = ("matrix_index", "gene_id", "chrom")
    cell_query = pl.scan_parquet(str(cells_path)).select(cell_columns)
    gene_query = pl.scan_parquet(str(genes_path)).select(gene_columns)
    cell_metadata = _collect_projected_metadata(
        cell_query,
        path=cells_path,
        metadata_name="cell",
        required_columns=cell_columns,
    )
    gene_metadata = _collect_projected_metadata(
        gene_query,
        path=genes_path,
        metadata_name="gene",
        required_columns=gene_columns,
    )

    normalized = normalize_sparse_single_cell(
        counts,
        cell_metadata,
        gene_metadata,
        cell_type_column=cell_type_column,
    )
    copy_events: tuple[SparseCopyEvent, ...] = ("npz_materialized", *normalized.copy_events)
    return replace(normalized, copy_events=copy_events)
