# pattern: Imperative Shell

from __future__ import annotations

import ctypes
import errno
import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jaxlib
import numpy as np
import polars as pl
import pyarrow
import scipy

import jax

from jaxqtl import __version__ as jaxqtl_version
from jaxqtl.state import StateFactorResult

from ._state_artifact_contract import (
    _ChromosomeManifest,
    _PayloadRecord,
    _ReplayProvenance,
    _StateArtifactChromosomeResult,
    _validate_cell_type_selection,
    _validate_state_factor_payload_numerics,
    _validated_chromosome_set,
    _validated_gene_chromosomes,
    ARTIFACT_TYPE,
    canonical_chromosome_key,
    canonical_payload_inventory,
    decode_manifest,
    encode_manifest,
    identifier_order_hash,
    INPUT_NAMES,
    PROVENANCE_PACKAGES,
    SCHEMA_VERSION,
    StateArtifactManifest,
    StateArtifactResult,
    THREAD_ENVIRONMENT_VARIABLES,
    validate_manifest,
)


_HASH_CHUNK_SIZE = 1024 * 1024
_STAGING_DIRECTORY_PREFIX = ".jaxqtl-state-artifact-staging-"
_DARWIN_RENAME_EXCL = 0x00000004
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1


def _preflight_state_artifact_destination(destination: str | os.PathLike[str]) -> Path:
    """Reject deterministic destination failures before artifact computation."""
    final_path = Path(destination)
    if final_path.name.startswith(_STAGING_DIRECTORY_PREFIX):
        raise ValueError(f"state artifact destination uses the reserved staging namespace: {final_path}")
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(f"state artifact destination already exists: {final_path}")
    parent = final_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"state artifact parent directory does not exist or is not a directory: {parent}")
    return final_path


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publication_os_error(error_number: int, final_path: Path) -> OSError:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        return FileExistsError(
            errno.EEXIST,
            f"state artifact destination already exists: {final_path}",
            final_path,
        )
    return OSError(error_number, os.strerror(error_number), final_path)


def _publication_platform() -> str:
    """Return the runtime platform used by the owned publication adapter."""
    return sys.platform


def _load_process_library() -> ctypes.CDLL:
    """Load the process C library used by the owned publication adapter."""
    return ctypes.CDLL(None, use_errno=True)


def _publish_directory_noreplace(staging: Path, final_path: Path) -> None:
    """Atomically publish without replacement, or fail closed if unsupported."""
    source = os.fsencode(staging)
    destination = os.fsencode(final_path)
    publication_platform = _publication_platform()
    if publication_platform not in {"darwin", "linux"} and not publication_platform.startswith("linux"):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported on this platform",
            final_path,
        )
    libc = _load_process_library()
    if publication_platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory publication is unavailable on this Darwin runtime",
                final_path,
            ) from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source, destination, _DARWIN_RENAME_EXCL)
    elif publication_platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory publication requires renameat2 on Linux",
                final_path,
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _LINUX_AT_FDCWD,
            source,
            _LINUX_AT_FDCWD,
            destination,
            _LINUX_RENAME_NOREPLACE,
        )
    if result != 0:
        raise _publication_os_error(ctypes.get_errno(), final_path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        frame.write_parquet(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _package_version(name: str) -> str:
    modules = {
        "numpy": np,
        "scipy": scipy,
        "polars": pl,
        "pyarrow": pyarrow,
        "jax": jax,
        "jaxlib": jaxlib,
    }
    if name == "jaxqtl":
        return jaxqtl_version
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return str(modules[name].__version__)


def _normalized_linear_algebra(configuration: Mapping[str, Any]) -> dict[str, Any]:
    build_dependencies = configuration.get("Build Dependencies", {})
    if not isinstance(build_dependencies, Mapping):
        build_dependencies = {}
    result: dict[str, Any] = {}
    for library in ("blas", "lapack"):
        raw = build_dependencies.get(library, {})
        if not isinstance(raw, Mapping):
            raw = {}
        result[library] = {
            "name": str(raw.get("name", "unknown")).lower(),
            "version": str(raw.get("version", "unknown")),
            "found": bool(raw.get("found", False)),
            "detection_method": str(raw.get("detection method", "unknown")).lower(),
            "ilp64": bool(raw.get("has ilp64", False)),
        }
    return result


def _collect_replay_provenance() -> _ReplayProvenance:
    return _ReplayProvenance(
        jaxqtl_version=jaxqtl_version,
        artifact_schema_version=SCHEMA_VERSION,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        package_versions=tuple((name, _package_version(name)) for name in PROVENANCE_PACKAGES),
        blas_lapack=(
            ("numpy", _normalized_linear_algebra(np.__config__.show(mode="dicts"))),
            ("scipy", _normalized_linear_algebra(scipy.show_config(mode="dicts"))),
        ),
        thread_environment=tuple((name, os.environ.get(name)) for name in THREAD_ENVIRONMENT_VARIABLES),
        platform="cpu",
    )


def _required_columns(frame: pl.DataFrame, columns: Sequence[str], *, context: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{context} must be an already-materialized Polars DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")
    if any(frame[column].null_count() for column in columns):
        raise ValueError(f"{context} required identifier columns cannot contain null values")


def _identifier_column(frame: pl.DataFrame, column: str, *, context: str) -> tuple[str, ...]:
    values = tuple(frame[column].to_list())
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{context} {column} values must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{context} {column} values must be unique")
    return values


def _ordered_matrix_indices(frame: pl.DataFrame, *, context: str) -> tuple[int, ...]:
    values = tuple(frame["matrix_index"].to_list())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{context} matrix_index values must be nonnegative integers")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{context} matrix_index values must be strictly increasing in matrix order")
    return values


def _validated_metadata(
    cell_metadata: pl.DataFrame,
    gene_metadata: pl.DataFrame,
    *,
    cell_type_column: str,
    selected_cell_type: str | None,
    allow_mixed_cell_types: bool,
    donor_ids: Sequence[str],
    donor_counts: Sequence[int],
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    if not isinstance(cell_type_column, str) or not cell_type_column:
        raise ValueError("cell_type_column must be a nonempty string")
    _required_columns(
        cell_metadata,
        ("matrix_index", "cell_id", "donor_id", cell_type_column),
        context="cell metadata",
    )
    _required_columns(gene_metadata, ("matrix_index", "gene_id", "chrom"), context="gene metadata")
    if cell_metadata.height == 0:
        raise ValueError("cell metadata cannot be empty")
    if gene_metadata.height == 0:
        raise ValueError("gene metadata cannot be empty")
    _ordered_matrix_indices(cell_metadata, context="cell metadata")
    _ordered_matrix_indices(gene_metadata, context="gene metadata")
    cell_ids = _identifier_column(cell_metadata, "cell_id", context="cell metadata")
    gene_ids = _identifier_column(gene_metadata, "gene_id", context="gene metadata")
    _validated_gene_chromosomes(gene_metadata["chrom"].to_list(), context="gene metadata chrom")
    _validate_cell_type_selection(
        cell_metadata[cell_type_column].to_list(),
        selected_cell_type=selected_cell_type,
        allow_mixed_cell_types=allow_mixed_cell_types,
    )
    cell_donors = tuple(cell_metadata["donor_id"].to_list())
    if any(not isinstance(value, str) or not value for value in cell_donors):
        raise ValueError("cell metadata donor_id values must be nonempty strings")

    normalized_donors = tuple(donor_ids)
    if any(not isinstance(value, str) or not value for value in normalized_donors):
        raise ValueError("donor_ids must contain nonempty strings")
    if not normalized_donors or len(set(normalized_donors)) != len(normalized_donors):
        raise ValueError("donor_ids must be nonempty and unique")
    normalized_counts = tuple(donor_counts)
    if len(normalized_counts) != len(normalized_donors):
        raise ValueError("donor_counts must align one-to-one with donor_ids")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in normalized_counts):
        raise ValueError("donor_counts must contain positive integers")
    if sum(normalized_counts) != len(cell_ids):
        raise ValueError("donor_counts must sum to the cell count")
    observed_counts = tuple(cell_donors.count(donor) for donor in normalized_donors)
    if observed_counts != normalized_counts or set(cell_donors) != set(normalized_donors):
        raise ValueError("donor_ids and donor_counts must match cell metadata exactly")
    first_observed = tuple(dict.fromkeys(cell_donors))
    if first_observed != normalized_donors:
        raise ValueError("donor_ids must follow first-retained-cell order")
    return (
        cell_metadata.clone(),
        gene_metadata.clone(),
        cell_ids,
        gene_ids,
        normalized_donors,
        normalized_counts,
    )


def _validated_input_hashes(input_paths: Mapping[str, str | os.PathLike[str]]) -> tuple[tuple[str, str], ...]:
    if set(input_paths) != set(INPUT_NAMES):
        raise ValueError("input_paths must contain exactly counts, cells, and genes")
    hashes = []
    for name in INPUT_NAMES:
        path = Path(input_paths[name])
        if not path.is_file():
            raise FileNotFoundError(f"{name} input path is not a regular file: {path}")
        hashes.append((name, _sha256_file(path)))
    return tuple(hashes)


def _payload_record(
    path: Path,
    *,
    root: Path,
    kind: Literal["npy", "parquet"],
    shape: tuple[int, ...] | None,
    dtype: str | None,
    rows: int | None,
) -> _PayloadRecord:
    return _PayloadRecord(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256_file(path),
        kind=kind,
        shape=shape,
        dtype=dtype,
        rows=rows,
    )


def _write_shared_metadata(
    staging: Path,
    *,
    cell_metadata: pl.DataFrame,
    donor_ids: tuple[str, ...],
    donor_counts: tuple[int, ...],
) -> tuple[_PayloadRecord, _PayloadRecord]:
    cells_path = staging / "cells.parquet"
    donors_path = staging / "donors.parquet"
    donors = pl.DataFrame(
        {
            "donor_index": list(range(len(donor_ids))),
            "donor_id": list(donor_ids),
            "cell_count": list(donor_counts),
        }
    )
    _write_parquet(cells_path, cell_metadata)
    _write_parquet(donors_path, donors)
    return (
        _payload_record(cells_path, root=staging, kind="parquet", shape=None, dtype=None, rows=cell_metadata.height),
        _payload_record(donors_path, root=staging, kind="parquet", shape=None, dtype=None, rows=donors.height),
    )


def _active_gene_metadata(gene_metadata: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("active gene indices must be a one-dimensional integer array")
    normalized = tuple(int(index) for index in indices)
    if not normalized:
        raise ValueError("each factor result must retain at least one gene")
    if any(index < 0 or index >= gene_metadata.height for index in normalized):
        raise ValueError("active gene indices are outside the gene metadata axis")
    if any(right <= left for left, right in zip(normalized, normalized[1:], strict=False)):
        raise ValueError("active gene indices must be unique and strictly increasing")
    return gene_metadata.gather(list(normalized))


def _validated_factor_result(
    result: StateFactorResult,
    *,
    chromosome: str,
    n_cells: int,
    donor_counts: tuple[int, ...],
    gene_metadata: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    if not isinstance(result, StateFactorResult):
        raise TypeError("factor_results must yield StateFactorResult values")
    diagnostics = result.diagnostics
    if (
        diagnostics.excluded_chromosome is None
        or canonical_chromosome_key(diagnostics.excluded_chromosome) != chromosome
    ):
        raise ValueError(f"factor result chromosome does not match requested chromosome {chromosome}")
    factors = np.asarray(result.factors)
    loadings = np.asarray(result.loadings)
    singular_values = np.asarray(result.singular_values)
    rank = diagnostics.rank
    active_gene_indices = np.asarray(diagnostics.active_gene_indices)
    active_genes = _active_gene_metadata(gene_metadata, active_gene_indices)
    if diagnostics.input_gene_count != gene_metadata.height:
        raise ValueError(f"chromosome {chromosome} input gene count disagrees with shared gene metadata")
    if diagnostics.transform_excluded_gene_count != gene_metadata.height - active_genes.height:
        raise ValueError(f"chromosome {chromosome} transform-excluded gene count disagrees with active genes")
    excluded_gene_chromosome = str(int(chromosome))
    expected_active_gene_indices = np.flatnonzero(
        np.asarray(gene_metadata["chrom"].to_list(), dtype=np.str_) != excluded_gene_chromosome
    )
    if not np.array_equal(active_gene_indices, expected_active_gene_indices):
        raise ValueError(f"chromosome {chromosome} active gene indices disagree with the LOCO exclusion")
    if factors.shape != (n_cells, rank) or tuple(diagnostics.factors_shape) != factors.shape:
        raise ValueError(f"chromosome {chromosome} factor shape disagrees with cells, rank, or diagnostics")
    if loadings.shape != (active_genes.height, rank) or tuple(diagnostics.loadings_shape) != loadings.shape:
        raise ValueError(f"chromosome {chromosome} loading shape disagrees with genes, rank, or diagnostics")
    if singular_values.shape != (rank,) or diagnostics.singular_values.shape != (rank,):
        raise ValueError(f"chromosome {chromosome} singular-value shape disagrees with rank or diagnostics")
    for name, array in (("factors", factors), ("loadings", loadings), ("singular values", singular_values)):
        if array.dtype != np.dtype(np.float64):
            raise TypeError(f"chromosome {chromosome} {name} must use the schema-v1 float64 dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"chromosome {chromosome} {name} must be finite")
    if not np.array_equal(singular_values, diagnostics.singular_values):
        raise ValueError(f"chromosome {chromosome} singular values disagree with diagnostics")
    if tuple(int(count) for count in diagnostics.donor_counts) != donor_counts:
        raise ValueError(f"chromosome {chromosome} donor counts disagree with shared donor metadata")
    return factors, loadings, singular_values, active_genes


def _chromosome_manifest(
    result: StateFactorResult,
    *,
    chromosome: str,
    active_genes: pl.DataFrame,
    approximation_metrics: Mapping[str, Any],
) -> _ChromosomeManifest:
    diagnostics = result.diagnostics
    return _ChromosomeManifest(
        chromosome=chromosome,
        n_cells=result.factors.shape[0],
        n_genes=result.loadings.shape[0],
        rank=diagnostics.rank,
        factors_dtype=result.factors.dtype.str,
        loadings_dtype=result.loadings.dtype.str,
        singular_values_dtype=result.singular_values.dtype.str,
        gene_order_hash=identifier_order_hash((("gene_id", tuple(active_genes["gene_id"].to_list())),)),
        pflog_diagnostics={
            "alpha": diagnostics.alpha,
            "alpha_source": diagnostics.alpha_source,
            "retained_gene_count": diagnostics.alpha_retained_gene_count,
            "excluded_gene_count": diagnostics.alpha_excluded_gene_count,
            "numerator": diagnostics.alpha_numerator,
            "denominator": diagnostics.alpha_denominator,
            "excluded_numerator": diagnostics.alpha_excluded_numerator,
            "excluded_denominator": diagnostics.alpha_excluded_denominator,
        },
        filtering_diagnostics={
            "input_gene_count": diagnostics.input_gene_count,
            "active_gene_count": int(diagnostics.active_gene_indices.size),
            "transform_excluded_gene_count": diagnostics.transform_excluded_gene_count,
        },
        donor_counts=tuple(int(count) for count in diagnostics.donor_counts),
        center_within_donor=diagnostics.center_within_donor,
        balance_donors=diagnostics.balance_donors,
        solver_configuration={
            "solver": diagnostics.solver,
            "seed": diagnostics.seed,
            "tol": diagnostics.tol,
            "maxiter": diagnostics.maxiter,
            "propack_kmax": diagnostics.propack_kmax,
            "arpack_ncv": diagnostics.arpack_ncv,
        },
        singular_values=tuple(float(value) for value in result.singular_values),
        convergence_residuals={
            "sigma_floor": diagnostics.sigma_floor,
            "residual_limit": diagnostics.residual_limit,
            "max_forward_residual": diagnostics.max_forward_residual,
            "max_adjoint_residual": diagnostics.max_adjoint_residual,
            "loading_orthogonality_error": diagnostics.loading_orthogonality_error,
        },
        approximation_metrics=dict(approximation_metrics),
    )


def _write_chromosome_payloads(
    staging: Path,
    *,
    chromosome: str,
    factors: np.ndarray,
    loadings: np.ndarray,
    singular_values: np.ndarray,
    active_genes: pl.DataFrame,
) -> tuple[_PayloadRecord, ...]:
    directory = staging / "chromosomes" / chromosome
    factors_path = directory / "factors.npy"
    loadings_path = directory / "loadings.npy"
    singular_values_path = directory / "singular_values.npy"
    genes_path = directory / "genes.parquet"
    _write_array(factors_path, factors)
    _write_array(loadings_path, loadings)
    _write_array(singular_values_path, singular_values)
    _write_parquet(genes_path, active_genes)
    return (
        _payload_record(
            factors_path,
            root=staging,
            kind="npy",
            shape=factors.shape,
            dtype=factors.dtype.str,
            rows=None,
        ),
        _payload_record(
            loadings_path,
            root=staging,
            kind="npy",
            shape=loadings.shape,
            dtype=loadings.dtype.str,
            rows=None,
        ),
        _payload_record(
            singular_values_path,
            root=staging,
            kind="npy",
            shape=singular_values.shape,
            dtype=singular_values.dtype.str,
            rows=None,
        ),
        _payload_record(genes_path, root=staging, kind="parquet", shape=None, dtype=None, rows=active_genes.height),
    )


def write_state_artifact(
    destination: str | os.PathLike[str],
    factor_results: Iterable[StateFactorResult],
    *,
    requested_chromosomes: Sequence[str],
    cell_metadata: pl.DataFrame,
    gene_metadata: pl.DataFrame,
    donor_ids: Sequence[str],
    donor_counts: Sequence[int],
    input_paths: Mapping[str, str | os.PathLike[str]],
    cell_type_column: str,
    selected_cell_type: str | None,
    allow_mixed_cell_types: bool,
    configuration: Mapping[str, Any],
    approximation_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> StateArtifactManifest:
    r"""Stream, validate, and atomically publish one state-factor artifact.

    **Arguments:**

    destination
        Previously nonexistent final directory.
    factor_results
        Ordered stream containing exactly one complete result per requested chromosome.
    requested_chromosomes
        One autosome or the complete ordered autosome sequence 1-22.
    cell_metadata, gene_metadata, donor_ids, donor_counts
        Canonical selected metadata and frozen cell/donor/gene ordering.
    input_paths
        Exact ``counts``, ``cells``, and ``genes`` raw inputs hashed in bounded chunks.
    cell_type_column, selected_cell_type, allow_mixed_cell_types
        Resolved selection contract.
    configuration
        Fully resolved replay configuration.
    approximation_metrics
        Optional per-chromosome approximation diagnostics.

    **Returns:**

    The validated manifest published at ``destination``.

    **Raises:**

    FileExistsError
        If the destination already exists.
    ValueError, TypeError, OSError
        If inputs, streamed results, writes, hashes, or staged validation fail. The staging
        directory is removed and no final artifact is exposed. Publication uses Darwin
        ``RENAME_EXCL`` or Linux ``RENAME_NOREPLACE``; other runtimes fail closed.
    """
    final_path = _preflight_state_artifact_destination(destination)
    parent = final_path.parent
    chromosomes = _validated_chromosome_set(requested_chromosomes, require_canonical=False)
    (
        canonical_cells,
        canonical_genes,
        cell_ids,
        _,
        canonical_donor_ids,
        canonical_donor_counts,
    ) = _validated_metadata(
        cell_metadata,
        gene_metadata,
        cell_type_column=cell_type_column,
        selected_cell_type=selected_cell_type,
        allow_mixed_cell_types=allow_mixed_cell_types,
        donor_ids=donor_ids,
        donor_counts=donor_counts,
    )
    input_sha256 = _validated_input_hashes(input_paths)
    donor_lookup = {donor: index for index, donor in enumerate(canonical_donor_ids)}
    cell_donor_index = np.asarray(
        [donor_lookup[donor] for donor in canonical_cells["donor_id"].to_list()],
        dtype=np.int64,
    )
    raw_metrics = {} if approximation_metrics is None else approximation_metrics
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("approximation_metrics must be a chromosome-keyed mapping")
    metrics: dict[str, Mapping[str, Any]] = {}
    for key, value in raw_metrics.items():
        canonical_key = canonical_chromosome_key(key)
        if canonical_key in metrics:
            raise ValueError(f"duplicate canonical approximation metric key: {canonical_key}")
        metrics[canonical_key] = value
    if set(metrics) - set(chromosomes):
        raise ValueError("approximation metrics contain an unrequested chromosome")

    staging = Path(tempfile.mkdtemp(prefix=_STAGING_DIRECTORY_PREFIX, dir=parent))
    published = False
    try:
        payloads = list(
            _write_shared_metadata(
                staging,
                cell_metadata=canonical_cells,
                donor_ids=canonical_donor_ids,
                donor_counts=canonical_donor_counts,
            )
        )
        chromosome_manifests: list[_ChromosomeManifest] = []
        iterator = iter(factor_results)
        for chromosome in chromosomes:
            try:
                result = next(iterator)
            except StopIteration as error:
                raise ValueError("factor result stream ended before all requested chromosomes completed") from error
            factors, loadings, singular_values, active_genes = _validated_factor_result(
                result,
                chromosome=chromosome,
                n_cells=len(cell_ids),
                donor_counts=canonical_donor_counts,
                gene_metadata=canonical_genes,
            )
            chromosome_manifest = _chromosome_manifest(
                result,
                chromosome=chromosome,
                active_genes=active_genes,
                approximation_metrics=metrics.get(chromosome, {}),
            )
            _validate_state_factor_payload_numerics(
                factors,
                loadings,
                singular_values,
                record=chromosome_manifest,
                donor_index=cell_donor_index,
            )
            payloads.extend(
                _write_chromosome_payloads(
                    staging,
                    chromosome=chromosome,
                    factors=factors,
                    loadings=loadings,
                    singular_values=singular_values,
                    active_genes=active_genes,
                )
            )
            chromosome_manifests.append(chromosome_manifest)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("factor result stream contains more results than requested chromosomes")

        manifest = StateArtifactManifest(
            artifact_type=ARTIFACT_TYPE,
            schema_version=SCHEMA_VERSION,
            cell_type_column=cell_type_column,
            selected_cell_type=selected_cell_type,
            allow_mixed_cell_types=allow_mixed_cell_types,
            input_sha256=input_sha256,
            cell_order_hash=identifier_order_hash((("cell_id", cell_ids),)),
            donor_order_hash=identifier_order_hash((("donor_id", canonical_donor_ids),)),
            requested_chromosomes=chromosomes,
            completed_chromosomes=chromosomes,
            n_cells=len(cell_ids),
            n_donors=len(canonical_donor_ids),
            configuration=dict(configuration),
            provenance=_collect_replay_provenance(),
            chromosomes=tuple(chromosome_manifests),
            payloads=tuple(payloads),
        )
        validate_manifest(manifest)
        _write_bytes(staging / "manifest.json", encode_manifest(manifest))
        _fsync_directory(staging)
        _load_state_artifact(staging, allow_staging=True)
        _publish_directory_noreplace(staging, final_path)
        published = True
        _fsync_directory(parent)
        return manifest
    except BaseException:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_frame(path: Path, *, required_columns: Sequence[str], context: str) -> pl.DataFrame:
    try:
        frame = pl.read_parquet(path)
    except Exception as error:
        raise ValueError(f"cannot read {context} Parquet payload: {path}") from error
    _required_columns(frame, required_columns, context=context)
    return frame


def _verified_array(path: Path, payload: _PayloadRecord) -> np.memmap:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot memory-map NumPy payload: {payload.path}") from error
    if not isinstance(array, np.memmap):
        raise ValueError(f"NumPy payload is not memory-mappable: {payload.path}")
    if array.shape != payload.shape:
        raise ValueError(f"NumPy payload shape disagrees with manifest: {payload.path}")
    if array.dtype.str != payload.dtype:
        raise ValueError(f"NumPy payload dtype disagrees with manifest: {payload.path}")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"NumPy payload must use the schema-v1 float64 dtype: {payload.path}")
    array.flags.writeable = False
    return array


def _load_state_artifact(
    root: Path,
    *,
    allow_staging: bool,
    expected_cell_ids: Sequence[str] | None = None,
    expected_donor_ids: Sequence[str] | None = None,
    expected_gene_ids: Mapping[str, Sequence[str]] | None = None,
    expected_configuration: Mapping[str, Any] | None = None,
) -> StateArtifactResult:
    if root.name.startswith(_STAGING_DIRECTORY_PREFIX) and not allow_staging:
        raise ValueError("state artifact loader rejects staging directories")
    if not root.is_dir():
        raise FileNotFoundError(f"state artifact directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"state artifact is missing manifest.json: {root}")
    manifest = decode_manifest(manifest_path.read_bytes())
    normalized_expected_genes = None
    if expected_gene_ids is not None:
        normalized_expected_genes = {}
        for key, value in expected_gene_ids.items():
            canonical_key = canonical_chromosome_key(key)
            if canonical_key in normalized_expected_genes:
                raise ValueError(f"duplicate canonical expected gene key: {canonical_key}")
            normalized_expected_genes[canonical_key] = value
    validate_manifest(
        manifest,
        expected_cell_ids=expected_cell_ids,
        expected_donor_ids=expected_donor_ids,
        expected_gene_ids=normalized_expected_genes,
        expected_configuration=expected_configuration,
    )

    expected_inventory = canonical_payload_inventory(manifest.requested_chromosomes)
    expected_files = {"manifest.json", *expected_inventory}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing = sorted(expected_files - actual_files)
    if missing:
        raise FileNotFoundError(f"state artifact is missing payload: {missing[0]}")
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise ValueError(f"state artifact contains an unexpected payload: {unexpected[0]}")
    payloads = {payload.path: payload for payload in manifest.payloads}
    for relative_path in expected_inventory:
        actual_hash = _sha256_file(root / relative_path)
        if actual_hash != payloads[relative_path].sha256:
            raise ValueError(f"payload SHA-256 mismatch: {relative_path}")

    cells = _load_frame(
        root / "cells.parquet",
        required_columns=("matrix_index", "cell_id", "donor_id", manifest.cell_type_column),
        context="cell metadata",
    )
    _validate_cell_type_selection(
        cells[manifest.cell_type_column].to_list(),
        selected_cell_type=manifest.selected_cell_type,
        allow_mixed_cell_types=manifest.allow_mixed_cell_types,
    )
    donors = _load_frame(
        root / "donors.parquet",
        required_columns=("donor_index", "donor_id", "cell_count"),
        context="donor metadata",
    )
    if cells.height != manifest.n_cells:
        raise ValueError("cell metadata row count disagrees with the manifest")
    if donors.height != manifest.n_donors:
        raise ValueError("donor metadata row count disagrees with the manifest")
    try:
        _ordered_matrix_indices(cells, context="cell metadata order")
        cell_ids = _identifier_column(cells, "cell_id", context="cell metadata order")
    except ValueError as error:
        raise ValueError("cell metadata order is incompatible with the manifest") from error
    if identifier_order_hash((("cell_id", cell_ids),)) != manifest.cell_order_hash:
        raise ValueError("cell metadata order is incompatible with the manifest")
    donor_indices = tuple(donors["donor_index"].to_list())
    if donor_indices != tuple(range(manifest.n_donors)):
        raise ValueError("donor metadata order is incompatible with the manifest")
    try:
        donor_ids = _identifier_column(donors, "donor_id", context="donor metadata order")
    except ValueError as error:
        raise ValueError("donor metadata order is incompatible with the manifest") from error
    if identifier_order_hash((("donor_id", donor_ids),)) != manifest.donor_order_hash:
        raise ValueError("donor metadata order is incompatible with the manifest")
    donor_counts = tuple(donors["cell_count"].to_list())
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in donor_counts):
        raise ValueError("donor metadata cell counts must be positive integers")
    cell_donors = tuple(cells["donor_id"].to_list())
    if any(not isinstance(donor, str) or not donor for donor in cell_donors):
        raise ValueError("cell metadata donor IDs must be nonempty strings")
    if set(cell_donors) != set(donor_ids):
        raise ValueError("cell and donor metadata must exactly cover the same donors")
    if sum(donor_counts) != manifest.n_cells:
        raise ValueError("donor metadata counts must sum to the artifact cell dimension")
    if donor_counts != tuple(cell_donors.count(donor) for donor in donor_ids):
        raise ValueError("donor metadata counts do not align with cell metadata")
    if tuple(dict.fromkeys(cell_donors)) != donor_ids:
        raise ValueError("donor metadata must follow first-retained-cell order")
    donor_lookup = {donor: index for index, donor in enumerate(donor_ids)}
    cell_donor_index = np.asarray([donor_lookup[donor] for donor in cell_donors], dtype=np.int64)
    for record in manifest.chromosomes:
        if donor_counts != record.donor_counts:
            raise ValueError(f"chromosome {record.chromosome} donor counts disagree with shared donor metadata")

    chromosome_manifests = {record.chromosome: record for record in manifest.chromosomes}
    loaded_chromosomes = []
    for chromosome in manifest.requested_chromosomes:
        prefix = f"chromosomes/{chromosome}"
        record = chromosome_manifests[chromosome]
        genes = _load_frame(
            root / prefix / "genes.parquet",
            required_columns=("matrix_index", "gene_id", "chrom"),
            context=f"chromosome {chromosome} gene metadata",
        )
        if genes.height != record.n_genes:
            raise ValueError(f"chromosome {chromosome} gene metadata row count disagrees with the manifest")
        try:
            _ordered_matrix_indices(genes, context="gene metadata order")
            gene_ids = _identifier_column(genes, "gene_id", context="gene metadata order")
        except ValueError as error:
            raise ValueError(f"chromosome {chromosome} gene metadata order is incompatible") from error
        gene_chromosomes = _validated_gene_chromosomes(
            genes["chrom"].to_list(),
            context=f"chromosome {chromosome} gene metadata chrom",
        )
        if identifier_order_hash((("gene_id", gene_ids),)) != record.gene_order_hash:
            raise ValueError(f"chromosome {chromosome} gene metadata order is incompatible with the manifest")
        if str(int(chromosome)) in gene_chromosomes:
            raise ValueError(f"chromosome {chromosome} gene metadata retains a gene from the excluded chromosome")
        factors = _verified_array(root / prefix / "factors.npy", payloads[f"{prefix}/factors.npy"])
        loadings = _verified_array(root / prefix / "loadings.npy", payloads[f"{prefix}/loadings.npy"])
        singular_values = _verified_array(
            root / prefix / "singular_values.npy", payloads[f"{prefix}/singular_values.npy"]
        )
        if not np.isfinite(factors).all() or not np.isfinite(loadings).all() or not np.isfinite(singular_values).all():
            raise ValueError(f"chromosome {chromosome} memory-mapped arrays must be finite")
        if not np.array_equal(singular_values, np.asarray(record.singular_values)):
            raise ValueError(f"chromosome {chromosome} singular values disagree with the manifest diagnostics")
        _validate_state_factor_payload_numerics(
            factors,
            loadings,
            singular_values,
            record=record,
            donor_index=cell_donor_index,
        )
        loaded_chromosomes.append(
            _StateArtifactChromosomeResult(
                chromosome=chromosome,
                factors=factors,
                loadings=loadings,
                singular_values=singular_values,
                gene_ids=gene_ids,
            )
        )
    return StateArtifactResult(
        root=root,
        manifest=manifest,
        cell_ids=cell_ids,
        donor_ids=donor_ids,
        chromosomes=tuple(loaded_chromosomes),
    )


def load_state_artifact(
    root: str | os.PathLike[str],
    *,
    expected_cell_ids: Sequence[str] | None = None,
    expected_donor_ids: Sequence[str] | None = None,
    expected_gene_ids: Mapping[str, Sequence[str]] | None = None,
    expected_configuration: Mapping[str, Any] | None = None,
) -> StateArtifactResult:
    r"""Load and fully validate one state-factor artifact with read-only mmap arrays.

    **Arguments:**

    root
        Published artifact directory. Staging-directory names are rejected.
    expected_cell_ids, expected_donor_ids, expected_gene_ids
        Optional exact identity/order contracts checked before returning arrays.
    expected_configuration
        Optional fully resolved configuration that must match exactly.

    **Returns:**

    A backend-neutral result containing read-only ``numpy.memmap`` arrays.

    **Raises:**

    FileNotFoundError
        If the artifact or a declared payload is missing.
    ValueError, TypeError
        If schema, hashes, shapes, dtypes, metadata order, alignment, or configuration is incompatible.
    """
    return _load_state_artifact(
        Path(root),
        allow_staging=False,
        expected_cell_ids=expected_cell_ids,
        expected_donor_ids=expected_donor_ids,
        expected_gene_ids=expected_gene_ids,
        expected_configuration=expected_configuration,
    )
