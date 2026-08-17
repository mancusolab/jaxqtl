# pattern: Functional Core

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast, Literal

import numpy as np


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "jaxqtl-state-factor"
CANONICAL_CHROMOSOMES = tuple(f"{chromosome:02d}" for chromosome in range(1, 23))
INPUT_NAMES = ("counts", "cells", "genes")
PROVENANCE_PACKAGES = ("numpy", "scipy", "polars", "pyarrow", "jax", "jaxlib", "jaxqtl")
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FIELDS = (
    "artifact_type",
    "schema_version",
    "selection",
    "inputs",
    "ordering",
    "requested_chromosomes",
    "completed_chromosomes",
    "dimensions",
    "configuration",
    "provenance",
    "chromosomes",
    "payloads",
)
_SELECTION_FIELDS = ("cell_type_column", "selected_cell_type", "allow_mixed_cell_types")
_ORDERING_FIELDS = ("cell_order_sha256", "donor_order_sha256")
_DIMENSION_FIELDS = ("cells", "donors")
_CHROMOSOME_FIELDS = (
    "dimensions",
    "dtypes",
    "gene_order_sha256",
    "pflog",
    "filtering",
    "donor_counts",
    "center_within_donor",
    "balance_donors",
    "solver",
    "singular_values",
    "convergence",
    "approximation",
)
_CHROMOSOME_DIMENSION_FIELDS = ("cells", "genes", "rank")
_DTYPE_FIELDS = ("factors", "loadings", "singular_values")
_PAYLOAD_FIELDS = ("path", "sha256", "kind", "shape", "dtype", "rows")
_PROVENANCE_FIELDS = (
    "jaxqtl_version",
    "artifact_schema_version",
    "python",
    "system",
    "package_versions",
    "blas_lapack",
    "thread_environment",
    "platform",
)
_PYTHON_FIELDS = ("implementation", "version")
_SYSTEM_FIELDS = ("os", "release", "machine", "processor")


@dataclass(frozen=True, slots=True)
class _ReplayProvenance:
    jaxqtl_version: str
    artifact_schema_version: int
    python_implementation: str
    python_version: str
    os_name: str
    os_release: str
    machine: str
    processor: str
    package_versions: tuple[tuple[str, str], ...]
    blas_lapack: tuple[tuple[str, Mapping[str, Any]], ...]
    thread_environment: tuple[tuple[str, str | None], ...]
    platform: Literal["cpu"]


@dataclass(frozen=True, slots=True)
class _PayloadRecord:
    path: str
    sha256: str
    kind: Literal["npy", "parquet"]
    shape: tuple[int, ...] | None
    dtype: str | None
    rows: int | None


@dataclass(frozen=True, slots=True)
class _ChromosomeManifest:
    chromosome: str
    n_cells: int
    n_genes: int
    rank: int
    factors_dtype: str
    loadings_dtype: str
    singular_values_dtype: str
    gene_order_hash: str
    pflog_diagnostics: Mapping[str, Any]
    filtering_diagnostics: Mapping[str, Any]
    donor_counts: tuple[int, ...]
    center_within_donor: bool
    balance_donors: bool
    solver_configuration: Mapping[str, Any]
    singular_values: tuple[float, ...]
    convergence_residuals: Mapping[str, Any]
    approximation_metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StateArtifactManifest:
    r"""Versioned, backend-neutral state-factor artifact contract."""

    artifact_type: str
    schema_version: int
    cell_type_column: str
    selected_cell_type: str | None
    allow_mixed_cell_types: bool
    input_sha256: tuple[tuple[str, str], ...]
    cell_order_hash: str
    donor_order_hash: str
    requested_chromosomes: tuple[str, ...]
    completed_chromosomes: tuple[str, ...]
    n_cells: int
    n_donors: int
    configuration: Mapping[str, Any]
    provenance: _ReplayProvenance
    chromosomes: tuple[_ChromosomeManifest, ...]
    payloads: tuple[_PayloadRecord, ...]


@dataclass(frozen=True, slots=True)
class _StateArtifactChromosomeResult:
    chromosome: str
    factors: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    gene_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StateArtifactResult:
    r"""Validated, read-only memory-mapped state-factor artifact."""

    root: Path
    manifest: StateArtifactManifest
    cell_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    chromosomes: tuple[_StateArtifactChromosomeResult, ...]

    def chromosome(self, chromosome: str) -> _StateArtifactChromosomeResult:
        r"""Return one canonical chromosome result or raise ``KeyError``."""
        key = canonical_chromosome_key(chromosome)
        for result in self.chromosomes:
            if result.chromosome == key:
                return result
        raise KeyError(f"chromosome {key} is not present in this state artifact")


def canonical_chromosome_key(chromosome: str) -> str:
    r"""Normalize an autosome label to the artifact directory key ``01``-``22``."""
    if not isinstance(chromosome, str) or not chromosome:
        raise ValueError("chromosome must be one of the canonical autosomes 1-22")
    raw = chromosome.strip()
    if not raw.isdigit():
        raise ValueError("chromosome must be one of the canonical autosomes 1-22")
    value = int(raw)
    if not 1 <= value <= 22:
        raise ValueError("chromosome must be one of the canonical autosomes 1-22")
    return f"{value:02d}"


def _validated_chromosome_set(chromosomes: Sequence[str], *, require_canonical: bool) -> tuple[str, ...]:
    if isinstance(chromosomes, (str, bytes)):
        raise TypeError("chromosomes must be an ordered sequence")
    values = tuple(chromosomes)
    normalized = tuple(canonical_chromosome_key(value) for value in values)
    if require_canonical and values != normalized:
        raise ValueError("manifest chromosomes must use canonical chromosome keys 01-22")
    if len(set(normalized)) != len(normalized):
        raise ValueError("requested chromosomes must not contain duplicates")
    if len(normalized) == 1:
        return normalized
    if normalized == CANONICAL_CHROMOSOMES:
        return normalized
    raise ValueError("a state artifact must request one or all 22 canonical chromosomes")


def canonical_payload_inventory(chromosomes: Sequence[str]) -> tuple[str, ...]:
    r"""Return the sole canonical payload layout, excluding ``manifest.json``."""
    keys = _validated_chromosome_set(chromosomes, require_canonical=True)
    inventory = ["cells.parquet", "donors.parquet"]
    for chromosome in keys:
        prefix = f"chromosomes/{chromosome}"
        inventory.extend(
            (
                f"{prefix}/factors.npy",
                f"{prefix}/loadings.npy",
                f"{prefix}/singular_values.npy",
                f"{prefix}/genes.parquet",
            )
        )
    return tuple(inventory)


def _hash_length_prefixed(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big", signed=False))
    hasher.update(encoded)


def identifier_order_hash(fields: Sequence[tuple[str, Sequence[str]]]) -> str:
    r"""Hash ordered UTF-8 identifier fields with explicit field and position framing."""
    hasher = hashlib.sha256()
    hasher.update(b"jaxqtl-identifier-order-v1\x00")
    seen: set[str] = set()
    for field_position, (field_name, values) in enumerate(fields):
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("identifier field names must be nonempty strings")
        if field_name in seen:
            raise ValueError(f"duplicate identifier field: {field_name}")
        seen.add(field_name)
        hasher.update(b"\x1eFIELD")
        hasher.update(field_position.to_bytes(8, "big", signed=False))
        _hash_length_prefixed(hasher, field_name)
        hasher.update(len(values).to_bytes(8, "big", signed=False))
        for value_position, value in enumerate(values):
            if not isinstance(value, str):
                raise TypeError("identifier values must be strings before hashing")
            hasher.update(b"\x1fVALUE")
            hasher.update(value_position.to_bytes(8, "big", signed=False))
            _hash_length_prefixed(hasher, value)
    return hasher.hexdigest()


def _require_exact_fields(value: object, fields: Sequence[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{context} fields are incompatible; missing={missing}, unknown={unknown}")
    return cast(Mapping[str, Any], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _validated_dtype(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a NumPy dtype string")
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a valid NumPy dtype string") from error
    if dtype.hasobject:
        raise ValueError(f"{name} cannot use object storage")
    return dtype.str


def _validated_json(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} floating values must be finite")
        return value
    if isinstance(value, Mapping):
        validated: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} JSON object keys must be strings")
            validated[key] = _validated_json(item, context=f"{context}.{key}")
        return validated
    if isinstance(value, (tuple, list)):
        return [_validated_json(item, context=f"{context}[]") for item in value]
    raise TypeError(f"{context} contains a value that is not JSON-compatible")


def _safe_payload_path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("payload paths must be safe canonical relative paths")
    posix = PurePosixPath(path)
    if posix.is_absolute() or PureWindowsPath(path).is_absolute() or posix.as_posix() != path:
        raise ValueError("payload paths must be safe canonical relative paths")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("payload paths must be safe canonical relative paths")
    return posix


def _validate_provenance(provenance: _ReplayProvenance) -> None:
    if provenance.artifact_schema_version != SCHEMA_VERSION:
        raise ValueError("provenance artifact schema version disagrees with the manifest schema version")
    required_strings = (
        provenance.jaxqtl_version,
        provenance.python_implementation,
        provenance.python_version,
        provenance.os_name,
        provenance.os_release,
        provenance.machine,
        provenance.processor,
    )
    if any(not isinstance(value, str) or not value for value in required_strings):
        raise ValueError("replay provenance string fields must be nonempty")
    if tuple(name for name, _ in provenance.package_versions) != PROVENANCE_PACKAGES:
        raise ValueError("replay provenance package inventory is incomplete or noncanonical")
    if any(not isinstance(version, str) or not version for _, version in provenance.package_versions):
        raise ValueError("replay provenance package versions must be nonempty strings")
    if tuple(name for name, _ in provenance.blas_lapack) != ("numpy", "scipy"):
        raise ValueError("replay provenance must contain normalized NumPy and SciPy BLAS/LAPACK information")
    for name, information in provenance.blas_lapack:
        _validated_json(information, context=f"provenance.blas_lapack.{name}")
    if tuple(name for name, _ in provenance.thread_environment) != THREAD_ENVIRONMENT_VARIABLES:
        raise ValueError("replay provenance thread environment inventory is incomplete or noncanonical")
    if any(value is not None and not isinstance(value, str) for _, value in provenance.thread_environment):
        raise TypeError("thread-count environment values must be strings or null")
    if provenance.platform != "cpu":
        raise ValueError("state artifact platform must be fixed to cpu")


def _validate_chromosome_manifest(record: _ChromosomeManifest, *, n_cells: int, n_donors: int) -> None:
    if record.chromosome not in CANONICAL_CHROMOSOMES:
        raise ValueError("manifest chromosomes must use canonical chromosome keys 01-22")
    if record.n_cells != n_cells:
        raise ValueError(f"chromosome {record.chromosome} cell dimension disagrees with the artifact")
    _positive_integer(record.n_genes, name=f"chromosome {record.chromosome} genes")
    _positive_integer(record.rank, name=f"chromosome {record.chromosome} rank")
    if record.rank >= min(record.n_cells, record.n_genes):
        raise ValueError(f"chromosome {record.chromosome} rank must be strictly truncated")
    _validated_dtype(record.factors_dtype, name="factors dtype")
    _validated_dtype(record.loadings_dtype, name="loadings dtype")
    _validated_dtype(record.singular_values_dtype, name="singular-values dtype")
    _validated_sha256(record.gene_order_hash, name="gene order hash")
    if len(record.donor_counts) != n_donors:
        raise ValueError(f"chromosome {record.chromosome} donor counts disagree with the donor dimension")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in record.donor_counts):
        raise ValueError("donor counts must contain positive integers")
    if sum(record.donor_counts) != n_cells:
        raise ValueError("donor counts must sum to the artifact cell dimension")
    if len(record.singular_values) != record.rank:
        raise ValueError("singular-values diagnostics must agree with rank")
    if any(not math.isfinite(value) or value <= 0.0 for value in record.singular_values):
        raise ValueError("singular values must be finite and strictly positive")
    if tuple(sorted(record.singular_values, reverse=True)) != record.singular_values:
        raise ValueError("singular values must be recorded in descending order")
    _validated_json(record.pflog_diagnostics, context="pflog diagnostics")
    _validated_json(record.filtering_diagnostics, context="filtering diagnostics")
    _validated_json(record.solver_configuration, context="solver configuration")
    _validated_json(record.convergence_residuals, context="convergence diagnostics")
    _validated_json(record.approximation_metrics, context="approximation metrics")


def _validate_payloads(manifest: StateArtifactManifest) -> None:
    paths = tuple(payload.path for payload in manifest.payloads)
    if len(paths) != len(set(paths)):
        raise ValueError("manifest contains a duplicate payload inventory entry")
    for payload in manifest.payloads:
        _safe_payload_path(payload.path)
        _validated_sha256(payload.sha256, name=f"payload {payload.path} hash")
        if payload.kind == "npy":
            if payload.shape is None or not payload.shape:
                raise ValueError(f"NumPy payload {payload.path} must declare a nonempty shape")
            if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in payload.shape):
                raise ValueError(f"NumPy payload {payload.path} shape dimensions must be positive integers")
            _validated_dtype(payload.dtype, name=f"payload {payload.path} dtype")
            if payload.rows is not None:
                raise ValueError(f"NumPy payload {payload.path} cannot declare Parquet rows")
        elif payload.kind == "parquet":
            if payload.shape is not None or payload.dtype is not None:
                raise ValueError(f"Parquet payload {payload.path} cannot declare NumPy shape or dtype")
            _positive_integer(payload.rows, name=f"payload {payload.path} rows")
        else:
            raise ValueError(f"payload {payload.path} has an unsupported kind")

    expected_inventory = canonical_payload_inventory(manifest.requested_chromosomes)
    if paths != expected_inventory:
        raise ValueError("manifest payloads must match the canonical payload inventory exactly")

    by_path = {payload.path: payload for payload in manifest.payloads}
    if by_path["cells.parquet"].rows != manifest.n_cells:
        raise ValueError("cell metadata row count disagrees with the cell dimension")
    if by_path["donors.parquet"].rows != manifest.n_donors:
        raise ValueError("donor metadata row count disagrees with the donor dimension")
    chromosome_records = {record.chromosome: record for record in manifest.chromosomes}
    for chromosome in manifest.requested_chromosomes:
        record = chromosome_records[chromosome]
        prefix = f"chromosomes/{chromosome}"
        factors = by_path[f"{prefix}/factors.npy"]
        loadings = by_path[f"{prefix}/loadings.npy"]
        singular_values = by_path[f"{prefix}/singular_values.npy"]
        genes = by_path[f"{prefix}/genes.parquet"]
        if factors.shape != (record.n_cells, record.rank):
            raise ValueError(f"chromosome {chromosome} factors shape disagrees with dimensions")
        if loadings.shape != (record.n_genes, record.rank):
            raise ValueError(f"chromosome {chromosome} loadings shape disagrees with dimensions")
        if singular_values.shape != (record.rank,):
            raise ValueError(f"chromosome {chromosome} singular-values shape disagrees with rank")
        if factors.dtype != np.dtype(record.factors_dtype).str:
            raise ValueError(f"chromosome {chromosome} factors dtype disagrees with diagnostics")
        if loadings.dtype != np.dtype(record.loadings_dtype).str:
            raise ValueError(f"chromosome {chromosome} loadings dtype disagrees with diagnostics")
        if singular_values.dtype != np.dtype(record.singular_values_dtype).str:
            raise ValueError(f"chromosome {chromosome} singular-values dtype disagrees with diagnostics")
        if genes.rows != record.n_genes:
            raise ValueError(f"chromosome {chromosome} gene metadata row count disagrees with dimensions")


def validate_manifest(
    manifest: StateArtifactManifest,
    *,
    expected_cell_ids: Sequence[str] | None = None,
    expected_donor_ids: Sequence[str] | None = None,
    expected_gene_ids: Mapping[str, Sequence[str]] | None = None,
    expected_configuration: Mapping[str, Any] | None = None,
) -> StateArtifactManifest:
    r"""Purely validate schema compatibility, completeness, and optional alignment."""
    if not isinstance(manifest, StateArtifactManifest):
        raise TypeError("manifest must be a StateArtifactManifest")
    if manifest.artifact_type != ARTIFACT_TYPE:
        raise ValueError(f"unsupported artifact type: {manifest.artifact_type!r}")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported state artifact schema version: {manifest.schema_version!r}")
    if not isinstance(manifest.cell_type_column, str) or not manifest.cell_type_column:
        raise ValueError("cell type column must be a nonempty string")
    if manifest.allow_mixed_cell_types:
        if manifest.selected_cell_type is not None:
            raise ValueError("mixed-cell opt-in cannot also record a selected cell type")
    elif not isinstance(manifest.selected_cell_type, str) or not manifest.selected_cell_type:
        raise ValueError("a selected cell type is required without mixed-cell opt-in")
    if tuple(name for name, _ in manifest.input_sha256) != INPUT_NAMES:
        raise ValueError("input hashes must contain counts, cells, and genes in canonical order")
    for name, digest in manifest.input_sha256:
        _validated_sha256(digest, name=f"{name} input hash")
    _validated_sha256(manifest.cell_order_hash, name="cell order hash")
    _validated_sha256(manifest.donor_order_hash, name="donor order hash")
    requested = _validated_chromosome_set(manifest.requested_chromosomes, require_canonical=True)
    if tuple(manifest.completed_chromosomes) != requested:
        raise ValueError("completed chromosomes must exactly match the complete requested chromosome set")
    n_cells = _positive_integer(manifest.n_cells, name="cells")
    n_donors = _positive_integer(manifest.n_donors, name="donors")
    if n_donors > n_cells:
        raise ValueError("donors cannot outnumber cells")
    _validated_json(manifest.configuration, context="configuration")
    _validate_provenance(manifest.provenance)
    chromosome_keys = tuple(record.chromosome for record in manifest.chromosomes)
    if chromosome_keys != requested:
        raise ValueError("chromosome diagnostics must exactly match requested chromosomes in order")
    for record in manifest.chromosomes:
        _validate_chromosome_manifest(record, n_cells=n_cells, n_donors=n_donors)
    _validate_payloads(manifest)

    if expected_cell_ids is not None:
        expected = tuple(expected_cell_ids)
        if identifier_order_hash((("cell_id", expected),)) != manifest.cell_order_hash:
            raise ValueError("state artifact cell order is incompatible with the expected cell order")
    if expected_donor_ids is not None:
        expected = tuple(expected_donor_ids)
        if identifier_order_hash((("donor_id", expected),)) != manifest.donor_order_hash:
            raise ValueError("state artifact donor order is incompatible with the expected donor order")
    if expected_gene_ids is not None:
        if set(expected_gene_ids) != set(requested):
            raise ValueError("expected gene order must cover every requested chromosome exactly")
        by_chromosome = {record.chromosome: record for record in manifest.chromosomes}
        for chromosome in requested:
            expected_hash = identifier_order_hash((("gene_id", tuple(expected_gene_ids[chromosome])),))
            if expected_hash != by_chromosome[chromosome].gene_order_hash:
                raise ValueError(f"state artifact gene order for chromosome {chromosome} is incompatible")
    if expected_configuration is not None:
        actual_json = _canonical_json_text(manifest.configuration)
        expected_json = _canonical_json_text(expected_configuration)
        if actual_json != expected_json:
            raise ValueError("state artifact configuration is incompatible with the expected configuration")
    return manifest


def _provenance_to_dict(provenance: _ReplayProvenance) -> dict[str, Any]:
    return {
        "jaxqtl_version": provenance.jaxqtl_version,
        "artifact_schema_version": provenance.artifact_schema_version,
        "python": {
            "implementation": provenance.python_implementation,
            "version": provenance.python_version,
        },
        "system": {
            "os": provenance.os_name,
            "release": provenance.os_release,
            "machine": provenance.machine,
            "processor": provenance.processor,
        },
        "package_versions": dict(provenance.package_versions),
        "blas_lapack": {name: copy.deepcopy(dict(value)) for name, value in provenance.blas_lapack},
        "thread_environment": dict(provenance.thread_environment),
        "platform": provenance.platform,
    }


def _chromosome_to_dict(record: _ChromosomeManifest) -> dict[str, Any]:
    return {
        "dimensions": {"cells": record.n_cells, "genes": record.n_genes, "rank": record.rank},
        "dtypes": {
            "factors": record.factors_dtype,
            "loadings": record.loadings_dtype,
            "singular_values": record.singular_values_dtype,
        },
        "gene_order_sha256": record.gene_order_hash,
        "pflog": copy.deepcopy(dict(record.pflog_diagnostics)),
        "filtering": copy.deepcopy(dict(record.filtering_diagnostics)),
        "donor_counts": list(record.donor_counts),
        "center_within_donor": record.center_within_donor,
        "balance_donors": record.balance_donors,
        "solver": copy.deepcopy(dict(record.solver_configuration)),
        "singular_values": list(record.singular_values),
        "convergence": copy.deepcopy(dict(record.convergence_residuals)),
        "approximation": copy.deepcopy(dict(record.approximation_metrics)),
    }


def manifest_to_dict(manifest: StateArtifactManifest) -> dict[str, Any]:
    r"""Convert a validated manifest to its exact JSON object contract."""
    validate_manifest(manifest)
    return {
        "artifact_type": manifest.artifact_type,
        "schema_version": manifest.schema_version,
        "selection": {
            "cell_type_column": manifest.cell_type_column,
            "selected_cell_type": manifest.selected_cell_type,
            "allow_mixed_cell_types": manifest.allow_mixed_cell_types,
        },
        "inputs": dict(manifest.input_sha256),
        "ordering": {
            "cell_order_sha256": manifest.cell_order_hash,
            "donor_order_sha256": manifest.donor_order_hash,
        },
        "requested_chromosomes": list(manifest.requested_chromosomes),
        "completed_chromosomes": list(manifest.completed_chromosomes),
        "dimensions": {"cells": manifest.n_cells, "donors": manifest.n_donors},
        "configuration": copy.deepcopy(dict(manifest.configuration)),
        "provenance": _provenance_to_dict(manifest.provenance),
        "chromosomes": {record.chromosome: _chromosome_to_dict(record) for record in manifest.chromosomes},
        "payloads": [
            {
                "path": payload.path,
                "sha256": payload.sha256,
                "kind": payload.kind,
                "shape": None if payload.shape is None else list(payload.shape),
                "dtype": payload.dtype,
                "rows": payload.rows,
            }
            for payload in manifest.payloads
        ],
    }


def _parse_provenance(raw: object) -> _ReplayProvenance:
    value = _require_exact_fields(raw, _PROVENANCE_FIELDS, context="provenance")
    python = _require_exact_fields(value["python"], _PYTHON_FIELDS, context="provenance.python")
    system = _require_exact_fields(value["system"], _SYSTEM_FIELDS, context="provenance.system")
    packages = _require_exact_fields(value["package_versions"], PROVENANCE_PACKAGES, context="package versions")
    blas = _require_exact_fields(value["blas_lapack"], ("numpy", "scipy"), context="BLAS/LAPACK provenance")
    threads = _require_exact_fields(
        value["thread_environment"], THREAD_ENVIRONMENT_VARIABLES, context="thread environment"
    )
    return _ReplayProvenance(
        jaxqtl_version=value["jaxqtl_version"],
        artifact_schema_version=value["artifact_schema_version"],
        python_implementation=python["implementation"],
        python_version=python["version"],
        os_name=system["os"],
        os_release=system["release"],
        machine=system["machine"],
        processor=system["processor"],
        package_versions=tuple((name, packages[name]) for name in PROVENANCE_PACKAGES),
        blas_lapack=tuple((name, copy.deepcopy(blas[name])) for name in ("numpy", "scipy")),
        thread_environment=tuple((name, threads[name]) for name in THREAD_ENVIRONMENT_VARIABLES),
        platform=value["platform"],
    )


def _parse_chromosome(chromosome: str, raw: object) -> _ChromosomeManifest:
    value = _require_exact_fields(raw, _CHROMOSOME_FIELDS, context=f"chromosome {chromosome}")
    dimensions = _require_exact_fields(
        value["dimensions"], _CHROMOSOME_DIMENSION_FIELDS, context=f"chromosome {chromosome} dimensions"
    )
    dtypes = _require_exact_fields(value["dtypes"], _DTYPE_FIELDS, context=f"chromosome {chromosome} dtypes")
    for name in ("pflog", "filtering", "solver", "convergence", "approximation"):
        if not isinstance(value[name], Mapping):
            raise TypeError(f"chromosome {chromosome} {name} must be a JSON object")
    donor_counts = value["donor_counts"]
    singular_values = value["singular_values"]
    if not isinstance(donor_counts, list) or not isinstance(singular_values, list):
        raise TypeError(f"chromosome {chromosome} donor counts and singular values must be JSON arrays")
    return _ChromosomeManifest(
        chromosome=chromosome,
        n_cells=dimensions["cells"],
        n_genes=dimensions["genes"],
        rank=dimensions["rank"],
        factors_dtype=dtypes["factors"],
        loadings_dtype=dtypes["loadings"],
        singular_values_dtype=dtypes["singular_values"],
        gene_order_hash=value["gene_order_sha256"],
        pflog_diagnostics=copy.deepcopy(value["pflog"]),
        filtering_diagnostics=copy.deepcopy(value["filtering"]),
        donor_counts=tuple(donor_counts),
        center_within_donor=value["center_within_donor"],
        balance_donors=value["balance_donors"],
        solver_configuration=copy.deepcopy(value["solver"]),
        singular_values=tuple(singular_values),
        convergence_residuals=copy.deepcopy(value["convergence"]),
        approximation_metrics=copy.deepcopy(value["approximation"]),
    )


def _parse_payload(raw: object) -> _PayloadRecord:
    value = _require_exact_fields(raw, _PAYLOAD_FIELDS, context="payload")
    shape = value["shape"]
    if shape is not None and not isinstance(shape, list):
        raise TypeError("payload shape must be a JSON array or null")
    return _PayloadRecord(
        path=value["path"],
        sha256=value["sha256"],
        kind=value["kind"],
        shape=None if shape is None else tuple(shape),
        dtype=value["dtype"],
        rows=value["rows"],
    )


def manifest_from_dict(raw: object) -> StateArtifactManifest:
    r"""Decode and validate an exact state-artifact JSON object."""
    value = _require_exact_fields(raw, _ARTIFACT_FIELDS, context="manifest")
    selection = _require_exact_fields(value["selection"], _SELECTION_FIELDS, context="selection")
    inputs = _require_exact_fields(value["inputs"], INPUT_NAMES, context="inputs")
    ordering = _require_exact_fields(value["ordering"], _ORDERING_FIELDS, context="ordering")
    dimensions = _require_exact_fields(value["dimensions"], _DIMENSION_FIELDS, context="dimensions")
    requested = value["requested_chromosomes"]
    completed = value["completed_chromosomes"]
    chromosomes = value["chromosomes"]
    payloads = value["payloads"]
    if not isinstance(requested, list) or not isinstance(completed, list):
        raise TypeError("requested and completed chromosomes must be JSON arrays")
    if not isinstance(chromosomes, Mapping):
        raise TypeError("chromosome diagnostics must be a JSON object")
    if not isinstance(payloads, list):
        raise TypeError("payload inventory must be a JSON array")
    manifest = StateArtifactManifest(
        artifact_type=value["artifact_type"],
        schema_version=value["schema_version"],
        cell_type_column=selection["cell_type_column"],
        selected_cell_type=selection["selected_cell_type"],
        allow_mixed_cell_types=selection["allow_mixed_cell_types"],
        input_sha256=tuple((name, inputs[name]) for name in INPUT_NAMES),
        cell_order_hash=ordering["cell_order_sha256"],
        donor_order_hash=ordering["donor_order_sha256"],
        requested_chromosomes=tuple(requested),
        completed_chromosomes=tuple(completed),
        n_cells=dimensions["cells"],
        n_donors=dimensions["donors"],
        configuration=copy.deepcopy(value["configuration"]),
        provenance=_parse_provenance(value["provenance"]),
        chromosomes=tuple(_parse_chromosome(chromosome, record) for chromosome, record in chromosomes.items()),
        payloads=tuple(_parse_payload(payload) for payload in payloads),
    )
    return validate_manifest(manifest)


def _canonical_json_text(value: Any) -> str:
    validated = _validated_json(value, context="JSON value")
    return json.dumps(validated, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def encode_manifest(manifest: StateArtifactManifest) -> bytes:
    r"""Encode a manifest as deterministic canonical UTF-8 JSON with a final newline."""
    return (_canonical_json_text(manifest_to_dict(manifest)) + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def decode_manifest(payload: bytes | str) -> StateArtifactManifest:
    r"""Decode canonical manifest JSON while rejecting duplicate object keys."""
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("manifest must be valid UTF-8 JSON") from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("manifest payload must be bytes or text")
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("manifest must be valid JSON") from error
    return manifest_from_dict(raw)
