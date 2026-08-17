# pattern: Mixed (unavoidable)
# Reason: The phase-owned test module covers the paired pure contract and real-filesystem shell boundary.

from __future__ import annotations

import errno
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from hypothesis import example, given, settings, strategies as st

import jaxqtl.io as qtl_io

from jaxqtl.state import StateFactorDiagnostics, StateFactorResult


_FULL_AUTOSOMES = tuple(f"{chromosome:02d}" for chromosome in range(1, 23))
_PROPERTY_SETTINGS = settings(max_examples=24, derandomize=True, deadline=None)


def _contract():
    spec = importlib.util.find_spec("jaxqtl.io._state_artifact_contract")
    assert spec is not None, "state artifact contract module is missing"
    return importlib.import_module("jaxqtl.io._state_artifact_contract")


def _provenance():
    contract = _contract()
    return contract._ReplayProvenance(
        jaxqtl_version="0.test",
        artifact_schema_version=1,
        python_implementation="CPython",
        python_version="3.11.0",
        os_name="TestOS",
        os_release="1",
        machine="test-machine",
        processor="test-processor",
        package_versions=tuple((name, "test") for name in contract.PROVENANCE_PACKAGES),
        blas_lapack=(
            ("numpy", {"blas": "test-blas", "lapack": "test-lapack"}),
            ("scipy", {"blas": "test-blas", "lapack": "test-lapack"}),
        ),
        thread_environment=tuple((name, None) for name in contract.THREAD_ENVIRONMENT_VARIABLES),
        platform="cpu",
    )


def _configuration(chromosomes: tuple[str, ...]) -> dict[str, Any]:
    is_loco = len(chromosomes) == 22
    return {
        "allow_mixed_cell_types": False,
        "balance_donors": True,
        "cell_type": "B",
        "center_within_donor": True,
        "exclude_chromosome": None if is_loco else "1",
        "loco": is_loco,
        "maxiter": 20,
        "ncv": None,
        "pflog_alpha": "auto",
        "platform": "cpu",
        "rank": 1,
        "seed": 7,
        "solver": "propack",
        "tol": 1e-8,
        "verbose": False,
    }


def _manifest(
    chromosomes: tuple[str, ...] = ("01",),
    *,
    cell_ids: tuple[str, ...] = ("cell-0", "cell-λ"),
    donor_ids: tuple[str, ...] = ("donor-0",),
    configuration: dict[str, Any] | None = None,
):
    contract = _contract()
    gene_ids = ("gene-0", "gene-β")
    cell_hash = contract.identifier_order_hash((("cell_id", cell_ids),))
    donor_hash = contract.identifier_order_hash((("donor_id", donor_ids),))
    gene_hash = contract.identifier_order_hash((("gene_id", gene_ids),))
    chromosome_records = tuple(
        contract._ChromosomeManifest(
            chromosome=chromosome,
            n_cells=len(cell_ids),
            n_genes=len(gene_ids),
            rank=1,
            factors_dtype="<f8",
            loadings_dtype="<f8",
            singular_values_dtype="<f8",
            gene_order_hash=gene_hash,
            pflog_diagnostics={
                "alpha": 1.25,
                "alpha_source": "auto",
                "retained_gene_count": 2,
                "excluded_gene_count": 0,
                "numerator": 5.0,
                "denominator": 4.0,
                "excluded_numerator": 0.0,
                "excluded_denominator": 0.0,
            },
            filtering_diagnostics={"active_gene_count": 2, "excluded_gene_count": 0},
            donor_counts=(len(cell_ids),),
            center_within_donor=True,
            balance_donors=True,
            solver_configuration={
                "solver": "propack",
                "seed": 7,
                "tol": 1e-8,
                "maxiter": 20,
                "propack_kmax": 20,
                "arpack_ncv": None,
            },
            singular_values=(2.5,),
            convergence_residuals={
                "sigma_floor": 2.5e-8,
                "residual_limit": 1e-7,
                "max_forward_residual": 1e-10,
                "max_adjoint_residual": 2e-10,
                "loading_orthogonality_error": 3e-10,
            },
            approximation_metrics={},
        )
        for chromosome in chromosomes
    )
    inventory = contract.canonical_payload_inventory(chromosomes)
    payloads = []
    for path in inventory:
        parts = path.split("/")
        if path == "cells.parquet":
            payloads.append(contract._PayloadRecord(path, "a" * 64, "parquet", None, None, len(cell_ids)))
        elif path == "donors.parquet":
            payloads.append(contract._PayloadRecord(path, "b" * 64, "parquet", None, None, len(donor_ids)))
        elif parts[-1] == "genes.parquet":
            payloads.append(contract._PayloadRecord(path, "c" * 64, "parquet", None, None, 2))
        elif parts[-1] == "factors.npy":
            payloads.append(contract._PayloadRecord(path, "d" * 64, "npy", (len(cell_ids), 1), "<f8", None))
        elif parts[-1] == "loadings.npy":
            payloads.append(contract._PayloadRecord(path, "e" * 64, "npy", (2, 1), "<f8", None))
        else:
            payloads.append(contract._PayloadRecord(path, "f" * 64, "npy", (1,), "<f8", None))
    return contract.StateArtifactManifest(
        artifact_type=contract.ARTIFACT_TYPE,
        schema_version=contract.SCHEMA_VERSION,
        cell_type_column="cell_type",
        selected_cell_type="B",
        allow_mixed_cell_types=False,
        input_sha256=(("counts", "1" * 64), ("cells", "2" * 64), ("genes", "3" * 64)),
        cell_order_hash=cell_hash,
        donor_order_hash=donor_hash,
        requested_chromosomes=chromosomes,
        completed_chromosomes=chromosomes,
        n_cells=len(cell_ids),
        n_donors=len(donor_ids),
        configuration=_configuration(chromosomes) if configuration is None else configuration,
        provenance=_provenance(),
        chromosomes=chromosome_records,
        payloads=tuple(payloads),
    )


def _manifest_dict(**changes: Any) -> dict[str, Any]:
    contract = _contract()
    value = contract.manifest_to_dict(_manifest())
    value.update(changes)
    return value


def test_schema_version_and_single_fixed_layout() -> None:
    contract = _contract()

    assert contract.SCHEMA_VERSION == 1
    assert contract.ARTIFACT_TYPE == "jaxqtl-state-factor"
    assert contract.canonical_payload_inventory(("01",)) == (
        "cells.parquet",
        "donors.parquet",
        "chromosomes/01/factors.npy",
        "chromosomes/01/loadings.npy",
        "chromosomes/01/singular_values.npy",
        "chromosomes/01/genes.parquet",
    )


def test_exact_json_field_contract_and_manifest_is_not_a_payload() -> None:
    contract = _contract()
    encoded = contract.manifest_to_dict(_manifest())

    assert tuple(encoded) == (
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
    assert tuple(encoded["selection"]) == ("cell_type_column", "selected_cell_type", "allow_mixed_cell_types")
    assert tuple(encoded["ordering"]) == ("cell_order_sha256", "donor_order_sha256")
    assert tuple(encoded["dimensions"]) == ("cells", "donors")
    assert tuple(encoded["chromosomes"]["01"]) == (
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
    assert "manifest.json" not in [payload["path"] for payload in encoded["payloads"]]


@_PROPERTY_SETTINGS
@given(st.text(max_size=12))
@example("角質-🧬")
def test_manifest_json_roundtrip_is_canonical(label: str) -> None:
    contract = _contract()
    manifest = _manifest()
    chromosome = replace(
        manifest.chromosomes[0],
        approximation_metrics={"label": label, "nested": {"z": 1, "a": [True, None]}},
    )
    manifest = replace(manifest, chromosomes=(chromosome,))

    encoded = contract.encode_manifest(manifest)
    decoded = contract.decode_manifest(encoded)

    assert decoded == manifest
    assert contract.encode_manifest(decoded) == encoded
    assert encoded.endswith(b"\n")
    assert b'"a"' in encoded


@_PROPERTY_SETTINGS
@given(st.lists(st.text(max_size=10), min_size=2, max_size=5, unique=True))
@example(["細胞-🧬", "cell-λ"])
def test_identifier_hash_is_deterministic_and_order_sensitive(values: list[str]) -> None:
    contract = _contract()
    forward = (("cell_id", tuple(values)),)
    reverse = (("cell_id", tuple(reversed(values))),)

    assert contract.identifier_order_hash(forward) == contract.identifier_order_hash(forward)
    assert contract.identifier_order_hash(forward) != contract.identifier_order_hash(reverse)
    assert contract.identifier_order_hash(forward) != contract.identifier_order_hash((("donor_id", tuple(values)),))


def test_identifier_hash_supports_empty_and_unicode_sequences_without_python_hash() -> None:
    contract = _contract()

    assert len(contract.identifier_order_hash((("cell_id", ()),))) == 64
    assert contract.identifier_order_hash((("cell_id", ("λ",)),)) != contract.identifier_order_hash(
        (("cell_id", ("lambda",)),)
    )


@_PROPERTY_SETTINGS
@given(st.sampled_from((("01",), _FULL_AUTOSOMES)))
@example(("01",))
@example(_FULL_AUTOSOMES)
def test_generated_valid_inventory_is_accepted(chromosomes: tuple[str, ...]) -> None:
    contract = _contract()
    manifest = _manifest(chromosomes)

    assert contract.validate_manifest(manifest) == manifest
    assert tuple(payload.path for payload in manifest.payloads) == contract.canonical_payload_inventory(chromosomes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"artifact_type": "other"}, "artifact type"),
        ({"schema_version": 2}, "schema version"),
        ({"requested_chromosomes": ["1"]}, "canonical chromosome"),
        ({"completed_chromosomes": []}, "complete"),
        ({"requested_chromosomes": ["01", "02"], "completed_chromosomes": ["01", "02"]}, "one or all 22"),
    ],
)
def test_rejects_incompatible_schema_and_chromosome_sets(change: dict[str, Any], message: str) -> None:
    contract = _contract()

    with pytest.raises(ValueError, match=message):
        contract.manifest_from_dict(_manifest_dict(**change))


@pytest.mark.parametrize(
    "unsafe_path",
    ["/absolute.npy", "../escape.npy", "chromosomes/01/../../escape.npy", "C:\\escape.npy"],
)
def test_rejects_unsafe_or_absolute_payload_paths(unsafe_path: str) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    encoded["payloads"][0]["path"] = unsafe_path

    with pytest.raises(ValueError, match="safe canonical relative path"):
        contract.manifest_from_dict(encoded)


def test_rejects_duplicate_or_missing_inventory_entries() -> None:
    contract = _contract()
    duplicate = _manifest_dict()
    duplicate["payloads"][1]["path"] = duplicate["payloads"][0]["path"]
    missing = _manifest_dict()
    missing["payloads"].pop()

    with pytest.raises(ValueError, match="duplicate payload"):
        contract.manifest_from_dict(duplicate)
    with pytest.raises(ValueError, match="canonical payload inventory"):
        contract.manifest_from_dict(missing)


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("factors.npy", "shape", [3, 1], "factors shape"),
        ("loadings.npy", "dtype", "<f4", "loadings dtype"),
        ("singular_values.npy", "shape", [2], "singular-values shape"),
        ("genes.parquet", "rows", 3, "gene metadata row count"),
    ],
)
def test_rejects_payload_shape_dtype_and_row_disagreement(
    filename: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    payload = next(item for item in encoded["payloads"] if item["path"].endswith(filename))
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        contract.manifest_from_dict(encoded)


def test_rejects_empty_axes_and_invalid_selection_contract() -> None:
    contract = _contract()
    empty = _manifest_dict(dimensions={"cells": 0, "donors": 1})
    mixed = _manifest_dict()
    mixed["selection"]["selected_cell_type"] = None

    with pytest.raises(ValueError, match="cells must be positive"):
        contract.manifest_from_dict(empty)
    with pytest.raises(ValueError, match="selected cell type"):
        contract.manifest_from_dict(mixed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(schema_version=True), "schema version.*integer"),
        (
            lambda manifest: manifest["selection"].update(allow_mixed_cell_types="yes"),
            "allow_mixed_cell_types.*boolean",
        ),
        (
            lambda manifest: manifest["chromosomes"]["01"].update(center_within_donor="yes"),
            "center_within_donor.*boolean",
        ),
        (lambda manifest: manifest["configuration"].update(rank=True), "configuration.rank.*integer"),
        (
            lambda manifest: manifest["chromosomes"]["01"]["solver"].update(solver="arpack"),
            "solver.*configuration",
        ),
        (
            lambda manifest: manifest["chromosomes"]["01"]["convergence"].pop("max_forward_residual"),
            "convergence.*fields",
        ),
        (
            lambda manifest: manifest["chromosomes"]["01"].update(singular_values=[True]),
            "singular values.*floating-point",
        ),
    ],
)
def test_decoder_rejects_invalid_exact_types_and_contradictory_replay_configuration(
    mutate,
    message: str,
) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    mutate(encoded)

    with pytest.raises((TypeError, ValueError), match=message):
        contract.decode_manifest(json.dumps(encoded))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("automatic-alpha", "automatic PFlog alpha"),
        ("sigma-floor", "sigma_floor.*derived"),
        ("residual-limit", "residual_limit.*derived"),
        ("singular-floor", "singular value.*sigma_floor"),
    ],
)
def test_decoder_rejects_derived_replay_diagnostics_outside_algorithm_contract(
    mutation: str,
    message: str,
) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    chromosome = encoded["chromosomes"]["01"]
    if mutation == "automatic-alpha":
        chromosome["pflog"]["alpha"] = 9.0
    elif mutation == "sigma-floor":
        chromosome["convergence"]["sigma_floor"] = 999.0
    elif mutation == "residual-limit":
        chromosome["convergence"]["residual_limit"] = 999.0
    else:
        encoded["configuration"]["tol"] = 2.0
        chromosome["solver"]["tol"] = 2.0
        chromosome["convergence"]["sigma_floor"] = 5.0
        chromosome["convergence"]["residual_limit"] = 20.0

    with pytest.raises(ValueError, match=message):
        contract.decode_manifest(json.dumps(encoded))


@pytest.mark.parametrize(
    ("field", "section", "message"),
    [
        ("alpha", "pflog", "automatic PFlog alpha"),
        ("sigma_floor", "convergence", "sigma_floor.*derived"),
        ("residual_limit", "convergence", "residual_limit.*derived"),
    ],
)
def test_decoder_rejects_one_ulp_mutations_of_derived_replay_diagnostics(
    field: str,
    section: str,
    message: str,
) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    diagnostics = encoded["chromosomes"]["01"][section]
    diagnostics[field] = math.nextafter(diagnostics[field], math.inf)

    with pytest.raises(ValueError, match=message):
        contract.decode_manifest(json.dumps(encoded))


@pytest.mark.parametrize(
    ("retained_gene_count", "numerator", "denominator"),
    [
        (1, -0.25, 0.5),
        (0, 0.0, 0.0),
    ],
)
def test_override_manifest_accepts_failed_automatic_fit_diagnostics(
    retained_gene_count: int,
    numerator: float,
    denominator: float,
) -> None:
    contract = _contract()
    encoded = _manifest_dict()
    encoded["configuration"]["pflog_alpha"] = 0.125
    diagnostics = encoded["chromosomes"]["01"]["pflog"]
    diagnostics.update(
        alpha=0.125,
        alpha_source="override",
        retained_gene_count=retained_gene_count,
        numerator=numerator,
        denominator=denominator,
    )

    decoded = contract.decode_manifest(json.dumps(encoded))

    assert decoded.chromosomes[0].pflog_diagnostics["alpha_source"] == "override"
    assert decoded.chromosomes[0].pflog_diagnostics["retained_gene_count"] == retained_gene_count


def test_pure_alignment_and_configuration_validation() -> None:
    contract = _contract()
    manifest = _manifest()
    gene_ids = {"01": ("gene-0", "gene-β")}

    assert (
        contract.validate_manifest(
            manifest,
            expected_cell_ids=("cell-0", "cell-λ"),
            expected_donor_ids=("donor-0",),
            expected_gene_ids=gene_ids,
            expected_configuration=_configuration(("01",)),
        )
        == manifest
    )
    with pytest.raises(ValueError, match="cell order"):
        contract.validate_manifest(manifest, expected_cell_ids=("cell-λ", "cell-0"))
    with pytest.raises(ValueError, match="donor order"):
        contract.validate_manifest(manifest, expected_donor_ids=("other",))
    with pytest.raises(ValueError, match="gene order"):
        contract.validate_manifest(manifest, expected_gene_ids={"01": ("gene-β", "gene-0")})
    with pytest.raises(ValueError, match="configuration"):
        contract.validate_manifest(manifest, expected_configuration={**_configuration(("01",)), "rank": 2})


def test_rejects_noncanonical_json_values_and_nonfinite_diagnostics() -> None:
    contract = _contract()
    manifest = _manifest()
    bad_chromosome = replace(manifest.chromosomes[0], approximation_metrics={"unsupported": {1, 2}})
    bad_config = replace(manifest, chromosomes=(bad_chromosome,))
    encoded = _manifest_dict()
    encoded["chromosomes"]["01"]["convergence"]["max_forward_residual"] = float("nan")

    with pytest.raises(TypeError, match="JSON"):
        contract.validate_manifest(bad_config)
    with pytest.raises(ValueError, match="finite"):
        contract.manifest_from_dict(encoded)


def test_decoder_rejects_unknown_or_missing_fields() -> None:
    contract = _contract()
    unknown = _manifest_dict(unplanned=True)
    missing = _manifest_dict()
    del missing["ordering"]

    with pytest.raises(ValueError, match="manifest fields"):
        contract.manifest_from_dict(unknown)
    with pytest.raises(ValueError, match="manifest fields"):
        contract.manifest_from_dict(missing)


def test_canonical_json_bytes_reject_duplicate_object_keys() -> None:
    contract = _contract()
    encoded = contract.encode_manifest(_manifest()).decode("utf-8")
    duplicated = encoded.replace('"artifact_type":', '"artifact_type":"duplicate","artifact_type":', 1)

    with pytest.raises(ValueError, match="duplicate JSON field"):
        contract.decode_manifest(duplicated.encode("utf-8"))


def test_encoded_manifest_is_standard_json() -> None:
    contract = _contract()
    decoded = json.loads(contract.encode_manifest(_manifest()))

    assert decoded["schema_version"] == 1
    assert decoded["requested_chromosomes"] == ["01"]


def _artifact_inputs(
    tmp_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, ...], tuple[int, ...], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cells = pl.DataFrame(
        {
            "matrix_index": [0, 1, 2],
            "cell_id": ["cell-0", "cell-λ", "cell-2"],
            "donor_id": ["donor-0", "donor-0", "donor-β"],
            "cell_type": ["B", "B", "B"],
            "quality": [0.9, 0.8, 0.7],
        }
    )
    genes = pl.DataFrame(
        {
            "matrix_index": [0, 1],
            "gene_id": ["gene-0", "gene-β"],
            "chrom": ["X", "MT"],
        }
    )
    inputs = {
        "counts": tmp_path / "source-counts.npz",
        "cells": tmp_path / "source-cells.parquet",
        "genes": tmp_path / "source-genes.parquet",
    }
    inputs["counts"].write_bytes((b"bounded-input-chunk-" * 5000) + b"counts")
    cells.write_parquet(inputs["cells"])
    genes.write_parquet(inputs["genes"])
    return cells, genes, ("donor-0", "donor-β"), (2, 1), inputs


def _factor_result(
    chromosome: str,
    *,
    bad_shape: bool = False,
    failed_automatic_fit_override: float | None = None,
) -> StateFactorResult:
    factors = np.asarray([[1.0], [0.0], [-1.0]], dtype=np.float64)
    if bad_shape:
        factors = factors[:2]
    loadings = np.asarray([[0.8], [0.6]], dtype=np.float64)
    singular_values = np.asarray([2.5], dtype=np.float64)
    active_gene_indices = np.asarray([0, 1], dtype=np.int64)
    donor_counts = np.asarray([2, 1], dtype=np.int64)
    for array in (factors, loadings, singular_values, active_gene_indices, donor_counts):
        array.flags.writeable = False
    diagnostics = StateFactorDiagnostics(
        alpha=1.25 if failed_automatic_fit_override is None else failed_automatic_fit_override,
        alpha_source="auto" if failed_automatic_fit_override is None else "override",
        alpha_retained_gene_count=2 if failed_automatic_fit_override is None else 0,
        alpha_excluded_gene_count=0,
        alpha_numerator=5.0 if failed_automatic_fit_override is None else 0.0,
        alpha_denominator=4.0 if failed_automatic_fit_override is None else 0.0,
        alpha_excluded_numerator=0.0,
        alpha_excluded_denominator=0.0,
        excluded_chromosome=chromosome,
        active_gene_indices=active_gene_indices,
        rank=1,
        solver="propack",
        seed=7,
        tol=1e-8,
        maxiter=20,
        propack_kmax=20,
        arpack_ncv=None,
        sigma_floor=2.5e-8,
        residual_limit=1e-7,
        max_forward_residual=1e-10,
        max_adjoint_residual=2e-10,
        loading_orthogonality_error=3e-10,
        singular_values=singular_values,
        donor_counts=donor_counts,
        singleton_donor_count=1,
        center_within_donor=True,
        balance_donors=True,
        operator_shape=(3, 2),
        transformed_shape=(3, 2),
        transformed_nnz=5,
        factors_shape=(3, 1),
        loadings_shape=(2, 1),
    )
    return StateFactorResult(
        factors=factors,
        loadings=loadings,
        singular_values=singular_values,
        diagnostics=diagnostics,
    )


def _writer_arguments(tmp_path: Path, chromosomes: tuple[str, ...] = ("1",)) -> tuple[Path, dict[str, Any]]:
    cells, genes, donor_ids, donor_counts, inputs = _artifact_inputs(tmp_path)
    destination = tmp_path / "state-artifact"
    return destination, {
        "requested_chromosomes": chromosomes,
        "cell_metadata": cells,
        "gene_metadata": genes,
        "donor_ids": donor_ids,
        "donor_counts": donor_counts,
        "input_paths": inputs,
        "cell_type_column": "cell_type",
        "selected_cell_type": "B",
        "allow_mixed_cell_types": False,
        "configuration": {
            "allow_mixed_cell_types": False,
            "balance_donors": True,
            "cell_type": "B",
            "center_within_donor": True,
            "exclude_chromosome": None if len(chromosomes) == 22 else "1",
            "loco": len(chromosomes) == 22,
            "rank": 1,
            "solver": "propack",
            "tol": 1e-8,
            "maxiter": 20,
            "seed": 7,
            "ncv": None,
            "pflog_alpha": "auto",
            "platform": "cpu",
            "verbose": False,
        },
    }


def _write_artifact(tmp_path: Path, chromosomes: tuple[str, ...] = ("1",)) -> Path:
    destination, arguments = _writer_arguments(tmp_path, chromosomes)
    qtl_io.write_state_artifact(
        destination,
        (_factor_result(chromosome) for chromosome in chromosomes),
        **arguments,
    )
    return destination


def _read_manifest_dict(destination: Path) -> dict[str, Any]:
    return json.loads((destination / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest_dict(destination: Path, manifest: dict[str, Any]) -> None:
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _update_payload_hash(destination: Path, relative_path: str) -> None:
    manifest = _read_manifest_dict(destination)
    payload = next(payload for payload in manifest["payloads"] if payload["path"] == relative_path)
    payload["sha256"] = _sha256(destination / relative_path)
    _write_manifest_dict(destination, manifest)


def test_backend_neutral_artifact_surface_is_reexported() -> None:
    assert hasattr(qtl_io, "StateArtifactManifest")
    assert hasattr(qtl_io, "StateArtifactResult")
    assert hasattr(qtl_io, "write_state_artifact")
    assert hasattr(qtl_io, "load_state_artifact")
    assert not hasattr(qtl_io, "_PayloadRecord")
    assert not hasattr(qtl_io, "SCHEMA_VERSION")


def test_single_chromosome_roundtrip_uses_read_only_memory_maps(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    manifest = qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert destination.is_dir()
    assert isinstance(manifest, qtl_io.StateArtifactManifest)
    loaded = qtl_io.load_state_artifact(destination)
    chromosome = loaded.chromosome("1")
    assert isinstance(loaded, qtl_io.StateArtifactResult)
    assert loaded.root == destination
    assert loaded.cell_ids == ("cell-0", "cell-λ", "cell-2")
    assert loaded.donor_ids == ("donor-0", "donor-β")
    assert chromosome.gene_ids == ("gene-0", "gene-β")
    assert isinstance(chromosome.factors, np.memmap)
    assert isinstance(chromosome.loadings, np.memmap)
    assert isinstance(chromosome.singular_values, np.memmap)
    assert not chromosome.factors.flags.writeable
    assert not chromosome.loadings.flags.writeable
    assert not chromosome.singular_values.flags.writeable
    np.testing.assert_array_equal(chromosome.factors, [[1.0], [0.0], [-1.0]])
    np.testing.assert_array_equal(chromosome.loadings, [[0.8], [0.6]])
    np.testing.assert_array_equal(chromosome.singular_values, [2.5])


def test_override_with_failed_automatic_fit_diagnostics_writes_and_loads(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    arguments["configuration"]["pflog_alpha"] = 0.125
    result = _factor_result("1", failed_automatic_fit_override=0.125)

    manifest = qtl_io.write_state_artifact(destination, iter([result]), **arguments)
    loaded = qtl_io.load_state_artifact(destination)

    assert manifest.chromosomes[0].pflog_diagnostics["retained_gene_count"] == 0
    assert loaded.manifest.chromosomes[0].pflog_diagnostics == manifest.chromosomes[0].pflog_diagnostics


@pytest.mark.parametrize(
    ("field", "section", "message"),
    [
        ("alpha", "pflog", "automatic PFlog alpha"),
        ("sigma_floor", "convergence", "sigma_floor.*derived"),
        ("residual_limit", "convergence", "residual_limit.*derived"),
    ],
)
def test_loader_rejects_one_ulp_mutations_of_derived_replay_diagnostics(
    tmp_path: Path,
    field: str,
    section: str,
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)
    diagnostics = manifest["chromosomes"]["01"][section]
    diagnostics[field] = math.nextafter(diagnostics[field], math.inf)
    _write_manifest_dict(destination, manifest)

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("cell_types", "message"),
    [
        (["T", "T", "T"], "selected cell type.*cell metadata"),
        (["B", "T", "B"], "mixed cell types.*opt-in"),
        (["B", "", "B"], "cell type values must be nonempty strings"),
    ],
)
def test_writer_rejects_cell_metadata_incompatible_with_selection_contract(
    tmp_path: Path,
    cell_types: list[str],
    message: str,
) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    arguments["cell_metadata"] = arguments["cell_metadata"].with_columns(pl.Series("cell_type", cell_types))

    with pytest.raises(ValueError, match=message):
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_writer_accepts_normalized_single_type_selection_contract(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)

    manifest = qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert manifest.selected_cell_type == "B"
    assert manifest.allow_mixed_cell_types is False
    assert qtl_io.load_state_artifact(destination).manifest == manifest


def test_writer_rejects_single_observed_type_recorded_as_mixed(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    arguments["selected_cell_type"] = None
    arguments["allow_mixed_cell_types"] = True
    arguments["configuration"] = {
        **arguments["configuration"],
        "cell_type": None,
        "allow_mixed_cell_types": True,
    }

    with pytest.raises(ValueError, match="mixed-cell opt-in requires at least two"):
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_published_destination_containing_staging_marker_roundtrips(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    destination = tmp_path / "published.staging-results"

    qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    loaded = qtl_io.load_state_artifact(destination)
    assert loaded.root == destination
    assert loaded.manifest.completed_chromosomes == ("01",)


def test_roundtrip_preserves_metadata_hashes_diagnostics_and_replay_provenance(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)

    assert pl.read_parquet(destination / "cells.parquet").to_dict(as_series=False) == {
        "matrix_index": [0, 1, 2],
        "cell_id": ["cell-0", "cell-λ", "cell-2"],
        "donor_id": ["donor-0", "donor-0", "donor-β"],
        "cell_type": ["B", "B", "B"],
        "quality": [0.9, 0.8, 0.7],
    }
    assert pl.read_parquet(destination / "donors.parquet").to_dict(as_series=False) == {
        "donor_index": [0, 1],
        "donor_id": ["donor-0", "donor-β"],
        "cell_count": [2, 1],
    }
    assert manifest["inputs"] == {
        name: _sha256(tmp_path / f"source-{name}.{'npz' if name == 'counts' else 'parquet'}")
        for name in ("counts", "cells", "genes")
    }
    assert len(manifest["payloads"]) == 6
    assert all(payload["sha256"] == _sha256(destination / payload["path"]) for payload in manifest["payloads"])
    chromosome = manifest["chromosomes"]["01"]
    assert chromosome["pflog"]["alpha"] == 1.25
    assert chromosome["filtering"] == {"active_gene_count": 2, "excluded_gene_count": 0}
    assert chromosome["donor_counts"] == [2, 1]
    assert chromosome["solver"]["solver"] == "propack"
    assert chromosome["singular_values"] == [2.5]
    assert chromosome["convergence"]["max_forward_residual"] == 1e-10
    provenance = manifest["provenance"]
    assert provenance["platform"] == "cpu"
    assert set(provenance["package_versions"]) == {"numpy", "scipy", "polars", "pyarrow", "jax", "jaxlib", "jaxqtl"}
    assert set(provenance["blas_lapack"]) == {"numpy", "scipy"}
    assert set(provenance["thread_environment"]) == set(_contract().THREAD_ENVIRONMENT_VARIABLES)


def test_22_chromosome_roundtrip_is_complete_and_ordered(tmp_path: Path) -> None:
    chromosomes = tuple(str(chromosome) for chromosome in range(1, 23))
    destination = _write_artifact(tmp_path, chromosomes)

    loaded = qtl_io.load_state_artifact(destination)
    assert loaded.manifest.requested_chromosomes == _FULL_AUTOSOMES
    assert loaded.manifest.completed_chromosomes == _FULL_AUTOSOMES
    assert tuple(result.chromosome for result in loaded.chromosomes) == _FULL_AUTOSOMES
    assert len(loaded.manifest.payloads) == 90
    assert (destination / "chromosomes" / "22" / "factors.npy").is_file()


def test_equivalent_writes_have_byte_identical_manifests(tmp_path: Path) -> None:
    first = _write_artifact(tmp_path / "first")
    second = _write_artifact(tmp_path / "second")

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_writer_refuses_existing_destination_without_modification(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_writer_atomically_refuses_destination_created_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = importlib.import_module("jaxqtl.io._state_artifact")
    destination, arguments = _writer_arguments(tmp_path)
    original_publish = getattr(artifact, "_publish_directory_noreplace", os.rename)
    published_inode: int | None = None

    def create_destination_then_publish(staging: Path, final_path: Path) -> None:
        nonlocal published_inode
        final_path.mkdir()
        published_inode = final_path.stat().st_ino
        original_publish(staging, final_path)

    monkeypatch.setattr(
        artifact,
        "_publish_directory_noreplace",
        create_destination_then_publish,
        raising=False,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert destination.is_dir()
    assert destination.stat().st_ino == published_inode
    assert list(destination.iterdir()) == []
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_writer_fails_closed_when_atomic_publication_platform_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = importlib.import_module("jaxqtl.io._state_artifact")
    destination, arguments = _writer_arguments(tmp_path)
    monkeypatch.setattr(artifact, "_publication_platform", lambda: "unsupported-test-os", raising=False)

    with pytest.raises(OSError, match="unsupported on this platform") as exc_info:
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert exc_info.value.errno == errno.ENOTSUP
    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


@pytest.mark.parametrize(
    ("platform_name", "message"),
    [
        ("darwin", "unavailable on this Darwin runtime"),
        ("linux", "requires renameat2 on Linux"),
    ],
)
def test_writer_fails_closed_when_atomic_publication_libc_symbol_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    message: str,
) -> None:
    artifact = importlib.import_module("jaxqtl.io._state_artifact")
    destination, arguments = _writer_arguments(tmp_path)
    monkeypatch.setattr(artifact, "_publication_platform", lambda: platform_name, raising=False)
    monkeypatch.setattr(artifact, "_load_process_library", lambda: object(), raising=False)

    with pytest.raises(OSError, match=message) as exc_info:
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert exc_info.value.errno == errno.ENOTSUP
    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_writer_rejects_canonically_duplicate_approximation_metric_keys(tmp_path: Path) -> None:
    destination, arguments = _writer_arguments(tmp_path)
    arguments["approximation_metrics"] = {"1": {"error": 0.1}, "01": {"error": 0.2}}

    with pytest.raises(ValueError, match="duplicate canonical approximation.*01"):
        qtl_io.write_state_artifact(destination, iter([_factor_result("1")]), **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_loader_rejects_staging_directory(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)
    staging = tmp_path / ".jaxqtl-state-artifact-staging-observable"
    shutil.copytree(destination, staging)

    with pytest.raises(ValueError, match="staging"):
        qtl_io.load_state_artifact(staging)


def test_factor_iterator_failure_after_one_chromosome_leaves_no_artifact_or_staging(tmp_path: Path) -> None:
    chromosomes = tuple(str(chromosome) for chromosome in range(1, 23))
    destination, arguments = _writer_arguments(tmp_path, chromosomes)

    def failing_results():
        yield _factor_result("1")
        raise RuntimeError("observable factor failure")

    with pytest.raises(RuntimeError, match="observable factor failure"):
        qtl_io.write_state_artifact(destination, failing_results(), **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


def test_validation_failure_after_one_chromosome_leaves_no_artifact_or_staging(tmp_path: Path) -> None:
    chromosomes = tuple(str(chromosome) for chromosome in range(1, 23))
    destination, arguments = _writer_arguments(tmp_path, chromosomes)
    results = iter([_factor_result("1"), _factor_result("2", bad_shape=True)])

    with pytest.raises(ValueError, match="factor shape"):
        qtl_io.write_state_artifact(destination, results, **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".jaxqtl-state-artifact-staging-*")) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema version"),
        ("missing-payload", "missing payload"),
        ("corrupt-bytes", "SHA-256"),
        ("manifest-hash", "SHA-256"),
    ],
)
def test_loader_rejects_schema_missing_and_corrupt_payloads(tmp_path: Path, mutation: str, message: str) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)
    factor_path = destination / "chromosomes/01/factors.npy"
    if mutation == "schema":
        manifest["schema_version"] = 2
        _write_manifest_dict(destination, manifest)
    elif mutation == "missing-payload":
        factor_path.unlink()
    elif mutation == "corrupt-bytes":
        with factor_path.open("ab") as stream:
            stream.write(b"corruption")
    else:
        payload = next(payload for payload in manifest["payloads"] if payload["path"].endswith("factors.npy"))
        payload["sha256"] = "0" * 64
        _write_manifest_dict(destination, manifest)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selection-type", "allow_mixed_cell_types.*boolean"),
        ("solver-contradiction", "solver.*configuration"),
    ],
)
def test_loader_rejects_invalid_types_and_contradictory_replay_configuration(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)
    if mutation == "selection-type":
        manifest["selection"]["allow_mixed_cell_types"] = "yes"
    else:
        manifest["chromosomes"]["01"]["solver"]["solver"] = "arpack"
    _write_manifest_dict(destination, manifest)

    with pytest.raises((TypeError, ValueError), match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("automatic-alpha", "automatic PFlog alpha"),
        ("sigma-floor", "sigma_floor.*derived"),
        ("residual-limit", "residual_limit.*derived"),
        ("singular-floor", "singular value.*sigma_floor"),
    ],
)
def test_loader_rejects_mutated_derived_replay_diagnostics(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)
    chromosome = manifest["chromosomes"]["01"]
    if mutation == "automatic-alpha":
        chromosome["pflog"]["alpha"] = 9.0
    elif mutation == "sigma-floor":
        chromosome["convergence"]["sigma_floor"] = 999.0
    elif mutation == "residual-limit":
        chromosome["convergence"]["residual_limit"] = 999.0
    else:
        manifest["configuration"]["tol"] = 2.0
        chromosome["solver"]["tol"] = 2.0
        chromosome["convergence"]["sigma_floor"] = 5.0
        chromosome["convergence"]["residual_limit"] = 20.0
    _write_manifest_dict(destination, manifest)

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("cell_types", "message"),
    [
        (["T", "T", "T"], "selected cell type.*cell metadata"),
        (["B", "T", "B"], "mixed cell types.*opt-in"),
        (["B", "", "B"], "cell type values must be nonempty strings"),
    ],
)
def test_loader_rejects_cell_payload_incompatible_with_manifest_selection(
    tmp_path: Path,
    cell_types: list[str],
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    cells_path = destination / "cells.parquet"
    cells = pl.read_parquet(cells_path).with_columns(pl.Series("cell_type", cell_types))
    cells.write_parquet(cells_path)
    _update_payload_hash(destination, "cells.parquet")

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


def test_loader_accepts_coordinated_normalized_single_type_selection(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)
    cells_path = destination / "cells.parquet"
    cells = pl.read_parquet(cells_path).with_columns(pl.lit("T").alias("cell_type"))
    cells.write_parquet(cells_path)
    manifest = _read_manifest_dict(destination)
    manifest["selection"]["selected_cell_type"] = "T"
    manifest["configuration"]["cell_type"] = "T"
    _write_manifest_dict(destination, manifest)
    _update_payload_hash(destination, "cells.parquet")

    loaded = qtl_io.load_state_artifact(destination)

    assert loaded.manifest.selected_cell_type == "T"
    assert loaded.manifest.allow_mixed_cell_types is False


def test_loader_rejects_single_observed_type_recorded_as_mixed(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)
    manifest = _read_manifest_dict(destination)
    manifest["selection"]["selected_cell_type"] = None
    manifest["selection"]["allow_mixed_cell_types"] = True
    manifest["configuration"]["cell_type"] = None
    manifest["configuration"]["allow_mixed_cell_types"] = True
    _write_manifest_dict(destination, manifest)

    with pytest.raises(ValueError, match="mixed-cell opt-in requires at least two"):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("permuted-counts", "chromosome.*donor counts"),
        ("unknown-cell-donor", "exactly cover"),
        ("missing-donor-coverage", "exactly cover"),
        ("first-observed-order", "first-retained-cell order"),
    ],
)
def test_loader_reconciles_donor_payload_with_cells_and_chromosome_diagnostics(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    cells_path = destination / "cells.parquet"
    donors_path = destination / "donors.parquet"
    cells = pl.read_parquet(cells_path)
    donors = pl.read_parquet(donors_path)
    if mutation == "permuted-counts":
        cells = cells.with_columns(pl.Series("donor_id", ["donor-0", "donor-β", "donor-β"]))
        donors = donors.with_columns(pl.Series("cell_count", [1, 2]))
    elif mutation == "unknown-cell-donor":
        cells = cells.with_columns(pl.Series("donor_id", ["donor-0", "unknown", "donor-β"]))
        donors = donors.with_columns(pl.Series("cell_count", [1, 1]))
    elif mutation == "missing-donor-coverage":
        cells = cells.with_columns(pl.Series("donor_id", ["donor-0", "donor-0", "donor-0"]))
        donors = donors.with_columns(pl.Series("cell_count", [3, 1]))
    else:
        cells = cells.with_columns(pl.Series("donor_id", ["donor-β", "donor-0", "donor-0"]))
    cells.write_parquet(cells_path)
    donors.write_parquet(donors_path)
    _update_payload_hash(destination, "cells.parquet")
    _update_payload_hash(destination, "donors.parquet")

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("filename", "array", "message"),
    [
        ("factors.npy", np.ones((2, 1), dtype=np.float64), "shape"),
        ("loadings.npy", np.ones((2, 1), dtype=np.float32), "dtype"),
        ("singular_values.npy", np.ones((2,), dtype=np.float64), "shape"),
    ],
)
def test_loader_rejects_array_shape_and_dtype_mutation(
    tmp_path: Path,
    filename: str,
    array: np.ndarray,
    message: str,
) -> None:
    destination = _write_artifact(tmp_path)
    relative_path = f"chromosomes/01/{filename}"
    with (destination / relative_path).open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    _update_payload_hash(destination, relative_path)

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("cells.parquet", "cell metadata order"),
        ("donors.parquet", "donor metadata order"),
        ("chromosomes/01/genes.parquet", "gene metadata order"),
    ],
)
def test_loader_rejects_shuffled_metadata(tmp_path: Path, relative_path: str, message: str) -> None:
    destination = _write_artifact(tmp_path)
    path = destination / relative_path
    frame = pl.read_parquet(path).reverse()
    frame.write_parquet(path)
    _update_payload_hash(destination, relative_path)

    with pytest.raises(ValueError, match=message):
        qtl_io.load_state_artifact(destination)


def test_loader_rejects_incompatible_expected_alignment_and_configuration(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="cell order"):
        qtl_io.load_state_artifact(destination, expected_cell_ids=("cell-2", "cell-λ", "cell-0"))
    with pytest.raises(ValueError, match="donor order"):
        qtl_io.load_state_artifact(destination, expected_donor_ids=("donor-β", "donor-0"))
    with pytest.raises(ValueError, match="gene order"):
        qtl_io.load_state_artifact(destination, expected_gene_ids={"01": ("gene-β", "gene-0")})
    with pytest.raises(ValueError, match="configuration"):
        qtl_io.load_state_artifact(destination, expected_configuration={"rank": 2})


def test_loader_rejects_canonically_duplicate_expected_gene_keys(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="duplicate canonical expected gene key.*01"):
        qtl_io.load_state_artifact(
            destination,
            expected_gene_ids={"1": ("wrong-order",), "01": ("gene-0", "gene-β")},
        )


def test_loader_rejects_unexpected_files_outside_the_fixed_layout(tmp_path: Path) -> None:
    destination = _write_artifact(tmp_path)
    (destination / "extra.bin").write_bytes(b"not declared")

    with pytest.raises(ValueError, match="unexpected payload"):
        qtl_io.load_state_artifact(destination)
