# pattern: Functional Core

from __future__ import annotations

import importlib
import importlib.util
import json

from dataclasses import replace
from typing import Any

import pytest

from hypothesis import example, given, settings, strategies as st


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
        configuration={"rank": 1, "mode": "loco" if len(chromosomes) == 22 else "single"}
        if configuration is None
        else configuration,
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
    manifest = _manifest(configuration={"label": label, "nested": {"z": 1, "a": [True, None]}})

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
            expected_configuration={"rank": 1, "mode": "single"},
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
        contract.validate_manifest(manifest, expected_configuration={"rank": 2, "mode": "single"})


def test_rejects_noncanonical_json_values_and_nonfinite_diagnostics() -> None:
    contract = _contract()
    bad_config = replace(_manifest(), configuration={"unsupported": {1, 2}})
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
