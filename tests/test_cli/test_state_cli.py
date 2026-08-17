# pattern: Imperative Shell

from __future__ import annotations

import json
import logging
import math

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scipy import sparse

from jaxqtl import cli
from jaxqtl.io import load_state_artifact


def _single_cell_inputs(tmp_path: Path, *, mixed: bool = False) -> dict[str, Path]:
    counts = sparse.csr_array(
        np.asarray(
            [
                [4, 0, 1, 0, 3, 2],
                [0, 5, 2, 1, 0, 4],
                [3, 1, 0, 6, 2, 0],
                [8, 2, 4, 0, 1, 3],
                [1, 7, 0, 2, 5, 0],
                [6, 0, 3, 4, 0, 2],
                [2, 4, 5, 0, 6, 1],
                [0, 7, 0, 3, 0, 6],
            ],
            dtype=np.int64,
        )
    )
    cell_types = ["B"] * counts.shape[0]
    if mixed:
        cell_types[-2:] = ["T", "T"]
    cells = pl.DataFrame(
        {
            "matrix_index": np.arange(counts.shape[0], dtype=np.int64),
            "cell_id": [f"cell-{index}" for index in range(counts.shape[0])],
            "donor_id": ["d0", "d0", "d0", "d1", "d1", "d1", "d2", "d3"],
            "cell_type": cell_types,
        }
    )
    genes = pl.DataFrame(
        {
            "matrix_index": np.arange(counts.shape[1], dtype=np.int64),
            "gene_id": [f"gene-{index}" for index in range(counts.shape[1])],
            "chrom": ["1", "2", "3", "X", "1", "2"],
        }
    )
    paths = {
        "counts": tmp_path / "counts.npz",
        "cells": tmp_path / "cells.parquet",
        "genes": tmp_path / "genes.parquet",
    }
    sparse.save_npz(paths["counts"], counts)
    cells.write_parquet(paths["cells"])
    genes.write_parquet(paths["genes"])
    return paths


def _argv(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    solver: str = "propack",
    out_name: str = "state-artifact",
) -> list[str]:
    argv = [
        "state-factor",
        "--counts",
        str(paths["counts"]),
        "--cells",
        str(paths["cells"]),
        "--genes",
        str(paths["genes"]),
        "--cell-type-column",
        "cell_type",
        "--rank",
        "1",
        "--solver",
        solver,
        "--tol",
        "1e-9",
        "--maxiter",
        "20",
        "--seed",
        "7",
        "--out",
        str(tmp_path / out_name),
        "--exclude-chromosome",
        "1",
    ]
    if solver == "arpack":
        argv.extend(["--ncv", "3"])
    return argv


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = cli.main(argv)
    return status, stdout.getvalue(), stderr.getvalue()


def _parse_error(argv: list[str]) -> tuple[int, str]:
    stderr = StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert isinstance(exc_info.value.code, int)
    return exc_info.value.code, stderr.getvalue()


def test_state_factor_help_documents_required_conditional_and_fixed_contract() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["state-factor", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(stdout.getvalue().split())
    for option in (
        "--counts",
        "--cells",
        "--genes",
        "--cell-type-column",
        "--rank",
        "--solver {propack,arpack}",
        "--tol",
        "--maxiter",
        "--seed",
        "--out",
        "--ncv",
        "--exclude-chromosome",
        "--loco",
        "--pflog-alpha",
        "--center-within-donor",
        "--no-center-within-donor",
        "--balance-donors",
        "--no-balance-donors",
        "--verbose",
    ):
        assert option in help_text
    assert "required" in help_text
    assert "required for ARPACK; forbidden for PROPACK" in help_text
    assert "fixed: cpu" in help_text
    assert "default: auto" in help_text
    assert "--platform" not in help_text


@pytest.mark.parametrize(
    "missing_option",
    [
        "--counts",
        "--cells",
        "--genes",
        "--cell-type-column",
        "--rank",
        "--solver",
        "--tol",
        "--maxiter",
        "--seed",
        "--out",
    ],
)
def test_state_factor_requires_every_unpinned_option(tmp_path: Path, missing_option: str) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths)
    option_index = argv.index(missing_option)
    del argv[option_index : option_index + 2]

    status, stderr = _parse_error(argv)

    assert status == 2
    assert missing_option in stderr


def test_state_factor_requires_exactly_one_exclusion_mode(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    neither = _argv(tmp_path, paths)
    del neither[-2:]
    both = [*_argv(tmp_path, paths), "--loco"]

    neither_status, neither_stderr = _parse_error(neither)
    both_status, both_stderr = _parse_error(both)

    assert neither_status == 2
    assert "required" in neither_stderr
    assert both_status == 2
    assert "not allowed with argument" in both_stderr


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "not-a-number"])
def test_state_factor_rejects_nonpositive_or_nonfinite_alpha(tmp_path: Path, value: str) -> None:
    paths = _single_cell_inputs(tmp_path)

    status, stderr = _parse_error([*_argv(tmp_path, paths), "--pflog-alpha", value])

    assert status == 2
    assert "--pflog-alpha" in stderr
    if value != "-inf":
        assert "must be 'auto' or a finite positive number" in stderr


@pytest.mark.parametrize(("option", "value"), [("--tol", "0"), ("--tol", "nan"), ("--maxiter", "0"), ("--rank", "0")])
def test_state_factor_rejects_invalid_positive_options(tmp_path: Path, option: str, value: str) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths)
    argv[argv.index(option) + 1] = value

    status, stderr = _parse_error(argv)

    assert status == 2
    assert "positive" in stderr


@pytest.mark.parametrize(
    ("solver", "extra", "message"),
    [
        ("propack", ["--ncv", "3"], "--ncv is forbidden for PROPACK"),
        ("arpack", [], "--ncv is required for ARPACK"),
    ],
)
def test_solver_specific_ncv_contract_is_actionable_after_load(
    tmp_path: Path,
    solver: str,
    extra: list[str],
    message: str,
) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths, solver="propack")
    argv[argv.index("--solver") + 1] = solver
    argv.extend(extra)

    status, _, stderr = _run(argv)

    assert status == 1
    assert message in stderr
    assert "Finished! Thank you!" not in stderr


@pytest.mark.parametrize(("solver", "ncv", "message"), [("arpack", "1", "rank < ncv"), ("arpack", "5", "ncv < min")])
def test_arpack_dimensions_are_validated_against_selected_shapes(
    tmp_path: Path,
    solver: str,
    ncv: str,
    message: str,
) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths, solver=solver)
    argv[argv.index("--ncv") + 1] = ncv

    status, _, stderr = _run(argv)

    assert status == 1
    assert message in stderr


def test_propack_rank_is_validated_against_selected_shapes(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths)
    argv[argv.index("--rank") + 1] = "4"

    status, _, stderr = _run(argv)

    assert status == 1
    assert "rank < min(M, q_active)" in stderr


def test_mixed_cell_types_require_selection_or_opt_in(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path, mixed=True)

    missing_status, _, missing_stderr = _run(_argv(tmp_path, paths, out_name="missing"))
    selected_status, _, _ = _run([*_argv(tmp_path, paths, out_name="selected"), "--cell-type", "B"])
    mixed_status, _, _ = _run([*_argv(tmp_path, paths, out_name="mixed"), "--allow-mixed-cell-types"])

    assert missing_status == 1
    assert "multiple cell types" in missing_stderr
    assert selected_status == 0
    assert mixed_status == 0


def test_state_factor_rejects_conflicting_cell_selection_at_parse_time(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path, mixed=True)

    status, stderr = _parse_error([*_argv(tmp_path, paths), "--cell-type", "B", "--allow-mixed-cell-types"])

    assert status == 2
    assert "not allowed with argument" in stderr


def test_state_factor_single_chromosome_end_to_end_roundtrip(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    destination = tmp_path / "state-artifact"

    status, stdout, stderr = _run(_argv(tmp_path, paths))

    assert status == 0
    assert stdout == ""
    assert stderr.count("Finished! Thank you!") == 1
    assert destination.is_dir()
    assert not Path(f"{destination}.log").exists()
    loaded = load_state_artifact(destination)
    assert loaded.manifest.requested_chromosomes == ("01",)
    assert loaded.manifest.completed_chromosomes == ("01",)
    assert loaded.manifest.selected_cell_type == "B"
    assert loaded.manifest.configuration == {
        "allow_mixed_cell_types": False,
        "balance_donors": True,
        "cell_type": "B",
        "center_within_donor": True,
        "exclude_chromosome": "1",
        "loco": False,
        "maxiter": 20,
        "ncv": None,
        "pflog_alpha": "auto",
        "platform": "cpu",
        "rank": 1,
        "seed": 7,
        "solver": "propack",
        "tol": 1e-9,
        "verbose": False,
    }
    result = loaded.chromosome("1")
    assert result.factors.shape == (8, 1)
    assert result.loadings.shape == (4, 1)
    assert result.singular_values.shape == (1,)


def test_state_factor_loco_dispatch_writes_all_autosomes(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths, out_name="loco")
    argv[-2:] = ["--loco"]

    status, _, stderr = _run(argv)

    assert status == 0, stderr
    loaded = load_state_artifact(tmp_path / "loco")
    assert loaded.manifest.requested_chromosomes == tuple(f"{index:02d}" for index in range(1, 23))
    assert len(loaded.chromosomes) == 22
    assert not (tmp_path / "loco.log").exists()


def test_runtime_failure_returns_one_without_success_completion(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = _argv(tmp_path, paths)
    argv[argv.index("--rank") + 1] = "99"

    status, stdout, stderr = _run(argv)

    assert status == 1
    assert stdout == ""
    assert "state-factor failed:" in stderr
    assert "Finished! Thank you!" not in stderr
    assert not (tmp_path / "state-artifact").exists()


def test_main_propagates_established_handler_failure_without_success_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _single_cell_inputs(tmp_path)

    def fail_handler(args, log) -> int:
        del args
        log.error("controlled established failure")
        return 1

    monkeypatch.setattr(cli, "_compute_expression_pcs", fail_handler)
    status, _, stderr = _run(
        [
            "compute-pcs",
            "--pheno",
            str(paths["cells"]),
            "--covar",
            str(paths["genes"]),
            "--num-pcs",
            "1",
            "--out",
            str(tmp_path / "pcs.tsv"),
        ]
    )

    assert status == 1
    assert stderr.count("controlled established failure") == 1
    assert "Finished! Thank you!" not in stderr


def test_cli_logging_context_owns_only_fresh_invocation_handlers(tmp_path: Path) -> None:
    import jaxqtl.log as log_module

    logger = logging.getLogger("jaxqtl.tests.cli-owned")
    external_stream = StringIO()
    external_handler = logging.StreamHandler(external_stream)
    logger.addHandler(external_handler)
    owned_handlers: list[logging.Handler] = []
    try:
        with log_module.cli_logging(logger.name, path=tmp_path / "owned", verbose=False) as active:
            owned_handlers = [handler for handler in active.handlers if getattr(handler, "_jaxqtl_cli_owned", False)]
            assert len(owned_handlers) == 2
            active.info("one line")

        assert logger.handlers == [external_handler]
        assert external_handler.stream is external_stream
        assert external_stream.getvalue().count("one line") == 1
        assert all(handler.stream is None for handler in owned_handlers if isinstance(handler, logging.FileHandler))
        assert (tmp_path / "owned.log").read_text().count("one line") == 1
    finally:
        logger.removeHandler(external_handler)
        external_handler.close()


@pytest.mark.parametrize("order", [("established", "state"), ("state", "established")])
def test_repeated_commands_in_both_orders_do_not_leak_or_duplicate_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    order: tuple[str, str],
) -> None:
    paths = _single_cell_inputs(tmp_path)
    calls: list[str] = []

    def established(args, log) -> int:
        calls.append("established")
        log.info("established marker")
        return 0

    def state(args, log) -> int:
        calls.append("state")
        log.info("state marker")
        return 0

    monkeypatch.setattr(cli, "_compute_expression_pcs", established)
    monkeypatch.setattr(cli, "_state_factor", state)
    established_out = tmp_path / "pcs.tsv"
    state_out = tmp_path / "state-dir"
    command = {
        "established": [
            "compute-pcs",
            "--pheno",
            str(paths["cells"]),
            "--covar",
            str(paths["genes"]),
            "--num-pcs",
            "1",
            "--out",
            str(established_out),
        ],
        "state": _argv(tmp_path, paths, out_name="state-dir"),
    }

    stderr = StringIO()
    with redirect_stderr(stderr):
        assert cli.main(command[order[0]]) == 0
        assert cli.main(command[order[1]]) == 0

    assert calls == list(order)
    assert stderr.getvalue().count("established marker") == 1
    assert stderr.getvalue().count("state marker") == 1
    assert stderr.getvalue().count("Finished! Thank you!") == 2
    assert Path(f"{established_out}.log").read_text().count("established marker") == 1
    assert not Path(f"{state_out}.log").exists()
    assert not any(getattr(handler, "_jaxqtl_cli_owned", False) for handler in logging.getLogger(cli.__name__).handlers)


@pytest.mark.parametrize("command", ["cis", "nominal", "trans", "compute-pcs"])
def test_established_subcommand_help_remains_available(command: str) -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main([command, "--help"])

    assert exc_info.value.code == 0
    assert "usage:" in stdout.getvalue()
    assert command in stdout.getvalue()


def test_state_manifest_json_is_finite_and_replay_complete(tmp_path: Path) -> None:
    paths = _single_cell_inputs(tmp_path)
    argv = [*_argv(tmp_path, paths), "--pflog-alpha", "0.2", "--no-balance-donors", "--verbose"]

    status, _, stderr = _run(argv)

    assert status == 0, stderr
    raw = json.loads((tmp_path / "state-artifact" / "manifest.json").read_text())
    configuration = raw["configuration"]
    assert configuration["pflog_alpha"] == pytest.approx(0.2)
    assert configuration["balance_donors"] is False
    assert configuration["center_within_donor"] is True
    assert configuration["verbose"] is True
    assert configuration["platform"] == "cpu"
    assert all(math.isfinite(value) for value in raw["chromosomes"]["01"]["singular_values"])
