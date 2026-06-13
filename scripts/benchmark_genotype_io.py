# pattern: Imperative Shell

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from benchmark_core import compare_qtl_frames, comparison_to_dict


_COMPARISON_MODE = "qtl_allele_aware"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    args: tuple[str, ...]
    result_suffixes: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_run-jaxqtl":
        return _run_jaxqtl_child(argv[1:])

    args = _parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = _select_cases(args.cases)
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "repeats": args.repeats,
        "compare_dir": None if args.compare_dir is None else str(Path(args.compare_dir).resolve()),
        "rtol": args.rtol,
        "atol": args.atol,
        "cases": [],
    }

    for case in selected_cases:
        case_runs: list[dict[str, Any]] = []
        for repeat in range(1, args.repeats + 1):
            run_dir = out_dir / case.name / f"repeat-{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            prefix = run_dir / "jaxqtl"
            run_args = tuple(arg.format(out=str(prefix), repo=str(repo_root)) for arg in case.args)
            metrics = _run_command(repo_root, run_args, run_dir)
            metrics["outputs"] = [str(prefix) + suffix for suffix in case.result_suffixes]
            if args.compare_dir is not None:
                metrics["comparisons"] = _compare_outputs(
                    current_prefix=prefix,
                    baseline_prefix=Path(args.compare_dir).resolve() / case.name / f"repeat-{repeat}" / "jaxqtl",
                    suffixes=case.result_suffixes,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            case_runs.append(metrics)
        report["cases"].append({"name": case.name, "runs": case_runs})

    report_path = out_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote benchmark report to {report_path}")
    return _exit_code(report)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small jaxQTL genotype IO downstream benchmarks.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", required=True, help="Directory for benchmark outputs and benchmark.json.")
    parser.add_argument(
        "--compare-dir",
        default=None,
        help="Optional prior benchmark output directory to compare Parquet outputs against.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["cis-poisson-acat", "nominal-poisson-score"],
        choices=sorted(_cases().keys()),
    )
    return parser.parse_args(argv)


def _cases() -> dict[str, BenchmarkCase]:
    common = (
        "--bfile",
        "{repo}/tutorial/input/chr22_N100",
        "--covar",
        "{repo}/tutorial/input/donor_features.tsv",
        "--pheno",
        "{repo}/tutorial/input/CD4_NC.N100.bed.gz",
        "--gene-list",
        "{repo}/tutorial/input/genelist_10",
        "--model",
        "poisson",
        "--test",
        "score",
        "--set-offset-from-libsize",
        "--normalize-covar",
        "--platform",
        "cpu",
        "--out",
        "{out}",
    )
    return {
        "cis-poisson-acat": BenchmarkCase(
            name="cis-poisson-acat",
            args=("cis", *common, "--acat"),
            result_suffixes=(".cis.score.acat.parquet.gz",),
        ),
        "nominal-poisson-score": BenchmarkCase(
            name="nominal-poisson-score",
            args=("nominal", *common),
            result_suffixes=(".nominal.score.parquet.gz",),
        ),
        "trans-poisson-score": BenchmarkCase(
            name="trans-poisson-score",
            args=("trans", *common),
            result_suffixes=(
                ".trans.score.variant.info.parquet.gz",
                ".trans.score.sumstats.parquet.gz",
            ),
        ),
    }


def _select_cases(names: list[str]) -> list[BenchmarkCase]:
    cases = _cases()
    return [cases[name] for name in names]


def _run_command(repo_root: Path, args: tuple[str, ...], run_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    child_metrics_path = run_dir / "child-metrics.json"
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--_run-jaxqtl",
        str(child_metrics_path),
        "--",
        *args,
    )
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=repo_root, env=env, stdout=stdout, stderr=stderr, check=False)
        elapsed = time.perf_counter() - start

    child_metrics = _read_child_metrics(child_metrics_path)
    return {
        "command": [sys.executable, "-m", "jaxqtl.cli", *args],
        "wrapper_command": list(command),
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "peak_rss_kib": child_metrics.get("peak_rss_kib"),
        "peak_rss_source": child_metrics.get("peak_rss_source"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _compare_outputs(
    *,
    current_prefix: Path,
    baseline_prefix: Path,
    suffixes: tuple[str, ...],
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    comparisons = []
    for suffix in suffixes:
        current_path = Path(str(current_prefix) + suffix)
        baseline_path = Path(str(baseline_prefix) + suffix)
        if not current_path.exists() or not baseline_path.exists():
            comparisons.append(
                {
                    "suffix": suffix,
                    "equal": False,
                    "reason": "missing output",
                    "current": str(current_path),
                    "baseline": str(baseline_path),
                }
            )
            continue

        current = pl.read_parquet(current_path)
        baseline = pl.read_parquet(baseline_path)
        comparison = compare_qtl_frames(current, baseline, rtol=rtol, atol=atol)
        comparisons.append(
            {
                "suffix": suffix,
                "current": str(current_path),
                "baseline": str(baseline_path),
                "comparison_mode": _COMPARISON_MODE,
                **comparison_to_dict(comparison),
            }
        )
    return comparisons


def _exit_code(report: dict[str, Any]) -> int:
    for case in report["cases"]:
        for run in case["runs"]:
            if run["returncode"] != 0:
                return 1
            for comparison in run.get("comparisons", []):
                if not comparison["equal"]:
                    return 2
    return 0


def _run_jaxqtl_child(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "--":
        raise SystemExit("--_run-jaxqtl expects: METRICS_PATH -- <jaxqtl args>")
    metrics_path = Path(argv[0])
    cli_args = argv[2:]

    from jaxqtl.cli import main as jaxqtl_main

    start = time.perf_counter()
    returncode = jaxqtl_main(cli_args)
    elapsed = time.perf_counter() - start
    metrics = {
        "wall_seconds": elapsed,
        "peak_rss_kib": _self_peak_rss_kib(),
        "peak_rss_source": "resource.RUSAGE_SELF.ru_maxrss",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return returncode


def _self_peak_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value / 1024)
    return int(value)


def _read_child_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
