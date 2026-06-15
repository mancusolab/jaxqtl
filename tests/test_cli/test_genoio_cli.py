# pattern: Imperative Shell

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from jax import numpy as jnp

import jaxqtl.map.cis as cis_map

from jaxqtl import cli
from jaxqtl.hypothesis import TestResult as AssocTestResult
from jaxqtl.map.data import CisData


class _LoggerStub:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _EmptyGenoioDataset:
    def samples(self) -> pl.DataFrame:
        return pl.DataFrame({"iid": []}, schema={"iid": pl.Utf8})

    def variants(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chrom": [],
                "pos": [],
                "id": [],
                "a0": [],
                "a1": [],
            },
            schema={
                "chrom": pl.Utf8,
                "pos": pl.Int64,
                "id": pl.Utf8,
                "a0": pl.Utf8,
                "a1": pl.Utf8,
            },
        )


class _LoadDatasetSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.result = _EmptyGenoioDataset()

    def __call__(self, source: str, path: str):
        self.calls.append((source, path))
        return self.result


def _args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {"bfile": None, "pfile": None, "vcf": None, "bgen": None, "geno": None}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _common_setup_args(cmd: str) -> SimpleNamespace:
    return SimpleNamespace(
        cmd=cmd,
        bfile="tutorial/input/chr22_N100",
        pfile=None,
        bgen=None,
        geno=None,
        vcf=None,
        pheno="tutorial/input/CD4_NC.N100.bed.gz",
        covar="tutorial/input/donor_features.tsv",
        covar_name=None,
        rm_covar=None,
        normalize_covar=True,
        one_hot=False,
        no_intercept=False,
        offset=None,
        offset_name_from_covar=None,
        set_offset_from_libsize=True,
        model="poisson",
        test="score",
        robust_se=False,
        spa=False,
        keep=None,
        exclude=None,
        min_indiv_expr_pct=None,
        min_gene_expr_pct=0.0,
        gene_list="tutorial/input/genelist_5",
        genes=None,
        window=500_000,
        acat=False,
        nperm=1000,
        max_iter=1000,
        tol=1e-3,
        step_size=1.0,
        seed=0,
        solver="cholesky",
        platform="cpu",
        verbose=False,
        out="jaxqtl",
    )


def test_bfile_constructs_genoio_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _LoadDatasetSpy()
    monkeypatch.setattr(cli, "load_genotype_dataset", spy)

    result = cli._load_genotype_data(_args(bfile="plink-prefix"), _LoggerStub())

    assert result is spy.result
    assert spy.calls == [("bfile", "plink-prefix")]


@pytest.mark.parametrize(
    ("arg_name", "source", "path"),
    [
        ("pfile", "pfile", "plink2-prefix"),
        ("vcf", "vcf", "input.vcf.gz"),
        ("bgen", "bgen", "input.bgen"),
    ],
)
def test_other_genoio_sources_construct_dataset(
    monkeypatch: pytest.MonkeyPatch, arg_name: str, source: str, path: str
) -> None:
    spy = _LoadDatasetSpy()
    monkeypatch.setattr(cli, "load_genotype_dataset", spy)

    result = cli._load_genotype_data(_args(**{arg_name: path}), _LoggerStub())

    assert result is spy.result
    assert spy.calls == [(source, path)]


def test_deprecated_geno_raises_without_constructing_genoio(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _LoadDatasetSpy()
    monkeypatch.setattr(cli, "load_genotype_dataset", spy)

    with pytest.raises(ValueError, match="--geno.*deprecated.*genoio-native inputs"):
        cli._load_genotype_data(_args(geno="legacy-prefix"), _LoggerStub())

    assert spy.calls == []


def test_cli_help_marks_legacy_genotype_inputs() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["cis", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(stdout.getvalue().split())
    assert "Prefix to PLINK2 PGEN/PVAR/PSAM triplets." in help_text
    assert "Path to an indexed VCF/BCF genotype file." in help_text
    assert "Path to a BGEN genotype file." in help_text


@pytest.mark.parametrize("cmd", ["cis", "nominal"])
def test_common_setup_bfile_uses_genoio_and_yields_cis_data(cmd: str) -> None:
    args = _common_setup_args(cmd)

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.genotype.__class__.__name__ == "Dataset"
    assert ready_data.sample_ids
    cis_data = next(ready_data.iter_cis(args.window))
    assert isinstance(cis_data, CisData)
    assert cis_data.num_snps > 0


def test_common_setup_bfile_uses_genoio_and_yields_trans_genotype_block() -> None:
    args = _common_setup_args("trans")

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.genotype.__class__.__name__ == "Dataset"
    assert ready_data.sample_ids
    G, variant_info = next(ready_data.iter_geno(chunk_size=2500))
    assert G.shape[0] > 0
    assert G.shape[1] > 0
    assert variant_info.height > 0


def test_nominal_cli_smoke_writes_genoio_score_schema(tmp_path: Path) -> None:
    out_prefix = tmp_path / "jaxqtl"

    return_code = cli.main(
        [
            "nominal",
            "--bfile",
            "tutorial/input/chr22_N100",
            "--covar",
            "tutorial/input/donor_features.tsv",
            "--pheno",
            "tutorial/input/CD4_NC.N100.bed.gz",
            "--gene-list",
            "tutorial/input/genelist_5",
            "--model",
            "poisson",
            "--test",
            "score",
            "--set-offset-from-libsize",
            "--normalize-covar",
            "--platform",
            "cpu",
            "--out",
            str(out_prefix),
        ]
    )

    nominal_output = tmp_path / "jaxqtl.nominal.score.parquet.gz"
    assert return_code == 0
    assert nominal_output.exists()
    assert pl.read_parquet(nominal_output, n_rows=0).columns == [
        "phenotype_id",
        "chrom",
        "snp",
        "pos",
        "a1",
        "a0",
        "tss_distance",
        "af",
        "ma_count",
        "beta",
        "se",
        "pvalue",
        "model_converged",
    ]


def test_map_cis_batches_streamed_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeData:
        def iter_cis(self, window):
            for idx in range(3):
                yield CisData(
                    jnp.ones((2, 1)),
                    jnp.array([[0.0], [1.0]]),
                    jnp.array([0.0, 1.0]),
                    jnp.array(0.0),
                    f"gene{idx}",
                    "1",
                    100,
                    101,
                    pl.DataFrame(
                        {
                            "chrom": ["1"],
                            "snp": [f"rs{idx}"],
                            "pos": [101 + idx],
                            "a1": ["A"],
                            "a0": ["G"],
                        }
                    ),
                    1,
                    200,
                )

    class FakeTest:
        model = SimpleNamespace(family=object())

        def __call__(self, X, G, y, offset):
            return AssocTestResult(
                beta=jnp.array([0.1]),
                se=jnp.array([0.2]),
                p=jnp.array([0.3]),
                z=jnp.array([0.4]),
                num_iters=jnp.array([1]),
                converged=jnp.array([True]),
                disp=jnp.array(0.0),
            )

    # Lower the private flush threshold so the test observes batching without a large fixture.
    monkeypatch.setattr(cis_map, "_MAP_CIS_BATCH_ROWS", 2)
    monkeypatch.setattr(cis_map.eqx, "filter_jit", lambda fn: fn)

    map_cis = getattr(cis_map, "map_cis")
    chunks = list(map_cis(FakeData(), FakeTest(), None, mode="nominal", verbose=False, log=_LoggerStub()))

    assert [chunk.height for chunk in chunks] == [2, 1]
    assert chunks[0]["phenotype_id"].to_list() == ["gene0", "gene1"]
    assert chunks[1]["phenotype_id"].to_list() == ["gene2"]


def test_cis_scan_streams_map_cis_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = SimpleNamespace(window=500_000, verbose=False, seed=0, out=str(tmp_path / "jaxqtl"))
    test = SimpleNamespace(name="score")
    perm_test = SimpleNamespace(name="acat")
    dat = SimpleNamespace(num_genes=2)

    monkeypatch.setattr(cli, "_common_setup", lambda args, log: (dat, None, None, test, perm_test))

    def cis_frame(phenotype_id, snp):
        return pl.DataFrame(
            {
                "phenotype_id": [phenotype_id],
                "chrom": ["1"],
                "num_var": [2],
                "snp": [snp],
                "a1": ["A"],
                "a0": ["G"],
                "pos": [101],
                "tss_distance": [1],
                "af": [0.1],
                "ma_count": [10],
                "beta": [0.2],
                "se": [0.03],
                "pvalue": [0.04],
                "pvalue_adj": [0.05],
                "adj_method": ["ACAT"],
                "model_converged": [True],
            }
        )

    def cis_chunks(*args, **kwargs):
        assert kwargs["mode"] == "cis"
        yield cis_frame("gene1", "rs1")
        yield cis_frame("gene2", "rs2")

    monkeypatch.setattr(cli, "map_cis", cis_chunks)

    return_code = cli._cis_scan(args, _LoggerStub())

    cis_output = tmp_path / "jaxqtl.cis.score.acat.parquet.gz"
    assert return_code == 0
    assert cis_output.exists()
    assert pl.read_parquet(cis_output)["phenotype_id"].to_list() == ["gene1", "gene2"]


def test_nominal_scan_streams_map_cis_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = SimpleNamespace(window=500_000, verbose=False, seed=0, out=str(tmp_path / "jaxqtl"))
    test = SimpleNamespace(name="score")
    dat = SimpleNamespace(num_genes=2)

    monkeypatch.setattr(cli, "_common_setup", lambda args, log: (dat, None, None, test, None))

    def nominal_frame(phenotype_id, snp, converged):
        return pl.DataFrame(
            {
                "phenotype_id": [phenotype_id],
                "chrom": ["1"],
                "snp": [snp],
                "pos": [101],
                "a1": ["A"],
                "a0": ["G"],
                "tss_distance": [1],
                "af": [0.1],
                "ma_count": [10],
                "beta": [0.2],
                "se": [0.03],
                "pvalue": [0.04],
                "model_converged": [converged],
            }
        )

    def nominal_chunks(*args, **kwargs):
        assert kwargs["mode"] == "nominal"
        yield nominal_frame("gene1", "rs1", True)
        yield nominal_frame("gene2", "rs2", False)

    monkeypatch.setattr(cli, "map_cis", nominal_chunks)

    return_code = cli._nominal_scan(args, _LoggerStub())

    nominal_output = tmp_path / "jaxqtl.nominal.score.parquet.gz"
    assert return_code == 0
    assert nominal_output.exists()
    assert pl.read_parquet(nominal_output)["phenotype_id"].to_list() == ["gene1", "gene2"]
