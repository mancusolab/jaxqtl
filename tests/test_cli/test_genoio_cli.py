# pattern: Imperative Shell

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from jaxqtl import cli
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
