# pattern: Imperative Shell

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from jaxqtl import cli
from jaxqtl.io import GenoioData
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


class _GenoioLoadSpy:
    calls: list[str] = []
    result = GenoioData(
        genotype=object(),
        sample_info=pl.DataFrame({"iid": []}, schema={"iid": pl.Utf8}),
        variant_info=pl.DataFrame(
            {
                "chrom": [],
                "snp": [],
                "pos": [],
                "a0": [],
                "a1": [],
            },
            schema={
                "chrom": pl.Utf8,
                "snp": pl.Utf8,
                "pos": pl.Int64,
                "a0": pl.Utf8,
                "a1": pl.Utf8,
            },
        ),
    )

    @classmethod
    def load(cls, prefix: str) -> GenoioData:
        cls.calls.append(prefix)
        return cls.result


class _VCFLoadSpy:
    calls: list[str] = []

    @classmethod
    def load(cls, path: str) -> object:
        cls.calls.append(path)
        raise RuntimeError("existing VCF path remains unsupported")


def _args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {"bfile": None, "geno": None, "vcf": None}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _common_setup_args(cmd: str) -> SimpleNamespace:
    return SimpleNamespace(
        cmd=cmd,
        bfile="tutorial/input/chr22_N100",
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


def test_bfile_constructs_genoio_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _GenoioLoadSpy.calls = []
    monkeypatch.setattr(cli, "GenoioData", _GenoioLoadSpy)

    result = cli._load_genotype_data(_args(bfile="plink-prefix"), _LoggerStub())

    assert result is _GenoioLoadSpy.result
    assert isinstance(result, GenoioData)
    assert _GenoioLoadSpy.calls == ["plink-prefix"]


def test_deprecated_geno_raises_without_constructing_genoio(monkeypatch: pytest.MonkeyPatch) -> None:
    _GenoioLoadSpy.calls = []
    monkeypatch.setattr(cli, "GenoioData", _GenoioLoadSpy)

    with pytest.raises(ValueError, match="--geno.*deprecated.*--bfile.*PLINK1 BED/BIM/FAM"):
        cli._load_genotype_data(_args(geno="legacy-prefix"), _LoggerStub())

    assert _GenoioLoadSpy.calls == []


def test_vcf_uses_existing_unsupported_path_without_constructing_genoio(monkeypatch: pytest.MonkeyPatch) -> None:
    _GenoioLoadSpy.calls = []
    _VCFLoadSpy.calls = []
    log = _LoggerStub()
    monkeypatch.setattr(cli, "GenoioData", _GenoioLoadSpy)
    monkeypatch.setattr(cli, "VCFData", _VCFLoadSpy)

    with pytest.raises(RuntimeError, match="existing VCF path remains unsupported"):
        cli._load_genotype_data(_args(vcf="input.vcf.gz"), log)

    assert _GenoioLoadSpy.calls == []
    assert _VCFLoadSpy.calls == ["input.vcf.gz"]
    assert log.errors == ["`--vcf PREFIX` is not fully supported yet."]


def test_cli_help_marks_legacy_genotype_inputs() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["cis", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(stdout.getvalue().split())
    assert "deprecated; use --bfile for PLINK1 BED/BIM/FAM prefixes" in help_text
    assert "unsupported/experimental; not a production genotype input" in help_text


@pytest.mark.parametrize("cmd", ["cis", "nominal"])
def test_common_setup_bfile_uses_genoio_and_yields_cis_data(cmd: str) -> None:
    args = _common_setup_args(cmd)

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert isinstance(ready_data.genotype, GenoioData)
    cis_data = next(ready_data.iter_cis(args.window))
    assert isinstance(cis_data, CisData)
    assert cis_data.num_snps > 0


def test_common_setup_bfile_uses_genoio_and_yields_trans_genotype_block() -> None:
    args = _common_setup_args("trans")

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert isinstance(ready_data.genotype, GenoioData)
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
