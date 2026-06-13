# pattern: Imperative Shell

from types import SimpleNamespace

import polars as pl
import pytest

from jaxqtl import cli
from jaxqtl.io import GenoioData


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

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
