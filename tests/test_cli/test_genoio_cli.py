# pattern: Imperative Shell

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import genoio
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


class _ChromosomeMetadataDataset:
    def __init__(self, chromosomes: list[str]) -> None:
        self._variants = pl.DataFrame({"chrom": chromosomes})

    def variants(self) -> pl.DataFrame:
        return self._variants


class _LoadDatasetSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.result = _EmptyGenoioDataset()

    def __call__(self, source: str, path: str):
        self.calls.append((source, path))
        return self.result


def _cis_data(gene_name: str = "gene1") -> CisData:
    return CisData(
        jnp.ones((3, 1)),
        jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]),
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array(0.0),
        gene_name,
        "1",
        100,
        101,
        pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "snp": [f"{gene_name}_rs1", f"{gene_name}_rs2"],
                "pos": [101, 102],
                "a1": ["A", "C"],
                "a0": ["G", "T"],
            }
        ),
        1,
        200,
    )


def _test_result(pvalues: list[float]) -> AssocTestResult:
    size = len(pvalues)
    return AssocTestResult(
        beta=jnp.arange(size, dtype=float) + 0.1,
        se=jnp.arange(size, dtype=float) + 0.2,
        p=jnp.asarray(pvalues),
        z=jnp.arange(size, dtype=float) + 0.3,
        num_iters=jnp.array(1),
        converged=jnp.array(True),
        disp=jnp.array(0.4),
    )


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
        dosage=False,
        chr=None,
        maf=None,
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
        tss_centered=False,
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


def test_common_setup_rejects_robust_score_test() -> None:
    args = _common_setup_args("cis")
    args.robust_se = True

    with pytest.raises(ValueError, match="--robust-se is only compatible with --test wald"):
        cli._common_setup(args, _LoggerStub())


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


def test_unique_chromosome_labels_deduplicates_labels() -> None:
    chromosomes = pl.Series("chrom", ["1", "1", "2"])

    assert cli._unique_chromosome_labels(chromosomes) == {"1", "2"}


def test_chromosome_mismatch_error_identifies_zero_overlap() -> None:
    expression = SimpleNamespace(pheno_meta=pl.DataFrame({"chrom": ["Chr1"]}))
    genotype = _ChromosomeMetadataDataset(["1"])

    with pytest.raises(ValueError, match="No chromosome labels overlap") as exc_info:
        cli._validate_chromosome_labels(expression, genotype)

    assert "Chr1" in str(exc_info.value)
    assert "1" in str(exc_info.value)


def test_chromosome_validation_accepts_genotype_chromosome_subset() -> None:
    expression = SimpleNamespace(pheno_meta=pl.DataFrame({"chrom": ["chr1", "chr22"]}))
    genotype = _ChromosomeMetadataDataset(["chr22"])

    overlap = cli._validate_chromosome_labels(expression, genotype)

    assert overlap == {"chr22"}


@pytest.mark.parametrize(
    ("phenotype_chromosomes", "genotype_chromosomes", "missing_source"),
    [
        (["chr1"], ["chr1", "chr22"], "phenotype"),
        (["chr1", "chr22"], ["chr1"], "genotype"),
    ],
)
def test_chromosome_validation_rejects_requested_chromosome_missing_from_input(
    phenotype_chromosomes: list[str], genotype_chromosomes: list[str], missing_source: str
) -> None:
    expression = SimpleNamespace(pheno_meta=pl.DataFrame({"chrom": phenotype_chromosomes}))
    genotype = _ChromosomeMetadataDataset(genotype_chromosomes)

    with pytest.raises(ValueError, match=rf"Requested chromosome 'chr22'.*{missing_source}"):
        cli._validate_chromosome_labels(expression, genotype, chromosome="chr22")


def test_cli_help_marks_legacy_genotype_inputs() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["cis", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(stdout.getvalue().split())
    assert "Prefix to PLINK2 PGEN/PVAR/PSAM triplets." in help_text
    assert "Path to an indexed VCF/BCF genotype file." in help_text
    assert "Path to a BGEN genotype file." in help_text
    assert "Deprecated:" in help_text
    assert "--maf" in help_text
    assert "--chr" in help_text
    assert "--tss-centered" in help_text


def test_common_setup_uses_genoio_minimum_maf_filter() -> None:
    args = _common_setup_args("trans")
    args.maf = 0.05

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.variant_filter.to_ir() == {
        "op": "and",
        "left": genoio.polymorphic().to_ir(),
        "right": genoio.maf(min=0.05).to_ir(),
    }


def test_common_setup_filters_expression_to_genotype_chromosome_overlap() -> None:
    args = _common_setup_args("trans")
    args.gene_list = None

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.expression.pheno_meta.get_column("chrom").unique().to_list() == ["22"]
    assert ready_data.expression.pheno.columns == [
        "iid",
        *ready_data.expression.pheno_meta.get_column("phenotype_id").to_list(),
    ]


def test_common_setup_applies_requested_chromosome_to_genotype_filter() -> None:
    args = _common_setup_args("trans")
    args.chr = "22"

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.expression.pheno_meta.get_column("chrom").unique().to_list() == ["22"]
    assert ready_data.variant_filter.to_ir() == {
        "op": "and",
        "left": genoio.polymorphic().to_ir(),
        "right": genoio.chrom("22").to_ir(),
    }


def test_cli_help_exposes_dosage_genotype_mode() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["cis", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(stdout.getvalue().split())
    assert "--dosage" in help_text
    assert "Read genotype dosages instead of hard calls." in help_text


def test_mapping_help_organizes_options_and_displays_defaults() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["cis", "--help"])

    assert exc_info.value.code == 0
    help_text = stdout.getvalue()
    headings = [
        "Genotypes:",
        "Covariates:",
        "Library-size adjustment (offsets):",
        "Model and variant testing:",
        "Gene-level testing:",
        "Filters:",
        "Phenotypes:",
        "Solver:",
        "Runtime:",
    ]
    heading_positions = [help_text.index(heading) for heading in headings]
    assert heading_positions == sorted(heading_positions)
    sections = {
        heading: help_text[start:end]
        for heading, start, end in zip(headings, heading_positions, [*heading_positions[1:], len(help_text)])
    }
    assert "--bfile" in sections["Genotypes:"]
    assert "--covar-name" in sections["Covariates:"]
    assert "--set-offset-from-libsize" in sections["Library-size adjustment (offsets):"]
    assert "--window" in sections["Phenotypes:"]
    assert "(default: 500000)" in help_text


def test_compute_pcs_help_organizes_options_and_displays_defaults() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
        cli.main(["compute-pcs", "--help"])

    assert exc_info.value.code == 0
    help_text = stdout.getvalue()
    headings = ["Inputs:", "PCA options:", "Runtime and output:"]
    heading_positions = [help_text.index(heading) for heading in headings]
    assert heading_positions == sorted(heading_positions)
    sections = {
        heading: help_text[start:end]
        for heading, start, end in zip(headings, heading_positions, [*heading_positions[1:], len(help_text)])
    }
    assert "--pheno" in sections["Inputs:"]
    assert "--num-pcs" in sections["PCA options:"]
    assert "--platform" in sections["Runtime and output:"]
    assert "(default: cpu)" in help_text


def test_help_formatter_preserves_literal_square_brackets() -> None:
    parser = cli.ap.ArgumentParser(formatter_class=cli._HelpFormatter)
    parser.add_argument("--example", help="Use values in [square brackets].")

    help_text = parser.format_help()

    assert "[square brackets]" in help_text
    assert cli._HelpFormatter.text_markup is False
    assert cli._HelpFormatter.help_markup is False
    assert cli._HelpFormatter.styles == {
        "argparse.args": "#578fa4",
        "argparse.groups": "#ff8700",
        "argparse.help": "#b9bcba",
        "argparse.metavar": "#00af87",
        "argparse.prog": "#808080",
        "argparse.syntax": "bold #b9bcba",
        "argparse.text": "#b9bcba",
        "argparse.default": "italic #b9bcba",
        "argparse.deprecated": "bold red",
    }
    assert r"(?P<deprecated>Deprecated:)" in cli._HelpFormatter.highlights


@pytest.mark.parametrize(
    ("dosage", "expected_mode"),
    [(False, "hardcall"), (True, "dosage")],
)
def test_common_setup_selects_genoio_genotype_mode(dosage: bool, expected_mode: str) -> None:
    args = _common_setup_args("trans")
    args.dosage = dosage

    ready_data, _, _, _, _ = cli._common_setup(args, _LoggerStub())

    assert ready_data.read_options.dosage == expected_mode


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


def test_map_cis_forwards_tss_centered_window_mode() -> None:
    class FakeData:
        requested_window = None

        def iter_cis(self, window, *, tss_centered=False):
            self.requested_window = (window, tss_centered)
            return iter(())

    data = FakeData()
    test = SimpleNamespace(model=SimpleNamespace(family=object()))

    map_cis = getattr(cis_map, "map_cis")
    list(
        map_cis(
            data,
            test,
            None,
            mode="nominal",
            window=123,
            tss_centered=True,
            verbose=False,
            log=_LoggerStub(),
        )
    )

    assert data.requested_window == (123, True)


def test_map_cis_yields_empty_nominal_frame_when_all_genes_are_skipped() -> None:
    data = SimpleNamespace(iter_cis=lambda window: iter(()))
    test = SimpleNamespace(model=SimpleNamespace(family=object()))

    map_cis = getattr(cis_map, "map_cis")
    chunks = list(map_cis(data, test, None, mode="nominal", verbose=False, log=_LoggerStub()))

    assert len(chunks) == 1
    assert chunks[0].height == 0
    assert chunks[0].columns == [
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


def test_map_cis_yields_empty_cis_frame_when_all_genes_are_skipped() -> None:
    data = SimpleNamespace(iter_cis=lambda window: iter(()))
    test = SimpleNamespace(model=SimpleNamespace(family=object()))
    gene_test = SimpleNamespace(name="acat")

    map_cis = getattr(cis_map, "map_cis")
    chunks = list(map_cis(data, test, gene_test, mode="cis", verbose=False, log=_LoggerStub()))

    assert len(chunks) == 1
    assert chunks[0].height == 0
    assert chunks[0].columns == [
        "phenotype_id",
        "chrom",
        "num_var",
        "snp",
        "a1",
        "a0",
        "pos",
        "tss_distance",
        "af",
        "ma_count",
        "beta",
        "se",
        "pvalue",
        "pvalue_adj",
        "adj_method",
        "model_converged",
        "result_valid",
        "failure_reason",
    ]


def test_process_cis_result_reports_no_finite_pvalues_as_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_choice(*args, **kwargs):
        pytest.fail("lead-SNP tie breaking must not run without finite p-values")

    monkeypatch.setattr(cis_map.rdm, "choice", fail_choice)

    result = cis_map._process_cis_result(
        _cis_data(),
        _test_result([float("nan"), float("inf")]),
        (jnp.array(float("nan")), None),
        cis_map.rdm.key(0),
    )

    assert result == {
        "phenotype_id": "gene1",
        "chrom": "1",
        "num_var": 2,
        "snp": None,
        "a1": None,
        "a0": None,
        "pos": None,
        "tss_distance": None,
        "af": None,
        "ma_count": None,
        "beta": None,
        "se": None,
        "pvalue": None,
        "pvalue_adj": None,
        "adj_method": "ACAT",
        "nb_alpha": None,
        "model_converged": None,
        "result_valid": False,
        "failure_reason": "no_finite_pvalues",
    }


def test_process_cis_result_keeps_beta_schema_for_no_finite_pvalues() -> None:
    result = cis_map._process_cis_result(
        _cis_data(),
        _test_result([float("nan"), float("nan")]),
        (jnp.array([float("nan"), float("nan")]), (object(), object(), object())),
        cis_map.rdm.key(0),
    )

    assert result["adj_method"] == "BETA"
    assert result["result_valid"] is False
    assert result["failure_reason"] == "no_finite_pvalues"
    for column in ["shape1", "shape2", "nc_estimate", "perm_converged"]:
        assert column in result
        assert result[column] is None

    frame = pl.DataFrame([result], schema=cis_map._empty_cis_columns(SimpleNamespace(name="beta")))
    assert frame.schema["shape1"] == pl.Float64
    assert frame.schema["perm_converged"] == pl.Boolean


def test_process_cis_result_selects_minimum_finite_pvalue() -> None:
    result = cis_map._process_cis_result(
        _cis_data(),
        _test_result([float("nan"), 0.02]),
        (jnp.array(0.03), None),
        cis_map.rdm.key(0),
    )

    assert result["snp"] == "gene1_rs2"
    assert result["pvalue"] == pytest.approx(0.02)
    assert result["result_valid"] is True
    assert result["failure_reason"] is None


@pytest.mark.parametrize("invalid_first", [True, False])
def test_map_cis_preserves_invalid_rows_and_parquet_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid_first: bool
) -> None:
    genes = [_cis_data("invalid"), _cis_data("valid")]
    if not invalid_first:
        genes.reverse()

    data = SimpleNamespace(iter_cis=lambda window: iter(genes))
    snp_test = SimpleNamespace(model=SimpleNamespace(family=object()))
    gene_test = SimpleNamespace(name="acat")
    invalid = (_test_result([float("nan"), float("nan")]), (jnp.array(float("nan")), None))
    valid = (_test_result([0.01, 0.02]), (jnp.array(0.03), None))
    results = iter([invalid, valid] if invalid_first else [valid, invalid])
    monkeypatch.setattr(cis_map, "map_cis_single", lambda *args, **kwargs: next(results))
    log = _LoggerStub()

    map_cis = getattr(cis_map, "map_cis")
    chunks = list(map_cis(data, snp_test, gene_test, mode="cis", verbose=False, log=log, seed=0))
    output_path = tmp_path / "cis.parquet"
    assert cli._write_scan_results(iter(chunks), output_path)
    output = pl.read_parquet(output_path)

    invalid_row = output.filter(pl.col("phenotype_id") == "invalid")
    valid_row = output.filter(pl.col("phenotype_id") == "valid")
    assert output.schema["snp"] == pl.Utf8
    assert output.schema["pos"] == pl.Int64
    assert output.schema["pvalue"] == pl.Float64
    assert invalid_row["snp"].item() is None
    assert invalid_row["pos"].item() is None
    assert invalid_row["pvalue"].item() is None
    assert invalid_row["result_valid"].item() is False
    assert invalid_row["failure_reason"].item() == "no_finite_pvalues"
    assert valid_row["result_valid"].item() is True
    assert valid_row["failure_reason"].item() is None
    assert log.warnings == ["No finite p-values for invalid over region 1:1-200; emitting an invalid result row."]


def test_cis_scan_streams_map_cis_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = SimpleNamespace(window=500_000, tss_centered=False, verbose=False, seed=0, out=str(tmp_path / "jaxqtl"))
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
                "result_valid": [True],
                "failure_reason": [None],
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
    args = SimpleNamespace(window=500_000, tss_centered=False, verbose=False, seed=0, out=str(tmp_path / "jaxqtl"))
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
