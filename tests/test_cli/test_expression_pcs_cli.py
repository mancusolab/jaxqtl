# pattern: Imperative Shell

import numpy as np
import polars as pl
import pytest

from jaxqtl import cli


@pytest.fixture
def expression_files(tmp_path):
    values = np.array(
        [
            [100.0, 4.0, 1.0, 9.0, 0.0],
            [1.0, 8.0, 3.0, 2.0, 0.0],
            [3.0, 5.0, 2.0, 4.0, 0.0],
            [8.0, 1.0, 4.0, 3.0, 0.0],
            [2.0, 4.0, 9.0, 6.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 5.0],
        ]
    )
    samples = [f"{i:03d}" for i in range(1, 7)]
    bed = tmp_path / "expression.bed"
    pl.DataFrame(
        {
            "chrom": ["1", "1", "2", "1", "2"],
            "start": [1, 2, 3, 4, 5],
            "end": [2, 3, 4, 5, 6],
            "gene_id": [f"g{i}" for i in range(1, 6)],
            **dict(zip(samples, values.tolist(), strict=True)),
        }
    ).write_csv(bed, separator="\t")
    covar = tmp_path / "covar.tsv"
    pl.DataFrame(
        {
            "iid": ["005", "003", "002", "004", "006", "999"],
            "age": [50, 30, 20, 40, 60, 99],
            "total": [190.0, 140.0, 120.0, 160.0, 170.0, 1000.0],
        }
    ).write_csv(covar, separator="\t")
    return bed, covar, values, samples


def _run(tmp_path, bed, *options):
    output = tmp_path / "pcs.tsv"
    cli.main(["compute-pcs", "--pheno", str(bed), "--num-pcs", "2", "--out", str(output), *options])
    pcs = pl.read_csv(output, separator="\t", schema_overrides={"iid": pl.String})
    variance = pl.read_csv(f"{output}.variance.tsv", separator="\t")
    return pcs, variance


def _check_svd(pcs, variance, transformed):
    standardized = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0)
    expected, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    actual = pcs.select("ExprPC1", "ExprPC2").to_numpy()
    np.testing.assert_allclose(np.abs(expected[:, :2].T @ actual), np.eye(2), atol=2e-3)
    ratios = singular[:2] ** 2 / np.sum(singular**2)
    assert variance.columns == ["component", "explained_variance_ratio", "cumulative_explained_variance_ratio"]
    assert variance["component"].to_list() == ["ExprPC1", "ExprPC2"]
    np.testing.assert_allclose(variance["explained_variance_ratio"].to_numpy(), ratios, rtol=2e-4)
    np.testing.assert_allclose(variance["cumulative_explained_variance_ratio"].to_numpy(), np.cumsum(ratios), rtol=2e-4)


@pytest.mark.parametrize("selection", ["--keep", "--exclude"])
def test_pca_sample_filters_and_standalone_output(tmp_path, expression_files, selection):
    bed, _, values, samples = expression_files
    ids = tmp_path / "samples.txt"
    ids.write_text("006\n004\n003\n002\n005\n" if selection == "--keep" else "001\n")
    pcs, variance = _run(tmp_path, bed, selection, str(ids), "--transform", "lognorm")
    assert pcs["iid"].to_list() == samples[1:]
    assert pcs.columns == ["iid", "ExprPC1", "ExprPC2"]
    counts = values[1:]
    sizes = counts.sum(axis=1)
    _check_svd(pcs, variance, np.log1p(counts * np.median(sizes) / sizes[:, None]))


@pytest.mark.parametrize("size_source", ["automatic", "file", "covar"])
def test_pca_aligns_cohort_before_filtering_and_normalizes_with_full_library_sizes(
    tmp_path,
    expression_files,
    size_source,
):
    bed, covar, values, samples = expression_files
    options = ["--covar", str(covar), "--transform", "lognorm", "--chr", "1", "--min-indiv-expr-pct", "0.4"]
    sizes = values[1:].sum(axis=1)
    if size_source == "file":
        sizes = np.array([120.0, 140.0, 160.0, 190.0, 170.0])
        path = tmp_path / "library.tsv"
        pl.DataFrame({"iid": samples[1:], "libsize": sizes}).reverse().write_csv(path, separator="\t")
        options += ["--libsize", str(path)]
    elif size_source == "covar":
        sizes = np.array([120.0, 140.0, 160.0, 190.0, 170.0])
        options += ["--libsize-name-from-covar", "total"]
    pcs, variance = _run(tmp_path, bed, *options)
    assert pcs["iid"].to_list() == samples[1:]
    assert pcs["age"].to_list() == [20, 30, 40, 50, 60]
    assert "total" in pcs.columns
    # g3/g5 contribute to library size and sample QC, even though --chr excludes them from PCA.
    counts = values[1:, [0, 1, 3]]
    _check_svd(pcs, variance, np.log1p(counts * np.median(sizes) / sizes[:, None]))


@pytest.mark.parametrize("selection", ["--genes", "--gene-list", "--rm-genes", "--exclude-gene-list"])
def test_pca_gene_selections(tmp_path, expression_files, selection):
    bed, _, values, _ = expression_files
    names = "g1,g2,g4" if selection in {"--genes", "--gene-list"} else "g3,g5"
    if selection.endswith("list"):
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text(names.replace(",", "\n") + "\n")
        argument = str(gene_file)
    else:
        argument = names
    pcs, variance = _run(tmp_path, bed, selection, argument)
    _check_svd(pcs, variance, values[:, [0, 1, 3]])


def test_pca_gene_prevalence_uses_selected_cohort(tmp_path, expression_files):
    bed, covar, values, _ = expression_files
    pcs, variance = _run(tmp_path, bed, "--covar", str(covar), "--min-gene-expr-pct", "0.2")
    # g5 is expressed in exactly 1/5 retained samples and is excluded by the strict threshold.
    _check_svd(pcs, variance, values[1:, :4])


def test_pca_output_keeps_iid_first_when_covariate_iid_is_not_first(tmp_path, expression_files):
    bed, covar, _, samples = expression_files
    pl.read_csv(covar, separator="\t", schema_overrides={"iid": pl.String}).select("age", "iid", "total").write_csv(
        covar, separator="\t"
    )
    pcs, _ = _run(tmp_path, bed, "--covar", str(covar))
    assert pcs.columns == ["iid", "age", "total", "ExprPC1", "ExprPC2"]
    assert pcs["iid"].to_list() == samples[1:]


@pytest.mark.parametrize("extra", [[], ["--transform", "tmm"], ["--keep", "a", "--exclude", "b"]])
def test_pca_parser_rejects_missing_component_count_or_unsupported_options(extra):
    args = ["compute-pcs", "--pheno", "unused.bed"]
    if extra:
        args += ["--num-pcs", "2", *extra]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("case", ["duplicate", "null", "collision", "disjoint"])
def test_pca_rejects_invalid_covariate_cohorts(tmp_path, expression_files, case):
    bed, covar, _, _ = expression_files
    samples = {
        "duplicate": ["002", "002"],
        "null": ["002", None],
        "collision": ["002", "003"],
        "disjoint": ["999", "998"],
    }[case]
    name = "ExprPC1" if case == "collision" else "age"
    pl.DataFrame({"iid": samples, name: [1, 2]}).write_csv(covar, separator="\t")
    with pytest.raises(ValueError, match="sample IDs|ExprPC|samples"):
        _run(tmp_path, bed, "--covar", str(covar))
    assert not (tmp_path / "pcs.tsv").exists()


def test_pca_ignores_nonnumeric_library_sizes_for_extra_samples(tmp_path, expression_files):
    bed, _, values, samples = expression_files
    sizes = values.sum(axis=1)
    path = tmp_path / "library.tsv"
    pl.DataFrame({"iid": [*samples, "999"], "libsize": [*map(str, sizes), "unavailable"]}).reverse().write_csv(
        path, separator="\t"
    )
    pcs, variance = _run(tmp_path, bed, "--transform", "lognorm", "--libsize", str(path))
    assert pcs["iid"].to_list() == samples
    _check_svd(pcs, variance, np.log1p(values * np.median(sizes) / sizes[:, None]))


@pytest.mark.parametrize("case", ["missing", "duplicate", "negative", "nonfinite", "nonnumeric"])
def test_pca_rejects_invalid_external_library_sizes(tmp_path, expression_files, case):
    bed, _, _, samples = expression_files
    sizes = [100.0] * len(samples)
    if case == "missing":
        samples, sizes = samples[:-1], sizes[:-1]
    elif case == "duplicate":
        samples = [samples[1], *samples[1:]]
    elif case == "negative":
        sizes[0] = -1.0
    elif case == "nonnumeric":
        sizes = ["unavailable", *map(str, sizes[1:])]
    else:
        sizes[0] = float("inf")
    path = tmp_path / "library.tsv"
    pl.DataFrame({"iid": samples, "libsize": sizes}).write_csv(path, separator="\t")
    with pytest.raises(ValueError, match="library sizes|sample IDs"):
        _run(tmp_path, bed, "--transform", "lognorm", "--libsize", str(path))


@pytest.mark.parametrize(
    "extra", [["--libsize", "unused.tsv"], ["--transform", "lognorm", "--libsize-name-from-covar", "total"]]
)
def test_pca_rejects_library_options_without_required_context(tmp_path, expression_files, extra):
    with pytest.raises(ValueError, match="lognorm|covar"):
        _run(tmp_path, expression_files[0], *extra)


@pytest.mark.parametrize("bad", [None, float("inf"), -1.0])
def test_pca_rejects_invalid_counts_before_gene_filtering(tmp_path, expression_files, bad):
    bed = expression_files[0]
    frame = pl.read_csv(bed, separator="\t")
    frame = frame.with_columns(
        pl.when(pl.col("gene_id") == "g5").then(pl.lit(bad)).otherwise(pl.col("001")).alias("001")
    )
    frame.write_csv(bed, separator="\t")
    with pytest.raises(ValueError, match="finite|nonnegative"):
        _run(tmp_path, bed, "--genes", "g1,g2,g4", "--transform", "log1p")


@pytest.mark.parametrize(
    "options",
    [
        ["--min-indiv-expr-pct", "1"],
        ["--min-gene-expr-pct", "1"],
        ["--rm-genes", "g1,g2,g3,g4,g5"],
        ["--chr", "missing"],
    ],
)
def test_pca_rejects_filters_leaving_no_usable_data(tmp_path, expression_files, options):
    with pytest.raises(ValueError, match="samples|genes|chromosome"):
        _run(tmp_path, expression_files[0], *options)
