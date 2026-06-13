# pattern: Imperative Shell

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import jax
import jax.numpy as jnp

from jaxqtl.io import GenoioData, PlinkData
from jaxqtl.map.data import CisData


REPO_ROOT = Path(__file__).resolve().parents[2]
GENO_PREFIX = REPO_ROOT / "tutorial" / "input" / "chr22_N100"


def _source_iids() -> list[str]:
    fam_path = GENO_PREFIX.with_suffix(".fam")
    return [line.split()[1] for line in fam_path.read_text().splitlines()]


def _sample_iids(sample_info: pl.DataFrame) -> list[str]:
    return sample_info.get_column("iid").to_list()


def _first_five_variant_region(data: GenoioData) -> tuple[str, int, int]:
    variant_info = data.variant_info.head(5)
    return (
        variant_info.get_column("chrom")[0],
        variant_info.get_column("pos")[0],
        variant_info.get_column("pos")[-1],
    )


def _full_variant_region(data: GenoioData) -> tuple[str, int, int]:
    return (
        data.variant_info.get_column("chrom")[0],
        data.variant_info.get_column("pos").min(),
        data.variant_info.get_column("pos").max(),
    )


def _assert_counted_allele_orientation(
    query_G: jax.Array,
    query_variant_info: pl.DataFrame,
    legacy_G: jax.Array,
    legacy_variant_info: pl.DataFrame,
) -> set[str]:
    legacy_rows = {row["snp"]: (index, row) for index, row in enumerate(legacy_variant_info.iter_rows(named=True))}
    observed_orientations: set[str] = set()

    for query_index, query_row in enumerate(query_variant_info.iter_rows(named=True)):
        legacy_index, legacy_row = legacy_rows[query_row["snp"]]
        query_values = np.asarray(query_G[:, query_index])
        legacy_values = np.asarray(legacy_G[:, legacy_index])
        finite = np.isfinite(query_values) & np.isfinite(legacy_values)
        assert finite.any()

        if query_row["a1"] == legacy_row["a1"]:
            observed_orientations.add("same")
            np.testing.assert_allclose(query_values[finite], legacy_values[finite])
        elif query_row["a1"] == legacy_row["a0"]:
            observed_orientations.add("opposite")
            np.testing.assert_allclose(query_values[finite], 2 - legacy_values[finite])
            assert np.any(query_values[finite] != legacy_values[finite])
        else:
            raise AssertionError(
                f"Counted allele {query_row['a1']} is not represented in legacy metadata " f"for {query_row['snp']}"
            )

    return observed_orientations


def test_sample_info_has_iid_and_preserves_source_order() -> None:
    data = GenoioData.load(str(GENO_PREFIX))

    assert "iid" in data.sample_info.columns
    assert _sample_iids(data.sample_info) == _source_iids()


def test_replace_individuals_freezes_reversed_iid_order() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    reversed_sample_info = data.sample_info.reverse()

    replaced = data.replace_individuals(reversed_sample_info)

    assert _sample_iids(replaced.sample_info) == list(reversed(_source_iids()))


def test_filter_individuals_matches_cli_keep_and_drop_expectations() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    source_iids = _source_iids()

    keep_iids = [source_iids[2], source_iids[0], source_iids[-1]]
    kept = data.filter_individuals(keep_iids, "keep")
    assert _sample_iids(kept.sample_info) == [source_iids[0], source_iids[2], source_iids[-1]]

    drop_iids = [source_iids[1], source_iids[3]]
    dropped = data.filter_individuals(drop_iids, "drop")
    assert _sample_iids(dropped.sample_info) == [iid for iid in source_iids if iid not in drop_iids]


def test_replace_individuals_rejects_duplicate_iids() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    duplicate_sample_info = data.sample_info.head(2).with_columns(pl.lit(_source_iids()[0]).alias("iid"))

    with pytest.raises(ValueError, match="duplicate IID"):
        data.replace_individuals(duplicate_sample_info)


def test_variant_info_uses_minimal_mapping_metadata_columns() -> None:
    data = GenoioData.load(str(GENO_PREFIX))

    assert data.variant_info.columns == ["chrom", "snp", "pos", "a0", "a1"]


def test_cis_data_uses_named_variant_metadata_without_legacy_columns() -> None:
    cis_info = pl.DataFrame(
        {
            "chrom": ["22", "22"],
            "snp": ["rs1", "rs2"],
            "pos": [110, 130],
            "a0": ["A", "G"],
            "a1": ["C", "T"],
        }
    )
    cis_data = CisData(
        X=jnp.ones((3, 1)),
        G=jnp.array([[0.0, 2.0], [1.0, 2.0], [2.0, 0.0]]),
        y=jnp.array([0.0, 1.0, 2.0]),
        offset=jnp.array(0.0),
        gene_name="gene1",
        chrom="22",
        gene_start=100,
        gene_end=120,
        cis_info=cis_info,
        start=50,
        end=150,
    )

    snp_info = cis_data.get_snp_info(0)
    assert snp_info.id == "rs1"
    assert snp_info.pos == 110
    assert snp_info.a1 == "C"
    assert snp_info.a0 == "A"
    assert snp_info.tss_distance == 10

    output = cis_data.get_cis_info()
    assert output.columns == ["chrom", "snp", "pos", "a1", "a0", "tss_distance", "af", "ma_count"]
    assert output.select(["chrom", "snp", "pos", "a1", "a0"]).to_dict(as_series=False) == {
        "chrom": ["22", "22"],
        "snp": ["rs1", "rs2"],
        "pos": [110, 130],
        "a1": ["C", "T"],
        "a0": ["A", "G"],
    }


def test_query_cis_returns_jax_matrix_and_normalized_metadata() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    chrom, start, end = _first_five_variant_region(data)

    G, variant_info = data.query_cis(chrom, start, end)

    assert isinstance(G, jax.Array)
    assert G.shape == (data.sample_info.height, 5)
    assert variant_info.columns == ["chrom", "snp", "pos", "a0", "a1"]
    assert variant_info.height == G.shape[1]
    assert variant_info.get_column("snp").to_list() == (data.variant_info.head(5).get_column("snp").to_list())
    assert bool(jnp.all(jnp.var(G, axis=0) > 0))


def test_query_cis_reorders_rows_to_frozen_iid_order() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    reversed_data = data.replace_individuals(data.sample_info.reverse())
    chrom, start, end = _first_five_variant_region(data)

    source_G, source_variant_info = data.query_cis(chrom, start, end)
    reversed_G, reversed_variant_info = reversed_data.query_cis(chrom, start, end)

    np.testing.assert_array_equal(np.asarray(reversed_G), np.asarray(source_G)[::-1, :])
    assert reversed_variant_info.equals(source_variant_info)


def test_query_cis_empty_region_returns_empty_jax_matrix_and_metadata() -> None:
    data = GenoioData.load(str(GENO_PREFIX))

    G, variant_info = data.query_cis("22", 1, 10)

    assert isinstance(G, jax.Array)
    assert G.shape == (data.sample_info.height, 0)
    assert variant_info.columns == ["chrom", "snp", "pos", "a0", "a1"]
    assert variant_info.height == 0


def test_query_cis_matches_plink_values_by_counted_allele_orientation() -> None:
    genoio_data = GenoioData.load(str(GENO_PREFIX))
    plink_data = PlinkData.load(str(GENO_PREFIX))
    chrom, start, end = _first_five_variant_region(genoio_data)

    genoio_G, genoio_variant_info = genoio_data.query_cis(chrom, start, end)
    plink_G, plink_variant_info = plink_data.query_cis(chrom, start, end)

    same_orientations = _assert_counted_allele_orientation(
        plink_G,
        plink_variant_info,
        plink_G,
        plink_variant_info,
    )
    observed_orientations = _assert_counted_allele_orientation(
        genoio_G,
        genoio_variant_info,
        plink_G,
        plink_variant_info,
    )

    assert same_orientations == {"same"}
    assert "opposite" in observed_orientations


def test_iter_geno_rejects_invalid_chunk_size() -> None:
    data = GenoioData.load(str(GENO_PREFIX))

    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        list(data.iter_geno(0))


def test_iter_geno_blocks_match_full_read_after_conversion_and_filtering() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    chrom, start, end = _full_variant_region(data)
    full_G, full_variant_info = data.query_cis(chrom, start, end)

    blocks = list(data.iter_geno(max(1, data.variant_info.height // 2)))

    assert len(blocks) > 1
    for block_G, block_variant_info in blocks:
        assert isinstance(block_G, jax.Array)
        assert block_G.ndim == 2
        assert block_G.shape[0] == data.sample_info.height
        assert block_variant_info.height == block_G.shape[1]
        assert block_variant_info.columns == ["chrom", "snp", "pos", "a0", "a1"]

    observed_G = jnp.concatenate([block_G for block_G, _ in blocks], axis=1)
    observed_variant_info = pl.concat([block_variant_info for _, block_variant_info in blocks])

    np.testing.assert_array_equal(np.asarray(observed_G), np.asarray(full_G))
    assert observed_variant_info.equals(full_variant_info)
