# pattern: Imperative Shell

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import genoio
import numpy as np
import polars as pl

import jax
import jax.numpy as jnp

from jaxqtl.io._geno_engine import filter_sample_ids, normalize_variant_info
from jaxqtl.io._pheno import ExpressionData
from jaxqtl.map.data import CisData, ReadyDataState


REPO_ROOT = Path(__file__).resolve().parents[2]
GENO_PREFIX = REPO_ROOT / "tutorial" / "input" / "chr22_N100"


class _SyntheticGenoioDataset:
    def __init__(self) -> None:
        self.block_calls: list[tuple[int, dict[str, object]]] = []
        self.region_calls: list[tuple[list[object], dict[str, object]]] = []
        self.returned_samples = pl.DataFrame({"iid": ["iid1", "iid2", "iid3"]})
        self.returned_variants = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "id": ["rs1", "rs2"],
                "pos": [10, 20],
                "a0": ["A", "C"],
                "a1": ["T", "G"],
            }
        )
        self.genotype = np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )

    def samples(self) -> pl.DataFrame:
        return self.returned_samples

    def variants(self) -> pl.DataFrame:
        return self.returned_variants

    def iter_blocks(self, size: int, **read_options: object) -> Iterator[tuple[np.ndarray, pl.DataFrame]]:
        self.block_calls.append((size, read_options))
        yield self.genotype, self.returned_variants

    def iter_regions(
        self, regions: list[object], **read_options: object
    ) -> Iterator[tuple[object, tuple[np.ndarray, pl.DataFrame]]]:
        self.region_calls.append((regions, read_options))
        for region in regions:
            yield region, (self.genotype, self.returned_variants)


def _expression() -> ExpressionData:
    pheno = pl.DataFrame(
        {
            "iid": ["iid2", "iid1", "iid3"],
            "gene1": [20.0, 10.0, 30.0],
        }
    )
    pheno_meta = pl.DataFrame(
        {
            "chrom": ["1"],
            "start": [5],
            "end": [15],
            "phenotype_id": ["gene1"],
        }
    )
    libsize = pl.DataFrame({"iid": ["iid2", "iid1", "iid3"], "libsize": [2.0, 1.0, 3.0]})
    return ExpressionData(pheno, pheno_meta, libsize)


def test_filter_sample_ids_preserves_genoio_source_order_for_keep_and_drop() -> None:
    samples = pl.DataFrame({"iid": ["iid1", "iid2", "iid3", "iid4"]})

    assert filter_sample_ids(samples, keep=["iid3", "iid1"]) == ("iid1", "iid3")
    assert filter_sample_ids(samples, drop=["iid2", "iid4"]) == ("iid1", "iid3")


def test_normalize_variant_info_preserves_genoio_counted_allele_convention() -> None:
    variants = pl.DataFrame(
        {
            "chrom": [22],
            "id": ["rs1"],
            "pos": [123],
            "a0": ["A"],
            "a1": ["G"],
        }
    )

    observed = normalize_variant_info(variants)

    assert observed.to_dict(as_series=False) == {
        "chrom": ["22"],
        "snp": ["rs1"],
        "pos": [123],
        "a0": ["A"],
        "a1": ["G"],
    }


def test_ready_data_state_aligns_to_genoio_source_order_and_uses_sample_pushdown() -> None:
    dataset = _SyntheticGenoioDataset()
    covar = pl.DataFrame({"iid": ["iid3", "iid1", "iid2"], "cov": [30.0, 10.0, 20.0]})

    ready = ReadyDataState.from_data(cast(genoio.Dataset, dataset), _expression(), covar, keep_samples=["iid3", "iid1"])
    G, variant_info = next(ready.iter_geno(chunk_size=128))

    assert ready.sample_ids == ("iid1", "iid3")
    assert ready.expression.pheno.get_column("iid").to_list() == ["iid1", "iid3"]
    np.testing.assert_array_equal(np.asarray(ready.covar), np.array([[10.0], [30.0]], dtype=np.float32))
    assert dataset.block_calls[0][0] == 128
    assert dataset.block_calls[0][1]["samples"] == ["iid1", "iid3"]
    assert dataset.block_calls[0][1]["missing"] == "impute"
    variants = cast(genoio.FilterExpr, dataset.block_calls[0][1]["variants"])
    assert variants.to_ir() == genoio.polymorphic().to_ir()
    assert isinstance(G, jax.Array)
    assert variant_info.get_column("snp").to_list() == ["rs1", "rs2"]


def test_ready_data_state_iter_cis_uses_genoio_regions_with_polymorphic_filter() -> None:
    dataset = _SyntheticGenoioDataset()
    covar = pl.DataFrame({"iid": ["iid1", "iid2", "iid3"], "cov": [1.0, 2.0, 3.0]})
    ready = ReadyDataState.from_data(cast(genoio.Dataset, dataset), _expression(), covar)

    cis_data = next(ready.iter_cis(window=5))

    assert isinstance(cis_data, CisData)
    assert cis_data.G.shape == (3, 2)
    regions, read_options = dataset.region_calls[0]
    assert len(regions) == 1
    region = cast(genoio.FilterExpr, regions[0])
    assert region.to_ir()["op"] == "and"
    assert read_options["samples"] == ["iid1", "iid2", "iid3"]
    assert read_options["missing"] == "impute"
    assert read_options["return_variants"] is True


def test_cis_data_uses_genoio_variant_metadata_without_legacy_columns() -> None:
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
