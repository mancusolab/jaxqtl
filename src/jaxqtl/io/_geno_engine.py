# pattern: Imperative Shell

from collections import Counter
from collections.abc import Iterator
from typing import Literal

import genoio
import numpy as np
import polars as pl

import jax.numpy as jnp

from jaxtyping import Array

from ._geno import GenotypeData


def _validate_sample_info(sample_info: pl.DataFrame) -> None:
    if "iid" not in sample_info.columns:
        raise ValueError("sample_info must include an 'iid' column")
    if sample_info.get_column("iid").n_unique() != sample_info.height:
        raise ValueError("sample_info contains duplicate IID values")


def _normalize_variant_info(variant_info: pl.DataFrame) -> pl.DataFrame:
    id_column = "id" if "id" in variant_info.columns else "snp"
    required_columns = {"chrom", id_column, "pos", "a0", "a1"}
    missing_columns = sorted(required_columns - set(variant_info.columns))
    if missing_columns:
        raise ValueError(f"variant_info is missing required columns: {missing_columns}")

    return variant_info.select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col(id_column).alias("snp"),
        pl.col("pos"),
        pl.col("a0"),
        pl.col("a1"),
    )


def _row_order_for_frozen_iids(returned_samples: pl.DataFrame, frozen_iids: list[str]) -> list[int]:
    if "iid" not in returned_samples.columns:
        raise ValueError("returned_samples must include an 'iid' column")

    returned_iids = returned_samples.get_column("iid").to_list()
    returned_iid_counts = Counter(returned_iids)
    duplicate_iids = sorted(iid for iid, count in returned_iid_counts.items() if count > 1)
    missing_iids = sorted(set(frozen_iids) - set(returned_iids))
    unexpected_iids = sorted(set(returned_iids) - set(frozen_iids))

    if returned_samples.height != len(frozen_iids) or missing_iids or unexpected_iids or duplicate_iids:
        raise ValueError(
            "returned_samples must match frozen IIDs exactly; "
            f"expected height {len(frozen_iids)}, got {returned_samples.height}; "
            f"missing IIDs: {missing_iids}; "
            f"unexpected IIDs: {unexpected_iids}; "
            f"duplicate IIDs: {duplicate_iids}"
        )

    row_index_by_iid = {iid: index for index, iid in enumerate(returned_iids)}
    return [row_index_by_iid[iid] for iid in frozen_iids]


def _to_jax_filtered_genotype(
    genotype: np.ndarray,
    variant_info: pl.DataFrame,
    row_order: list[int],
) -> tuple[Array, pl.DataFrame]:
    genotype = genotype[row_order, :]
    genotype_jax = jnp.asarray(genotype)
    normalized_variant_info = _normalize_variant_info(variant_info)

    if genotype_jax.shape[1] == 0:
        return genotype_jax, normalized_variant_info

    snp_var = jnp.var(genotype_jax, axis=0)
    keep = ~jnp.isnan(snp_var) & (snp_var > 0)
    genotype_jax = genotype_jax[:, keep]
    normalized_variant_info = normalized_variant_info.filter(np.asarray(keep))

    return genotype_jax, normalized_variant_info


class GenoioData(GenotypeData):
    """genoio-backed genotype data adapter."""

    genotype: genoio.Dataset
    sample_info: pl.DataFrame
    variant_info: pl.DataFrame

    @property
    def dataset(self) -> genoio.Dataset:
        """Underlying genoio dataset."""
        return self.genotype

    @classmethod
    def load(cls, prefix: str) -> "GenoioData":
        """Load a PLINK1 genotype dataset through genoio."""
        dataset = genoio.bfile(prefix)
        sample_info = dataset.samples()
        _validate_sample_info(sample_info)
        return cls(dataset, sample_info, _normalize_variant_info(dataset.variants()))

    def replace_individuals(self, sample_info: pl.DataFrame) -> "GenoioData":
        """Return a copy of the dataset with sample order frozen by `sample_info`."""
        _validate_sample_info(sample_info)
        return GenoioData(self.dataset, sample_info, self.variant_info)

    def filter_individuals(self, individuals: list[str], how: Literal["keep", "drop"]) -> "GenoioData":
        """Keep or drop individuals by IID, preserving current source/sample order semantics."""
        if how not in ["keep", "drop"]:
            raise ValueError("`how` must be 'keep' or 'drop'")

        pl_how = "semi" if how == "keep" else "anti"
        iid_frame = pl.DataFrame({"iid": individuals}, schema={"iid": self.sample_info.schema["iid"]})
        subset = self.sample_info.join(iid_frame, on="iid", how=pl_how)
        return self.replace_individuals(subset)

    def query_cis(self, chrom: str, start: int, end: int) -> tuple[Array, pl.DataFrame]:
        """Extract genotypes and variant metadata for a chromosome interval."""
        frozen_iids = self.sample_info.get_column("iid").to_list()
        genotype, returned_samples, variant_info = self.dataset.read(
            variants=genoio.region(f"{chrom}:{start}-{end}"),
            samples=frozen_iids,
            missing="nan",
            return_samples=True,
            return_variants=True,
        )

        row_order = _row_order_for_frozen_iids(returned_samples, frozen_iids)
        return _to_jax_filtered_genotype(genotype, variant_info, row_order)

    def iter_geno(self, chunk_size: int) -> Iterator[tuple[Array, pl.DataFrame]]:
        """Yield genotype blocks and variant metadata in fixed-size chunks."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        frozen_iids = self.sample_info.get_column("iid").to_list()

        def blocks() -> Iterator[tuple[Array, pl.DataFrame]]:
            for genotype, returned_samples, variant_info in self.dataset.iter_blocks(
                size=chunk_size,
                samples=frozen_iids,
                missing="nan",
                return_samples=True,
                return_variants=True,
            ):
                row_order = _row_order_for_frozen_iids(returned_samples, frozen_iids)
                yield _to_jax_filtered_genotype(genotype, variant_info, row_order)

        return blocks()
