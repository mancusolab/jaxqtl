# pattern: Imperative Shell

from collections.abc import Iterator
from typing import Literal

import genoio
import polars as pl

from jaxtyping import Array

from ._geno import GenotypeData


def _validate_sample_info(sample_info: pl.DataFrame) -> None:
    if "iid" not in sample_info.columns:
        raise ValueError("sample_info must include an 'iid' column")
    if sample_info.get_column("iid").n_unique() != sample_info.height:
        raise ValueError("sample_info contains duplicate IID values")


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
        return cls(dataset, sample_info, dataset.variants())

    def replace_individuals(self, sample_info: pl.DataFrame) -> "GenoioData":
        """Return a copy of the dataset with sample order frozen by `sample_info`."""
        _validate_sample_info(sample_info)
        return GenoioData(self.dataset, sample_info, self.variant_info)

    def filter_individuals(self, individuals: list[str], how: Literal["keep", "drop"]) -> "GenoioData":
        """Keep or drop individuals by IID, preserving current source/sample order semantics."""
        if how not in ["keep", "drop"]:
            raise ValueError("`how` must be have value of 'keep' or 'drop'")

        pl_how = "semi" if how == "keep" else "anti"
        iid_frame = pl.DataFrame({"iid": individuals}, schema={"iid": self.sample_info.schema["iid"]})
        subset = self.sample_info.join(iid_frame, on="iid", how=pl_how)
        return self.replace_individuals(subset)

    def query_cis(self, chrom: str, start: int, end: int) -> tuple[Array, pl.DataFrame]:
        """Extract genotypes and variant metadata for a chromosome interval."""
        raise NotImplementedError("GenoioData.query_cis is implemented in a later genoio-engine task")

    def iter_geno(self, chunk_size: int) -> Iterator[tuple[Array, pl.DataFrame]]:
        """Yield genotype blocks and variant metadata in fixed-size chunks."""
        raise NotImplementedError("GenoioData.iter_geno is implemented in a later genoio-engine task")
