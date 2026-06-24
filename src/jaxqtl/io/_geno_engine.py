# pattern: Imperative Shell

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import genoio
import numpy as np
import polars as pl


GenotypeSource = Literal["bfile", "pfile", "vcf", "bgen"]


@dataclass(frozen=True, slots=True)
class GenotypeReadOptions:
    """Mapping read options passed directly to genoio."""

    kind: Literal["geno", "haplo"] = "geno"
    dosage: Literal["hardcall", "dosage"] = "hardcall"
    sparse: bool | Literal["csc", "csr"] = False
    missing: Literal["impute", "nan", "raise"] = "impute"
    dtype: object = np.float32

    def kwargs(
        self,
        *,
        samples: tuple[str, ...],
        variants: genoio.FilterExpr | None = None,
    ) -> dict[str, object]:
        options: dict[str, object] = {
            "kind": self.kind,
            "dosage": self.dosage,
            "sparse": self.sparse,
            "samples": list(samples),
            "missing": self.missing,
            "dtype": self.dtype,
            "return_variants": True,
        }
        if variants is not None:
            options["variants"] = variants
        return options


def load_genotype_dataset(source: GenotypeSource, path: str) -> genoio.Dataset:
    """Open a genotype source through genoio without wrapping the returned Dataset."""
    match source:
        case "bfile":
            return genoio.bfile(path)
        case "pfile":
            return genoio.pfile(path)
        case "vcf":
            return genoio.vcf(path)
        case "bgen":
            return genoio.bgen(path)
        case _:
            raise ValueError(f"Unsupported genotype source: {source!r}")


def default_variant_filter() -> genoio.FilterExpr:
    """Default mapping filter: genoio handles monomorphic variants before return."""
    return genoio.polymorphic()


def region_filter(chrom: str, start: int, end: int, base_filter: genoio.FilterExpr) -> genoio.FilterExpr:
    """Compose a cis region predicate with the global variant predicate."""
    return genoio.region(f"{chrom}:{start}-{end}") & base_filter


def normalize_sample_info(sample_info: pl.DataFrame) -> pl.DataFrame:
    """Return validated IID-only sample metadata in the existing row order."""
    if "iid" not in sample_info.columns:
        raise ValueError("genotype samples must include an 'iid' column")
    samples = sample_info.select("iid")
    if samples.get_column("iid").n_unique() != samples.height:
        raise ValueError("genotype samples contain duplicate IID values")
    return samples


def filter_sample_ids(
    sample_info: pl.DataFrame,
    *,
    keep: list[str] | None = None,
    drop: list[str] | None = None,
) -> tuple[str, ...]:
    """Apply keep/drop IID filters while preserving genoio source order."""
    if keep is not None and drop is not None:
        raise ValueError("Cannot specify both keep and drop sample filters")

    samples = normalize_sample_info(sample_info)
    if keep is not None:
        selected = samples.join(pl.DataFrame({"iid": keep}), on="iid", how="semi", maintain_order="left")
    elif drop is not None:
        selected = samples.join(pl.DataFrame({"iid": drop}), on="iid", how="anti", maintain_order="left")
    else:
        selected = samples

    return tuple(selected.get_column("iid").to_list())


def normalize_variant_info(variant_info: pl.DataFrame) -> pl.DataFrame:
    """Return mapping metadata with genoio allele semantics preserved."""
    id_column = "id" if "id" in variant_info.columns else "snp"
    required_columns = {"chrom", id_column, "pos", "a0", "a1"}
    missing_columns = sorted(required_columns - set(variant_info.columns))
    if missing_columns:
        raise ValueError(f"variant metadata is missing required columns: {missing_columns}")

    return variant_info.select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col(id_column).alias("snp"),
        pl.col("pos"),
        pl.col("a0"),
        pl.col("a1"),
    )
