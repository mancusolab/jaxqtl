import warnings

from abc import abstractmethod
from typing import Any, cast, Literal, NamedTuple

import numpy as np
import pandas as pd
import polars as pl

from cyvcf2 import VCF
from dask.array.core import Array as DaskArray
from pandas_plink import read_plink

import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array


def _impute_geno(geno, bim, fam):
    # to make sure that the bim index is continuous
    bim = bim.reset_index(drop=True)
    # if we observe SNPs have nan value for all participants (although not likely), drop them
    (del_idx,) = jnp.where(jnp.all(jnp.isnan(geno), axis=0))
    # this is the first time we modify the bim and geno
    # it's okay just to directly drop them
    bim = bim.drop(del_idx).reset_index(drop=True)
    geno = jnp.delete(geno, del_idx, 1)

    # if we observe SNPs that partially have nan value, impute them with column mean
    col_mean = jnp.nanmean(geno, axis=0)
    # it gives the dimension index of the nan value
    imp_idx = jnp.where(jnp.isnan(geno))
    if len(imp_idx[1]) != 0:
        # based on the column index of imp_idx, we used jnp.take to get the (multiple) value
        geno = geno.at[imp_idx].set(jnp.take(col_mean, imp_idx[1]))
    return geno, bim, fam


class GenoState(NamedTuple):
    genotype: Array
    bim: pd.DataFrame
    fam: pd.DataFrame


class GenotypeData(eqx.Module):
    """Read genotype or count data from different file format"""

    genotype: eqx.AbstractVar[Any]
    sample_info: eqx.AbstractVar[pl.DataFrame]
    variant_info: eqx.AbstractVar[pl.DataFrame]

    @classmethod
    @abstractmethod
    def load(cls, prefix: str) -> "GenotypeData":
        ...

    @abstractmethod
    def replace_individuals(self, sample_info: pl.DataFrame) -> "GenotypeData":
        ...

    @abstractmethod
    def query_cis(self, chrom: str, start: int, end: int) -> tuple[Array, pl.DataFrame]:
        ...


class PlinkData(GenotypeData):
    """Read raw genotype data from plink triplets
    prefix: chr22.bed, also accept chr*.bed (read everything)

    Note: read bed file is much faster than VCF file (parser)

    bim: chrom          snp   cm       pos a0 a1  i (one-based)
    fam: fid  iid father mother gender trait  i
    bed: zero-based
    """

    genotype: DaskArray
    sample_info: pl.DataFrame
    variant_info: pl.DataFrame

    sample_idx: Array = eqx.field(init=False)

    def __post_init__(self):
        self.sample_idx = self.sample_info.get_column("i").to_numpy()

    @classmethod
    def load(cls, prefix: str):
        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            bim, fam, bed = read_plink(prefix, verbose=False)
            genotype = cast(DaskArray, bed)
            sample_info = pl.DataFrame(fam)
            variant_info = pl.DataFrame(bim)
        return PlinkData(genotype, sample_info, variant_info)

    def filter_individuals(self, individuals: list[str], how: Literal["keep", "drop"]):
        if how not in ["keep", "drop"]:
            raise ValueError("`how` must be have value of 'keep' or 'drop'")

        pl_how = "semi" if how == "keep" else "anti"
        iid_series = pl.Series("iid", individuals)
        subset = self.sample_info.join(iid_series, on="iid", how=pl_how)
        return PlinkData(self.genotype, subset, self.variant_info)

    def replace_individuals(self, sample_info: pl.DataFrame):
        return PlinkData(self.genotype, sample_info, self.variant_info)

    def query_cis(self, chrom: str, start: int, end: int) -> tuple[Array, pl.DataFrame]:
        # subset cis variants
        cis_var_info = self.variant_info.filter(
            (pl.col("chrom") == str(chrom)) & (pl.col("pos") >= start) & (pl.col("pos") <= end)
        )

        # pull the variant indices as a NumPy array
        cis_idx = cis_var_info.get_column("i").to_numpy()

        # subset geno cis variants at the specified samples
        G = jnp.asarray(self.genotype[cis_idx, :][:, self.sample_idx].compute().T)  # (n, p)

        # drop monomorphnic sites
        snp_var = jnp.var(G, axis=0)
        keep = ~jnp.isnan(snp_var) & (snp_var > 0)
        G = G[:, keep]
        cis_var_info = cis_var_info.filter(np.asarray(keep))  # back to numpy for polars :(

        return G, cis_var_info

    def __call__(self, bed_path: str) -> GenoState:
        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            bim, fam, bed = read_plink(bed_path, verbose=False)
        fam = fam.set_index("iid", drop=False)
        G = jnp.asarray(bed.compute().T)  # nxp
        G, bim, fam = _impute_geno(G, bim, fam)
        return GenoState(G, bim, fam)


class VCFData(GenotypeData):
    def __call__(self, vcf_path: str) -> GenoState:
        """read genotype from VCF file
        Note: slower than PlinkReader()
        Recommend converting VCF file to bed file first using command:
        `plink2 --vcf example.vcf.gz --make-bed --out ex`

        """

        # read VCF files
        vcf = VCF(vcf_path, gts012=True)  # can add samples=[]
        fam = pd.DataFrame(vcf.samples).rename(columns={0: "iid"})  # individuals
        fam.set_index("iid", drop=False)

        genotype = []
        bim_list = []

        for idx, var in enumerate(vcf):
            genotype.append(var.gt_types)
            # var.ALT is a list of alternative allele
            bim_list.append([var.CHROM, var.ID, 0.0, var.POS, var.ALT[0], var.REF, idx])

        vcf.close()

        #  chrom        snp       cm     pos a0 a1  i
        bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "cm", "pos", "alt", "ref", "i"])

        G = jnp.asarray(genotype).T
        G, bim, fam = _impute_geno(G, bim, fam)

        return GenoState(G, bim, fam)
