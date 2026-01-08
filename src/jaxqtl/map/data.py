from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array

from ..io.geno import GenotypeData
from ..io.pheno import ExpressionData


class SNPInfo(eqx.Module):
    """Container for cis-variant metadata used in mapping results."""

    id: str
    pos: int
    a1: str
    a0: str
    tss_distance: int
    af: float
    ma_count: int


class CisData(eqx.Module):
    """Batch of genotype/phenotype data for a single gene-level cis window."""

    # individual-level info
    X: Array
    G: Array
    y: Array
    offset: Array

    # (gene/feature)-level info
    gene_name: str
    chrom: str
    gene_start: int
    gene_end: int

    # variant-level info
    cis_info: pl.DataFrame

    # analysis-level info; ie our window around gene start/end
    start: int
    end: int

    @property
    def num_snps(self) -> int:
        """Number of SNPs present in the cis-genotype matrix."""
        num_snp = 0 if self.G is None else self.G.shape[1]
        return num_snp

    def get_af_summary(self, idx: int) -> tuple[Array, Array]:
        """Compute allele frequency and minor allele count for a single variant."""
        g = self.G[:, idx]
        n = len(g)
        counts = jnp.sum(g, axis=0)  # count REF allele
        af = counts / (2.0 * n)
        flag = af <= 0.5
        ma_counts = jnp.where(flag, counts, 2 * n - counts)

        return af, ma_counts

    def get_snp_info(self, idx: int) -> SNPInfo:
        """Return SNPInfo for a variant at the provided column index."""
        af, ma_count = self.get_af_summary(idx)
        # chrom, snp, cm, pos, a0, a1, index
        _, snp_id, _, snp_pos, a0, a1, _ = self.cis_info.row(idx)
        tss_distance = snp_pos - self.gene_start
        return SNPInfo(snp_id, snp_pos, a1, a0, tss_distance, float(af), int(ma_count))

    def get_cis_info(self) -> pl.DataFrame:
        """Return cis variant information augmented with AF and minor allele counts."""
        n, p = self.G.shape
        counts = jnp.sum(self.G, axis=0)  # count REF allele
        af = counts / (2.0 * n)
        flag = af <= 0.5
        ma_counts = jnp.where(flag, counts, 2 * n - counts)
        local = self.cis_info.with_columns(
            (pl.col("pos") - pl.lit(self.gene_start)).alias("tss_distance"),
            pl.Series("af", np.array(af)),
            pl.Series("ma_count", np.array(ma_counts, dtype=int)),
        ).select(["chrom", "snp", "pos", "a1", "a0", "tss_distance", "af", "ma_count"])

        return local


@dataclass
class ReadyDataState:
    """Aligned genotype, expression, covariates, and offsets ready for mapping."""

    genotype: GenotypeData
    expression: ExpressionData
    covar: Array
    offset: Array

    @property
    def num_genes(self) -> int:
        """Number of phenotypes available after alignment."""
        return self.expression.pheno_meta.height

    def iter_cis(self, window: int) -> Iterator[CisData]:
        """Iterate over genes and yield per-gene cis windows with matched genotype."""
        for data in self.expression:
            y, gene_name, chrom, gene_start, gene_end = data
            start = max(0, gene_start - window)
            end = gene_end + window

            # query cis-variant info
            # note: if no variants taken, then G has shape (n,0), cis_var_info has shape (0, 7); both 2-dim
            G, cis_var_info = self.genotype.query_cis(chrom, start, end)

            yield CisData(
                self.covar, G, y, self.offset, gene_name, str(chrom), gene_start, gene_end, cis_var_info, start, end
            )

        return

    def iter_geno(self, chunk_size: int) -> Iterator[tuple[Array, pl.DataFrame]]:
        """Yield genotype blocks and metadata in chunks."""
        yield from self.genotype.iter_geno(chunk_size)

    @classmethod
    def from_data(
        cls,
        genotype: GenotypeData,
        expression: ExpressionData,
        covar: pl.DataFrame,
        offset: pl.DataFrame | None = None,
    ) -> "ReadyDataState":
        """Align genotype, expression, covariates, and optional offset on IID and return a ReadyDataState."""
        dfs = [genotype.sample_info, expression.pheno, covar]
        if offset is not None:
            dfs.append(offset)

        aligned_dfs = align_on_iid(dfs, iid_col="iid")
        if offset is not None:
            geno_samples, expression_samples, covar, offset = aligned_dfs
        else:
            geno_samples, expression_samples, covar = aligned_dfs

        # create new object with the subsetted individuals
        # we need this method bc we dont know what kind of geno data we're looking at here (PLINK, VCF, etc)
        genotype = genotype.replace_individuals(geno_samples)

        # at this point we have only 1 kind of expression object so just make a new one
        expression = ExpressionData(expression_samples, expression.pheno_meta, expression.libsize)

        # convert covariates to jax.numpy at this point
        covar = covar.select(pl.all().exclude("iid")).to_jax()

        # offset should only have two columns by construction at this point
        if offset is not None:
            assert offset.width == 2, "Offset dataframe should only have two columns at this point."
            offset = offset[:, 1].to_jax()
        else:
            # otherwise just zero it out
            offset = jnp.array(0.0)

        return ReadyDataState(
            genotype=genotype,
            expression=expression,
            covar=covar,
            offset=offset,
        )


def align_on_iid(
    dfs: list[pl.DataFrame],
    iid_col: str = "iid",
) -> list[pl.DataFrame]:
    """Align multiple DataFrames on a shared IID column, preserving the order of the first frame."""
    # first df determines final ordering (minus dropped iids)
    base_iids = dfs[0].get_column(iid_col).to_list()

    # compute iid intersection across all dfs
    common_iids = set(base_iids)
    for df in dfs[1:]:
        other = set(df.get_column(iid_col).to_list())
        common_iids &= other

    # filter base_iids to keep order but drop missing ones
    ordered_common_iids = [iid for iid in base_iids if iid in common_iids]

    # construct canonical iid frame in *base order*
    iid_df = pl.DataFrame({iid_col: ordered_common_iids})

    # align all dfs using left join on the canonical ordering
    aligned = []
    for df in dfs:
        aligned.append(iid_df.join(df, on=iid_col, how="left"))

    return aligned
