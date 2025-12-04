from dataclasses import dataclass
from typing import Optional

import polars as pl

import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from .geno import GenotypeData
from .pheno import ExpressionData


class SNPInfo(eqx.Module):
    id: str
    pos: int
    a1: str
    a0: str
    tss_distance: int
    af: float
    ma_count: int


class CisData(eqx.Module):
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
        num_snp = 0 if self.G is None else self.G.shape[1]
        return num_snp

    def get_af_summary(self, idx: int) -> tuple[Array, Array]:
        g = self.G[:, idx]
        n = len(g)
        counts = jnp.sum(g, axis=0)  # count REF allele
        af = counts / (2.0 * n)
        flag = af <= 0.5
        ma_counts = jnp.where(flag, counts, 2 * n - counts)

        return af, ma_counts

    def get_snp_info(self, idx: int) -> SNPInfo:
        af, ma_count = self.get_af_summary(idx)
        # chrom, snp, cm, pos, a0, a1, index
        _, snp_id, _, snp_pos, a0, a1, _ = self.cis_info.row(idx)
        tss_distance = snp_pos - self.gene_start
        return SNPInfo(snp_id, snp_pos, a1, a0, tss_distance, float(af), int(ma_count))


@dataclass
class ReadyDataState:
    genotype: GenotypeData  # sample x genes
    expression: ExpressionData
    covar: Array  # sample x covariates
    offset: ArrayLike

    @property
    def num_genes(self) -> int:
        return self.expression.pheno_meta.height

    def iter_cis(self, window: int):
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


def align_pheno_covar(
    pheno: pl.LazyFrame,
    covar: pl.LazyFrame,
    offset: Optional[pl.LazyFrame] = None,
):
    # store this once to avoid typos etc
    IID = "iid"
    iid_col = pl.col(IID)

    # pull out common iids across pheno/covar
    common_iids = pheno.select(iid_col).join(covar.select(iid_col), on=IID, how="inner")

    # if offset is provided then repeat
    if offset is not None:
        common_iids = common_iids.join(offset.select(iid_col), on=IID, how="inner")

    pheno = pheno.join(common_iids, on=IID, how="semi")
    covar = covar.join(common_iids, on=IID, how="semi")

    if offset is not None:
        offset = offset.join(common_iids, on=IID, how="inner")

    return pheno, covar, offset


def align_on_iid(
    dfs: list[pl.DataFrame],
    iid_col: str = "iid",
) -> list[pl.DataFrame]:
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


def create_readydata(
    genotype: GenotypeData,
    expression: ExpressionData,
    covar: pl.DataFrame,
    offset: Optional[pl.DataFrame] = None,
) -> ReadyDataState:
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
    expression = ExpressionData(expression_samples, expression.pheno_meta)

    # convert covariates to jax.numpy at this point
    covar = jnp.asarray(covar.select(pl.all().exclude("iid")).to_numpy())

    # offset should only have two columns by construction at this point
    if offset is not None:
        assert offset.width == 2, "Offset dataframe should only have two columns at this point."
        offset = jnp.asarray(offset[:, 1].to_numpy())
    else:
        # otherwise just zero it out
        offset = jnp.array(0.0)

    return ReadyDataState(
        genotype=genotype,
        expression=expression,
        covar=covar,
        offset=offset,
    )
