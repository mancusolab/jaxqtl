from typing import List

import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import Array


class ExpressionData:
    data: pd.DataFrame

    def __init__(self, dat: pd.DataFrame):
        self.count = dat
        self.gene_list = dat.columns.to_list()

    def __iter__(self):
        pass

    def __getitem__(self, gene_name: str) -> Array:
        """Get pseudo bulk data for one cell type
        Remove genes with all zeros count across samples
        """
        nobs = self.count.shape[0]
        onegene = self.count[gene_name]
        return jnp.float64(onegene).reshape((nobs, 1))


class GeneMetaData:
    """Store gene meta data
    Gene name, chrom, start, rend
    """

    data: pd.DataFrame

    def __init__(
        self,
        gene_list: List,
        gene_path: str = "../example/data/ensembl_allgenes.chr22.txt",
    ):
        gene_map = pd.read_csv(gene_path, delimiter="\t")
        gene_map.columns = [
            "chr",
            "gene_start",
            "gene_end",
            "symbol",
            "tss_start_min",
            "strand",
            "gene_type",
            "ensemble_id",
            "refseq_id",
        ]
        gene_map["tss_start_max"] = gene_map["tss_start_min"]

        gene_map = gene_map.groupby(
            ["chr", "ensemble_id", "strand"], as_index=False
        ).agg({"tss_start_min": "min", "tss_start_max": "max"})
        # Merge genes from dat with this gene map
        gene_map["found"] = np.where(gene_map.ensemble_id.isin(gene_list), 1, 0)
        # eQTL scan only for genes with known tss
        gene_map = gene_map.loc[gene_map.found == 1]

        self.gene_map = gene_map
        self.gene_notfound = set(gene_list) - set(gene_map.ensemble_id)

    def __iter__(self):
        for _, gene in self.gene_map.iterrows():
            gene_name = gene.ensemble_id
            chrom = gene.chr
            start_min = gene.tss_start_min
            end_max = gene.tss_start_max
            yield gene_name, chrom, start_min, end_max
