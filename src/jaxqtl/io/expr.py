import pandas as pd

import jax.numpy as jnp
from jax import Array


class ExpressionData:
    data: pd.DataFrame

    def __init__(self, dat: pd.DataFrame):
        self.count = dat

    def __iter__(self):
        pass

    def __getitem__(self, gene_name: str) -> Array:
        """Get count data for one gene"""
        nobs = self.count.shape[0]
        onegene = self.count[gene_name]
        return jnp.float64(onegene).reshape((nobs, 1))


class GeneMetaData:
    """Store gene meta data
    Gene name, chrom, start, rend
    bed file is zero-based, start = end-1
    """

    data: pd.DataFrame

    def __init__(
        self,
        pos_df: pd.DataFrame,
    ):
        """position df for genes
        chr   start   end  phenotype_id
        Note: start = end (tss position)
        """

        self.gene_map = pos_df

    def __iter__(self):
        for _, gene in self.gene_map.iterrows():
            gene_name = gene.phenotype_id  # ensemble_id
            chrom = gene.chr
            start_min = gene.start
            end_max = gene.end
            yield gene_name, chrom, start_min, end_max
