from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import decoupler as dc
import equinox as eqx
import pandas as pd
import scanpy as sc

# from jaxqtl.io.expr import ExpressionData

# from anndata import AnnData


@dataclass
class SingleCellFilter:
    """Filtering metric for single cell data"""

    min_cells: int
    min_genes: int
    n_genes: int
    percent_mt: int = 5  # 5 means 5%
    norm_target_sum: float = 1000000.0
    cell_type: str = "CD14-positive monocyte"
    bulk_method: str = "mean"
    bulk_min_prop: float = 0.2
    bulk_min_smpls: int = 2
    bulk_min_cells: int = 0
    bulk_min_count: int = 0


class PhenoIO(eqx.Module, metaclass=ABCMeta):
    """Read genotype or count data from different file format"""

    @abstractmethod
    def __call__(self, pheno_path: str):
        pass

    @abstractmethod
    def process(self, dat: Any, filter_opt: SingleCellFilter) -> Any:
        pass


class H5AD(PhenoIO):
    def __call__(self, pheno_path: str):
        return sc.read_h5ad(pheno_path)

    def process(self, dat, filter_opt) -> pd.DataFrame:
        """
        dat.X: n_obs (cell) x n_vars (genes)
        dat.var_name = 'ensembl_id'
        ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

        Returns:
            pseudo bulk RNA seq data for all cell types
        """
        # filter cells by min number of genes expressed (in place)
        sc.pp.filter_cells(dat, min_genes=200)
        # filter genes by min number of cells expressed (in place)
        sc.pp.filter_genes(dat, min_cells=3)

        #  filter cells with too many genes expressed
        dat = dat[dat.obs["n_genes"] < 2500, :]

        #  filter cells that have >5% mitochondrial counts
        dat = dat[dat.obs["percent.mt"] < 5, :]

        # normalize by total UMI count: scale factor (in-place change), CPM if target=1e6
        # every cell has the same total count
        sc.pp.normalize_total(dat, target_sum=1e6)

        # mean count for given cell type within individual and create a view
        dat.bulk = dc.get_pseudobulk(
            dat,
            sample_col="donor_id",
            groups_col="cell_type",
            mode="mean",
            min_prop=0.2,  # filter gene
            min_smpls=2,  # filter gene
            min_cells=0,  # filter sample
            min_counts=0,  # filter sample
        )

        # create pd.Dataframe
        count = pd.DataFrame(dat.bulk.X)  # sample_cell x gene
        count = count.set_index([dat.bulk.obs["donor_id"], dat.bulk.obs["cell_type"]])
        count.columns = dat.bulk.var.index
        return count
