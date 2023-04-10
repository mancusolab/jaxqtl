import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import decoupler as dc
import equinox as eqx
import pandas as pd
import scanpy as sc
from anndata import AnnData

from jaxqtl.io.expr import GeneMetaData


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
    def process(self, dat: Any, filter_opt: Optional[SingleCellFilter]) -> Any:
        pass


class H5AD(PhenoIO):
    def __call__(self, pheno_path: str):
        return sc.read_h5ad(pheno_path)

    def process(
        self, dat: AnnData, filter_opt: Optional[SingleCellFilter]
    ) -> pd.DataFrame:
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

    @staticmethod
    def write_bed(
        pheno: pd.DataFrame,
        out_dir: str = "./example/data",
        cell_type: str = "CD14-positive monocyte",
    ):
        """
        After creating pseudo-bulk using process(), create bed file for each cell type (group)
        """
        pheno_onetype = pheno[pheno.index.get_level_values("cell_type") == cell_type]
        # drop genes with all zero expressions
        pheno_onetype = pheno_onetype.loc[:, (pheno_onetype != 0).any(axis=0)]
        gene_list = pheno_onetype.columns.values

        # remove cell type index
        pheno_onetype = pheno_onetype.reset_index(level="cell_type", drop=True)
        # transpose s.t samples on columns
        bed = pheno_onetype.T
        bed = bed.reset_index()

        gene_map = GeneMetaData(gene_list).gene_map
        gene_map["end"] = (
            (gene_map.tss_start_min + gene_map.tss_start_max) / 2
        ).round()
        gene_map["start"] = gene_map["end"] - 1
        gene_map_to_merge = gene_map[["chr", "ensemble_id", "start", "end"]]

        out = pd.merge(
            gene_map_to_merge, bed, left_on="ensemble_id", right_on="ensembl_id"
        )
        out = out.drop("ensemble_id", axis=1)
        out = out.rename(columns={"ensembl_id": "phenotype_id"})

        cell_type_outname = re.sub("[^0-9a-zA-Z]+", "_", cell_type)
        out.to_csv(out_dir + "/" + cell_type_outname + ".bed.gz", index=False, sep="\t")


class PheBedReader(PhenoIO):
    """Read phenotype from bed format
    must in this following format (same as tensorqtl / fastqtl):
    chr, tss_start, tss_end, phenotype_id, sample_id_1, sample_id_2, ...

    tss_start = tss - 1 (0-base format)
    tss_end = tss (1-base)
    """

    def __call__(self, pheno_path):
        if pheno_path.endswith((".bed.gz", ".bed")):
            phenotype_df = pd.read_csv(
                pheno_path, sep="\t", index_col=3, dtype={"#chr": str, "#Chr": str}
            )
        elif pheno_path.endswith(".parquet"):
            phenotype_df = pd.read_parquet(pheno_path)
            phenotype_df.set_index(phenotype_df.columns[3], inplace=True)
        else:
            raise ValueError("Unsupported file type.")
        phenotype_df.rename(
            columns={
                i: i.lower().replace("#chr", "chr") for i in phenotype_df.columns[:3]
            },
            inplace=True,
        )

        return phenotype_df

    def process(
        self, phenotype_df: pd.DataFrame, filter_opt: Optional[SingleCellFilter]
    ) -> pd.DataFrame:
        phenotype_df["start"] += 1  # change to 1-based

        return phenotype_df
