import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import decoupler as dc
import equinox as eqx
import pandas as pd
import scanpy as sc
from anndata import AnnData


@dataclass
class SingleCellFilter:
    """Filtering metric for single cell data"""

    min_cells: int = 3
    min_genes: int = 200
    n_genes: int = 2500
    percent_mt: int = 5  # 5 means 5%
    norm_target_sum: float = 1e6
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
    def process(self, dat: Any, filter_opt=SingleCellFilter) -> pd.DataFrame:
        pass


class H5AD(PhenoIO):
    def __call__(self, pheno_path: str):
        """Read raw count file in H5AD format"""
        return sc.read_h5ad(pheno_path)

    def process(self, dat: AnnData, filter_opt=SingleCellFilter) -> pd.DataFrame:
        """Filter single cell data and create pseudo-bulk
        dat.X: n_obs (cell) x n_vars (genes)
        dat.var_name = 'ensembl_id'
        ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

        Returns:
            pseudo bulk RNA seq data for all cell types
            index by ['donor_id', 'cell_type']
        """
        # filter cells by min number of genes expressed (in place)
        sc.pp.filter_cells(dat, min_genes=SingleCellFilter.min_genes)
        # filter genes by min number of cells expressed (in place)
        sc.pp.filter_genes(dat, min_cells=SingleCellFilter.min_cells)

        #  filter cells with too many genes expressed
        dat = dat[dat.obs["n_genes"] < SingleCellFilter.n_genes, :]

        #  filter cells that have >5% mitochondrial counts
        dat = dat[dat.obs["percent.mt"] < SingleCellFilter.percent_mt, :]

        # normalize by total UMI count: scale factor (in-place change), CPM if target=1e6
        # every cell has the same total count
        sc.pp.normalize_total(dat, target_sum=SingleCellFilter.norm_target_sum)

        # mean count for given cell type within individual and create a view
        dat.bulk = dc.get_pseudobulk(
            dat,
            sample_col="donor_id",
            groups_col="cell_type",
            mode=SingleCellFilter.bulk_method,  # take mean across cells for each individual
            min_prop=SingleCellFilter.bulk_min_prop,  # filter gene
            min_smpls=SingleCellFilter.bulk_min_smpls,  # filter gene
            min_cells=SingleCellFilter.bulk_min_cells,  # filter sample
            min_counts=SingleCellFilter.bulk_min_count,  # filter sample
        )

        # create pd.Dataframe
        count = pd.DataFrame(dat.bulk.X)  # sample_cell x gene
        count = count.set_index([dat.bulk.obs["donor_id"], dat.bulk.obs["cell_type"]])
        count.columns = dat.bulk.var.index
        return count

    @staticmethod
    def write_bed(
        pheno: pd.DataFrame,
        gtf_bed_path: str = "./example/data/Homo_sapiens.GRCh37.87.bed.gz",
        out_dir: str = "./example/data",
        cell_type: str = "CD14-positive monocyte",
    ):
        """
        After creating pseudo-bulk using process(), create bed file for each cell type
        """
        pheno_onetype = pheno[pheno.index.get_level_values("cell_type") == cell_type]

        # drop genes with all zero expressions
        pheno_onetype = pheno_onetype.loc[:, (pheno_onetype != 0).any(axis=0)]

        # remove cell type index
        pheno_onetype = pheno_onetype.reset_index(level="cell_type", drop=True)

        # transpose s.t samples on columns, put ensembl_id back to column
        bed = pheno_onetype.T
        bed = bed.reset_index()

        # load gtf file for locating tss
        gene_map = load_gene_gft_bed(gtf_bed_path)

        out = pd.merge(gene_map, bed, left_on="ensemble_id", right_on="ensembl_id")
        out = out.drop("ensemble_id", axis=1)
        out = out.rename(columns={"ensembl_id": "phenotype_id"})

        cell_type_outname = re.sub("[^0-9a-zA-Z]+", "_", cell_type)
        out.to_csv(out_dir + "/" + cell_type_outname + ".bed.gz", index=False, sep="\t")


class PheBedReader(PhenoIO):
    """Read phenotype from bed format
    must in this following format (same as tensorqtl / fastqtl):
    chr, start, end, phenotype_id, sample_id_1, sample_id_2, ...

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
        self,
        phenotype_df: pd.DataFrame,
        filter_opt=SingleCellFilter,
    ) -> pd.DataFrame:
        phenotype_df["start"] += 1  # change to 1-based

        return phenotype_df


def load_gene_gft_bed(gtf_bed_path: str) -> pd.DataFrame:
    gene_map = pd.read_csv(gtf_bed_path, delimiter="\t")
    gene_map.columns = [
        "chr",
        "start",
        "end",
        "ensemble_id",
    ]

    return gene_map
