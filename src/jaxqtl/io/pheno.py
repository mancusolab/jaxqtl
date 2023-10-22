import os
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import decoupler as dc
import equinox as eqx
import numpy as np
import pandas as pd
import qtl.io
import qtl.norm
import scanpy as sc
from anndata import AnnData
from scipy.sparse import diags


# TODO: need find out commonly used parameters
@dataclass
class SingleCellFilter:
    """Filtering metric for single cell data"""

    id_col: str = "donor_id"
    celltype_col: str = "cell_type"
    mt_col: str = "percent.mt"
    geneid_col: str = "ensemble_id"
    min_cells: int = 3
    min_genes: int = 200
    max_genes: int = 2500  # can decide this based on plotting
    percent_mt: int = 5  # 5 means 5%
    norm_target_sum: float = 1e5  # not recommended
    bulk_method: str = "mean"
    bulk_min_prop: float = (
        0.0  # Minimum proportion of cells that express a gene in a sample.
    )
    bulk_min_smpls: int = 0  # Minimum number of samples with >= proportion of cells with expression than min_prop
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

    def process(
        self,
        dat: AnnData,
        filter_opt=SingleCellFilter,
        divide_size_factor: bool = True,
        norm_fix_L: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter single cell data and create pseudo-bulk
        dat.X: n_obs (cell) x n_vars (genes)
        dat.var_name = 'ensembl_id'
        ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

        Returns:
            pseudo bulk RNA seq data for all cell types
            index by ['donor_id', 'cell_type']
        """
        # TODO: check these result, make col names consistent
        # filter cells by min number of genes expressed (in place)
        sc.pp.filter_cells(dat, min_genes=filter_opt.min_genes)

        #  filter cells with too many genes expressed (in place)
        sc.pp.filter_cells(dat, max_genes=filter_opt.max_genes)

        # filter genes by min number of cells expressed (in place)
        sc.pp.filter_genes(dat, min_cells=filter_opt.min_cells)

        # filter cells that have >5% mitochondrial counts
        # here return the actual sparse matrix instead of View for shifted_transformation_nolog()
        # dat = dat[dat.obs[filter_opt.mt_col] < filter_opt.percent_mt, :].copy()
        dat = dat[dat.obs[filter_opt.mt_col] < filter_opt.percent_mt, :]

        # normalize total
        if norm_fix_L is not None:
            sc.pp.normalize_total(dat, target_sum=norm_fix_L)  # fixed L
        if divide_size_factor:
            dat = adjust_size_factor(dat)

        # mean count for given cell type within individual and create a view
        # first compute and then filter
        dat.bulk = dc.get_pseudobulk(
            dat,
            sample_col=filter_opt.id_col,
            groups_col=filter_opt.celltype_col,
            mode=filter_opt.bulk_method,  # take mean across cells for each individual
            min_cells=filter_opt.bulk_min_cells,  # exclude sample with < min cells from calc
            min_counts=filter_opt.bulk_min_count,  # exclude sample < min # summed count from calc
            min_prop=filter_opt.bulk_min_prop,  # selects genes that expressed across > % cells in each sample
            min_smpls=filter_opt.bulk_min_smpls,  # this condition is met across a minimum number of samples
        )

        # create pd.Dataframe
        count = pd.DataFrame(dat.bulk.X)  # sample_cell x gene
        count = count.set_index(
            [dat.bulk.obs[filter_opt.id_col], dat.bulk.obs[filter_opt.celltype_col]]
        )
        count.columns = dat.bulk.var.index  # use var.index as gene names

        return count

    @staticmethod
    def write_bed(
        pheno: pd.DataFrame,
        filter_opt=SingleCellFilter,
        gtf_bed_path: str = "../example/data/Homo_sapiens.GRCh37.87.bed.gz",
        out_dir: str = "../example/local/phe_bed",
        celltype_path: str = "../example/data/celltype.tsv",
        autosomal_only: bool = True,
    ):
        """After creating pseudo-bulk using process(), create bed file for each cell type"""

        cell_type_list = (
            pd.read_csv(celltype_path, sep="\t", header=None).iloc[:, 0].to_list()
        )

        for cell_type in cell_type_list:
            pheno_onetype = pheno[
                pheno.index.get_level_values(filter_opt.celltype_col) == cell_type
            ]

            # remove cell type index
            pheno_onetype = pheno_onetype.reset_index(
                level=filter_opt.celltype_col, drop=True
            )

            # transpose s.t samples on columns, put ensembl_id back to column
            bed = pheno_onetype.T
            bed = bed.reset_index()

            # load gtf file for locating tss
            gene_map = load_gene_gft_bed(gtf_bed_path)

            if autosomal_only:
                gene_map = gene_map.loc[
                    gene_map.chr.isin([str(i) for i in range(1, 23)])
                ]

            # inner join
            out = pd.merge(
                gene_map, bed, left_on="ensemble_id", right_on=filter_opt.geneid_col
            )
            out = out.drop("ensemble_id", axis=1)
            out = out.rename(columns={"ensembl_id": "phenotype_id", "chr": "#Chr"})

            cell_type_outname = re.sub("[^0-9a-zA-Z]+", "_", cell_type)
            out.to_csv(
                os.path.join(out_dir, f"{cell_type_outname}.bed.gz"),
                index=False,
                sep="\t",
            )


class PheBedReader(PhenoIO):
    """Read phenotype from bed format
    filename: *.bed, *.bed.gz

    must have following headers (same as tensorqtl / fastqtl):
    #Chr, start, end, phenotype_id, sample_id_1, sample_id_2, ...

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

        # convert all columns to lower case and rename #Chr to chr
        phenotype_df.rename(
            columns={
                i: i.lower().replace("#chr", "chr") for i in phenotype_df.columns[:3]
            },
            inplace=True,
        )

        # ensure chr values are str
        phenotype_df["chr"] = phenotype_df["chr"].astype(str)

        # set index name
        phenotype_df.index.name = "phenotype_id"

        phenotype_df["start"] += 1  # change to 1-based

        return phenotype_df

    def process(self, dat: Any, filter_opt=SingleCellFilter) -> pd.DataFrame:
        pass


def load_gene_gft_bed(gtf_bed_path: str) -> pd.DataFrame:
    """Read gft bed file"""
    gene_map = pd.read_csv(gtf_bed_path, delimiter="\t")
    gene_map.columns = [
        "chr",
        "start",
        "end",
        "ensemble_id",
    ]

    return gene_map


def adjust_size_factor(adata: AnnData):
    """Suggested by AE & Huber 2023 paper
    size factor = (sum_g Y_gc) / L
    where L = (sum_gc Y_gc) / (number of cells)

    adapt code from: https://github.com/mousepixels/sanbomics_scripts/blob/main/shifted_transformation.ipynb
    """
    # TODO: need do this by cell type? right now this divide by average across all cells all cell type
    # X: cell x gene
    size_factors = adata.X.sum(axis=1) / np.mean(adata.X.sum(axis=1))  # (num cell x 1)

    # array.A1 returns self as a flattened array, same as array.ravel()
    adata.X = diags(1.0 / size_factors.A1).dot(adata.X)
    # adata.X = adata.X.toarray()  # convert to dense array
    # adata.X = adata.X + y0
    # adata.X.data = (
    #     adata.X.data + y0
    # )  # !!! add y0 to non-sparse values, not sure if need add y0 to zeros raw count

    return adata


def bed_transform_y(pheno_path: str, method: str = "log1p"):
    """
    count_df: rows are genes, columns are individual ID
    """
    count_df = pd.read_csv(pheno_path, sep="\t", dtype={"#chr": str, "#Chr": str})
    # filter genes with zero expression
    count_df = count_df[count_df.iloc[:, 4:].sum(axis=1) > 0]

    if method == "log1p":
        count_df.iloc[:, 4:] = np.log1p(count_df.iloc[:, 4:])  # prevent log(0)
    elif method == "tmm":
        # Note: don't filter before TMM
        # use edger TMM method to calculate size factor and convert to counts per million
        tmm_counts_df = qtl.norm.edger_cpm(
            count_df.iloc[:, 4:], normalized_lib_sizes=True
        )
        # # mask is filter by gene
        # inverse normal transformation on each gene (row)
        norm_df = qtl.norm.inverse_normal_transform(tmm_counts_df)
        count_df.iloc[:, 4:] = norm_df
    elif method == "qn":
        pass
        # qn_df = qtl.norm.normalize_quantiles(tpm_df.loc[mask])
        # norm_df = qtl.norm.inverse_normal_transform(qn_df)
    else:
        raise ValueError(f"Unsupported mode {method}")

    return count_df
