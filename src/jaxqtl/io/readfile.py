from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional

import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata._core.anndata import AnnData
from cyvcf2 import VCF
from pandas_plink import read_plink1_bin

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node, register_pytree_node_class

pd.set_option("display.max_rows", 100000)


class PlinkState(NamedTuple):
    bed: np.ndarray
    bim: pd.DataFrame
    fam: pd.DataFrame


class CleanDataState(NamedTuple):
    """
    count: filtered cells and genes for given cell type, contains sample features
    """

    genotype: pd.DataFrame  # nxp, index by sample iid, column names are variant names chr:pos:ref:alt
    var_info: pd.DataFrame
    count: AnnData  # nxG for one cell type, count.var has gene names
    covar: jnp.ndarray  # nxcovar, covariates for the same individuals


@register_pytree_node_class
class IO(ABC):
    """
    Read genotype or count data from different file format
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def __call__(self, prefix: str) -> Array:
        """
        Read files
        """
        pass

    def tree_flatten(self):
        children = ()
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class Plink(IO):
    """
    bim:
    """

    def __call__(self, prefix: str) -> PlinkState:
        # Append prefix with suffix
        self.bed_path = prefix + ".bed"
        self.bim_path = prefix + ".bim"
        self.fam_path = prefix + ".fam"

        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        # print(G.a0.sel(variant="variant0").values)
        # print(G.sel(sample="1", variant="variant0").values)
        G = read_plink1_bin(self.bed_path, self.bim_path, self.fam_path, verbose=False)
        genotype = np.array(G.values)  # sample x variants

        var_info = pd.DataFrame()
        sample_info = pd.DataFrame()

        return PlinkState(genotype, var_info, sample_info)


class CYVCF2(IO):
    def __call__(self, prefix: str) -> PlinkState:
        """
        need genotype: sample ID, genotype in dose values
        need number of variants
        """
        vcf_path = prefix + ".vcf.gz"
        varnum_path = prefix + ".numvar"
        # read VCF files
        vcf = VCF(vcf_path, gts012=True)  # can add samples=[]
        sample_info = pd.DataFrame(vcf.samples).rename(
            columns={0: "iid"}
        )  # individuals
        nobs = len(vcf.samples)

        # use "bcftools stats file.vcf > file.stats" to count this
        # need a file to provide number of variants
        num_var = pd.read_csv(varnum_path, header=None)
        num_var = int(num_var[0].values)  # convert to int

        genotype = np.ones((num_var, nobs)) * -9
        var_list = [[]] * num_var  # type: List[List]

        idx = 0
        for var in vcf:
            genotype[idx] = var.gt_types
            # var.ALT is a list of alternative allele
            var_list[idx] = [var.CHROM, var.ID, var.POS, var.ALT[0], var.REF]
            idx += 1

        vcf.close()

        var_info = pd.DataFrame(
            var_list, columns=["chrom", "chr_pos", "pos", "alt", "ref"]
        )
        var_info["ID"] = var_info.chr_pos + ":" + var_info.ref + ":" + var_info.alt

        genotype = pd.DataFrame(genotype.T).set_index([vcf.samples])
        genotype.columns = var_info.ID.values

        # drop multi-allelic SNPs: not sure if need do this
        var_info_nodup = var_info.drop_duplicates(["chr_pos", "ref"], keep=False)

        # drop SNPs in genotype but not in var_info (only bi-allelic)
        genotype = genotype.drop(
            columns=list(set(genotype.columns) - set(var_info_nodup.ID.values))
        )

        # check order
        # allsorted = np.sum(genotype.columns == var_info_nodup.ID) == genotype.shape[1]
        # convert to REF dose
        # genotype = 2 - genotype

        return PlinkState(genotype, var_info, sample_info)


def process_count(dat: AnnData, cell_type: str = "CD14-positive monocyte") -> AnnData:
    """
    dat: n_obs (cell) x n_vars (genes)
    dat.var_name = 'ensembl_id'

    No log transformation on the count data!
    should we include preprocessing here or expect user to provide clean data
    ready to run GLM

    ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    """
    # start with: 11043 x 36571
    # filter cells by min number of genes expressed (in place)
    sc.pp.filter_cells(dat, min_genes=200)  # 11043 x 36571
    # filter genes by min number of cells expressed (in place)
    sc.pp.filter_genes(dat, min_cells=3)  # 11043 × 17841

    #  filter cells with too many genes expressed
    dat = dat[dat.obs["n_genes"] < 2500, :]  # 11026 × 17841

    #  filter cells that have >5% mitochondrial counts
    dat = dat[dat.obs["percent.mt"] < 5, :]  # 10545 × 17841

    # normalize by total UMI count: scale factor (in-place change), CPM if target=1e6
    # every cell has the same total count
    sc.pp.normalize_total(dat, target_sum=1e4)

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

    # subset to one cell type
    dat_onetype = dat.bulk[dat.bulk.obs["cell_type"] == cell_type]
    # filter out genes with zeros reads
    dat_onetype = dat_onetype[:, np.sum(dat_onetype.X, axis=0) > 0]

    return dat_onetype


def read_data(
    file_type: IO,
    geno_path: str,
    pheno_path: str,
    covar_path: Optional[str],
    cell_type: str = "CD14-positive monocyte",
):
    """
    Genotype data: plink file
    pheno_path: h5ad file path, including covariates
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0
    """
    genotype, var_info, sample_info = file_type(geno_path)
    covar = pd.read_csv(covar_path, delimiter="\t")  # use donor_id
    covar = covar.set_index("donor_id")

    # inner join and keep the order of left df
    # X = pd.merge(sample_info, covar, left_index=True, right_index=True)  # inner join

    dat = sc.read_h5ad(pheno_path)
    count = process_count(dat, cell_type=cell_type)
    donor_id = count.obs.donor_id.values

    # filter genotype and covariates, ordered?
    genotype = genotype.filter(items=donor_id, axis=0)
    covar = covar.filter(items=donor_id, axis=0)

    check_genotype_order = np.sum(genotype.index == donor_id) == len(donor_id)
    check_covar_order = np.sum(covar.index == donor_id) == len(donor_id)

    if check_genotype_order and check_covar_order:
        return CleanDataState(genotype, var_info, count, jnp.asarray(covar))
