from abc import ABC, abstractmethod
from typing import NamedTuple

import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata._core.anndata import AnnData
from cyvcf2 import VCF
from pandas_plink import read_plink

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node, register_pytree_node_class

pd.set_option("display.max_rows", 100000)


class PlinkState(NamedTuple):
    genotype: Array
    bim: pd.DataFrame
    fam: pd.DataFrame


class CleanDataState(NamedTuple):
    """
    count: filtered cells and genes for given cell type, contains sample features
    """

    genotype: Array  # nxp, index by sample iid, column names are variant names chr:pos:ref:alt
    bim: pd.DataFrame  # variant on rows
    count: AnnData  # nxG for one cell type, count.var has gene names
    covar: pd.DataFrame  # nxcovar, covariates for the same individuals


@register_pytree_node_class
class IO(ABC):
    """Read genotype or count data from different file format"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @abstractmethod
    def __call__(self, prefix: str) -> PlinkState:
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


class PlinkReader(IO):
    """Read genotype data from plink triplets
    prefix: chr22.bed, also accept chr*.bed (read everything)

    Note: read bed file is much faster than VCF file (parser)
    """

    def __call__(self, prefix: str) -> PlinkState:
        # Append prefix with suffix
        self.bed_path = prefix + ".bed"
        self.bim_path = prefix + ".bim"
        self.fam_path = prefix + ".fam"

        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        bim, fam, bed = read_plink(prefix, verbose=False)
        G = jnp.asarray(bed.compute())

        # TODO: add imputation etc...

        return PlinkState(G, bim, fam)


class VCFReader(IO):
    def __call__(self, vcf_path: str) -> PlinkState:
        """
        need genotype: sample ID, genotype in dose values
        need number of variants
        """

        # read VCF files
        vcf = VCF(vcf_path, gts012=True)  # can add samples=[]
        fam = pd.DataFrame(vcf.samples).rename(columns={0: "iid"})  # individuals

        genotype = []
        bim_list = []

        for idx, var in enumerate(vcf):
            genotype.append(var.gt_types)
            # var.ALT is a list of alternative allele
            bim_list.append([var.CHROM, var.ID, 0.0, var.POS, var.ALT[0], var.REF, idx])

        vcf.close()

        # convert to jax array
        genotype = jnp.asarray(genotype)

        #  chrom        snp       cm     pos a0 a1  i
        bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "pos", "alt", "ref"])

        # check order
        # allsorted = np.sum(genotype.columns == var_info_nodup.ID) == genotype.shape[1]
        # convert to REF dose
        # genotype = 2 - genotype

        return PlinkState(genotype, bim, fam)


def process_count(dat: AnnData, cell_type: str = "CD14-positive monocyte") -> AnnData:
    """
    dat: n_obs (cell) x n_vars (genes)
    dat.var_name = 'ensembl_id'

    No log transformation on the count data!
    should we include preprocessing here or expect user to provide clean data
    ready to run GLM

    ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
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

    # subset to one cell type
    dat_onetype = dat.bulk[dat.bulk.obs["cell_type"] == cell_type]
    # filter out genes with zeros reads
    dat_onetype = dat_onetype[:, np.sum(dat_onetype.X, axis=0) > 0]

    return dat_onetype


def read_data(
    file_type: IO,
    geno_path: str,
    pheno_path: str,
    covar_path: str,
    cell_type: str = "CD14-positive monocyte",
) -> CleanDataState:
    """Read genotype, phenotype and covariates, including interaction terms
    Genotype data: plink triplet, vcf
    pheno_path: h5ad file path, including covariates
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0
    """
    genotype, var_info, sample_info = file_type(geno_path)
    covar = pd.read_csv(covar_path, delimiter="\t")
    covar = covar.set_index("donor_id")

    dat = sc.read_h5ad(pheno_path)
    count = process_count(dat, cell_type=cell_type)
    donor_id = count.obs.donor_id.values

    # filter genotype and covariates, ordered?
    genotype = genotype.filter(items=donor_id, axis=0)
    covar = covar.filter(items=donor_id, axis=0)

    # check_genotype_order = np.sum(genotype.index == donor_id) == len(donor_id)
    # check_covar_order = np.sum(covar.index == donor_id) == len(donor_id)

    # if check_genotype_order and check_covar_order:
    return CleanDataState(genotype, var_info, count, covar)
