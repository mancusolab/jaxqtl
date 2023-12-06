from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import qtl.io
import qtl.norm

from sklearn.decomposition import PCA

import jax.numpy as jnp

from jax import Array

from jaxqtl.io.expr import ExpressionData, GeneMetaData
from jaxqtl.log import get_log


@dataclass
class ReadyDataState:
    geno: Array  # sample x genes
    bim: pd.DataFrame
    pheno: ExpressionData
    pheno_meta: GeneMetaData
    covar: Array  # sample x covariate

    def filter_geno(self, maf_threshold: float = 0.0, *chrom):
        if len(chrom) > 0:
            # filter bim by chr
            self.bim = self.bim.loc[self.bim.chrom.isin(chrom)]

        assert 0 <= maf_threshold <= 1, "maf threshold must be in range [0, 1]"
        if maf_threshold > 0.0:
            af = np.array(self.geno.mean(axis=0) / 2)
            maf = np.where(af > 0.5, 1 - af, af)  # convert to maf
            self.bim = self.bim.loc[maf > maf_threshold]

        # pull filtered geno by bim file
        self.geno = jnp.take(self.geno, jnp.array(self.bim.i), axis=1)
        assert self.geno.shape[1] == len(self.bim), "genotype and bim file do not have same shape"

        # reset "i" for pulling genotype by position
        self.bim.i = np.arange(0, self.geno.shape[1])

    def transform_y(self, mode: str = "log1p"):
        """
        tmm: normalize between individuals to make them comparable (differential library size)
        """
        if mode == "log1p":
            self.pheno.count = np.log1p(self.pheno.count)  # prevent log(0)
        elif mode == "tmm":
            # use edger TMM method to calculate size factor and convert to counts per million
            tmm_counts_df = qtl.norm.edger_cpm(self.pheno.count.iloc[:, 4:], normalized_lib_sizes=True)
            # # mask is filter by gene
            # inverse normal transformation on each gene (row)
            norm_df = qtl.norm.inverse_normal_transform(tmm_counts_df)
            self.pheno.count.iloc[:, 4:] = norm_df
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def add_covar_pheno_PC(self, k: int):
        count_std = self.pheno.count.copy(deep=True)
        count_std = (count_std - count_std.mean()) / count_std.std()  # standardize genes

        pca_pheno = PCA(n_components=k)
        pca_pheno.fit(count_std)
        PCs = pca_pheno.fit_transform(count_std)  # nxk
        self.covar = jnp.hstack((self.covar, PCs))  # append k expression PCs in pheno

    def filter_gene(self, geneexpr_percent_cutoff: float = 0.0, gene_list: Optional[List] = None):
        """Filter genes to be mapped"""
        if gene_list is not None:
            gene_list_insample = list(set(self.pheno_meta.gene_map.phenotype_id).intersection(set(gene_list)))
            # filter pheno
            self.pheno_meta.gene_map = self.pheno_meta.gene_map.loc[
                self.pheno_meta.gene_map.phenotype_id.isin(gene_list_insample)
            ]
            # subset by column name
            self.pheno.count = self.pheno.count[gene_list_insample]

            assert set(self.pheno_meta.gene_map.phenotype_id) == set(
                self.pheno.count.columns
            ), "gene map does not agree with pheno count matrix after gene list selection"

        # filter genes not expressed across samples
        total_n = len(self.pheno.count.index.unique())  # number of individuals
        geneexpr_percent = (self.pheno.count > 0).sum(axis=0) / total_n
        self.pheno.count = self.pheno.count.loc[:, geneexpr_percent > geneexpr_percent_cutoff]
        self.pheno_meta.gene_map = self.pheno_meta.gene_map.loc[
            self.pheno_meta.gene_map.phenotype_id.isin(self.pheno.count.columns)
        ]
        assert set(self.pheno_meta.gene_map.phenotype_id) == set(
            self.pheno.count.columns
        ), "gene map does not agree with pheno count matrix after gene expression percent filtering"


def create_readydata(
    geno: pd.DataFrame,
    bim: pd.DataFrame,
    pheno: pd.DataFrame,
    covar: pd.DataFrame,
    log=None,
    autosomal_only: bool = True,
    ind_list: Optional[List] = None,
) -> ReadyDataState:
    """Read genotype, phenotype and covariates, including interaction terms
    Genotype data: plink triplet, vcf
    pheno_path: bed file
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0

    All these files must contain the same set of individuals, otherwise only complete data is retained.
    Internally we check ordering and guarantee they are in the same order as phenotype data
    """
    if log is None:
        log = get_log()

    # keep genes in autosomals
    if autosomal_only:
        pheno = pheno.loc[pheno.chr.isin([str(i) for i in range(1, 23)])]

    # put gene name (index) back to columns
    pos_df = pheno[["chr", "start", "end"]].reset_index()
    pheno.drop(["chr", "start", "end"], axis=1, inplace=True)

    # transpose to sample x genes
    pheno = pheno.T
    pheno.columns.name = None  # remove column name due to tranpose
    if ind_list is not None:
        sample_id_subset = ind_list
    else:
        sample_id_subset = pheno.index.to_list()

    # find complete data of individuals
    sample_id = set.intersection(
        set(pheno.index.to_list()),
        set(geno.index.to_list()),
        set(covar.index.to_list()),
        set(sample_id_subset),
    )
    sample_id = list(sample_id)

    # subset and order genotype, covariates and pheno
    pheno = pheno.loc[pheno.index.isin(sample_id)].sort_index()
    geno = geno.loc[geno.index.isin(sample_id)].sort_index()
    covar = covar.loc[covar.index.isin(sample_id)].sort_index()

    # ensure sample order in genotype and covar are same as count
    assert (geno.index == pheno.index).all(), "samples are not sorted in genotype and count matrix"

    assert (covar.index == pheno.index).all(), "samples are not sorted in covariate and count matrix"

    log.info("Finished loading raw data.")

    return ReadyDataState(
        geno=jnp.float64(geno),
        bim=bim,
        pheno=ExpressionData(pheno),
        pheno_meta=GeneMetaData(pos_df),
        covar=jnp.float64(covar),
    )
