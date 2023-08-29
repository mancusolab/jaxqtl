# import numpy.testing as nptest

# import pandas as pd

# import jax.numpy as jnp
from jax.config import config

from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader  # , VCFReader
from jaxqtl.io.pheno import PheBedReader, bed_transform_y

# from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log

# from jaxqtl.io.readfile import read_data

config.update("jax_enable_x64", True)

geno_path = "../example/data/chr22.n94"
covar_path = "../example/data/donor_features.n94.tsv"
pheno_path = "../example/data/n94_CD14_positive_monocyte.bed.gz"
# genelist_path = "../example/data/genelist.tsv"
genelist_path = "../example/data/genelist_chr22.tsv"

reader = PlinkReader()
dat = reader(geno_path)

log = get_log()

# raw genotype data and impute for genotype data
log.info("Load genotype.")
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

log.info("Load covariates.")
covar = covar_reader(covar_path)

log.info("Load phenotype.")
pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

transform_method = "tmm"  # log1p, tmm
pheno_transform = bed_transform_y(pheno_path, mode=transform_method)
pheno_transform.to_csv(
    f"../example/data/n94_CD14_positive_monocyte.{transform_method}.bed.gz",
    index=False,
    sep="\t",
)

# # run Mapping #
# dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)
#
# maf_threshold = 0.0
# dat.filter_geno(maf_threshold, "22")
#
# # filter phenotype (5 genes)
# gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()
#
# # before filter gene list, calculate library size and set offset
# total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
# offset_eta = jnp.log(total_libsize)
#
# dat.transform_y(mode="log1p")
#
# # dat.filter_gene(gene_list=[gene_list[0]])  # filter to one gene
# dat.filter_gene(gene_list=gene_list[50:70])
# # dat.filter_gene(gene_list=['ENSG00000184113'])

# # Check shape of data loaded, i.e. sample size is the same
# def assert_sampleN_eq(Data, rtol=1e-10):
#     nptest.assert_allclose(Data.genotype.shape[0], Data.count.X.shape[0], rtol=rtol)
#
#
# def test_read_data():
#     dat = read_data(CYVCF2(), geno_path, pheno_path)
#     return assert_sampleN_eq(dat)
