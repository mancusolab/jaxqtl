# import numpy.testing as nptest

from jax.config import config

from jaxqtl.io.readfile import CYVCF2, read_data

config.update("jax_enable_x64", True)

geno_path = "../example/data/chr22"
pheno_path = "../example/data/Countdata.h5ad"
covar_path = "../example/data/donor_features.tsv"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"


G, count, covar = read_data(CYVCF2(), geno_path, pheno_path, covar_path)

# # Check shape of data loaded, i.e. sample size is the same
# def assert_sampleN_eq(Data, rtol=1e-10):
#     nptest.assert_allclose(Data.genotype.shape[0], Data.count.X.shape[0], rtol=rtol)
#
#
# def test_read_data():
#     dat = read_data(CYVCF2(), geno_path, pheno_path)
#     return assert_sampleN_eq(dat)
