import numpy.testing as nptest

from jax.config import config

from jaxqtl.io.readfile import read_data

config.update("jax_enable_x64", True)

geno_path = "./tests/data/onek1k"
pheno_path = "./tests/data/Countdata.h5ad"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"


# Check shape of data loaded, i.e. sample size is the same
def assert_sampleN_eq(Data, rtol=1e-10):
    nptest.assert_allclose(Data.genotype.shape[0], Data.count.X.shape[0], rtol=rtol)


def test_read_data():
    dat = read_data(geno_path, pheno_path)
    return assert_sampleN_eq(dat)
