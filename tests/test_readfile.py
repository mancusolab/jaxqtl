import numpy.testing as nptest

from jax.config import config

from jaxqtl.load.readfile import readraw

config.update("jax_enable_x64", True)

geno_path = "./tests/data/onek1k"
pheno_path = "./tests/data/Countdata.h5ad"
cov_path = "./tests/data/onek1kpca.eigenvec"


# Check shape of data loaded, i.e. sample size is the same
def assert_sampleN_eq(Data, rtol=1e-10):
    nptest.assert_allclose(
        Data.genotype.shape[0], len(Data.count.obs.donor_id.unique()), rtol=rtol
    )


def test_readraw():
    RawData = readraw(geno_path, pheno_path, cov_path)
    return assert_sampleN_eq(RawData)
