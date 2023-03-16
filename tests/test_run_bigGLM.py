from util import assert_beta_array_eq

from jax.config import config

from jaxqtl.infer.glm import run_bigGLM, run_bigGLM_sm
from jaxqtl.load.readfile import read_data

config.update("jax_enable_x64", True)

geno_path = "./tests/data/onek1k"
pheno_path = "./tests/data/Countdata.h5ad"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"

cell_type = "CD14-positive monocyte"
dat = read_data(geno_path, pheno_path, cell_type)


def test_run_bigGLM():
    # 940 samples x 12733 genes
    smstate = run_bigGLM_sm(dat, test_run=10)
    glmstate = run_bigGLM(dat, test_run=100)
    assert_beta_array_eq(glmstate, smstate)
