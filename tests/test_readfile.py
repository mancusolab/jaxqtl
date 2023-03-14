# import importlib.resources as pkg_resources
# import os

from jax.config import config

from jaxqtl.load.readfile import readraw

config.update("jax_enable_x64", True)

pheno_path = "TBD"
prefix_path = "./tests/data/onek1k"


def test_readraw():
    RawData = readraw(prefix_path, pheno_path)
    print(RawData.genotype)
    return None
