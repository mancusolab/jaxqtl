# import importlib.resources as pkg_resources
# import os

from jax.config import config

from jaxqtl.load.readfile import readraw

config.update("jax_enable_x64", True)

# TODO: better way to handle this, for now the data is in /tests folder
# prefix_path = os.fspath(pkg_resources.path("jaxqtl.data", "onek1k"))
pheno_path = "TBD"
prefix_path = "./data/onek1k"


def test_readraw():
    RawData = readraw(prefix_path, pheno_path)
    print(RawData.genotype)
    return None
