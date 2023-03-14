"""
Genotype data: plink file

h5ad file

"""
from typing import NamedTuple

from pandas_plink import read_plink1_bin

import jax.numpy as jnp


class PlinkState(NamedTuple):
    bed: jnp.ndarray
    bim: jnp.ndarray
    fam: jnp.ndarray


class RawDataState(NamedTuple):
    genotype: jnp.ndarray
    phenotype: jnp.ndarray
    features: jnp.ndarray


def readraw(geno_path: str, pheno_path: str):
    """
    pheno_path: h5ad file path, including covariates
    """
    # Append prefix with suffix
    bed_path = geno_path + ".bed"
    bim_path = geno_path + ".bim"
    fam_path = geno_path + ".fam"

    G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)

    genotype = jnp.array(G.values)  # sample x variants

    phenotype = jnp.array([0.0])
    features = jnp.array([0.0])

    # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
    # print(G.a0.sel(variant="variant0").values)
    # print(G.sel(sample="1", variant="variant0").values)

    return RawDataState(genotype, phenotype, features)
