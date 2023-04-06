from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd
from cyvcf2 import VCF
from pandas_plink import read_plink

from jax import Array, numpy as jnp
from jax._src.tree_util import register_pytree_node, register_pytree_node_class


class PlinkState(NamedTuple):
    genotype: Array
    bim: pd.DataFrame
    fam: pd.DataFrame


@register_pytree_node_class
class GenoIO(ABC):
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


class PlinkReader(GenoIO):
    """Read genotype data from plink triplets
    prefix: chr22.bed, also accept chr*.bed (read everything)

    Note: read bed file is much faster than VCF file (parser)
    """

    def __call__(self, bed_path: str) -> PlinkState:
        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        bim, fam, bed = read_plink(bed_path, verbose=False)
        G = jnp.asarray(bed.compute())

        # TODO: add imputation for missing genotype etc...

        return PlinkState(G, bim, fam)


class VCFReader(GenoIO):
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
        bim = pd.DataFrame(
            bim_list, columns=["chrom", "snp", "cm", "pos", "alt", "ref", "i"]
        )

        # convert to REF dose
        # genotype = 2 - genotype

        return PlinkState(genotype, bim, fam)
