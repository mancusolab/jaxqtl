from abc import ABCMeta, abstractmethod
from typing import NamedTuple

import equinox as eqx
import pandas as pd
from cyvcf2 import VCF
from pandas_plink import read_plink


class PlinkState(NamedTuple):
    genotype: pd.DataFrame
    bim: pd.DataFrame
    fam: pd.DataFrame


class GenoIO(eqx.Module, metaclass=ABCMeta):
    """Read genotype or count data from different file format"""

    @abstractmethod
    def __call__(self, path: str) -> PlinkState:
        """
        Read files
        """
        pass


class PlinkReader(GenoIO):
    """Read genotype data from plink triplets
    prefix: chr22.bed, also accept chr*.bed (read everything)

    Note: read bed file is much faster than VCF file (parser)

    bim: chrom          snp   cm       pos a0 a1  i
    fam: fid  iid father mother gender trait  i
    """

    def __call__(self, bed_path: str) -> PlinkState:
        # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
        bim, fam, bed = read_plink(bed_path, verbose=False)
        G = pd.DataFrame(bed.compute().T)  # nxp

        # TODO: add imputation for missing genotype etc...
        # G = G.fillna(G.mean())  # really slow

        G = G.set_index(fam.iid)
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

        #  chrom        snp       cm     pos a0 a1  i
        bim = pd.DataFrame(
            bim_list, columns=["chrom", "snp", "cm", "pos", "alt", "ref", "i"]
        )

        G = pd.DataFrame(genotype).T
        # G = G.fillna(G.mean())  # really slow

        G = G.set_index(fam.iid)

        # convert to REF dose
        # genotype = 2 - genotype

        return PlinkState(G, bim, fam)
