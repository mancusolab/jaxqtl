import gzip

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd

from cyvcf2 import VCF
from pandas_plink import read_plink

import equinox as eqx


class PlinkState(NamedTuple):
    genotype: pd.DataFrame
    bim: pd.DataFrame
    fam: pd.DataFrame


class GenoIO(eqx.Module, metaclass=ABCMeta):
    """Read genotype or count data from different file format"""

    @abstractmethod
    def __call__(self, path: str) -> PlinkState:
        pass


class PlinkReader(GenoIO):
    """Read raw genotype data from plink triplets
    prefix: chr22.bed, also accept chr*.bed (read everything)

    Note: read bed file is much faster than VCF file (parser)

    bim: chrom          snp   cm       pos a0 a1  i (one-based)
    fam: fid  iid father mother gender trait  i
    bed: zero-based
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
        """read genotype from VCF file
        Note: slower than PlinkReader()
        Recommend converting VCF file to bed file first using command:
        `plink2 --vcf example.vcf.gz --make-bed --out ex`

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
        bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "cm", "pos", "alt", "ref", "i"])

        G = pd.DataFrame(genotype).T
        # G = G.fillna(G.mean())  # really slow

        G = G.set_index(fam.iid)

        return PlinkState(G, bim, fam)


# A function to convert gtf to tss bed
def gtf_to_tss_bed(annotation_gtf, feature="gene", exclude_chrs=[], phenotype_id="gene_id"):
    """Parse genes and TSSs from GTF and return DataFrame for BED output
    This function is from: https://github.com/broadinstitute/pyqtl/blob/master/qtl/io.py
    """

    chrom = []
    start = []
    end = []
    gene_id = []
    gene_name = []

    if annotation_gtf.endswith(".gz"):
        opener = gzip.open(annotation_gtf, "rt")
    else:
        opener = open(annotation_gtf, "r")

    with opener as gtf:
        for row in gtf:
            row = row.strip().split("\t")
            if row[0][0] == "#" or row[2] != feature:
                continue  # skip header
            chrom.append(row[0])

            # TSS: gene start (0-based coordinates for BED)
            if row[6] == "+":
                start.append(np.int64(row[3]) - 1)
                end.append(np.int64(row[3]))
            elif row[6] == "-":
                start.append(np.int64(row[4]) - 1)  # last base of gene
                end.append(np.int64(row[4]))
            else:
                raise ValueError("Strand not specified.")

            attributes = defaultdict()
            for a in row[8].replace('"', "").split(";")[:-1]:
                kv = a.strip().split(" ")
                if kv[0] != "tag":
                    attributes[kv[0]] = kv[1]
                else:
                    attributes.setdefault("tags", []).append(kv[1])

            gene_id.append(attributes["gene_id"])
            gene_name.append(attributes["gene_name"])

    if phenotype_id == "gene_id":
        bed_df = pd.DataFrame(
            data={"chr": chrom, "start": start, "end": end, "gene_id": gene_id},
            columns=["chr", "start", "end", "gene_id"],
            index=gene_id,
        )
    elif phenotype_id == "gene_name":
        bed_df = pd.DataFrame(
            data={"chr": chrom, "start": start, "end": end, "gene_id": gene_name},
            columns=["chr", "start", "end", "gene_id"],
            index=gene_name,
        )
    # drop rows corresponding to excluded chromosomes
    mask = np.ones(len(chrom), dtype=bool)
    for k in exclude_chrs:
        mask = mask & (bed_df["chr"] != k)
    bed_df = bed_df[mask]

    # sort by start position
    bed_df = bed_df.groupby("chr", sort=False, group_keys=False).apply(lambda x: x.sort_values("start"))

    return bed_df
