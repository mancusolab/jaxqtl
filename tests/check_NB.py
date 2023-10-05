import pandas as pd

import jax.numpy as jnp
from jax.config import config

from jaxqtl.families.distribution import NegativeBinomial
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map import map_cis_score

pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

chr = "3"
# indir = "../example/data/NK/"
indir = "../example/local/NK/"
geno_path = indir + "chr3"
covar_path = indir + "donor_features.all.6PC.tsv"
pheno_path = indir + "NK.bed.gz"
genelist_path = indir + "chr3_ENSG00000177463"
out = indir + "chr3_ENSG00000177463"

log = get_log()

# raw genotype data and impute for genotype data
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

covar = covar_reader(covar_path)

genelist = pd.read_csv(genelist_path, header=None, sep="\t").iloc[:, 0].to_list()

dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)

# before filter gene list, calculate library size and set offset
total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

# filter genes with no expressions at all
dat.filter_gene(geneexpr_percent_cutoff=0.0)

# add expression PCs to covar, genotype PC should appended to covar outside jaxqtl
dat.add_covar_pheno_PC(k=2)

# filter gene list
dat.filter_gene(gene_list=genelist)

outdf_cis_score = map_cis_score(
    dat,
    family=NegativeBinomial(),
    standardize=True,
    offset_eta=offset_eta,
    compute_qvalue=False,
    n_perm=1000,
    log=log,
)

outdf_cis_score.to_csv(out + ".cis_score.tsv.gz", sep="\t", index=False)
