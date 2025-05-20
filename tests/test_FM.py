import timeit

import pandas as pd

import jax.numpy as jnp

from jax import config

from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.finemap import map_finemap


pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

geno_path = "../example/local/NK_new/chr22"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
pheno_path = "../example/local/NK_new/NK.bed.gz"
# genelist_path = "../example/local/NK_new/ENSG00000198125"
genelist_path = "../example/data/genelist_spatest.tsv"
log = get_log()

# raw genotype data and impute for genotype data
log.info("Load genotype.")
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

log.info("Load covariates.")
# covar = covar_reader(covar_path, addcovar_path, covar_test)
covar = covar_reader(covar_path)

log.info("Load phenotype.")
pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

# run Mapping #
dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)

maf_threshold = 0.0
dat.filter_geno(maf_threshold, "22")

# add phenotype PCs
dat.filter_gene(geneexpr_percent_cutoff=0.0)

dat.add_covar_pheno_PC(k=2, add_covar=None)

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()

# before filter gene list, calculate library size and set offset
total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

# dat.filter_gene(gene_list=[gene_list[0]])  # filter to one gene

# ENSG00000188677 has one CS one single SNP: chr22_44424108_C_T_b37
# ENSG00000075234 has 2 CS, one of size 12 and the other 5
# ENSG00000099889 has 1 CS with 2 SNP
# dat.filter_gene(gene_list=["ENSG00000075234", "ENSG00000188677", "ENSG00000099889"])
dat.filter_gene(gene_list=["ENSG00000099889"])

covar = dat.covar

# ~4s
start = timeit.default_timer()
mapcis_out_score_nb = map_finemap(
    dat, set_L=5, step_size=0.05, out_path="../example/local/NK_new", covar=covar, offset_eta=offset_eta
)
stop = timeit.default_timer()
print("Time: ", stop - start)
