import pandas as pd

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import NegativeBinomial
from jaxqtl.infer.utils import WaldTest
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.nominal import map_nominal_covar


pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

geno_path = "../example/local/NK_new/chr22"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
addcovar_path = "../example/local/NK_new/prs.tsv"
covar_test = "score"
pheno_path = "../example/local/NK_new/NK.chr22.bed.gz"
genelist_path = "../example/local/NK_new/ENSG00000100181"
# genelist_path = "../example/data/genelist_spatest.tsv"

log = get_log()

# raw genotype data and impute for genotype data
log.info("Load genotype.")
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

log.info("Load covariates.")
covar = covar_reader(covar_path, addcovar_path, covar_test)

log.info("Load phenotype.")
pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

# run Mapping #
dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)

dat.filter_gene(geneexpr_percent_cutoff=0.0)

dat.add_covar_pheno_PC(k=2, add_covar=addcovar_path)

maf_threshold = 0.0
dat.filter_geno(maf_threshold, "22")

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()

# before filter gene list, calculate library size and set offset
total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

# dat.filter_gene(gene_list=gene_list)
dat.filter_gene(gene_list=["ENSG00000100181"])

mapnom_covar = map_nominal_covar(dat, family=NegativeBinomial(), test=WaldTest(), offset_eta=offset_eta, robust_se=True)
