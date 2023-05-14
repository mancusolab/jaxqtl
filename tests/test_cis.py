import os
import timeit

import pandas as pd

import jax.numpy as jnp
from jax.config import config

from jaxqtl.families.distribution import Poisson  # , Gaussian
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map import map_cis, map_cis_nominal

pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)


geno_path = "../example/data/chr22.n94"
covar_path = "../example/data/donor_features.n94.tsv"
pheno_path = "../example/data/CD14_positive_monocyte.bed.gz"
genelist_path = "../example/data/genelist.tsv"

log = get_log()

# raw genotype data and impute for genotype data
log.info("Load genotype.")
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

log.info("Load covariates.")
covar = covar_reader(covar_path)

log.info("Load phenotype.")
pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

# run Mapping #
dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)

maf_threshold = 0.0
dat.filter_geno(maf_threshold, "22")

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()

# before filter gene list, calculate library size and set offset
total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

dat.filter_gene(gene_list=[gene_list[0]])  # filter to one gene

# n=94, one gene cis mapping, 2592 variants, 1min 22s (80s)
# 109 s
start = timeit.default_timer()
mapcis_out = map_cis(dat, family=Poisson(), offset_eta=offset_eta)
stop = timeit.default_timer()
print("Time: ", stop - start)


# n=94, one gene nominal mapping, 2592 variants, 916 ms
map_cis_nominal(
    dat,
    family=Poisson(),
    offset_eta=offset_eta,
    out_path="../example/result/dat_n94_test",
)

prefix = "dat_n94_test"
out_dir = "../example/result"
pairs_df = pd.read_parquet(os.path.join(out_dir, f"{prefix}.cis_qtl_pairs.22.parquet"))
