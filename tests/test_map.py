import timeit

import pandas as pd

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.utils import WaldTest
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.cis import map_cis
from jaxqtl.map.nominal import map_nominal


pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

geno_path = "../example/local/NK_new/chr22"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
# addcovar_path = "../example/local/NK_new/prs.tsv"
# covar_test = "score"
pheno_path = "../example/local/NK_new/NK.bed.gz"  # NK.tmm.bed.gz
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

# filter genes with no expressions at all
dat.filter_gene(geneexpr_percent_cutoff=0.0)

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()

# before filter gene list, calculate library size and set offset
total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

dat.add_covar_pheno_PC(2)

# dat.filter_gene(gene_list=[gene_list[0]])  # filter to one gene
dat.filter_gene(gene_list=["ENSG00000273289"])

n_obs = dat.pheno.count.shape[0]

# ENSG00000273289
# start = timeit.default_timer()
# mapcis_out_score_nb = map_cis(
#     dat,
#     family=NegativeBinomial(),
#     test=ScoreTest(),
#     offset_eta=offset_eta,
#     n_perm=1000,
#     compute_qvalue=False,
# )
# stop = timeit.default_timer()
# print("Time: ", stop - start)
# # mapcis_out_score_nb.to_csv("../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False)

# mapcis_out_score_lm = map_cis(
#     dat,
#     family=Gaussian(),
#     test=WaldTest(),
#     offset_eta=jnp.zeros((n_obs, 1)),
#     # offset_eta=offset_eta,
#     n_perm=1000,
#     compute_qvalue=False,
#     beta_estimator=InferBetaLM(),
#     seed=1,
# )
#
# mapnom_covar = map_nominal_covar(
#     dat, family=NegativeBinomial(), test=WaldTest(), offset_eta=offset_eta, robust_se=False
# )

out_nb = map_nominal(dat, family=NegativeBinomial(), offset_eta=offset_eta, test=WaldTest())
# cond_snp="22:51216564")
# out_nb.to_csv("../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False)

out_lm = map_nominal(dat, family=Gaussian(), offset_eta=0.0, test=WaldTest())

# ~250s
start = timeit.default_timer()
mapcis_out_wald = map_cis(
    dat,
    family=Poisson(),
    offset_eta=offset_eta,
    n_perm=1000,
    robust_se=False,
    compute_qvalue=False,
)
stop = timeit.default_timer()
print("Time: ", stop - start)
# mapcis_out_wald.to_csv("./example/result/n94_waldtest_pois_res.tsv", sep="\t", index=False)
