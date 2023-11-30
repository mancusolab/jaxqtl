import timeit

import pandas as pd
from utils import assert_array_eq

import jax.numpy as jnp
from jax.config import config

from jaxqtl.families.distribution import NegativeBinomial, Poisson  # , Gaussian
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map import (
    map_cis,
    map_cis_nominal,
    map_cis_nominal_score,
    map_fit_intercept_only,
    write_parqet,
)

pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

geno_path = "../example/local/NK_new/chr22"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
pheno_path = "../example/local/NK_new/NK.bed.gz"
genelist_path = "../example/local/NK_new/ENSG00000198125"


# geno_path = "../example/data/chr22.n94"
# covar_path = "../example/data/donor_features.n94.tsv"
# pheno_path = "../example/data/n94_CD14_positive_monocyte.bed.gz"
# genelist_path = "../example/data/genelist.tsv"
# genelist_path = "../example/data/genelist_chr22.tsv"

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
# dat.filter_gene(gene_list=gene_list[50:55])
# dat.filter_gene(gene_list=['ENSG00000184113'])

# run mapping #

# read Rres for score test and wald test
R_res = pd.read_csv("../example/data/n94_wald_scoretest_pois_Rres.tsv", sep="\t")


# score test
def test_cis_scoretest():
    start = timeit.default_timer()

    outdf = map_cis_nominal_score(dat, family=Poisson(), offset_eta=offset_eta)
    stop = timeit.default_timer()
    print("Time: ", stop - start)
    write_parqet(
        outdf=outdf, method="scoretest", out_path="./example/result/dat_n94_test"
    )

    assert_array_eq(outdf.Z, jnp.array(R_res["Z_scoretest"]))
    assert_array_eq(outdf.pval_nominal, jnp.array(R_res["pval_scoretest"]))


# Wald test and slope, se estimates
def test_cis_waldtest():
    # n=94, one gene nominal mapping, 2592 variants, 916 ms
    # 0.86s vs. 0.90
    start = timeit.default_timer()
    outdf = map_cis_nominal(
        dat, family=Poisson(), offset_eta=offset_eta, robust_se=False
    )
    stop = timeit.default_timer()
    print("Time: ", stop - start)
    write_parqet(outdf=outdf, method="wald", out_path="./example/result/dat_n94_test")

    assert_array_eq(outdf.slope, jnp.array(R_res["slope"]))
    assert_array_eq(outdf.se, jnp.array(R_res["se"]))


# ~4s
start = timeit.default_timer()
mapcis_out_score_nb = map_cis(
    dat,
    family=NegativeBinomial(),
    offset_eta=offset_eta,
    n_perm=1000,
    compute_qvalue=False,
)
stop = timeit.default_timer()
print("Time: ", stop - start)
# mapcis_out_score_nb.to_csv(
#     "../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False
# )

out = map_cis_nominal(dat, family=NegativeBinomial(), offset_eta=offset_eta)
out.to_csv("../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False)


# ~250s
start = timeit.default_timer()
mapcis_out_wald = map_cis(
    dat,
    family=Poisson(),
    offset_eta=offset_eta,
    robust_se=False,
    n_perm=1000,
    compute_qvalue=True,
)
stop = timeit.default_timer()
print("Time: ", stop - start)
mapcis_out_wald.to_csv(
    "./example/result/n94_waldtest_pois_res.tsv", sep="\t", index=False
)

# fit intercept only model for each gene to check model assumptions
mapcis_intercept_only_mu = map_fit_intercept_only(
    dat, family=Poisson(), offset_eta=offset_eta
)
