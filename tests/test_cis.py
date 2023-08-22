import os
import timeit

import pandas as pd
from utils import assert_array_eq

import jax.numpy as jnp
from jax.config import config

from jaxqtl.families.distribution import Poisson  # , Gaussian
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map import map_cis_nominal, map_cis_nominal_scoretest  # , map_cis

pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)


geno_path = "./example/data/chr22.n94"
covar_path = "./example/data/donor_features.n94.tsv"
pheno_path = "./example/data/n94_CD14_positive_monocyte.bed.gz"
genelist_path = "./example/data/genelist.tsv"

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

# # n=94, one gene cis mapping, 2592 variants
# # 80 s vs. 60s
# start = timeit.default_timer()
# mapcis_out_1000 = map_cis(
#     dat,
#     family=Poisson(),
#     offset_eta=offset_eta,
#     robust_se=False,
#     n_perm=1000,
#     add_qval=True,
# )
# stop = timeit.default_timer()
# print("Time: ", stop - start)

# # 500 looks ok
# start = timeit.default_timer()
# mapcis_out_500 = map_cis(
#     dat, family=Poisson(), offset_eta=offset_eta, robust_se=False, n_perm=500
# )
# stop = timeit.default_timer()
# print("Time: ", stop - start)
#
# start = timeit.default_timer()
# mapcis_out_100 = map_cis(
#     dat, family=Poisson(), offset_eta=offset_eta, robust_se=False, n_perm=100
# )
# stop = timeit.default_timer()
# print("Time: ", stop - start)
#


# read Rres for score test and wald test
R_res = pd.read_csv("./example/data/n94_wald_scoretest_pois_Rres.tsv", sep="\t")


# score test
def test_cis_scoretest():
    start = timeit.default_timer()
    map_cis_nominal_scoretest(
        dat,
        family=Poisson(),
        offset_eta=offset_eta,
        out_path="./example/result/dat_n94_test_scoretest",
    )
    stop = timeit.default_timer()
    print("Time: ", stop - start)

    prefix = "dat_n94_test_scoretest"
    out_dir = "./example/result"
    pairs_df_scoretest = pd.read_parquet(
        os.path.join(out_dir, f"{prefix}.cis_qtl_pairs.22.scoretest.parquet")
    )

    assert_array_eq(pairs_df_scoretest.Z, jnp.array(R_res["Z_scoretest"]))


# Wald test and slope, se estimates
def test_cis_waldtest():
    # n=94, one gene nominal mapping, 2592 variants, 916 ms
    # 0.86s vs. 0.90
    start = timeit.default_timer()
    map_cis_nominal(
        dat,
        family=Poisson(),
        offset_eta=offset_eta,
        out_path="./example/result/dat_n94_test_wald",
        robust_se=False,
    )
    stop = timeit.default_timer()
    print("Time: ", stop - start)

    prefix = "dat_n94_test_wald"
    out_dir = "./example/result"
    pairs_df_wald = pd.read_parquet(
        os.path.join(out_dir, f"{prefix}.cis_qtl_pairs.22.parquet")
    )

    assert_array_eq(pairs_df_wald.slope, jnp.array(R_res["slope"]))
    assert_array_eq(pairs_df_wald.se, jnp.array(R_res["se"]))
