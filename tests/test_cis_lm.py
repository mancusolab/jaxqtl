import timeit

import pandas as pd

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import Gaussian
from jaxqtl.infer.permutation import InferBetaLM
from jaxqtl.infer.utils import WaldTest_lm
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.cis import map_cis


pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

dat_dir = "../../Software/tensorqtl/example"
geno_path = f"{dat_dir}/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18"
covar_path = f"{dat_dir}/data/GEUVADIS.445_samples.covariates.txt"
pheno_path = f"{dat_dir}/data/GEUVADIS.445_samples.expression.bed.gz"  # NK.tmm.bed.gz
genelist_path = f"{dat_dir}/data/chr18_genelist"

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
dat = create_readydata(geno, bim, pheno, covar, autosomal_only=False)

maf_threshold = 0.0
dat.filter_geno(maf_threshold, "18")

# filter genes with no expressions at all
# dat.filter_gene(geneexpr_percent_cutoff=0.0)

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t", header=None).iloc[:, 0].to_list()

dat.filter_gene(gene_list=[gene_list[1]])  # filter to one gene
# dat.filter_gene(gene_list=["ENSG00000273289"])

n_obs = dat.pheno.count.shape[0]
offset_eta = jnp.zeros((n_obs, 1))

# ENSG00000273289
start = timeit.default_timer()
mapcis_out = map_cis(
    dat,
    family=Gaussian(),
    test=WaldTest_lm(),
    beta_estimator=InferBetaLM(),
    offset_eta=offset_eta,
    n_perm=1000,
    compute_qvalue=False,
)
stop = timeit.default_timer()
print("Time: ", stop - start)

print("done")
