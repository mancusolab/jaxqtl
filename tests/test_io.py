import jax.numpy as jnp

from jax import config

from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader  # , VCFReader
from jaxqtl.io.pheno import bed_transform_y, PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log


config.update("jax_enable_x64", True)

celltype = "Mono_C"
gene = "ENSG00000188536"
geno_path = "../example/local/yazar2022_RAsnps"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
addcovar_path = "../example/local/NK_new/prs.tsv"
covar_test = "score"
pheno_path = f"../../jaxqtl_project/data/OneK1K/pheno/celltype16/{celltype}.bed.gz"

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

dat.filter_gene(geneexpr_percent_cutoff=0.0)

dat.add_covar_pheno_PC(k=2, add_covar=None)

total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
offset_eta = jnp.log(total_libsize)

dat.filter_gene(gene_list=[gene])

# transformation by TMM and inverse rank normal transformation
transform_method = "tmm"  # log1p, tmm
pheno_transform = bed_transform_y(pheno_path, method=transform_method)
pheno_transform.to_csv(
    f"../example/data/n94_CD14_positive_monocyte.{transform_method}.bed.gz",
    index=False,
    sep="\t",
)
