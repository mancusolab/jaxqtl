[![PyPI-Server](https://img.shields.io/pypi/v/jaxqtl.svg)](https://pypi.org/project/jaxqtl/)
[![Github](https://img.shields.io/github/stars/mancusolab/jaxqtl?style=social)](https://github.com/mancusolab/jaxqtl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# jaxQTL

``jaxQTL`` is a scalable software for large-scale eQTL mapping using count-based models!

We present jaxQTL for single-cell eQTL mapping using highly efficient count-based model (i.e., negative binomial or Poisson).
Our software is implemented using `Just-in-time` (JIT)
via [JAX](https://github.com/google/jax) in Python, which generates and compiles heavily optimized
C++ code in real time and operates seamlessly on CPU, GPU or TPU.
jaxQTL is a command line tool.
Please see example below and full documentation.

For preprint, please see:

Zhang, Z., Kim, A., Suboc, N., Mancuso, N., and Gazal, S. (2025).
Efficient count-based models improve power and robustness for large-scale single-cell eQTL mapping. _medRxiv_
(https://www.medrxiv.org/content/10.1101/2025.01.18.25320755v1)

We are currently working on more detailed documentations.
Feel free to contact me (zzhang39@usc.edu) if you need help on running our tool and further analysis.

[**Installation**](#Installation)
| [**Example**](#Example)
| [**Notes**](#notes)
| [**Support**](#support)
| [**Other Software**](#other-software)

## sc-eQTL model

jaxQTL requires pseudo-bulking by sum for each pre-annotated cell type from the single-cell data matrix.
For a focal gene in a given cell type,
jaxQTL can fit a GLM count-based model (Poisson or negative binomial) between gene expression and a SNP variant as:

$$\log(E(y_i | X_i, g_i, L_i)) = g_i \beta_{g} + X_i \beta + \log(L_i) $$,
where $i$ denotes individual $i$,
$y_i$ is the pseudo-bulking UMI count for a given gene,
$g_i$ is the genotype of one SNP variant (e.g., allele count of alternative alleles),
$X_i$ is covariate vector (e.g., age, sex, genotype PCs, expression PCs),
$L_i$ is the total UMI count sum for individual $i$ across all genes within this cell type, i.e., library size.

To account for overdispersion observed in single-cell count data, jaxQTL modeled the conditional variance as
$$ Var(y_i | X_i, g_i, L_i) = \mu_i + \alpha \mu_i^2$$,
where $\mu_i = E(y_i | X_i, g_i, L_i)$ and $\alpha$ is the overdispersion parameter in negbinom model.
When $\alpha=0$, negbinom is equivalent as Poisson model.

For eQTL mapping, we focus on estimating the SNP effect size, its standard error under specified model,
and the test statistics for $\beta_g \ne 0$.
While both the Wald test and score test are implemented in jaxQTL,
jaxQTL employs the score test in assessing the nonzero cis-SNP effect $g$
on gene expression count $y$ for its improved computational efficiency.

## Interpretation on genetic effect

Compared to linear model applied to normalized count data,
count-based model provides interpretation on the original data scale.
The effect sizes estimated by Poisson or negbinom reflect a change in the transcription rate
(or proportion if including library size offsets).

## Installation

We recommend first create a conda environment and have `pip` installed.
```bash
# download use http address
git clone https://github.com/mancusolab/jaxqtl.git
# or use ssh agent
git clone git@github.com:mancusolab/jaxqtl.git

cd jaxqtl
pip install -e .

# install other required packages
pip install lineax qtl
```

## Working example

Here we provide a working example for cis-eQTL mapping using downsampled OneK1K dataset (N=100).
Now we focus on identifying lead SNP for 10 genes in CD4_NC cell type and obtain
its calibrated P value using permutations.

### input format

The input data format for expression data and covariate files are the very similar as described in
[tensorQTL](https://github.com/broadinstitute/tensorqtl). See example data in `./tutorial/input/`.

Four input files are required for eQTL analyses with jaxQTL: genotypes, phenotypes, covariates, and gene list.

* Phenotypes are provided in BED format, with a single header line starting with # and the first four columns corresponding to:
`chr`, `start`, `end`, `phenotype_id`, with the remaining columns corresponding to samples (the identifiers must match those in the genotype input).
The BED file can specify the center of the cis-window (usually the TSS), with `start == end-1`,
or alternatively, start and end positions, in which case the cis-window is [start-window, end+window]

* Covariates are provided as a tab-delimited tsv file dataframe (samples x covariates) with column headers.

* Genotypes can be provided in PLINK1 bed/bim/fam format.

* A single-column (no header) file specifying gene identifiers.
This means to break all genes on each chromosome to chunks (recommend 50-100 genes each) so that
run jaxQTL in parallel using HPC (e.g., slurm) array jobs.

We will accommodate for other data formats such as PLINK2 pgen/pvar/psam format in the future versions.

### run jaxQTL in shell

We provide two scripts for using jaxQTL in command line interface for cis scan and nominal scan:
* `./tutorial/code/run_jaxqtl_cis.sh`:: cis-eQTL mapping with permutation
* `./tutorial/code/run_jaxqtl_nominal.sh`: all pairwise summary statistics of cis-SNPs-gene

For example in cis-eQTL mapping with permutation calibration, we first specify parameters and paths:
```bash
data_path="./tutorial/input"
out_path="./tutorial/output"

celltype="CD4_NC"

# # genelist to perform cis-eQTL mapping
chr=22
chunk_file="genelist_10"

# choose test method: score test (recommended) or wald
test_method="score"

# choose cis or nominal scan
mode="cis"
window=500000 # default extend 500kb on either side, i.e., [start-window, end+window]

# jaxQTL by default compute expression PCs using the entire data provided in *.bed.gz
# to disable this, set this to 0
num_expression_pc=2

pheno="${data_path}/${celltype}.N100.bed.gz"
geno="${data_path}/chr${chr}" # prefix for plink triplet files
covar="${data_path}/donor_features.all.6PC.tsv"

# choose gene list for eQTL mapping
genelist="${data_path}/${chunk_file}"

# choose eQTL model: NB for negative binomial, poisson, gaussian
model="NB"

# if using permutation method to calibrate gene-level p value, set number of permutation
nperm=1000

# prefix for output file
out="${out_path}/${celltype}_chr${chr}_${chunk_file}_jaxqtl_${model}"
```

Then run jaxQTL using:
```bash
jaxqtl \
 --geno ${geno} \
 --covar ${covar} \
 --pheno ${pheno} \
 --model ${model} \
 --mode ${mode} \
 --genelist ${genelist} \
 --test-method ${test_method} \
 --nperm ${nperm} \
 --addpc ${num_expression_pc} \
 --standardize \
 --out ${out}
```
For all available flags, please use ``jaxqtl -h``.

### output format

See output in `./tutorial/output`. Specifically,
* `phenotype_id`: phenotype id
* `chrom`: chromosome of gene
* `num_var`: number of variants tested in the cis-window
* `variant_id`: identifier for lead SNP among cis-SNPs in the format of chr*_pos_ref_alt_b37, i.e., smallest p-value
* `tss_distance`: distance to transcription starting site (TSS)
* `ma_count`: minor allele count
* `af`: allele frequency of alternative allele (effect allele)
* `beta_shape1`: Parameter of the fitted Beta distribution (These values are centered around 1)
* `beta_shape2`: Parameter of the fitted Beta distribution
* `beta_converged`: whether fitting Beta distribution succeed (1 or 0)
* `true_nc`: non-central parameter for chi2 distribution to compute p values
* `opt_status`: whether optimizer for identifying true_nc exited successfully
* `pval_nominal`: nominal p-value of the association between the phenotype and variant
* `slope`: regression slope
* `slope_se`: standard error of the regression slope
* `pval_beta`: Beta-approximated empirical p-value
* `alpha_cov`: overdispersion parameter fitted for covariate-only model
* `model_converged`: whether the covariate-only model converged (1 or 0)

## Genome-wide sc-eQTL mapping

## Note

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
