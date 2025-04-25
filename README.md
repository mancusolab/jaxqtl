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
(https://www.medrxiv.org/content/10.1101/2025.01.18.25320755v2)

We are currently working on more detailed documentations.
Feel free to contact me (zzhang39@usc.edu) if you need help on running our tool and further analysis.

[**Installation**](#Installation)
| [**Example**](#Example)
| [**Notes**](#notes)
| [**Support**](#support)
| [**Other Software**](#other-software)

## sc-eQTL model

jaxQTL requires pseudobulking by sum for each annotated cell type from the single-cell data.
For a focal gene in a given cell type,
jaxQTL can fit a count-based model, either Poisson or negative binomial (NB) model,
between gene expression read count and a SNP variant as:

$$\log(E(y_i | X_i, g_i, l_i)) = g_i \beta_{g} + X_i \beta + \log(l_i)$$

where $i$ denotes individual $i$,
$y_i$ is the pseudobulk read count for a given gene,
$g_i$ is the genotype of one SNP variant (e.g., allele count of alternative alleles),
$X_i$ is the covariate vector (e.g., age, sex, genotype PCs, expression PCs),
$l_i$ is the total read count for individual $i$ across all genes within this cell type, i.e., library size.

To account for overdispersion observed in single-cell count data, jaxQTL modeled the conditional variance as

$$Var(y_i | X_i, g_i, L_i) = \mu_i + \alpha \mu_i^2$$

where $\mu_i = E(y_i | X_i, g_i, l_i)$ and $\alpha$ is the overdispersion parameter in NB model.
When $\alpha=0$, NB reduced to Poisson model.

For cis-eQTL mapping, we focus on estimating the SNP effect size, its standard error under specified model,
and the test statistics for $\beta_g \ne 0$.
While both the Wald test and score test are available in jaxQTL,
we recommend using the score test to assess the nonzero cis-SNP effect $g$
on read counts $y$ for its improved computational efficiency.

## Interpretation on genetic effect

Compared to linear model applied to normalized read counts,
count-based model provides interpretation on the original count data scale.
The effect sizes estimated by Poisson or NB model reflect a change in the transcription rate
or proportion if including library size offsets.

## Installation

We recommend first create a conda environment and have `pip` installed.
```bash
# download use http address
git clone https://github.com/mancusolab/jaxqtl.git

# create conda environment
conda create -n jaxqtl python=3.10.9
conda activate jaxqtl

cd jaxqtl
pip install -e .

# install other required packages
pip install lineax qtl
```

## Example

Here we provide a working example for cis-eQTL mapping using down-sampled OneK1K dataset (N=100).
Now we focus on identifying lead SNP for 10 genes in CD4_NC cell type and obtain
its calibrated P value using permutations.

### input format

The input data format for expression data and covariate files are the same as described in
[tensorQTL](https://github.com/broadinstitute/tensorqtl). See example data in `./tutorial/input/`.

Four input files are required for eQTL analyses with jaxQTL: genotypes, phenotypes, covariates, and gene list (optional).

* Phenotypes are provided in BED format, with a single header line starting with # and the first four columns corresponding to:
`chr`, `start`, `end`, `phenotype_id`, with the remaining columns corresponding to samples (the identifiers must match those in the genotype input).
The BED file can specify the center of the cis-window (usually the TSS), with `start == end-1`,
or alternatively, start and end positions, in which case the cis-window is [start-window, end+window]

* Covariates are provided as a tab-delimited tsv file dataframe (samples x covariates) with column headers.

* Genotypes can be provided in PLINK1 bed/bim/fam format. (We will accomodate other formats in the future verssions)

* Optional: A single-column (no header) file specifying gene identifiers.
This means to break all genes on each chromosome to chunks (recommend 200-300 genes) so that
run jaxQTL using HPC (e.g., slurm) array jobs to distribute in parallel.

__Important note for the phenotype file__:\
In order to adjust for library size (total count per person) correctly in the count-based model,
you can use either of following:\
- include all genes in the phenotype file, then jaxQTL can compute library size on the fly (default)
- include a filtered subset of genes in the phenotype file, but compute the $\log$(library size) by yourself in file
(see example `./tutorial/input/CD4_NC.N100.offset.tsv` ) and feed the path into jaxQTL using  `--offset ${offset_path}`

### run jaxQTL in command line

We provide two scripts for using jaxQTL in command line interface for cis-eQTL and nominal scan:
* `./tutorial/code/run_jaxqtl_cis.sh`:: cis-eQTL mapping with permutation
* `./tutorial/code/run_jaxqtl_nominal.sh`: all pairwise summary statistics of cis-SNPs for each gene

For example in cis-eQTL mapping with permutation calibration, we first specify parameters and paths:
```bash
data_path="./tutorial/input"
out_path="./tutorial/output"

celltype="CD4_NC"

# genelist to perform cis-eQTL mapping
chr=22
chunk_file="genelist_10"

# choose test method: score test (recommended) or wald
test_method="score"

# choose cis or nominal scan
mode="cis"
window=500000 # default extend 500kb on either side, i.e., [start-window, end+window]

# jaxQTL by default compute expression PCs using the entire data provided in *.bed.gz
# to disable adding expression PCs, set this to 0
num_expression_pc=2

pheno="${data_path}/${celltype}.N100.bed.gz"
geno="${data_path}/chr${chr}" # prefix for plink triplet files
covar="${data_path}/donor_features.tsv"

# choose gene list for eQTL mapping
genelist="${data_path}/${chunk_file}"

# choose eQTL model: NB for negative binomial, poisson, gaussian
model="NB"

# if using permutation method to calibrate gene-level p value, set number of permutation
nperm=1000

# choose platform: cpu, gpu, tpu
platform="cpu"

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
 --window ${window} \
 --genelist ${genelist} \
 --test-method ${test_method} \
 --nperm ${nperm} \
 --addpc ${num_expression_pc} \
 --standardize \
 -p ${platform} \
 --out ${out}
```
This took about ~1 min on my PC of macOS Apple M1 chip.

For all available flags, please use ``jaxqtl -h``.

### output format

See output results in `./tutorial/output`. The columns are,
* `phenotype_id`: phenotype id of gene
* `chrom`: chromosome of gene
* `num_var`: number of variants tested in the cis-window
* `variant_id`: identifier for lead SNP with the smallest p-value
* `tss_distance`: distance to transcription starting site (TSS)
* `ma_count`: minor allele count
* `af`: allele frequency of alternative allele (effect allele)
* `beta_shape1`: Parameter of the fitted Beta distribution
* `beta_shape2`: Parameter of the fitted Beta distribution
* `beta_converged`: whether fitting Beta distribution converged (1 or 0)
* `true_nc`: non-central parameter for chi2 distribution to compute p values
* `opt_status`: whether optimizer for identifying true_nc exited successfully
* `pval_nominal`: nominal p-value of the association between the gene and SNP
* `slope`: regression slope
* `slope_se`: standard error of the regression slope
* `pval_beta`: Beta-approximated empirical p-value
* `alpha_cov`: overdispersion parameter fitted for covariate-only model if `--test-method score`
* `model_converged`: whether the GLM model converged (1 or 0)

## Genome-wide sc-eQTL mapping

### General advice
To efficiently run sc-eQTL mapping genome-wide using jaxQTL count-based models especially for large cohort (N>1000),
we recommend run jaxQTL by chromosome and
break genes (total ~20,000) into chunks, with chunk size of 200 - 300 genes.
For our cis-eQTL analysis on OneK1K (max N=982),
the run time for each chunk of 50 genes is around 1 -2 hrs depending on which chromosome (chr6 usually takes longer).
In this way, we can distribute the tasks over multiple nodes in parallel using job scheduler on HPC.
You may consider increase the chunk size to maximize efficiency on CPU node or even more if using GPU (50 genes takes ~20 mins).

If sample size is small (e.g., N=200), you may consider not splitting by chunks because it's fast enough.
We recommend trying a few genes and check the log file to estimate the run time.

We provide a shell script example to submit batch jobs on HPC: `./tutoral/code/run_jaxqtl_cis_all.sh`.
Please see below for detailed instructions:

### 1. Organize directory structure
Suppose we are located in the working directory of a project. Here is an example of directory structure:

```{bash}
.
├── code/
├── data/
│   ├── features/
│   │   └── donor_features.tsv
│   ├── geno/
│   │   ├── chr1.bed
│   │   ├── chr1.bim
│   │   └── chr1.fam
│   │   └── ...
│   └── pheno/
│       ├── B_IN.bed.gz
│       ├── CD4_NC.bed.gz
│       └── CD8_NC.bed.gz
│       └── ...
└── result/
    └── cis/
```

### 2. Create gene list chunks and output directories

Assuming the directory structure as above, we provide a script for creating
1) gene list directories and gene list files `/data/genelist/${celltype}/chr{chr_idx}/*`, for example:
```{bash}
./data/genelist/CD4_NC/chr1
├── chunk_1
├── chunk_2
├── chunk_3
...
```

2) a parameter file for the example shell script to distribute jobs

```{bash}
CD4_NC	1	chunk_1
CD4_NC	1	chunk_2
....
CD4_NC	2	chunk_1
CD4_NC	2	chunk_2
...
```

3) result directories for the corresponding cell type and chromosomes `/result/cis/${celltype}/chr{chr_idx}/*`
```{bash}
./result/cis/CD4_NC/
├── chr1/
├── chr2/
├── chr3/
...
```

First put this script `./tutoral/code/create_genelist_dir.R` in your `/code` directory.
Then run `Rscript create_genelist_dir.R`

### 3. Run jaxQTL on HPC using array jobs

We used slurm job schedule on our HPC. An example sbatch script can be found in `./tutoral/code/run_jaxqtl_cis_all.sh` \
To submit jobs, use `sbatch run_jaxqtl_cis_all.sh`

### 4. Collect results

After all cis-eQTL mapping are completed, you can prepare results for analysis by:
1) combine all chunk results from one cell type into one single file
2) filter by converged GLM model `model_converged > 0` and converged beta approximation `beta_converged > 0`
3) calculate FDR-controlled pvalues on `pval_beta` using
[qvalue](https://bioconductor.org/packages/release/bioc/html/qvalue.html) method in R
4) identify eGenes (genes with at least one eQTL) by qvalue < FDR level, e.g., 0.05

## Note

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
