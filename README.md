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
| [**Repository Structure**](#repository-structure)
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

Set up a local environment with `uv`.
```bash
# download use http address
git clone https://github.com/mancusolab/jaxqtl.git

# create and enter a local virtualenv
cd jaxqtl
uv venv
source .venv/bin/activate  # or use uv run

# install runtime + dev dependencies
uv sync --frozen --extra dev

# (optional) install docs dependencies
uv sync --frozen --extra docs

# build docs locally
uv run zensical build --strict --clean
rm -rf site
```

## Example

This section uses the example files in `tutorial/input/` to run the current CLI on a small chr22/CD4_NC dataset.
The examples below focus on 10 genes from the down-sampled OneK1K data (`N=100`).

### input format

Four inputs are relevant for the examples below: genotypes, phenotypes, covariates, and an optional gene list.

- Phenotypes: BED-like table. The first four columns must encode chromosome, start, end, and phenotype ID. The loader accepts common aliases such as `#Chr` and `Geneid`; `tutorial/input/CD4_NC.N100.bed.gz` is a working example.
- Covariates: tab-delimited table with exactly one IID-like column such as `iid` or `#iid`. `tutorial/input/donor_features.tsv` matches the current reader.
- Genotypes: PLINK1 BED/BIM/FAM with `--bfile`, PLINK2 PGEN/PVAR/PSAM with `--pfile`, indexed VCF/BCF with `--vcf`, or BGEN with `--bgen`. The examples below use the PLINK1 prefix `tutorial/input/chr22_N100`.
- Gene list: optional single-column file with one gene ID per line. `tutorial/input/genelist_10` restricts the run to 10 genes.

__Important note for the phenotype file__:\
In order to adjust for library size correctly in count-based models, pass one of the following:
- `--set-offset-from-libsize` when the phenotype file still contains the genes needed to compute library size on the fly
- `--offset ./tutorial/input/CD4_NC.N100.offset.tsv` when you want to use a precomputed offset file

### run jaxQTL in command line

The current CLI uses subcommands rather than a single `--mode` flag. The examples below show the direct interface that matches `src/jaxqtl/cli.py`.

To compute expression PCs ahead of time, use the dedicated `compute-pcs` command:
```bash
jaxqtl compute-pcs \
 --pheno ./tutorial/input/CD4_NC.N100.bed.gz \
 --covar ./tutorial/input/donor_features.tsv \
 --num-pcs 2 \
 --out ./tutorial/output/CD4_NC.N100.covar_with_expr_pcs.tsv
```

This writes a covariate table with `ExprPC0`, `ExprPC1`, ... appended to the original covariates. Use that file as `--covar` if you want expression PCs in the downstream scan.

For fast cis-eQTL mapping with permutation calibration:
```bash
data_path="./tutorial/input"
out_prefix="./tutorial/output/CD4_NC_chr22_genelist_10_jaxqtl_nb"

jaxqtl cis \
 --bfile "${data_path}/chr22_N100" \
 --covar "${data_path}/donor_features.tsv" \
 --pheno "${data_path}/CD4_NC.N100.bed.gz" \
 --gene-list "${data_path}/genelist_10" \
 --model nb \
 --test score \
 --set-offset-from-libsize \
 --normalize-covar \
 --nperm 1000 \
 --out "${out_prefix}"
```

For *faster* cis-eQTL mapping with SPA tail calibration and ACAT gene-level aggregation:

```bash
data_path="./tutorial/input"
out_prefix="./tutorial/output/CD4_NC_chr22_genelist_10_jaxqtl_nb"

jaxqtl cis \
 --bfile "${data_path}/chr22_N100" \
 --covar "${data_path}/donor_features.tsv" \
 --pheno "${data_path}/CD4_NC.N100.bed.gz" \
 --gene-list "${data_path}/genelist_10" \
 --model nb \
 --test score \
 --spa \
 --acat \
 --set-offset-from-libsize \
 --normalize-covar \
 --out "${out_prefix}"
```

This writes `./tutorial/output/CD4_NC_chr22_genelist_10_jaxqtl_nb.cis.score.spa.acat.parquet.gz`.

For a nominal cis scan over the same genes:

```bash
data_path="./tutorial/input"
out_prefix="./tutorial/output/CD4_NC_chr22_genelist_10_jaxqtl_nb"

jaxqtl nominal \
 --bfile "${data_path}/chr22_N100" \
 --covar "${data_path}/donor_features.tsv" \
 --pheno "${data_path}/CD4_NC.N100.bed.gz" \
 --gene-list "${data_path}/genelist_10" \
 --model nb \
 --test wald \
 --set-offset-from-libsize \
 --normalize-covar \
 --out "${out_prefix}"
```

For all available flags, please use ``jaxqtl -h``.

### output format

See example outputs in `./tutorial/output`. The current CLI writes Parquet files:

- `jaxqtl cis ...` writes `${out}.cis.{test}.{perm|acat}.parquet.gz`
- `jaxqtl nominal ...` writes `${out}.nominal.{test}.parquet.gz`
- `jaxqtl trans ...` writes `${out}.trans.{test}.variant.info.parquet.gz` and `${out}.trans.{test}.sumstats.parquet.gz`

The cis output contains:

- `phenotype_id`, `chrom`, `num_var`, `snp`, `a1`, `a0`, `pos`, `tss_distance`, `af`, `ma_count`
- `beta`, `se`, `pvalue`, `pvalue_adj`, `adj_method`, `nb_alpha`, `model_converged`
- `shape1`, `shape2`, `nc_estimate`, `perm_converged` when using permutation-based calibration rather than `--acat`

The nominal output contains per-variant statistics for each tested gene:

- `phenotype_id`, `chrom`, `snp`, `pos`, `a1`, `a0`, `tss_distance`, `af`, `ma_count`
- `beta`, `se`, `pvalue`, `nb_alpha`, `model_converged`

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

The repository includes `./tutorial/code/run_jaxqtl_cis_all.sh` as a batch-workflow template. Use the current `jaxqtl cis ...` invocation shown above inside your scheduler script.
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

The repository also includes `./tutorial/code/create_genelist_dir.R` as one way to generate these directories.
Then run `Rscript create_genelist_dir.R`

### 3. Run jaxQTL on HPC using array jobs

We used slurm job schedule on our HPC. An example sbatch script can be found in `./tutorial/code/run_jaxqtl_cis_all.sh` \
To submit jobs, use `sbatch run_jaxqtl_cis_all.sh`

### 4. Collect results

After all cis-eQTL mapping are completed, you can prepare results for analysis by:
1) combine all chunk results from one cell type into one single file
2) filter by converged GLM model `model_converged > 0` and, when using permutation-based calibration, converged adjustment fits `perm_converged > 0`
3) calculate FDR-controlled pvalues on `pvalue_adj` using
[qvalue](https://bioconductor.org/packages/release/bioc/html/qvalue.html) method in R
4) identify eGenes (genes with at least one eQTL) by qvalue < FDR level, e.g., 0.05

## Note

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
