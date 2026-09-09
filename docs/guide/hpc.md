# Run large scans

Large cis scans can be divided by chromosome and gene list, then submitted as independent scheduler jobs. Start with
small chunks and use observed runtime and memory to choose the final chunk size.

The repository includes two project-layout templates:

- [`tutorial/code/create_genelist_dir.R`](https://github.com/mancusolab/jaxqtl/blob/main/tutorial/code/create_genelist_dir.R),
  which demonstrates gene-list chunk creation.
- [`tutorial/code/run_jaxqtl_cis_all.sh`](https://github.com/mancusolab/jaxqtl/blob/main/tutorial/code/run_jaxqtl_cis_all.sh),
  which demonstrates a Slurm array running one cis scan per parameter row.

!!! warning "Adapt the batch template before submitting jobs"

    They contain site-specific placeholders for the partition, notification address, working directory, and array
    size. Review every `#SBATCH` directive and path before submitting a job.

!!! warning "Do not recompute offsets from a restricted phenotype file"

    Chromosome- or chunk-specific phenotype files do not contain the full library. Supply offsets computed from the
    original unfiltered count matrix rather than using `--set-offset-from-libsize` in each job. Using `--gene-list`
    with a full phenotype file is safe because jaxQTL computes library size before selecting genes.

## Recommended workflow

1. Split phenotypes by chromosome if the input pipeline supports chromosome-specific files.
2. Divide gene IDs into scheduler-sized lists.
3. Run one `jaxqtl cis` process per chromosome and gene-list pair.
4. Concatenate compatible Parquet results after every job finishes.
5. Follow [Post-process cis results](postprocessing.md) to filter failures and control the study-level FDR.

## Parameter-file layout

The Slurm template reads three whitespace-delimited fields from each row:

```text
CD4_NC  1  chunk_1
CD4_NC  1  chunk_2
CD4_NC  2  chunk_1
```

The fields identify the cell type, chromosome, and gene-list filename. With the template directory layout, the first
row resolves to:

```text
data/geno/chr1
data/pheno/CD4_NC.bed.gz
data/genelist/CD4_NC/chr1/chunk_1
result/cis/CD4_NC/chr1/chunk_1.nb.cis.score.perm.parquet.gz
```

Set the Slurm array range to the number of rows in the parameter file. Before submission, run one row interactively
from the intended working directory and confirm that its input files and output directory exist.

```bash
bash tutorial/code/run_jaxqtl_cis_all.sh 1
sbatch tutorial/code/run_jaxqtl_cis_all.sh
```

## Compilation and memory

Score, SPA, and Wald cis and nominal scans use fixed-size genotype blocks. The final block is padded internally,
and padded variants do not contribute to reported associations or gene-level tests. Genes with different numbers
of cis variants share compiled fitting and testing kernels when their sample dimensions, covariate dimensions,
dtypes, and test settings match.

The first initialization and test blocks include JAX compilation time. Permutation scans also compile batched
initialization and scoring functions. Estimate runtime from several genes with representative window sizes,
including startup costs. Each scheduler process has its own in-memory compilation cache; starting more jobs
repeats those startup costs.

The standard cis and nominal paths reuse compiled kernels without periodic cache clearing. This avoids
recompiling them during a scan. Compilation can still occur when sample or covariate dimensions, dtypes, or
settings change; direct Python calls using variable-size arrays have their own compilation behavior.

Fixed blocks bound the size of the numerical test kernels, not total process memory. Input data, host genotype
windows, assembled results, permutation state batches, and compiled executables also consume memory. Measure
peak resident memory with the same backend and settings intended for the full analysis.
