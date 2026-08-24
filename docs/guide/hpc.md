# Scale genome-wide scans

Large cis scans can be divided by chromosome and gene list, then submitted as independent scheduler jobs. Start with
small chunks and use observed runtime and memory to choose the final chunk size.

The repository includes:

- `tutorial/code/create_genelist_dir.R`, which demonstrates gene-list chunk creation.
- `tutorial/code/run_jaxqtl_cis_all.sh`, which demonstrates a batch cis workflow.

!!! note "Separate compilation time from scan time"

    JAX compiles a new numerical program for new array shapes. The first gene or genotype block can therefore be
    slower than later work with the same shape. Estimate production runtime from multiple genes, not the first one.

## Recommended workflow

1. Split phenotypes by chromosome if the input pipeline supports chromosome-specific files.
2. Divide gene IDs into scheduler-sized lists.
3. Run one `jaxqtl cis` process per chromosome and gene-list pair.
4. Concatenate Parquet results after every job finishes.
5. Filter on `result_valid` before interpreting lead associations.
6. For permutation calibration, inspect both `model_converged` and `perm_converged`.
7. Apply the study-level FDR procedure to `pvalue_adj`.

!!! warning "Do not recompute offsets from a restricted gene file"

    Chromosome- or chunk-specific phenotype files do not contain the full library. Supply offsets computed from the
    original unfiltered count matrix rather than using `--set-offset-from-libsize` in each job.
