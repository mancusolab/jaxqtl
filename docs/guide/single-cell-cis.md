# Single-cell cis-eQTL workflow

jaxQTL runs cell-type-specific eQTL scans from donor-level pseudobulk expression. It starts after cell quality control,
donor assignment, and cell-type annotation. It does not read cell-level AnnData or Seurat objects, assign cell types,
or create pseudobulk counts.

The workflow is:

1. Create one donor-by-gene phenotype matrix per cell type.
2. Prepare donor covariates and genotypes.
3. Choose an offset for the count model.
4. Run one cis scan per cell type.
5. Filter and correct the results for multiple testing.

## 1. Create one pseudobulk matrix per cell type

For each cell type:

1. Choose which donor–cell-type combinations pass your cell-level quality-control rules.
2. Sum the expression value for each gene across the retained cells from each donor.
3. Omit donors with no retained cells or zero total abundance for that cell type.
4. Write one phenotype file for the cell type.

Summation preserves the library-size exposure used by the count model. Do not average cells. Fractional abundance
estimates from a quantifier are valid; the phenotype values do not need to be integers.

!!! note "jaxQTL does not store the cell-type label"

    Put the cell type in the phenotype filename and output prefix. Run a separate command for every cell type.

### Add gene coordinates

The phenotype file is a BED-like matrix. Each row contains a gene and each sample column contains one donor's
pseudobulk value:

```text
#Chr  start  end    Geneid          donor_1  donor_2
1     29553  29554  ENSG00000243485 12.5     8.0
1     69090  69091  ENSG00000186092 31.0     14.0
```

Calculate the strand-aware transcription start site (TSS) before writing the file. Encode it as a one-base,
zero-based BED interval: `start = TSS - 1` and `end = TSS`, where `TSS` is the one-based genomic position.

Chromosome labels must match the genotype source. For example, do not mix `chr1` and `1`. Reported `tss_distance`
uses genomic coordinate direction; its sign is not reoriented by gene strand.

!!! warning "Keep the full gene matrix when computing offsets"

    `--set-offset-from-libsize` sums every gene present in the phenotype file. If you write a chromosome-only or
    otherwise restricted file, compute the log-library-size offset from the full pseudobulk matrix and pass it with
    `--offset`.

## 2. Prepare donors, covariates, and genotypes

Donor IDs must use the same spelling in the phenotype columns, genotype samples, covariate table, and optional offset
table. jaxQTL analyzes their intersection in genotype order. Compare the IDs before running so an unexpected mismatch
does not reduce the sample size.

A typical covariate table may contain ancestry PCs, age, sex, study batches, and other prespecified variables. If the
analysis uses expression PCs, compute them separately for each cell type because each pseudobulk matrix is different.

```text
iid      age  sex  genotype_pc1
donor_1  42   F    -0.013
donor_2  57   M     0.021
```

See [Covariates and sample alignment](covariates.md) for encoding and normalization options and
[Genotype data](genotypes.md) for supported genotype formats.

## 3. Choose the offset

For Poisson and Negative Binomial models, the usual pseudobulk offset is the log of the donor's total abundance for
that cell type.

- Use `--set-offset-from-libsize` when the phenotype file contains the full, unfiltered gene matrix.
- Use `--offset PATH` when the phenotype file has already been restricted. The file must contain `iid` and one
  log-scale offset column.

Never pass raw library sizes as offsets. See [Offsets](offsets.md) for the full contract.

## 4. Run one cis scan

This command fits a Negative Binomial model, uses a score test, and calibrates the gene-level p-value by permutation:

```bash
mkdir -p results

jaxqtl cis \
  --bfile genotype/chr1 \
  --pheno pseudobulk/CD4_NC.bed.gz \
  --covar covariates.tsv \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --one-hot \
  --normalize-covar \
  --maf 0.05 \
  --nperm 1000 \
  --out results/CD4_NC.chr1
```

The `0.05` MAF threshold is an example, not a jaxQTL default. Choose and report the threshold required by the study.
Use the same analysis specification for every cell type unless the study design requires otherwise.

The command writes `results/CD4_NC.chr1.cis.score.perm.parquet.gz`. See [Cis mapping](cis.md) for ACAT and SPA
alternatives.

## 5. Process results

Use [Post-process cis results](postprocessing.md) to retain valid converged results and apply a study-level
false-discovery-rate procedure.

jaxQTL does not add a cell-type column to the output. Add that label from the input filename or job metadata before
combining results across cell types.

Define the multiple-testing family before looking at results. For example, decide whether FDR is controlled separately
within each cell type or jointly across all tested cell types, then record that choice with the final results.

## Before running production data

- Each phenotype file represents one cell type.
- Every retained donor has cells and positive total abundance for that cell type.
- TSS intervals and chromosome labels match the genotype build and naming convention.
- Donor IDs match across all inputs.
- The offset was computed from the full gene set.
- Covariates and expression filters were chosen before association testing.
- The MAF threshold, cis window, calibration method, and FDR family are recorded.
