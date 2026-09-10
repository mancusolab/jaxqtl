# Genotype data

jaxQTL uses [genoio](https://mancusolab.github.io/genoio/) for genotype input/output.

## Choose an input format

Each mapping command requires exactly one of the following source options. For PLINK inputs, pass the shared
filename prefix; for VCF/BCF and BGEN, pass the file path.

| Source option | Required input | With `--dosage` |
| --- | --- | --- |
| `--bfile PREFIX` | PLINK1: `.bed`, `.bim`, and `.fam` | Unsupported; hard calls only |
| `--pfile PREFIX` | PLINK2: `.pgen`, `.pvar` (or `.pvar.zst`), and `.psam` | Reads stored biallelic dosages |
| `--vcf PATH` | Indexed VCF or BCF | Reads `FORMAT/DS` instead of `FORMAT/GT` |
| `--bgen PATH` | BGEN | Required; reads expected allele counts from probabilities |

BGEN support covers v1.2+ Layout 2 biallelic diploid records. Sample IDs must be embedded in the BGEN file or
provided in a same-prefix `.sample` file so they can be aligned with phenotype and covariate IDs.

## Choose hard calls or dosages

**`--dosage` is a separate flag used alongside a source option**, for example `--vcf cohort.vcf.gz --dosage`
or `--bgen cohort.bgen --dosage`. It takes no value and does not select a file format.

Without it, jaxQTL requests hard-call allele counts (0, 1, or 2 for diploid genotypes). Dosages are expected allele
counts and can be fractional, preserving uncertainty in imputed genotypes. The reader does not convert dosages
into hard calls; VCF dosage reads require `DS`, rather than deriving values from `GP`.

To use imputed VCF dosages in the [Quickstart](quickstart.md#run-a-cis-scan), replace its
`--bfile tutorial/input/chr22_N100` argument with `--vcf cohort.vcf.gz --dosage` and supply matching phenotype
and covariate files.

!!! note "The counted allele is `a1`"

    jaxQTL preserves the `genoio` allele convention: genotype values count `a1`. Output effect estimates therefore
    describe a one-unit increase in the `a1` count or dosage.

## Filter and match inputs

Use `--maf FLOAT` to exclude variants below a minimum minor-allele frequency. The filter is applied by the genotype
reader before cis regions or trans blocks are converted to JAX arrays.

By default, mapping retains expression phenotypes only on exact chromosome labels present in the genotype input. Use
`--chr LABEL` to restrict both expression phenotypes and genotype variants to one chromosome. Labels are matched
exactly: for example, `chr22` and `22` are different labels. The requested label must occur in both inputs.

Variant metadata must provide chromosome, position, identifier, and `a0`/`a1` alleles through the selected
`genoio` adapter. jaxQTL computes allele frequency and minor-allele count from the loaded genotype values.
