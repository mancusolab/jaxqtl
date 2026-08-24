# Genotype I/O

The CLI supports genotype inputs through:

- `--bfile` (PLINK1 BED/BIM/FAM prefix)
- `--pfile` (PLINK2 PGEN/PVAR/PSAM prefix)
- `--vcf` (indexed VCF/BCF)
- `--bgen` (BGEN)

`--geno` is deprecated and raises an error. Use one of the above flags instead.

Genotype adapters provide:

- per-variant metadata (chromosome, position, alleles, IDs)
- efficient iterators over genotype matrices for cis windows or trans chunks

The mapping layer converts genotype blocks to JAX arrays and preserves `a1` as the counted allele.

!!! note "Metadata accompany every read"

    jaxQTL requests variant metadata with genotype values so output allele and position fields remain aligned with
    matrix columns.

## API

::: jaxqtl.io.GenotypeReadOptions

---

::: jaxqtl.io.load_genotype_dataset
