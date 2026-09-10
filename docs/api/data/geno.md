# Genotype I/O

See [Genotype data](../../guide/genotypes.md) for supported sources and file requirements.

Genotype adapters provide:

- per-variant metadata (chromosome, position, alleles, IDs)
- efficient iterators over genotype matrices for cis windows or trans chunks

The mapping layer converts genotype blocks to JAX arrays and preserves `a1` as the counted allele.

!!! note "Metadata accompany every read"

    jaxQTL requests variant metadata with genotype values so output allele and position fields remain aligned with
    matrix columns.

## Genotype loading

::: jaxqtl.io.GenotypeReadOptions

---

::: jaxqtl.io.load_genotype_dataset
