# Genotype Data

The production CLI genotype path in this implementation is PLINK1 BED/BIM/FAM through `--bfile`.
`--geno` is deprecated and now raises an error; use `--bfile` for PLINK1 BED/BIM/FAM prefixes.
`--vcf` remains unsupported in production CLI behavior for this migration.

PLINK2, VCF/BCF, BGEN, dosage, sparse, and haplotype inputs are future extension paths behind later design work.

Genotype adapters provide:

- per-variant metadata (chromosome, position, alleles, IDs)
- efficient iterators over genotype matrices for cis windows or trans chunks

Developer note: the internal engine converts `genoio` dense NumPy hardcall reads into JAX arrays and returns metadata with
`a1` as the counted allele.

!!! note

    The genotype IO modules depend on optional IO dependencies (e.g. `polars`, `pandas`, PLINK/VCF readers). In
    minimal documentation builds, the API is described here conceptually; consult the source for concrete classes:
    `src/jaxqtl/io/geno.py`.

## Concepts

### Genotype matrix

Genotypes are represented as a JAX array `G` with shape `(n, m)`:

- `n`: individuals
- `m`: variants

Entries typically represent allele counts (0/1/2).

### Variant metadata

Mapping outputs include variant metadata such as:

- `chrom`, `pos`, `snp` (ID)
- alleles `a0`/`a1`
- allele frequency and minor allele count
