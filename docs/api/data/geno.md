# Genotype Data

The production CLI supports genotype inputs through:

- `--bfile` (PLINK1 BED/BIM/FAM prefix)
- `--pfile` (PLINK2 PGEN/PVAR/PSAM prefix)
- `--vcf` (indexed VCF/BCF)
- `--bgen` (BGEN)

`--geno` is deprecated and raises an error. Use one of the above flags instead.

Genotype adapters provide:

- per-variant metadata (chromosome, position, alleles, IDs)
- efficient iterators over genotype matrices for cis windows or trans chunks

Developer note: the internal engine converts `genoio` dense NumPy hardcall reads into JAX arrays and returns metadata with
`a1` as the counted allele.

!!! note

    The genotype IO modules depend on optional IO dependencies (e.g. `polars`, `pandas`, PLINK/VCF readers). In
    minimal documentation builds, the API is described here conceptually; consult the source for concrete classes:
    `src/jaxqtl/io/_geno_engine.py` and `src/jaxqtl/io/__init__.py`.

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
