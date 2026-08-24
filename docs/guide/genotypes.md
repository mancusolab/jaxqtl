# Genotype data

Each mapping command requires exactly one genotype source:

| Option | Input |
| --- | --- |
| `--bfile PREFIX` | PLINK 1 BED/BIM/FAM files |
| `--pfile PREFIX` | PLINK 2 PGEN/PVAR/PSAM files |
| `--vcf PATH` | Indexed VCF or BCF file |
| `--bgen PATH` | BGEN file |

`--dosage` requests dosages instead of hard calls. Without it, jaxQTL reads hard-call allele counts.

!!! note "The counted allele is `a1`"

    jaxQTL preserves the `genoio` allele convention: genotype values count `a1`. Output effect estimates therefore
    describe a one-unit increase in the `a1` count or dosage.

Use `--maf FLOAT` to exclude variants below a minimum minor-allele frequency. The filter is applied by the genotype
reader before cis regions or trans blocks are converted to JAX arrays.

The legacy `--geno` option is rejected. Use one of the four format-specific options above.

See [Input formats](../reference/inputs.md#genotypes) for the required companion files and metadata fields.
