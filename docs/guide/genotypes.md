# Genotype data

Each mapping command requires exactly one genotype source:

| Option | Input |
| --- | --- |
| `--bfile PREFIX` | `PREFIX.bed`, `PREFIX.bim`, and `PREFIX.fam` |
| `--pfile PREFIX` | `PREFIX.pgen`, `PREFIX.pvar`, and `PREFIX.psam` |
| `--vcf PATH` | Indexed VCF or BCF file |
| `--bgen PATH` | BGEN file |

`--dosage` requests dosages instead of hard calls. Without it, jaxQTL reads hard-call allele counts.

!!! note "The counted allele is `a1`"

    jaxQTL preserves the `genoio` allele convention: genotype values count `a1`. Output effect estimates therefore
    describe a one-unit increase in the `a1` count or dosage.

Use `--maf FLOAT` to exclude variants below a minimum minor-allele frequency. The filter is applied by the genotype
reader before cis regions or trans blocks are converted to JAX arrays.

By default, mapping retains expression phenotypes only on exact chromosome labels present in the genotype input. Use
`--chr LABEL` to restrict both expression phenotypes and genotype variants to one chromosome. Labels are matched
exactly: for example, `chr22` and `22` are different labels. The requested label must occur in both inputs.

Variant metadata must provide chromosome, position, identifier, and `a0`/`a1` alleles through the selected
`genoio` adapter. jaxQTL computes allele frequency and minor-allele count from the loaded genotype values.
