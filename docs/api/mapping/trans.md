# Trans mapping

`jaxqtl` supports trans-eQTL mapping by streaming genotype variants in chunks and testing each variant against many
phenotypes.

## Chunked scanning

Trans scans can be large (millions of variants times thousands of phenotypes). `jaxqtl` supports streaming:

- iterate over genotype blocks of size `chunk_size`
- compute association statistics for all phenotypes in each block
- write or post-process block outputs incrementally

## API

::: jaxqtl.map.map_trans

---

::: jaxqtl.map.get_trans_schemas
