# Trans Mapping

Trans mapping tests variants genome-wide against many phenotypes.

## Chunked execution

Trans scans are streamed in genotype chunks to limit memory usage. A typical workflow is:

1. Choose a chunk size (e.g. 5000 variants).
2. Iterate over chunks and write results incrementally.

See `API → molQTL Mapping → Trans Mapping` for the API entrypoints.
