# Post-process cis results

A cis scan reports at most one row per phenotype that reaches association testing. It retains a row when testing
occurs but produces no finite variant-level p-value. Validate and filter the rows before applying a study-level
multiple-testing procedure.

## Combine compatible result files

Combine only files produced with the same model, variant-level test, and gene-level calibration method. Their schemas
then have the same model- and calibration-specific columns.

Cell type is not an output column. For a cell-type-specific study, add it from the phenotype filename or job metadata
before concatenating files from different cell types.

```python
from pathlib import Path

import polars as pl

pattern = "**/*.cis.score.spa.acat.parquet.gz"
# For Beta permutation, use "**/*.cis.score.perm.parquet.gz".
paths = sorted(Path("result/cis").glob(pattern))
if not paths:
    raise FileNotFoundError("no cis result files found under result/cis")

results = pl.scan_parquet(paths)
```

## Select interpretable results

Require valid, converged fits and finite `pvalue_adj` values in [0, 1]. Beta-permutation results also require
`perm_converged`:

```python
columns = results.collect_schema().names()
valid = results.filter(
    pl.col("result_valid")
    & pl.col("model_converged")
    & pl.col("pvalue_adj").is_finite()
    & pl.col("pvalue_adj").is_between(0.0, 1.0)
)
if "perm_converged" in columns:
    valid = valid.filter(pl.col("perm_converged"))

valid = valid.collect()
if valid.is_empty():
    raise ValueError("no valid converged cis results remain after filtering")

valid.write_parquet("result/cis/combined.valid.parquet")
```

!!! warning "Check model adequacy before interpreting discoveries"

    These filters remove failed fits and calibrations. They do not establish that the response family, covariates,
    offset, or testing procedure is appropriate for the study.

Keep the rejected rows for diagnostics. The `failure_reason` column distinguishes an invalid association from a valid
association with a large p-value.

## Control the study-level false discovery rate

`pvalue_adj` is the gene-level p-value produced within each cis window. After collecting all intended phenotypes for
the analysis, apply the study's chosen false-discovery-rate procedure across `pvalue_adj`. Do not substitute the lead
variant's nominal `pvalue` for this step.

jaxQTL does not impose a particular FDR implementation. Record the method, tested phenotype set, and threshold with
the final results so the discovery set can be reproduced.

For analyses with multiple cell types, define the testing family before inspecting results. State whether FDR is
controlled separately within each cell type or jointly across all tested cell types.

See [Output schemas](../reference/outputs.md) for the available columns and
[Troubleshooting](troubleshooting.md) for skipped and invalid-result behavior.
