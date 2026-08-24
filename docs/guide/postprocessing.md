# Post-process cis results

A cis scan reports at most one row per phenotype that reaches association testing. It retains a row when testing
occurs but produces no finite variant-level p-value. Validate and filter the rows before applying a study-level
multiple-testing procedure.

## Combine compatible result files

Combine only files produced with the same model, variant-level test, and gene-level calibration method. Their schemas
then have the same model- and calibration-specific columns.

```python
from pathlib import Path

import polars as pl

paths = sorted(Path("result/cis").glob("**/*.cis.score.perm.parquet.gz"))
if not paths:
    raise FileNotFoundError("no cis result files found under result/cis")

results = pl.concat((pl.read_parquet(path) for path in paths), how="vertical")
```

## Select interpretable results

Apply the validity and convergence checks in this order:

1. Keep rows where `result_valid` is true.
2. Keep rows where `model_converged` is true.
3. For Beta-permutation results, keep rows where `perm_converged` is true.

```python
required = {"result_valid", "model_converged", "pvalue_adj"}
missing = required.difference(results.columns)
if missing:
    raise ValueError(f"missing required cis columns: {sorted(missing)}")

valid = results.filter(
    pl.col("result_valid")
    & pl.col("model_converged")
)
if "perm_converged" in valid.columns:
    valid = valid.filter(pl.col("perm_converged"))

if valid.is_empty():
    raise ValueError("no valid converged cis results remain after filtering")

valid.write_parquet("result/cis/combined.valid.parquet")
```

!!! warning "Convergence is a numerical check"

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

See [Output schemas](../reference/outputs.md) for the available columns and
[Failures and convergence](../reference/diagnostics.md) for skipped and invalid-result behavior.
