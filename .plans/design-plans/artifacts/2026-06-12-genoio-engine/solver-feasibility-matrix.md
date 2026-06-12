# Solver Feasibility Matrix

## Context
- Plan slug: `genoio-engine`
- Generated date: `2026-06-12`

| Candidate | Problem Form Fit (root/least-squares/minimize) | AD Compatibility | Constraint Handling | Status/Error Mapping | Feasible (yes/no) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A | N/A | No solver changes are part of this genotype IO design. |

## Decision
- Preferred solver path: keep existing jaxQTL solver behavior.
- Reason: design changes genotype IO only.
- Benchmark or validation requirement before implementation: downstream Parquet equivalence through `scripts/benchmark_genotype_io.py`.
