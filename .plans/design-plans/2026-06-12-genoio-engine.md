# Genoio Genotype IO Engine Design

## Status
Ready for Implementation Planning

## Handoff Decision
- Current decision: ready for implementation planning
- Ready for implementation: yes
- Blocking items: none

## Metadata
- Date: 2026-06-12
- Slug: genoio-engine
- Artifact Directory: `.plans/design-plans/artifacts/2026-06-12-genoio-engine`

## Summary
jaxQTL will replace its current PLINK1-centered genotype IO path with a new internal engine backed by `genoio.Dataset`. The first implementation is deliberately narrow: existing `--bfile` workflows should route through the new engine while preserving current cis, nominal, and trans mapping contracts, output schemas, and JAX-facing array behavior. The engine owns all `genoio` interaction, metadata normalization, sample-order validation, missing/monomorphic handling, and NumPy-to-JAX conversion so downstream mapping code remains unchanged.

The migration is gated by a standalone benchmark harness rather than pytest alone. Before changing genotype IO, the current engine must produce a saved baseline of downstream Parquet outputs plus wall-time and peak-RSS metrics. After migration, the same harness must compare new outputs against the baseline and record performance deltas. Broader format support for PLINK2, VCF/BCF, BGEN, dosage, sparse, or haplotype reads stays out of scope until a later design/update defines their JAX-facing representations and CLI contracts.

## Problem Statement
jaxQTL currently has a narrow genotype IO layer centered on a complete PLINK1 reader and a partial VCF placeholder. This limits format coverage and leaves genotype loading behavior split across local parsing code, while a separate `genoio` package now provides a unified engine for PLINK1, PLINK2, VCF/BCF, and BGEN reads. The design must replace jaxQTL's internal genotype IO engine with a `genoio.Dataset` backed abstraction without changing association-test semantics or broadening the CLI surface prematurely.

## Definition of Done
Design a new internal genotype IO engine for jaxQTL built around `genoio.Dataset`, replacing the current `PlinkData`/partial `VCFData` implementation behind the existing genotype interface. The first implementation should wire the existing `--bfile` CLI path through the new engine so current cis, nominal, and trans workflows exercise `genoio` end to end. The design should preserve current jaxQTL mapping contracts (`query_cis`, chunked genotype iteration, IID alignment, variant metadata shape) while creating a clean path to add PLINK2, VCF/BCF, BGEN, dosage, and filter pushdown later. Out of scope for this design: a broad CLI redesign, full exposure of every `genoio` backend, sparse genotype plumbing, and changing association-test behavior.

Before replacing genotype IO, capture a saved current-engine benchmark baseline and use it after the `genoio` migration to compare downstream Parquet equivalence plus wall-time and peak-memory metrics.

## Goals and Non-Goals
### Goals
- Introduce a cohesive `genoio`-backed genotype engine boundary inside `jaxqtl.io` while preserving the current mapping-facing genotype contract.
- Route the existing `--bfile` CLI path through the new engine in the first implementation so PLINK1 workflows exercise `genoio` end to end.
- Preserve dense JAX array inputs for cis, nominal, and trans mapping code even though `genoio` returns NumPy arrays at the IO boundary.
- Normalize `genoio` sample and variant metadata into the jaxQTL schemas currently consumed by `ReadyDataState`, `map_cis`, and `map_trans`.
- Require before/after benchmark evidence: downstream Parquet equivalence plus wall-time and peak-RSS metrics from the standalone benchmark harness.

### Non-Goals
- Do not redesign the full genotype CLI surface in this design.
- Do not expose PLINK2, VCF/BCF, BGEN, dosage, haplotype, or sparse reads as first-class jaxQTL features in the first implementation.
- Do not change association-test, GLM, score/Wald/SPA, ACAT, or permutation behavior.
- Do not move jaxQTL mapping/inference code away from JAX arrays.
- Do not turn the benchmark harness into pytest coverage.

## Existing Patterns
- `ReadyDataState.from_data()` aligns genotype samples, expression, covariates, and offsets on `iid`, then freezes genotype order by calling `replace_individuals()` on the genotype backend.
- Current mapping code expects genotype matrices with shape `(n_samples, n_variants)` and already operates on JAX arrays returned by genotype IO.
- Current cis mapping handles empty cis windows by receiving a zero-variant matrix and skipping the gene in `map_cis()`.
- Current trans mapping streams genotype blocks through `ReadyDataState.iter_geno()` to limit memory pressure in large scans.
- The `io` package is already a durable seam with genotype, phenotype, and utility modules. A new genotype-engine module is justified as a distinct side-effect boundary and a stable compatibility contract, not as a file-per-concept split.
- The benchmark harness is intentionally outside pytest because it writes outputs, records performance metrics, and compares before/after artifacts rather than asserting small unit behavior.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: the target behavior is an internal port from jaxQTL's current genotype IO interface to a new `genoio`-backed implementation, preserving downstream mapping outputs.
- User selection confirmation: user confirmed `genoio-engine` scope and requested benchmark/equivalence baseline before genotype IO changes.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| N/A | N/A | N/A | No statistical model specification changes are part of this IO-engine design. | high |

## Model Option Analysis (Required When `suggested-model`)
| Candidate ID | Model Family | When It Fits | Key Assumptions | Failure Modes | Supporting Citation(s) | Selection Status |
| --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: replace jaxQTL's genotype IO backend with a `genoio.Dataset` backed engine while preserving current `--bfile` CLI behavior and current mapping output contracts.
- Source selection confirmation: local `jaxqtl` codebase and local sibling `genoio` package were inspected during design kickoff.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | `/Users/nicholas/Projects/jaxqtl` | working tree | Current consumer code and CLI surface. |
| PORT-SRC-2 | local-directory | `/Users/nicholas/Projects/genoio` | working tree | New genotype IO engine API. |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface (`cli`/`api`/`numerics`/`io`) | Current Behavior | Target Behavior | Evidence Plan (tests/golden outputs) |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | cli | `jaxqtl --bfile` uses current PLINK1 reader through `PlinkData`. | `jaxqtl --bfile` uses a `genoio` backed engine with the same downstream output schema. | Benchmark harness in `scripts/benchmark_genotype_io.py`; compare Parquet outputs against saved baseline. |
| PORT-BHV-2 | io | `GenotypeData.query_cis()` returns `(n_samples, n_variants)` dense genotype matrix plus jaxQTL-shaped variant metadata. | New engine preserves this contract while delegating reads and metadata to `genoio.Dataset`. | Focused engine tests using tutorial fixtures and/or `genoio` fixtures. |
| PORT-BHV-3 | io | `GenotypeData.iter_geno()` yields variant chunks for trans scans. | New engine preserves chunked iteration semantics through `genoio.Dataset.iter_blocks()`. | Unit/integration tests plus trans benchmark case when needed. |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | jaxQTL genotype IO | `PlinkData` is the complete current genotype backend; `VCFData` is partial and not a full `GenotypeData` implementation. | `src/jaxqtl/io/_geno.py:79`, `src/jaxqtl/io/_geno.py:169` | confirmed |
| PORT-INV-2 | jaxQTL mapping | Mapping code depends on `query_cis`, `iter_geno`, `sample_info`, and jaxQTL variant metadata columns. | `src/jaxqtl/map/data.py:91`, `src/jaxqtl/map/cis.py:40`, `src/jaxqtl/map/trans.py:19` | confirmed |
| PORT-INV-3 | genoio API | `genoio.Dataset` supports dense reads, sample/variant metadata, region iteration, and block iteration. | `/Users/nicholas/Projects/genoio/src/genoio/_api.py:76`, `/Users/nicholas/Projects/genoio/src/genoio/_api.py:446`, `/Users/nicholas/Projects/genoio/src/genoio/_api.py:556` | confirmed |
| PORT-INV-4 | benchmark baseline | A standalone benchmark harness now captures current downstream outputs, wall time, and peak RSS for before/after comparison. | `scripts/benchmark_genotype_io.py:1`, `scripts/benchmark_core.py:1` | addition |

## External Research Findings (When Triggered)
| Claim ID | Claim | Source URL | Source Type | Access Date | Confidence (high/med/low) |
| --- | --- | --- | --- | --- | --- |
| N/A | No external research was triggered; design inputs came from local jaxQTL and genoio codebases. | N/A | N/A | 2026-06-12 | high |

## Mathematical Sanity Checks
- Summary: no statistical model or inference equation changes are part of this design. The migration is an IO-engine replacement whose mathematical sanity criterion is downstream output parity for unchanged association workflows.
- Blocking issues: none for model mathematics.
- Accepted risks: floating-point results should remain identical or within tolerance for the first `--bfile` dense hardcall path. Any difference must be explained by genotype matrix values, sample order, variant metadata, or missing/monomorphic handling before it is accepted.

Detailed artifacts:
- `.plans/design-plans/artifacts/2026-06-12-genoio-engine/model-symbol-table.md`
- `.plans/design-plans/artifacts/2026-06-12-genoio-engine/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: preserve current JAX inference behavior.
- Chosen strategy: no solver strategy change.
- Why this strategy: genotype IO replacement should be benchmarked independently from solver or statistical changes.

## Solver Translation Feasibility
- Summary: not applicable; the design does not translate or replace solvers.
- Blocking constraints: none.
- Custom-solver rationale (if chosen): not applicable.

Detailed artifact:
- `.plans/design-plans/artifacts/2026-06-12-genoio-engine/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: the CLI continues accepting the existing `--bfile` path for first-phase integration. The CLI constructs a new `genoio`-backed genotype engine for this path.
- Rejection rules: invalid or unreadable genotype sources should fail through public `genoio` exceptions with added jaxQTL operation context where useful. Unsupported new formats remain out of scope for the first implementation unless added behind internal constructors only.

### Pipeline
- Contract: `ReadyDataState` and mapping code continue to depend on a genotype backend exposing `sample_info`, `replace_individuals(sample_info)`, `query_cis(chrom, start, end)`, and `iter_geno(chunk_size)`.
- Validation-first checks: the genotype engine verifies IID availability, canonical sample order after alignment, dense matrix shape, and normalized variant metadata columns before returning data to mapping.

### Numerics
- Contract: `query_cis()` and `iter_geno()` return dense JAX arrays with shape `(n_samples, n_variants)`. NumPy arrays from `genoio` are converted at the adapter boundary with `jnp.asarray(...)`.
- Result/status semantics: no new convergence or statistical status semantics are introduced. Existing mapping result columns and downstream p-value/effect-size semantics are preserved.

### Egress
- Contract: current cis, nominal, and trans Parquet output schemas remain unchanged for the migrated `--bfile` path.
- Output/exit-code mapping: CLI commands keep their existing success/failure behavior. The benchmark harness separately exits nonzero if subprocess execution fails or if Parquet comparison fails.

## Data Conversion and Copy Strategy
For each source format, record copy mode (`zero-copy`, `mmap`, `single-copy fallback`) and rationale.

- PLINK1 via `--bfile`: `genoio.Dataset.read()`/`iter_blocks()` returns NumPy arrays; jaxQTL converts to JAX arrays at the engine boundary. Treat this as a single-copy fallback into JAX device/host representation, matching the existing pattern where `pandas_plink` output is converted with `jnp.asarray(...)`.
- Future PLINK2, VCF/BCF, and BGEN: same engine boundary should apply when surfaced later. Format-specific optimizations must not leak into mapping code.
- Sparse and dosage reads: out of scope for first implementation; later designs must define the JAX representation and downstream statistical semantics before enabling them.

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
- Sources: genotype sample metadata from `genoio`, expression phenotypes, covariates, and optional offsets.
- Entity key(s) (for example subject/sample ID): `iid`.
- Join type and rationale: preserve existing `align_on_iid()` behavior, which intersects all sources and preserves the genotype/source base order through a canonical IID frame.
- Duplicate-key policy: duplicate IID handling should remain explicit; if duplicates appear in genotype metadata or aligned sample data, the engine or alignment layer should fail before array conversion.
- Missing-key policy: samples missing from any required source are dropped by intersection, matching current behavior.
- Row-order freeze policy: after `ReadyDataState.from_data()` aligns sources, `replace_individuals(sample_info)` freezes the exact genotype IID list used for all subsequent `genoio` reads. Returned sample metadata from reads must match this list.
- Reconciliation accounting (matched/dropped/retained counts): implementation should log or make inspectable the number of retained samples and variants where practical, without changing output schemas.
- Conversion boundary (where reconciled tabular data becomes arrays/PyTrees): genotype matrices become JAX arrays inside the genotype engine; covariates and phenotype data continue converting inside `ReadyDataState` and `ExpressionData`.

## Validation Strategy
- Boundary checks: validate source readability through `genoio`, required sample/variant metadata columns, expected dense matrix layout, and normalized metadata schema.
- Shape/range/domain checks: ensure genotype matrices are two-dimensional, row count equals retained sample count, metadata height equals matrix column count, and empty regions use `(n_samples, 0)` rather than `None`.
- Multi-input alignment checks (key uniqueness, overlap expectations, deterministic row ordering): verify aligned IIDs are unique and that `genoio` read results match the frozen IID order. If `genoio` returns source order for a requested subset, the engine must either request samples in a way that preserves the frozen canonical order or reorder/validate before returning.
- Failure semantics: fail fast with `ValueError` for jaxQTL contract violations and preserve public `genoio` exception types for source/read failures when possible.

## Testing and Verification Strategy
- TDD scope: adapter-level tests for the new genotype engine must be written before or alongside implementation. They should cover sample metadata, `replace_individuals()`, region reads, empty regions, block iteration, JAX conversion, metadata normalization, and monomorphic filtering.
- Regression strategy: focused integration tests should exercise `ReadyDataState.from_data()` and at least one small mapping or CLI smoke path through `--bfile`. The standalone benchmark harness is the before/after downstream-equivalence and performance gate.
- Verification commands:
  - `pytest -p no:capture <focused engine and integration test paths>`
  - `python scripts/benchmark_genotype_io.py --out-dir /tmp/jaxqtl-genoio-baseline --repeats 3`
  - `python scripts/benchmark_genotype_io.py --out-dir /tmp/jaxqtl-genoio-after --compare-dir /tmp/jaxqtl-genoio-baseline --repeats 3`

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Baseline Benchmark Capture
**Goal:** Capture current-engine downstream output and performance before genotype IO changes.

**Components:**
- Standalone genotype IO benchmark harness in `scripts/benchmark_genotype_io.py` and comparison core in `scripts/benchmark_core.py`.
- Saved local baseline output directory containing benchmark JSON and Parquet outputs for default benchmark cases.

**Dependencies:** None.

**Done when:** the benchmark harness runs successfully against the current `--bfile` path, records wall time and peak RSS, and stores current cis and nominal Parquet outputs for later comparison.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Genoio Engine Contract
**Goal:** Add a cohesive `genoio`-backed genotype engine that satisfies the current jaxQTL genotype backend contract without changing mapping code.

**Components:**
- New genotype-engine boundary inside `jaxqtl.io`, likely `src/jaxqtl/io/_geno_engine.py`, responsible for `genoio.Dataset` ownership, sample freezing, dense reads, JAX conversion, metadata normalization, monomorphic filtering, and block/region iteration.
- Compatibility exports from `jaxqtl.io` so the CLI can construct the new engine without exposing unnecessary internals.

**Dependencies:** Phase 1 baseline exists.

**Done when:** adapter-level tests prove the new engine returns JAX arrays and normalized metadata for sample metadata, sample replacement, cis region reads, empty regions, and block iteration.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: CLI `--bfile` Integration
**Goal:** Route the existing PLINK1 CLI path through the new `genoio` engine.

**Components:**
- CLI genotype setup in `src/jaxqtl/cli.py` constructs the new engine for `--bfile`.
- Legacy `--geno` behavior is either kept as an alias to the same engine or explicitly handled according to current deprecation behavior.
- Existing `ReadyDataState` and mapping modules remain contract-compatible and should not learn about `genoio` directly.

**Dependencies:** Phase 2 engine contract and tests.

**Done when:** focused CLI or mapping integration tests pass through the new engine for PLINK1 without changing output schema.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Downstream Equivalence Gate
**Goal:** Prove that the migrated `--bfile` path preserves downstream outputs and records performance deltas.

**Components:**
- Benchmark comparison run using the Phase 1 baseline directory.
- Review of `benchmark.json` wall-time and peak-RSS metrics for cis and nominal default cases.
- Optional trans benchmark run when the implementation touches chunk iteration or trans-specific behavior.

**Dependencies:** Phase 3 CLI integration.

**Done when:** benchmark comparison exits successfully, all default Parquet outputs compare equal within configured tolerances, and performance metrics are recorded for review.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Legacy Cleanup and Documentation
**Goal:** Remove or quarantine obsolete genotype IO paths and document the new engine boundary.

**Components:**
- Legacy `PlinkData`/partial `VCFData` handling in `src/jaxqtl/io/_geno.py` is either removed, reduced to compatibility aliases, or clearly marked as deprecated depending on implementation risk.
- README or developer-facing notes describe the current supported first-phase behavior: existing `--bfile`, dense hardcall, current output schemas, and future extension path.
- Known stale test collection issue in `tests/test_sim_pbeta_acat.py` is fixed, quarantined, or documented as an unrelated blocker before broad test runs are claimed.

**Dependencies:** Phase 4 downstream equivalence gate.

**Done when:** the codebase no longer has misleading partial genotype backends on the active path, documentation matches actual behavior, and verification commands are not blocked by unrelated stale tests without an explicit note.
<!-- END_PHASE_5 -->

## Simulation And Inference-Consistency Validation
- In scope: no
- Simulate entrypoint/signature: not applicable
- Inputs: not applicable
- Outputs: not applicable
- Seed/RNG policy: not applicable

### Assumption Alignment
| Inference Assumption | Simulation Rule | Mismatch Risk | Mitigation |
| --- | --- | --- | --- |
| No statistical inference change | Not applicable | IO migration could still perturb inputs through ordering, missing handling, or metadata changes | Benchmark downstream Parquet equivalence and adapter tests for sample/variant ordering |

### Planned Validation Experiments
| Experiment ID | Type (recovery/SBC/PPC) | Success Criterion | Notes |
| --- | --- | --- | --- |
| N/A | N/A | N/A | Simulation validation is out of scope because no model or inference assumptions change. |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | `genoio` returns retained samples in source order, while jaxQTL must preserve aligned IID order. | High | Freeze canonical IID order after alignment and validate returned read samples against it before array return. | implementation |
| R2 | Variant metadata column names differ (`id` vs `snp`) and allele semantics must remain output-compatible. | High | Centralize metadata normalization and test output schema/equivalence. | implementation |
| R3 | Missing and monomorphic genotype handling may differ from current PLINK1 path. | High | Use explicit missing policy, retain current monomorphic filtering, and compare downstream outputs. | implementation |
| R4 | Performance metrics include JAX startup/JIT variance. | Medium | Treat Parquet equivalence as hard gate; track wall time/RSS for review instead of hard timing failure. | implementation |
| R5 | Stale `tests/test_sim_pbeta_acat.py` breaks full pytest collection. | Medium | Fix/quarantine before claiming broad test-suite success, or use focused test paths with explicit caveat. | implementation |

## Additional Considerations
- Future format support should extend the new engine boundary rather than changing mapping code. PLINK2, VCF/BCF, BGEN, dosage, sparse reads, and filter pushdown should each define explicit JAX-facing representation and output-contract expectations before CLI exposure.
- The design intentionally avoids strict performance thresholds. If later implementation work needs thresholds, set them in the implementation plan based on repeated baseline measurements.

## Acceptance Criteria
### genoio-engine.AC1: Baseline Benchmark Gate
- `genoio-engine.AC1.1`: A current-engine baseline can be produced with `scripts/benchmark_genotype_io.py` for the default benchmark cases.
- `genoio-engine.AC1.2`: The benchmark report includes command, return code, wall time, peak RSS, output paths, and comparison data when `--compare-dir` is used.
- `genoio-engine.AC1.3`: After migration, benchmark comparison against the saved baseline succeeds for default Parquet outputs within configured tolerances.

### genoio-engine.AC2: Engine Contract
- `genoio-engine.AC2.1`: The new engine exposes the current mapping-facing contract: `sample_info`, `replace_individuals(sample_info)`, `query_cis(chrom, start, end)`, and `iter_geno(chunk_size)`.
- `genoio-engine.AC2.2`: `query_cis()` and `iter_geno()` return dense JAX arrays with shape `(n_samples, n_variants)`.
- `genoio-engine.AC2.3`: Empty cis regions return a `(n_samples, 0)` JAX array and empty normalized variant metadata.
- `genoio-engine.AC2.4`: Returned sample order matches the aligned IID order frozen by `ReadyDataState.from_data()`.

### genoio-engine.AC3: Metadata and Filtering Parity
- `genoio-engine.AC3.1`: Variant metadata returned to mapping is normalized to jaxQTL's current internal/output expectations, including `snp`, `chrom`, `pos`, `a0`, and `a1`.
- `genoio-engine.AC3.2`: Monomorphic or nonfinite variant columns are filtered consistently with current cis and trans behavior.
- `genoio-engine.AC3.3`: Missing genotype handling is explicit and preserves current downstream output parity for the migrated PLINK1 path.

### genoio-engine.AC4: CLI Integration Scope
- `genoio-engine.AC4.1`: Existing `jaxqtl cis --bfile ...`, `jaxqtl nominal --bfile ...`, and `jaxqtl trans --bfile ...` paths construct and use the new engine.
- `genoio-engine.AC4.2`: Current output schemas for migrated `--bfile` workflows remain unchanged.
- `genoio-engine.AC4.3`: PLINK2, VCF/BCF, BGEN, dosage, sparse, and haplotype support are not exposed as new production CLI behavior in this first migration unless covered by a later design/update.

### genoio-engine.AC5: Verification and Cleanup
- `genoio-engine.AC5.1`: Focused pytest coverage for the engine and integration paths runs with `pytest -p no:capture`.
- `genoio-engine.AC5.2`: The implementation does not claim full-suite success while `tests/test_sim_pbeta_acat.py` still breaks collection.
- `genoio-engine.AC5.3`: Active genotype IO code and documentation do not leave the partial legacy VCF backend looking like supported production behavior.

## Glossary
- **`genoio`**: external genotype IO package that provides `Dataset` objects for PLINK1, PLINK2, VCF/BCF, and BGEN sources.
- **Genoio engine**: new jaxQTL internal adapter that owns `genoio.Dataset`, converts NumPy matrices to JAX arrays, and presents the current mapping-facing genotype contract.
- **`--bfile`**: existing jaxQTL CLI argument for a PLINK1 BED/BIM/FAM prefix. This is the first CLI path to migrate.
- **Mapping-facing contract**: the genotype backend surface consumed by `ReadyDataState` and mapping code: `sample_info`, `replace_individuals()`, `query_cis()`, and `iter_geno()`.
- **Parquet equivalence**: benchmark comparison that verifies generated downstream result files match a saved baseline within configured numeric tolerances.
- **Peak RSS**: peak resident set size reported by the benchmark harness for the child jaxQTL process.
- **Dense hardcall**: first-phase genotype representation: a dense matrix of called allele counts, not sparse, dosage, or haplotype data.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-06-12 | N/A | Draft | Plan created | |
| 2026-06-12 | Draft | Ready for Implementation Planning | Definition of Done, architecture option, design sections, and acceptance criteria validated by user. | Codex |
