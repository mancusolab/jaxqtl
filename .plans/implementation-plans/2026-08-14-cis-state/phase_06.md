# Cis-by-state Phase 6 Implementation Plan

**Goal:** Connect raw focal counts, LOCO state artifacts, genotypes, required donor covariates, absorbed inference, explicit calibration, and bounded atomic output without changing existing pseudobulk workflows.

**Architecture:** Add a pure cis-state I/O contract for reconciliation, buckets, row schemas, and manifests; keep Parquet/artifact access and atomic publication in one I/O shell; add a distinct streaming map shell for genoio regions and per-gene execution. Compose the Phase 3-5 array cores once per gene through the existing calibration module and expose a dedicated CLI adapter.

**Tech Stack:** Python 3.11+, JAX 0.10+, Equinox, Polars, PyArrow, SciPy sparse, genoio 0.2+, NumPy, pytest, Hypothesis, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 6 of 7.

**Codebase verified:** 2026-08-14 17:41 PDT

---

## Review Profile

- Profile: `full`
- Why this profile: This phase joins every numerical and inference component through multi-input reconciliation, a public array API, genoio streaming, atomic output, and a dedicated CLI.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC5: CLI and artifacts

- **cis-state.AC5.1 Success:** `state-factor` writes a complete versioned artifact for one chromosome or chromosomes 1-22.
- **cis-state.AC5.2 Failure:** Incomplete, incompatible, or misaligned factor artifacts are rejected.
- **cis-state.AC5.3 Success:** Cell, gene, donor, factor, genotype, and covariate reconciliation is deterministic and fully accounted.
- **cis-state.AC5.4 Success:** `cis-state` writes bounded Parquet batches containing inference and failure diagnostics.
- **cis-state.AC5.5 Contract:** CLI success, usage failure, and runtime failure map to documented exit codes.
- **cis-state.AC5.6 Regression:** Existing commands, outputs, and tests remain unchanged.

Phase 7 owns default calibration selection. Phase 6 requires the method explicitly.

---

<!-- START_TASK_1 -->
### Task 1: Pin the genoio workflow contract

**Verifies:** None

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/pyproject.toml`
- Modify: `/Users/nicholas/Projects/jaxqtl/uv.lock`

**Implementation:**
- Raise the direct runtime dependency from `genoio>=0.1.0` to `genoio>=0.2`, the minimum version whose source-ordered sample filtering, lazy `iter_regions`, hard-call/dosage selection, missing-data policy, and returned-variant contracts are used by this phase.
- Regenerate the lock without unrelated dependency churn and record the resolved genoio version in cis-state run manifests. Do not add a second genotype adapter or reimplement genoio parsing.

**Verification:**
- Run: `uv lock`
- Expected: lock generation succeeds and records genoio 0.2 or newer as the jaxqtl runtime requirement.
- Run: `uv run python -c "from importlib.metadata import version; print(version('genoio'))"`
- Expected: the installed version is 0.2 or newer.

**Commit:** `build: require genoio 0.2`

Stage only `/Users/nicholas/Projects/jaxqtl/pyproject.toml` and `/Users/nicholas/Projects/jaxqtl/uv.lock`; never stage `/Users/nicholas/Projects/jaxqtl/state.md`.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Extend sparse ingress for focal-gene and cell-level columns

**Verifies:** cis-state.AC5.2, cis-state.AC5.3

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell_contract.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_single_cell.py` (unit/integration)

**Implementation:**
- Preserve the Phase 1 functional-core classification in `_single_cell_contract.py` and imperative-shell classification in `_single_cell.py`.
- Extend `load_sparse_single_cell` with ordered optional `extra_cell_columns` and `extra_gene_columns`. Project these columns in the same lazy Parquet scans and collect each metadata file once; reject duplicate requests and collisions with required fields.
- For this workflow, load `tss` as an extra gene column and validate it as non-null positive integer 1-based coordinates. Rows remain cells and columns remain genes in matrix-index order; do not reread, join, or transpose the sparse count matrix.
- Load optional cell covariate, offset, and library-size columns from the same cells Parquet selected by the user. Keep the initial interface Parquet-only; do not introduce TSV/AnnData input in this branch.
- Preserve the exact raw-file hashes, original matrix indices, selected-cell provenance, and Phase 2 state-artifact alignment identities. Extra projected values may not redefine any canonical identity/order hash.

**Testing:**
- Add failing tests first for ordered extra projection, missing/duplicate extra names, `tss` null/noninteger/nonpositive values, and exact preservation of matrix order and canonical hashes.
- Verify only requested columns are collected, shuffled metadata is restored by `matrix_index`, and one CSR column can be densified to the correct focal `(M,)` outcome without materializing an `M x genes` dense array.
- Exercise generated valid/invalid projected column lists with bounded Hypothesis strategies; keep iterative numerical behavior out of property tests.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_single_cell.py -x`
- Expected red: extra-column and focal-`tss` assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_single_cell.py`
- Expected green: original ingress tests and new projection tests pass.

**Commit:** `feat: project cis-state metadata columns`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Compose the array-level cis-state test once per gene

**Verifies:** cis-state.AC5.4

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_calibration.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cis_state_workflow.py` (unit/integration)

**Implementation:**
- Preserve `_cluster_calibration.py` as `# pattern: Functional Core` and the import-only shell classification in `hypothesis/__init__.py`.
- Export immutable backend-neutral `CisStateInteractionResult` and `test_cis_state_interaction(y, G, S, donor_index, offset, donor_covariates, key, *, covariates=None, min_informative_donors, solver, nb_config, variant_block_size, variant_capacity=None, calibration, calibration_config)`.
- Make this public composition boundary enforce the complete inference dtype contract once: require `jax.config.x64_enabled`, promote accepted real outcome/design/genotype/state/covariate/offset inputs to float64 before the first fit, preserve validated integer donor indices, and reject bool/complex arrays. All nested Phase 3-5 calls receive float64, and compact diagnostics record x64 enabled, resolved backend, and dtype.
- Keep `donor_covariates` required. Validate that `calibration` is explicitly `cluster-bootstrap` or `cluster-moment` and is paired with the matching Phase 5 config; provide no method default in this phase and accept no covariance, variant-weight, or state-weight arguments.
- Fit `fit_absorbed_nb` exactly once. If the null is not inferentially successful or has fewer than the explicitly required informative donors, return a stable failure with `pvalue=NaN`; otherwise build the Phase 4 score state once and run only the selected Phase 5 calibration without refitting.
- If `variant_capacity` is supplied, require a positive multiple of `variant_block_size` at least as large as input `p`, host-pad genotype columns with finite zeros to that capacity, and pass an `input_variant_mask` that marks padding false. Capacity padding is neutral storage, not a scientific monomorphic/drop count. With `None`, pad only the final computational block.
- Return the common statistic/p-value/calibration/status plus compact immutable null, score, and calibration diagnostic summaries. Do not nest the full Phase 3/4/5 result objects: keep their cell-level arrays, `M x d` nuisance design, per-variant scores, and donor contributions private and ephemeral. Keep the compact public host result structure fixed across numeric failure statuses; transformed internals communicate runtime outcomes only through array-valued codes, with enum/message conversion after `device_get`.

**Testing:**
- Write API composition tests first and observe assertion failures before implementation.
- Use one narrow injected/private counter only for the scientific fit-once/no-null-refit invariant; do not assert other helper call structure. Make numerical/output equivalence and unchanged null diagnostics the primary assertions, and verify method/config mismatches fail before numerics.
- Compare direct unpadded execution with multiple valid capacities; assert identical scientific variant counts, scales, statistic, and calibrated result for the same key. Cover capacity underflow, nonmultiple capacity, all-padding prohibition, null failure, informative-donor failure, and both calibration branches.
- Verify donor covariates are required, no weight/covariance parameters are accepted, failures carry no nominal p-value, private numerical kernel PyTrees are eager/JIT stable, host code-to-message mapping is complete, and the public high-level host result contains no cell-, variant-, or donor-contribution arrays. Do not claim the string-bearing public wrapper itself is jittable.
- In isolated subprocess tests, disable x64 and require failure before fitting/random draws; with x64 enabled require float32 direct inputs to promote to float64 and match explicit float64 results. Reject direct internal float32 launches and assert compact dtype/backend diagnostics.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cis_state_workflow.py -x`
- Expected red: composition, fit-once, and capacity-mask cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cis_state_workflow.py`
- Expected green: array-level workflow tests pass.

**Commit:** `feat: compose cis-state interaction test`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Define deterministic reconciliation, bucket, row, and run-manifest contracts

**Verifies:** cis-state.AC5.2, cis-state.AC5.3, cis-state.AC5.4

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_cis_state_contract.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_cis_state_io.py` (unit)

**Implementation:**
- Start `_cis_state_contract.py` with `# pattern: Functional Core`; keep the public import file classified as an imperative shell. Colocate immutable ready-data/accounting records, pure validation/reconciliation, bucket selection, fixed result-row schema/formatting, run-manifest schema version `1`, JSON conversion, and hash contracts. Do not create schema, type, validation, or manifest leaf modules.
- Define `VariantBucketConfig(block_size, capacities)` with positive, sorted, unique capacities that are multiples of the block size. Provide the engineering-only initial CLI setting `block_size=64`, capacities `(64, 128, 256, 512, 1024, 2048, 4096)`; label it compile-shape control, not a scientific default. Select the smallest capacity at least as large as regional `p`; never truncate, and return an accounted `variant_bucket_overflow` when none fits.
- Validate donor covariates from one Parquet table with unique, nonempty, case-sensitive `donor_id` and one or more explicitly ordered numeric finite columns. Require complete coverage of every final donor; report/exclude covariate-only rows. Categorical values, including sex, must be encoded numerically upstream.
- Accept optional explicitly ordered numeric finite cell covariates from the cells Parquet. Define mutually exclusive offsets: a finite already-log `offset_column`, a positive finite `library_size_column` transformed with log, or a default log of full raw-CSR row sums computed once before focal/state-gene filtering. Require positive derived sums for every retained cell.
- Reconcile in this order: validate state artifact against exact raw counts/cells/genes SHA-256 and canonical orders; get genoio `iid` sample IDs in source order; reject duplicate source IDs; inner-intersect with artifact donors while preserving genotype-source order; retain cells in artifact/count order; remap their donors to dense indices in final genotype order; then align required donor covariates. Account separately for genotype-only, artifact-only, and covariate-only donors and all excluded cells.
- Normalize each returned variant as canonical chromosome, positive 1-based integer position, uppercase nonempty distinct `a0/a1`, and a nonempty `snp`. Replace a missing/empty/`.` ID with `chrom:pos:a0:a1`. Define scientific identity as `(chrom, pos, a0, a1)` with `a1` counted; reject repeated identities even if their source IDs differ, while allowing distinct allele pairs at the same position.
- Pin autosomal LOCO factor lookup to canonical artifact key `01`-`22` and require that requested/completed exclusion. Unsupported focal chromosomes such as X/Y/MT format an `unsupported_chromosome` per-gene row rather than substituting a genome-wide factor.
- Define one fixed version-1 row schema containing gene ID/chromosome/TSS/cis interval; statistic/p-value/calibration/status/failure/message; `interpretation_threshold`, stable `interpretation_level_warning_code`, and Boolean `calibration_resolution_warning`; cell/donor/state/variant counts; null dispersion/convergence/iteration/rank diagnostics; kernel scales and raw-efficient discrepancy; and nullable/NaN fields for both multiplier and moment diagnostics. Every requested gene produces exactly one row, including missing variants and numeric failures.
- Define a run manifest that records input identities/hashes, state schema/manifest hash and selection, full resolved configs/seed/platform/dtype/package versions, interpretation threshold plus separate level-warning-code and calibration-resolution-warning counts, reconciliation accounting, bucket/observed compile shapes, requested/processed/success/failed counts, failure frequencies, result schema/row count/hash, and `completed`, `completed_with_failures`, or `all_failed` status.

**Testing:**
- Add pure contract tests first and confirm missing behavior fails.
- Verify genotype-source-order intersection, stable cell order/remapping, exact accounting, ordered donor/cell covariates, missing/duplicate/null/nonfinite/type failures, all three offset modes and mutual exclusion, exact raw hash/selection mismatch, and canonical autosomal factor selection.
- Parameterize bucket boundary/overflow cases and prove no truncation. Use bounded Hypothesis strategies for valid bucket configs, manifest encode/decode, fixed-row formatting, order-hash sensitivity, and malformed inventories.
- Verify every success/failure formatter produces the identical Polars/PyArrow schema and one row per focal gene; unsupported chromosomes and missing variants retain their gene identity with `pvalue=NaN`.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_cis_state_io.py -x`
- Expected red: reconciliation, schema, bucket, and manifest cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_cis_state_io.py`
- Expected green: pure cis-state I/O contract tests pass.

**Commit:** `feat: define cis-state run contract`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Load workflow inputs and publish bounded run artifacts atomically

**Verifies:** cis-state.AC5.2, cis-state.AC5.3, cis-state.AC5.4

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_cis_state.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_cis_state_io.py` (integration)

**Implementation:**
- Start `_cis_state.py` with `# pattern: Imperative Shell`; preserve `io/__init__.py` as an import-only shell. Keep all filesystem, lazy Parquet collection, state-artifact loading/mmap access, hashing, PyArrow writer, staging, cleanup, and publication here.
- Load and validate the state manifest first, derive its recorded `cell_type_column` and selection/mixed policy, then load counts/cells/genes once through the extended sparse ingress and select cells by that artifact-owned contract. Expose no second cell-type column or selection option in the cis-state workflow. Validate exact input hashes/order and build the pure ready-data contract. Compute default library sizes from the complete raw count matrix once.
- Expose a chromosome accessor that validates canonical completion/exclusion and returns the requested factor mmap plus metadata. Load one chromosome at a time; do not require the focal gene to appear among the held-out factorization's retained genes.
- Publish a destination directory containing `manifest.json` and `results.parquet`. Refuse an existing destination, create a unique sibling staging directory on the same filesystem, open one `pyarrow.parquet.ParquetWriter` with the fixed schema, cast each bounded batch to that schema, write bounded row groups, and fully close the writer before hashing.
- After close, hash `results.parquet`, write the manifest last, reload and validate schema/row count/hash/status from staging, then use `os.replace(staging, final)` for atomic visibility. Do not claim crash durability. On run-level input/write/validation failure, close resources, remove staging, and leave no final directory.
- If all requested genes produce accounted failures, still publish the complete diagnostic artifact with manifest status `all_failed`; return that status to the CLI so it exits `1`. A mixture of success/failure rows is a completed run.

**Testing:**
- Add real temporary Parquet/state-artifact round-trip tests first and observe failure.
- Verify projected reads, raw/state hash and order rejection, chromosome mmap selection, focal outcome extraction, donor/cell reconciliation, offsets/library sizes, and retention of bounded sparse storage.
- Write multiple small result batches and verify a single fixed-schema Parquet file with bounded row groups, exact row count/hash, all manifest fields, and full reload validation for completed/mixed/all-failed artifacts.
- Inject failures before writer creation, after one row group, before/after close, during hashing, manifest validation, and publication. Require no final path and cleaned sibling staging. Verify existing-target refusal and corrupted/truncated/schema-shifted reload rejection.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_cis_state_io.py -x`
- Expected red: ingress/publication integration cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_cis_state_io.py`
- Expected green: pure and shell I/O tests pass.

**Commit:** `feat: add atomic cis-state run artifacts`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Stream genes and genoio regions through the cis-state workflow

**Verifies:** cis-state.AC5.3, cis-state.AC5.4

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/map/cis_state.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/map/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cis_state_workflow.py` (integration)

**Implementation:**
- Start `map/cis_state.py` with `# pattern: Imperative Shell`; add `# pattern: Imperative Shell` to `map/__init__.py` and preserve it as an import-only shell. Do not extend pseudobulk `ReadyDataState`/`map_cis` or create a standalone `cis_state/` package.
- Implement `map_cis_state(ready, *, genotype, window, test_config, buckets, batch_rows, key, log) -> Iterator[pl.DataFrame]`. Consume focal genes in deterministic chromosome/TSS/matrix-index order so one validated LOCO factor mmap can be cached per active chromosome and released before the next; preserve canonical gene identity in result rows.
- For gene `h`, request 1-based region `chrom:max(1, tss-window)-tss+window` through genoio's lazy `iter_regions`. Supply final donor IDs for pushdown, `sparse=False`, floating dtype, `missing="nan"`, the caller's explicit hard-call/dosage mode, and `return_variants=True`. Do not use `default_variant_filter`, imputation, or prefilter monomorphic variants because Phase 4 owns fixed-cohort diagnostics.
- Respect genoio 0.2 semantics: dense matrices are samples x variants; retained samples/variants stay in source order; requested sample duplicates are invalid; variant metadata is aligned and `a1` is the counted allele. Normalize `id` to `snp` and apply Task 4's exact `(chrom,pos,a0,a1)` identity, missing-ID, allele, and duplicate rules. Default to the repository's existing `hardcall` mode and allow explicit `dosage`; treat a source that cannot satisfy the selected mode as actionable failure and never relabel one representation as the other.
- Select the smallest variant bucket without truncation, pad on the host, and call `test_cis_state_interaction` once. Derive its key with `jax.random.fold_in(master_key, canonical_gene_matrix_index)` so filtering, failure, or execution order cannot perturb later genes.
- Convert every expected regional/data/numerical failure to its fixed diagnostic row, append one row per gene, and yield a DataFrame whenever `batch_rows` is reached plus one final short batch. Never collect all genes, call `jax.clear_caches`, or allocate focal counts beyond one `(M,)` column.
- Track observed bucket/static shape combinations for the manifest and assert compile-shape growth is bounded by configured capacities and static test/calibration configurations.

**Testing:**
- Add streaming tests first and observe failures before implementation. Use a small source-pinned genoio fixture to verify lazy one-result-per-region behavior, source-ordered sample filtering, returned matrix/variant alignment, dosage selection, and counted allele.
- Exercise a two-chromosome LOCO fixture, including a canonical exclusion with zero retained state genes, and prove the correct exclusion is loaded/reused for each gene, focal `y` is a single CSR-column extraction, final donor/cell orders stay aligned, and no genome-wide fallback occurs for unsupported chromosomes.
- Verify missing variant IDs receive the canonical replacement, allele case/validity and 1-based positions are enforced, repeated `(chrom,pos,a0,a1)` identities fail even under different IDs, and different allele pairs at one position remain distinct.
- Verify folded gene keys are unchanged by earlier failures/skips, bucket capacities and input masks, overflow rows, missing/duplicate variants, all important null/score/calibration failures, fixed row schema, bounded DataFrame batches, bounded live chromosome mappings, and observed compile-shape counts.
- Record allocations and reject dense full-count, `M x p`, `M x (p*r)`, `N x (p*r)`, full `T`, and donor-covariance materializations.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cis_state_workflow.py -x`
- Expected red: genoio alignment, LOCO selection, streaming, and failure-row cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cis_state_workflow.py`
- Expected green: array and streaming workflow tests pass.

**Commit:** `feat: stream cis-state gene tests`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Add the dedicated `cis-state` CLI adapter and exit contracts

**Verifies:** cis-state.AC5.1, cis-state.AC5.2, cis-state.AC5.3, cis-state.AC5.4, cis-state.AC5.5, cis-state.AC5.6

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/cli.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cli/test_cis_state_cli.py` (integration/system)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cli/test_state_cli.py` (regression)

**Implementation:**
- Preserve `cli.py` as `# pattern: Imperative Shell`. Add `cis-state` directly as a dedicated parser/handler; do not use `_create_common_subp` or `_common_setup`, which assume pseudobulk outcomes, donor-level dense designs, intercept injection, and per-variant tests.
- Require `--counts`, `--cells`, `--genes`, `--state-artifact`, one existing genoio source group, `--donor-covariates` Parquet, at least one ordered `--donor-covariate`, `--out`, `--calibration {cluster-bootstrap,cluster-moment}`, and positive `--min-informative-donors`. Do not expose `--cell-type-column`, `--cell-type`, or mixed-selection flags: the validated state artifact owns all three values.
- Pin common effective options for Phase 6: `--genotype-mode=hardcall`; `--window=500000` positive base pairs on each side; `--solver=qr` with choices `qr|cholesky`; NB config equal to Phase 3's exact `config=None` contract unless any NB flag is supplied, in which case require a complete replacement set; `--seed=0`; `--variant-block-size=64`; `--variant-capacities=64,128,256,512,1024,2048,4096`; `--platform=cpu` with explicit `gpu` opt-in; `--batch-rows=128`; `--interpretation-threshold=0.05`; no cell covariates; no gene filter (all supported autosomal genes); derived full-count library-size offsets; and `--verbose=false`. Require positive/finite/range validation before I/O and record every resolved value in the run manifest/help summary.
- Before importing/initializing inference devices or converting inputs, enable JAX x64 explicitly, resolve the requested CPU/GPU platform, and verify a probe array is genuinely float64 on that backend. If x64 cannot be enabled or honored, return runtime status `1` before numerics/artifact creation. The CLI never exposes a lower-precision mode; record `jax_enable_x64=true`, requested/resolved platform/device, and `float64` in the run manifest.
- Accept optional ordered repeated `--cell-covariate` names from the cells Parquet; mutually exclusive `--offset-column`/`--library-size-column` from that same file; explicit gene IDs/list; and the pinned common overrides above. Keep metadata/covariates Parquet-only in this release. An explicit option always replaces its one common default except NB flags, which replace the full NB config as a set; CLI flags override no environment variables or artifact scientific settings.
- For `cluster-bootstrap`, require positive `--bootstrap-replicates` and `--bootstrap-block-size`; for `cluster-moment`, require positive `--moment-probes>=2`, positive `--moment-probe-block-size`, `--moment-approximation {two-moment,three-moment}`, `--invalid-three-moment {fallback-two-moment,fail}`, and positive finite `--moment-validity-rtol`. The method-independent interpretation threshold is not duplicated among method flags. Reject incompatible or missing method-specific flags through argparse before loading any file so usage errors exit `2`. Until Phase 7, never infer calibration or state-factor solver defaults from runtime speed.

| `cis-state` option group | Required/default contract | Validation and precedence |
| --- | --- | --- |
| counts/cells/genes/state artifact/donor covariates/out/genotype source | required | Paths exist, exactly one genotype source, destination absent; artifact supplies cell-selection column/policy |
| ordered donor covariate names/min informative donors | at least one name and positive threshold required | CLI order is preserved; no inferred covariate or threshold |
| calibration | required in Phase 6 | Exactly one method and its complete method-specific option set; no partial merge |
| genotype/window/gene filter | hardcall; 500000; all supported autosomal genes | Explicit dosage/window/gene selection replaces the corresponding default |
| cell covariates/offset | none; full-count-derived library offset | Explicit ordered cell covariates and exactly one offset/library-size column replace independently |
| absorbed solver/NB | QR; exact Phase 3 `config=None` | Solver override is independent; any NB override requires the complete NB field set |
| seed/platform | 0; CPU | Explicit integer seed or GPU replaces; no environment-derived values |
| blocks/capacities/batch rows | 64; pinned capacity tuple; 128 | Positive values; capacities sorted/unique/block multiples; explicit values replace |
| interpretation threshold | 0.05 | Positive finite in `(0,1)`; method-independent and recorded in every row/manifest |
| verbosity | false | Logging-only; no scientific effect |

- Keep the handler thin: assemble immutable configs, load/reconcile inputs, stream mapped batches into the atomic writer, log a final summary, and return `0` when at least one gene succeeds. Return `1` for structural/runtime failure or a successfully published `all_failed` diagnostic run.
- Use Phase 2's isolated command-logger context with no disk path for this directory-producing command so no sidecar log appears outside or pre-creates the atomic artifact. Preserve established command logging and schemas; close/remove owned handlers on every return. Ensure `main` propagates handler statuses, prints success completion only for `0`, and never prints it for `all_failed`/runtime status `1`.

**Testing:**
- Write parser/handler tests first and observe assertion failures before implementation.
- Verify the exact effective-option/help/precedence table, every required argument, absence of cell-selection flags, artifact-derived cell-type column, genotype mutual exclusion, hard-call/dosage selection and unsupported-source failure, ordered repeated covariates, offset/library-size exclusion, complete NB replacement semantics, common interpretation threshold, calibration-specific conditional flags, bucket validation, and exit `2` before I/O for all usage errors.
- Run deterministic end-to-end fixtures for both calibration methods. Reload the published manifest/results; verify success `0`, mixed success/failure `0`, all-failed published artifact plus `1`, structural/runtime no-artifact plus `1`, and exact stderr/console logging behavior.
- Run CLI device setup in isolated subprocesses: simulate disabled/unavailable x64 and require status `1` with no artifact; for CPU and an available x64 GPU require manifest `jax_enable_x64=true`, resolved device/platform, and dtype `float64`. Pass float32 fixture inputs through the public composition path and verify promotion, while direct internal float32 launch tests remain failures.
- Re-run state-factor one/22-chromosome, incompatible artifact, success/runtime/usage status tests. Invoke new/established commands in both orders and repeatedly; assert no handler leakage or duplicate stderr, new commands have no sidecars, established sidecars/output remain unchanged, status `1` has no success message, and unrelated logger handlers survive. Assert existing `cis`, `nominal`, `trans`, and `compute-pcs` parser, output, return status, stdout/stderr, and log behavior remain unchanged.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cli/test_cis_state_cli.py -x`
- Expected red: parser, dispatch, artifact, and exit-code cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cli/test_cis_state_cli.py tests/test_cli/test_state_cli.py`
- Expected green: both new dedicated commands and existing-command regressions pass.

**Commit:** `feat: add cis-state command`
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Run Phase 6 integration, regression, and quality gates

**Verifies:** cis-state.AC5.1, cis-state.AC5.2, cis-state.AC5.3, cis-state.AC5.4, cis-state.AC5.5, cis-state.AC5.6

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Audit that initial input formats remain sparse `.npz` plus Parquet metadata/covariates, that donor covariates are required, and that no public weight/covariance option was introduced.
- Confirm all cell/donor/variant reorders have explicit accounting, all per-gene failures become rows, exact state-input identities are validated before numerics, and destination visibility is atomic only after writer close and reload validation.
- Review bounded allocation/streaming logs, fixed variant bucket shapes, global folded keys, source-file FCIS classifications, and preservation of user-owned `/Users/nicholas/Projects/jaxqtl/state.md` as untracked/unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_io/test_cis_state_io.py tests/test_cis_state_workflow.py tests/test_cli/test_cis_state_cli.py tests/test_io/test_state_artifact.py tests/test_cluster_calibration.py tests/test_cli/test_state_cli.py`
- Expected: workflow, calibration, artifact, and both command suites pass.
- Run: `uv run pytest -p no:capture`
- Expected: the complete suite passes with no new warnings or regressions.
- Run: `uv run ruff check src tests scripts`
- Expected: no lint errors.
- Run: `uv run ruff format --check src tests scripts`
- Expected: all files already formatted.
- Run: `uv run ty check src tests scripts`
- Expected: no type errors.
- Run: `rg -n '^# pattern: Imperative Shell$' src/jaxqtl/map/cis_state.py src/jaxqtl/map/__init__.py src/jaxqtl/cli.py`
- Expected: every modified mapping/CLI shell has an explicit FCIS classification.
- Run: `git diff --check`
- Expected: no whitespace errors.
- Verifies: cis-state.AC5.1, cis-state.AC5.2, cis-state.AC5.3, cis-state.AC5.4, cis-state.AC5.5, cis-state.AC5.6.

**Commit:** None unless verification finds a required fix; repeat the owning task's tests and make a narrow correction commit.
<!-- END_TASK_8 -->
