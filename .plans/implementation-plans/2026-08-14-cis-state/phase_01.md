# Cis-by-state Phase 1 Implementation Plan

**Goal:** Establish a validated sparse single-cell count boundary and exact matrix-free PFlog, CLR-centering, donor-centering, and donor-balancing operations.

**Architecture:** Keep filesystem access and Polars materialization in a thin imperative adapter, with sparse normalization and metadata reconciliation in a pure ingress core. Put PFlog sufficient statistics and the composite operator in two cohesive functional-core modules under the new `jaxqtl.state` package; the SciPy `LinearOperator` and factorization adapter remain Phase 2 work.

**Tech Stack:** Python 3.11+, NumPy, SciPy 1.17+, Polars, pytest, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 1 of 7.

**Codebase verified:** 2026-08-14 15:52 PDT

---

## Review Profile

- Profile: `full`
- Why this profile: This phase creates public sparse-ingress contracts and a new state package while implementing numerically sensitive PFlog and adjoint operator algebra, so architecture, numerics, CLI/API, and inference boundary review are all required.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC1: State construction

- **cis-state.AC1.2 Success:** PFlog forward, adjoint, and block operations match dense references without materializing transformed `M x q` data.
- **cis-state.AC1.3 Success/Failure:** Fast `alpha_-h` equals explicit chromosome-exclusion fitting; invalid estimates fail unless explicitly overridden.
- **cis-state.AC1.4 Success:** The excluded chromosome affects neither shifted-log values nor the CLR center.
- **cis-state.AC1.5 Success:** Donor-centered features have zero donor means within tolerance.
- **cis-state.AC1.6 Edge:** Donor-balanced covariance matches the explicit formula; singleton donors contribute no covariance.

---

<!-- START_TASK_1 -->
### Task 1: Declare SciPy as a direct runtime dependency

**Verifies:** None

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/pyproject.toml`
- Modify: `/Users/nicholas/Projects/jaxqtl/uv.lock`

**Implementation:**
- Add `scipy>=1.17` to the project runtime dependencies. SciPy 1.17 is the first direct bound selected for the repository's declared CPython 3.11-3.14 matrix and is already the resolved transitive version in the current lock.
- Regenerate `uv.lock`. Review the lock diff and confirm that SciPy becomes direct `jaxqtl` metadata without an unintended resolver downgrade or unrelated dependency churn.

**Verification:**
- Run: `uv lock`
- Expected: lock generation succeeds and retains a SciPy version satisfying `>=1.17`.
- Run: `uv run python -c "import scipy; from scipy.sparse import csr_array; print(scipy.__version__, csr_array((1, 1)).format)"`
- Expected: the command prints SciPy 1.17 or newer and `csr`.
- Verifies: None; this is dependency setup for later functional tasks.

**Commit:** `build: declare scipy runtime dependency`

Stage only `/Users/nicholas/Projects/jaxqtl/pyproject.toml` and `/Users/nicholas/Projects/jaxqtl/uv.lock`; do not stage the user-owned `/Users/nicholas/Projects/jaxqtl/state.md`.
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: Add the pure sparse single-cell ingress contract

**Verifies:** None; this is tested boundary preparation for the Phase 1 numerical acceptance criteria.

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell_contract.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_single_cell.py` (unit)

**Implementation:**
- Start `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell_contract.py` with `# pattern: Functional Core`.
- Colocate the immutable `SparseSingleCellData` result and semantic copy-accounting contract with the pure normalization/reconciliation functions; do not create separate type, validation, exception, or sparse-helper modules.
- Normalize only SciPy CSR and CSC sparse arrays/matrices to a canonical `scipy.sparse.csr_array`. Reject BSR, COO, DIA, dense, boolean, non-2D, empty-axis, negative, fractional, or nonfinite count inputs. Require every integer-valued count to be at most `2**53`, checked before float64 arithmetic. Canonicalize duplicate coordinates with checked group accumulation: use Python-integer accumulation for integer storage and `math.fsum` plus an exact-integer check for floating storage, reject a group sum outside `[0,2**53]` or the source dtype, then rebuild the canonical CSR. Re-run integer/nonnegative/finite/maximum validation after duplicate consolidation. Preserve integer count storage at ingress.
- Record semantic events rather than promising buffer identity: `npz_materialized` is added by the file adapter in Task 3; the pure core reports `sparse_family_normalized`, `csc_to_csr`, and `canonicalized` only when those operations apply.
- Treat rows as cells and columns as genes. Reconcile already-materialized cell and gene Polars frames by required integer `matrix_index`, reject null/duplicate/gapped/out-of-range indices, sort into exact `0..M-1` and `0..q-1` order, and validate unique nonempty `cell_id`, `gene_id`, and `donor_id` values plus a caller-selected nonempty cell-type column.
- Require a `chrom` gene column. Normalize case-insensitive optional `chr` prefixes, canonicalize `M` to `MT`, accept `1`-`22`, `X`, `Y`, and `MT`, and reject unsupported or empty labels. Phase 2 will restrict focal LOCO requests to autosomes 1-22.
- Keep Polars objects at the adapter boundary; the numerical state modules receive the canonical CSR counts and NumPy identifier/index arrays derived after row order is frozen.

**Testing:**
- Follow red-green-refactor. First add behavior tests that access the wished-for API through the existing `jaxqtl.io` package and fail by assertion because it is absent; do not accept a collection/import error as the red state. Run the targeted command and confirm the expected missing-behavior failure before adding source code.
- Use in-memory CSR/CSC arrays and shuffled Polars frames to verify deterministic matrix-index ordering, normalized chromosomes, stable copy events, exact shape/orientation, integer-count preservation, and canonical CSR output.
- Parameterize failures for unsupported sparse formats, dense input, boolean/negative/fractional/nonfinite counts, count values immediately below/at/above the `2**53` exact-float64 boundary, and duplicate-coordinate sums immediately below/at/above that bound including source-dtype overflow. Cover empty axes, axis-length mismatch, missing/gapped/out-of-range/duplicate indices, null or duplicate identifiers, null/empty donor and cell-type values, and unsupported chromosomes.
- Do not assert internal helper calls or use mocks; assert returned data, ordering, diagnostics, and actionable exception messages.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_single_cell.py -x`
- Expected red: an assertion fails because the public sparse-ingress behavior is not yet available.
- Run green: `uv run pytest -p no:capture tests/test_io/test_single_cell.py`
- Expected green: all pure-ingress tests pass.

**Commit:** `feat: add sparse single-cell ingress contract`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Add the NPZ and lazy-Parquet ingress adapter

**Verifies:** None; this is tested boundary preparation for the Phase 1 numerical acceptance criteria.

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_single_cell.py` (integration)

**Implementation:**
- Start `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell.py` with `# pattern: Imperative Shell`; retain the existing classification in `io/__init__.py`.
- Implement `load_sparse_single_cell(counts_path, cells_path, genes_path, *, cell_type_column) -> SparseSingleCellData` as a thin gather/process boundary: load counts once with `scipy.sparse.load_npz`, build projected lazy scans with `pl.scan_parquet`, collect each metadata table once, then call the Task 2 pure core.
- Select only the required columns plus `cell_type_column` before collection. Do not convert through pandas or Arrow, join independent sources positionally, or materialize intermediate frames.
- Add `npz_materialized` to the returned semantic copy events and rely on exact axis coverage to diagnose normal orientation mistakes. Document that square matrices cannot reveal an accidental transpose from `.npz` alone.
- Re-export `SparseSingleCellData` and `load_sparse_single_cell` from `jaxqtl.io`; keep `ExpressionData` unchanged.

**Testing:**
- Before implementation, extend the existing test file with real temporary CSR and CSC `.npz` files plus temporary Parquet cell/gene tables; confirm the targeted run fails for the missing loader behavior.
- Verify projected ingress, one deterministic materialized result per metadata file, shuffled metadata reordering, CSR/CSC normalization, semantic copy reports, and actionable file/schema failures. Use real SciPy and Polars I/O through `tmp_path`, not mocks.
- Include a transposed non-square count matrix fixture whose metadata axis lengths make the orientation error explicit.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_single_cell.py -x`
- Expected red: the newly added loader behavior fails before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_single_cell.py`
- Expected green: all ingress unit and integration tests pass.

**Commit:** `feat: load sparse single-cell inputs`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_4 -->
### Task 4: Implement PFlog sufficient statistics and strict LOCO alpha estimation

**Verifies:** cis-state.AC1.3

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/_pflog.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_state/test_pflog.py` (unit)

**Implementation:**
- Classify `_pflog.py` as `# pattern: Functional Core`; classify the import-only `state/__init__.py` and root package initializer as `# pattern: Imperative Shell`.
- Colocate immutable sufficient-statistics and alpha-diagnostic results in `_pflog.py`. Expose a small public surface through `jaxqtl.state`: the statistics constructor and `estimate_pflog_alpha`; do not expose SciPy implementation objects.
- Compute float64 per-gene means and unbiased sample variances (`ddof=1`) in `O(nnz + q)`, including implicit zeros, using an exact-integer translation before conversion. For a fully stored gene choose its first canonical integer count as anchor; for a gene with any implicit zero choose anchor `0`. Subtract that anchor from every stored count in checked signed-integer arithmetic, then convert the bounded differences to float64, compute their compensated mean, and evaluate centered sum of squares as `sum_stored(d-mean_d)^2 + (M-nnz_g)*(0-anchor-mean_d)^2`. Translation leaves variance invariant and preserves unit-scale variation in large nearly constant dense columns; compute `mu=anchor+mean_d` with compensated addition and report its float64 representation/error scale. Define `variance_roundoff_tol = 64*eps(float64)*max(sum_squares,1)/(M-1)`: snap a negative variance to zero only when its magnitude is no larger than this tolerance, and otherwise return a numerical failure. Match CIPHER commit `11f0fca33b891d83032ea7788c216d02b67ca6e5` for `ddof=1` and use genes with finite positive mean and variance, so all-zero and exactly constant genes are excluded while sub-Poisson genes may contribute negative numerator terms.
- Cache per-gene `A_g = mu_g^2 * (v_g - mu_g)` and `B_g = mu_g^4` plus compensated chromosome bins and totals. Preserve the mathematical diagnostic identity `(A-A_h)/(B-B_h)`, but evaluate a LOCO numerator and denominator by pairwise/`math.fsum` accumulation of retained per-chromosome bins and any retained unassigned bin rather than subtracting nearly equal totals; never revisit cells. Report both total/held-out and directly accumulated retained values and fail if they disagree beyond `64*eps*max(1,sum_abs_terms)`.
- Return diagnostics containing the retained/excluded gene counts, numerator, denominator, excluded chromosome, source (`auto` or `override`), and alpha. For any requested canonical autosome `1`-`22` absent from the retained state-gene set, define `A_h = B_h = 0`, report zero excluded genes, and compute the valid no-op exclusion; reject only noncanonical requested chromosome labels. Reject fewer than two cells and nonfinite or nonpositive numerator/denominator/alpha with actionable errors.
- Accept an explicit override only when it is finite and strictly positive. Do not inherit CIPHER's minimum-gene fallbacks, median fallback, or `[1e-8, 1e8]` clipping; every invalid automatic estimate remains visible.

**Testing:**
- Follow red-green-refactor through the public `jaxqtl.state` API, obtaining an assertion failure rather than a collection error before implementation.
- Use hand-derived and large-nearly-constant sparse fixtures, including `[2**53-1,2**53]`, to verify translated means, exact reference `ddof=1` variances, the scale-aware negative-roundoff rule, implicit-zero contributions, exact constant/all-zero filtering, per-gene A/B terms, compensated chromosome sums, and diagnostics.
- Compare every cached fast `alpha_-h` with an explicit reference that physically removes chromosome `h` columns and recomputes the through-origin fit. Include a fixed reference fixture derived from the CIPHER formula.
- Verify that sub-Poisson genes remain in the fit; a canonical absent autosome gives zero chromosome sums and the same alpha as the genomewide retained-gene fit; and invalid/nonpositive numerator, degenerate denominator, too few cells, noncanonical chromosome, and nonfinite/nonpositive overrides fail rather than clip. Verify a valid override is returned with source and diagnostics intact.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_state/test_pflog.py -x`
- Expected red: an assertion fails because the new state API is absent.
- Run green: `uv run pytest -p no:capture tests/test_state/test_pflog.py`
- Expected green: all PFlog statistics, LOCO, override, and failure tests pass.

**Commit:** `feat: estimate strict LOCO PFlog alpha`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Implement the implicit PFlog composite operator

**Verifies:** cis-state.AC1.2, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/_operator.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_state/test_pflog.py` (unit)

**Implementation:**
- Classify `_operator.py` as `# pattern: Functional Core`. Colocate the concrete `PFLogOperator`, its immutable configuration/diagnostics, and the `pflog_operator` factory; do not create base, protocol, helper, or type-only modules.
- Construct a new sparse transformed-values buffer equivalent to `log1p(4 * alpha * counts.data)` and preserve the CSR sparsity structure. For positive stored counts, evaluate it as `logaddexp(0, log(4)+log(alpha)+log(count))` so finite valid inputs cannot overflow in the intermediate product; stored zeros remain exactly zero. Apply chromosome exclusion before both transformation and the feature dimension used by CLR.
- Represent `Z = C_D L J_q` only through vector and block products. Forward order is feature CLR centering, sparse `L` multiplication, optional within-donor centering by segment reductions, then optional left multiplication by `sqrt(w_c)`. The adjoint reverses that order and applies the symmetric centers correctly.
- Implement `matvec`, `rmatvec`, `matmat`, and `rmatmat` with stable one- and two-dimensional shape validation. Do not construct cell-by-cell, cell-by-donor, or dense transformed cell-by-gene arrays.
- Validate a finite positive alpha and a dense, contiguous, integer donor index covering `[0, N)`. Use `np.bincount` or `np.add.at` for donor reductions rather than incidence/centering matrices.
- For balancing, use total aligned donor count `N` and cell weight `1 / (N * (n_i - 1))` for donors with at least two cells; assign singleton cells zero weight. Keep centering and balancing as explicit booleans with defaults enabled.
- Expose only the domain operator/factory through `jaxqtl.state`; defer `scipy.sparse.linalg.LinearOperator` wrapping and truncated factorization to Phase 2.

**Testing:**
- Add dense-reference tests first and confirm they fail for the absent operator behavior before implementation.
- Construct the explicit small matrix `sqrt(D_w) C_D L J_q` only inside tests. Compare vector, adjoint-vector, matrix, and adjoint-matrix products; test the inner-product adjoint identity separately.
- Verify that excluded-chromosome values cannot affect transformed values or CLR centers, donor-centered outputs have zero within-donor means, balanced covariance equals the explicit donor formula, and singleton donors contribute zero covariance.
- Verify centered/unbalanced and uncentered/unbalanced toggles, nonsorted but dense donor membership, invalid/gapped donor indices, alpha failures, extreme finite alpha/count products against a high-precision scalar reference, and vector/block shape errors.
- Guard the sparse contract by using a CSR fixture whose dense conversion methods fail if called and by asserting that transformed storage scales with input `nnz`, while comparisons use a separately constructed test-only dense reference.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_state/test_pflog.py -x`
- Expected red: the new operator cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_state/test_pflog.py`
- Expected green: all PFlog and operator tests pass.

**Commit:** `feat: add implicit PFlog state operator`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Run Phase 1 regression and quality gates

**Verifies:** cis-state.AC1.2, cis-state.AC1.3, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Review every created or modified application source file for its required functional-core or imperative-shell classification.
- Confirm the public signatures contain no variant-weight or state-weight arguments and that `ExpressionData` and existing CLI code are unchanged.
- Inspect `git status --short` before staging or committing anything further; `/Users/nicholas/Projects/jaxqtl/state.md` must remain untracked and unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_io/test_single_cell.py tests/test_state/test_pflog.py`
- Expected: all Phase 1 tests pass.
- Run: `uv run pytest -p no:capture`
- Expected: the complete suite passes with no new warnings or regressions.
- Run: `uv run ruff check src tests scripts`
- Expected: no lint errors.
- Run: `uv run ruff format --check src tests scripts`
- Expected: all files already formatted.
- Run: `uv run ty check src tests scripts`
- Expected: no type-check errors.
- Run: `git diff --check`
- Expected: no whitespace errors.
- Verifies: cis-state.AC1.2, cis-state.AC1.3, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6 through the combined Phase 1 regression run.

**Commit:** None unless verification reveals a required fix; any fix must repeat its task's red-green verification and use a narrowly scoped commit.
<!-- END_TASK_6 -->
