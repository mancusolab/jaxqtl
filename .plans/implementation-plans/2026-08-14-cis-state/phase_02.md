# Cis-by-state Phase 2 Implementation Plan

**Goal:** Produce complete, versioned chromosome-specific state-factor artifacts without densifying transformed counts or whitening factors.

**Architecture:** Extend the Phase 1 domain operator with one cohesive factorization module that privately adapts it to SciPy `LinearOperator`/`svds`, then form unweighted factors through the original operator. Keep artifact schema and alignment validation pure while a separate I/O shell performs hashed, atomic directory writes and memory-mapped loads; expose the workflow through a dedicated thin CLI handler.

**Tech Stack:** Python 3.11+, NumPy, SciPy 1.17+, Polars, Hypothesis, argparse, pytest, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 2 of 7.

**Codebase verified:** 2026-08-14 16:03 PDT

---

## Review Profile

- Profile: `full`
- Why this profile: This phase crosses numerical factorization, durable artifact/API design, atomic I/O, and a new CLI, so every specialist review is required.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC1: State construction

- **cis-state.AC1.1 Success:** Valid sparse counts and metadata produce factors for one selected cell type; mixed types require explicit opt-in.
- **cis-state.AC1.2 Success:** PFlog forward, adjoint, and block operations match dense references without materializing transformed `M x q` data.
- **cis-state.AC1.3 Success/Failure:** Fast `alpha_-h` equals explicit chromosome-exclusion fitting; invalid estimates fail unless explicitly overridden.
- **cis-state.AC1.4 Success:** The excluded chromosome affects neither shifted-log values nor the CLR center.
- **cis-state.AC1.5 Success:** Donor-centered features have zero donor means within tolerance.
- **cis-state.AC1.6 Edge:** Donor-balanced covariance matches the explicit formula; singleton donors contribute no covariance.
- **cis-state.AC1.7 Success:** Truncated factors recover the reference leading subspace and satisfy `S = Z V_r` without whitening.

### cis-state.AC5: CLI and artifacts

- **cis-state.AC5.1 Success:** `state-factor` writes a complete versioned artifact for one chromosome or chromosomes 1-22.
- **cis-state.AC5.2 Failure:** Incomplete, incompatible, or misaligned factor artifacts are rejected.

---

<!-- START_TASK_1 -->
### Task 1: Add Hypothesis for bounded contract properties

**Verifies:** None

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/pyproject.toml`
- Modify: `/Users/nicholas/Projects/jaxqtl/uv.lock`

**Implementation:**
- Add `hypothesis>=6.165` to the `dev` optional dependency group and `[tool.hatch.envs.test].dependencies`, then regenerate the lock. This keeps both the uv CI path and documented `hatch run test:run` environment able to collect the new tests. Use Hypothesis across sparse normalization and artifact round-trip boundaries; do not use generated cases for iterative solver convergence thresholds.
- Keep generated cases small and deterministic with bounded composite strategies, a fixed `max_examples`, `derandomize=True`, and `deadline=None`. Retain explicit examples for empty, singleton, repeated-spectrum, corruption, and failure cases.

**Verification:**
- Run: `uv lock`
- Expected: lock generation succeeds without unrelated resolver churn.
- Run: `uv run --extra dev python -c "import hypothesis; print(hypothesis.__version__)"`
- Expected: Hypothesis 6.165 or newer imports from the development environment.
- Run: `hatch run test:run --collect-only`
- Expected: the Hatch test environment collects the existing suite successfully; the preceding direct import proves Hypothesis is installed before later tasks add Hypothesis-backed tests.
- Verifies: None; this is test infrastructure.

**Commit:** `test: add hypothesis dependency`

Stage only `/Users/nicholas/Projects/jaxqtl/pyproject.toml` and `/Users/nicholas/Projects/jaxqtl/uv.lock`; never stage `/Users/nicholas/Projects/jaxqtl/state.md`.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Select one cell type and freeze donor indexing

**Verifies:** cis-state.AC1.1

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_single_cell_contract.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_single_cell.py` (unit)

**Implementation:**
- Preserve the Phase 1 `# pattern: Functional Core` classification in `_single_cell_contract.py` and the shell classification in `io/__init__.py`.
- Implement `select_single_cell_data(data, *, cell_type=None, allow_mixed_cell_types=False)` in the ingress core. Return a canonical selected container with CSR row selection, selected cell metadata in matrix order, retained original matrix indices, donor IDs in first-retained-cell order, dense zero-based `donor_index`, and donor cell counts.
- If the metadata contains one cell type, accept an omitted `cell_type`; if it contains multiple types, require an explicit selected value or `allow_mixed_cell_types=True`. Reject an unknown/empty selected value and the contradictory combination of explicit selection plus mixed opt-in.
- Reindex matrix rows after selection without changing the immutable provenance column for original matrix positions. Keep cell type and donor reconciliation outside the factorization module and CLI.

**Testing:**
- Add tests first through the public `jaxqtl.io` surface and confirm an assertion failure for the missing selection behavior.
- Verify single-type default selection, explicit selection from mixed data, explicit mixed opt-in, preserved cell ordering, CSR sparsity, original-index provenance, first-retained-cell donor order, dense donor indices, and donor counts.
- Verify mixed-without-opt-in, unknown/empty selection, contradictory options, zero selected cells, and metadata/count row mismatch failures.
- Add bounded Hypothesis properties for selection/order normalization idempotence and donor-index density; include explicit one-cell and one-donor examples and avoid `.filter()`-heavy strategies.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_single_cell.py -x`
- Expected red: the new selection cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_single_cell.py`
- Expected green: all ingress and selection tests pass.

**Commit:** `feat: select state-factor cells`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement deterministic matrix-free state factorization

**Verifies:** cis-state.AC1.2, cis-state.AC1.3, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6, cis-state.AC1.7

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/_factor.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/_operator.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/state/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_state/test_factor.py` (unit)

**Implementation:**
- Start `_factor.py` with `# pattern: Functional Core`. Colocate the private SciPy adapter, `StateFactorDiagnostics`, immutable backend-neutral `StateFactorResult`, `construct_state_factor`, and streaming `iter_loco_state_factors`; do not create `_scipy.py`, `_backend.py`, `_result.py`, `_loco.py`, or validation/helper leaf modules.
- If needed, extend the Phase 1 operator with shared transformed sparse storage plus explicit balanced and unbalanced views. Learn right singular vectors from `sqrt(D_w) Z`, then compute returned factors as `S = Z V_r` through the unweighted view. Never divide weighted scores by cell weights, because singleton weights are zero, and never construct a second transformed CSR buffer.
- Wrap the balanced domain operator privately as `scipy.sparse.linalg.LinearOperator` with forward, transpose, vector, and block actions. Require the caller to choose `solver="propack"` or `solver="arpack"` until Phase 7 benchmarks select the public default; do not silently inherit SciPy's default.
- Enforce the public truncated-factor contract `1 <= rank < min(M, q_active)` for both PROPACK and ARPACK, even though SciPy can expose a full-rank PROPACK edge. Pass a reproducible `rng=np.random.default_rng(seed)`, solver tolerance, and solver-specific iteration configuration. Treat PROPACK `maxiter` as a Krylov-dimension setting in public names/diagnostics; allow `ncv` only for ARPACK.
- Request right singular vectors, sort singular values descending, reorder loadings identically, and canonicalize each loading/factor pair's sign by making the largest-absolute loading entry nonnegative. Treat repeated singular values as subspace-identifiable rather than column-identifiable.
- Require a finite strictly positive relative solver tolerance `tol`. With `dmax=max(M,q_active)`, `eps=eps(float64)`, and sorted `sigma_max`, define the requested-triplet floor `sigma_floor=max(tol, eps*dmax)*sigma_max`; reject a nonfinite/nonpositive `sigma_max` or any requested `sigma_j <= sigma_floor`. Reconstruct balanced left vectors as `u_j = A v_j / sigma_j` only after that check. Define forward and adjoint relative residual limits and the loading-orthogonality limit as `residual_limit=max(10*tol,100*eps*dmax)`; compute residual denominators with `max(sigma_j, ||Av_j||, tiny)` and `max(sigma_j, ||A.Tu_j||, tiny)`. Reject any diagnostic above its limit, invalid rank/configuration, `ArpackNoConvergence`, PROPACK `LinAlgError`, or incomplete solve; never return partial triplets.
- Return read-only NumPy arrays plus diagnostics recording alpha, excluded chromosome, active gene indices, solver, seed, tolerances, solver dimensions, residual maxima, orthogonality error, singular values, donor counts, centering/balancing flags, and operator dimensions.
- Compute PFlog sufficient statistics once per selected cell set. `iter_loco_state_factors` must yield one result at a time in caller-specified chromosome order, using cached strict fast alpha estimates and excluding each chromosome before transformation/CLR. Its default chromosome tuple is `("1", ..., "22")`.

**Testing:**
- Write dense-reference factorization tests first and confirm a behavioral failure before implementation.
- On fixed small spectra, compare singular values, loading projectors/subspace angles, and `S S.T` with dense SVD; verify descending order and paired sign canonicalization only where singular values are distinct.
- Verify `S = Z V_r` from the unbalanced operator, not weighted `u * sigma`; donor means remain zero, singleton rows remain present in `S`, and donor-balanced covariance matches the Phase 1 contract.
- Compare fast and explicit LOCO alpha/factor subspaces, and perturb held-out chromosome values to prove no alpha, shifted-log, or CLR leakage.
- Include a requested canonical autosome with no retained state genes: require zero excluded-gene/sum diagnostics and a complete factor result equal to the no-op exclusion. Run this through the default 1-22 iterator so complete LOCO artifacts do not depend on every autosome being represented.
- Verify deterministic replay with a fixed seed; reject rank zero, `rank == min(M,q_active)` for both solvers, and rank above the active dimension; and cover the largest valid truncated rank, repeated-spectrum projector behavior, exact scale-aware triplet/residual/orthogonality thresholds at both sides of each boundary, nonfinite/near-zero triplets, invalid `ncv`/iteration options, and both solver failure classes.
- Guard against dense transformed allocation by using operator/count fixtures that fail on dense conversion and by recording every observed two-dimensional allocation shape; test-only dense references remain separate.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_state/test_factor.py -x`
- Expected red: factorization behavior fails before implementation.
- Run green: `uv run pytest -p no:capture tests/test_state/test_factor.py`
- Expected green: all factorization, LOCO, solver-boundary, and failure tests pass.

**Commit:** `feat: add donor-balanced state factorization`
<!-- END_TASK_3 -->

<!-- START_SUBCOMPONENT_A (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Define the versioned state-artifact contract

**Verifies:** None; this pure schema task is exercised by the complete artifact behavior in Task 5.

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_state_artifact_contract.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_state_artifact.py` (unit)

**Implementation:**
- Start `_state_artifact_contract.py` with `# pattern: Functional Core`. Colocate schema version `1`, manifest/result dataclasses, canonical relative payload inventory, JSON conversion, deterministic identifier hashing, and pure compatibility/alignment validation.
- Pin one layout for a single exclusion or 22-part LOCO artifact: root `manifest.json`, `cells.parquet`, `donors.parquet`, and `chromosomes/<chrom>/factors.npy`, `loadings.npy`, `singular_values.npy`, and `genes.parquet`. Use canonical chromosome directory keys `01`-`22`; do not add a second schema for genome-wide factors in this branch.
- The manifest records the exact `cell_type_column`, selected cell type/mixed opt-in, counts and metadata input SHA-256 hashes, canonical cell/donor/gene-order hashes, requested and completed chromosomes, dimensions/dtypes, PFlog and filtering diagnostics, donor counts, centering/balancing, solver configuration/seed, singular values, convergence residuals, approximation metrics, and every payload SHA-256. Record replay provenance explicitly: jaxqtl/schema versions; Python implementation/version; OS/release/machine/processor; NumPy, SciPy, Polars, PyArrow, JAX, and jaxlib versions; normalized NumPy/SciPy BLAS/LAPACK vendor information; relevant thread-count environment variables; and the fixed `cpu` platform. The manifest does not hash itself.
- Define deterministic identifier hashing with length-prefixed UTF-8 values and explicit field/order separators; never use Python `hash()`.
- Reject unsupported schema versions, noncanonical or missing chromosome keys, incomplete requested/completed sets, unsafe/absolute payload paths, duplicate inventory entries, shape/dtype disagreement, cell/donor/gene order mismatch, and incompatible configuration before returning a validated contract.

**Testing:**
- Add pure manifest/normalization tests first and confirm the missing behavior fails before implementation.
- Use fixed examples for every rejection rule and exact JSON field contract.
- Use bounded Hypothesis strategies for `decode(encode(x)) == x`, canonical JSON round-trip, identifier-hash determinism/sensitivity to order, and validator acceptance of generated valid inventories. Explicitly include Unicode IDs, empty collections where invalid, and one/22 chromosome cases.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_state_artifact.py -x`
- Expected red: pure artifact-contract cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_state_artifact.py`
- Expected green: schema and property tests pass.

**Commit:** `feat: define state artifact schema`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Write and load state artifacts atomically

**Verifies:** cis-state.AC5.1, cis-state.AC5.2

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_state_artifact.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_state_artifact.py` (integration)

**Implementation:**
- Start `_state_artifact.py` with `# pattern: Imperative Shell`; preserve the shell classification in `io/__init__.py`.
- Implement a writer that creates a unique sibling staging directory on the destination filesystem, streams one factor result at a time, writes/fully closes `.npy` and Parquet payloads, computes raw-input and payload SHA-256 values in bounded chunks, writes `manifest.json` last, reloads and validates the staging artifact, then atomically renames it to a previously nonexistent final directory.
- Refuse an existing final path rather than overwrite it. On any factorization, write, hash, or validation failure, remove the staging directory and leave no visible final artifact. A LOCO final path must not appear until all requested chromosomes succeed.
- Store arrays as individual `.npy` files so the loader can use `np.load(..., mmap_mode="r")`; never package factors into `.npz`. Store shared cell/donor ordering once and per-chromosome retained gene ordering beside loadings.
- Implement a loader that rejects staging directories and validates schema, declared completeness, payload presence and SHA-256, shape/dtype, metadata row counts/index order, canonical cell/donor/gene hashes, and optional expected cell/donor/configuration identities before returning read-only memory-mapped factors/loadings/singular values.
- Re-export only the backend-neutral artifact result/manifest plus writer/loader through `jaxqtl.io`.

**Testing:**
- Add real temporary-directory round-trip tests first and verify they fail before the shell exists.
- Verify single-chromosome and 22-chromosome writes, memory-mapped loads, exact metadata/order preservation, complete hashes/diagnostics, and deterministic manifest validation.
- Parameterize unsupported schema, missing chromosome/payload, corrupt bytes/hash, shape/dtype mutation, shuffled cell/donor/gene metadata, incompatible expected IDs/configuration, existing destination, and staging-directory rejection.
- Inject a deterministic writer/factor-iterator failure after at least one chromosome and verify no final directory remains and the unique staging directory is cleaned. Test observable filesystem outcomes rather than internal calls.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_io/test_state_artifact.py -x`
- Expected red: writer/loader cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_io/test_state_artifact.py`
- Expected green: all artifact contract, atomicity, corruption, and round-trip tests pass.

**Commit:** `feat: add atomic state artifacts`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_6 -->
### Task 6: Add the dedicated `state-factor` CLI adapter

**Verifies:** cis-state.AC1.1, cis-state.AC5.1, cis-state.AC5.2

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/cli.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/log.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cli/test_state_cli.py` (integration)

**Implementation:**
- Preserve `cli.py` as `# pattern: Imperative Shell` and add `# pattern: Imperative Shell` to `log.py` before modifying it. Add a dedicated parser and `_state_factor` handler; do not route through `_create_common_subp` or `_common_setup`, which assume pseudobulk genotypes and dense donor-level phenotypes.
- Require `--counts`, `--cells`, `--genes`, `--cell-type-column`, `--rank`, `--solver {propack,arpack}`, positive finite `--tol`, positive `--maxiter`, integer `--seed`, and `--out`. Accept optional `--cell-type` or mutually exclusive `--allow-mixed-cell-types`; require exactly one of `--exclude-chromosome {1..22}` or `--loco`. ARPACK additionally requires `--ncv` with `rank < ncv < min(M,q_active)` after shapes are known; PROPACK rejects `--ncv`. The command exposes no device choice and always records/uses CPU.
- Pin the remaining effective options: `--pflog-alpha=auto`; `--center-within-donor` and `--balance-donors` true through BooleanOptionalAction; no selected cell type when the table contains exactly one type; and `--verbose=false`. There is no solver/tolerance/iteration default before Phase 7 evidence. Parser help must label required, conditional, and fixed/default values, and the artifact manifest records the fully resolved configuration rather than only caller-supplied flags.

| `state-factor` option group | Required/default contract | Validation and precedence |
| --- | --- | --- |
| counts/cells/genes/cell-type-column/rank/out | required | Existing paths; positive rank; destination absent |
| solver/tol/maxiter/seed | all required | Solver is `arpack` or `propack`; positive finite tolerance; positive iteration control; integer seed; no environment fallback |
| ncv | required only for ARPACK; forbidden for PROPACK | After shape load, `rank < ncv < min(M,q_active)` |
| exclusion mode | exactly one of one autosome or LOCO | Canonical `1`-`22` only |
| cell selection | omitted for one observed type; otherwise exactly selected type or mixed opt-in | Explicit selection and mixed opt-in are mutually exclusive |
| PFlog alpha | `auto` | Explicit override must be positive finite and replaces `auto` |
| donor centering/balancing | both true | Explicit BooleanOptionalAction flags replace independently |
| platform/verbosity | fixed CPU; verbosity false | No device flag; `--verbose` only changes logging level |

- Add a reusable CLI-only logging context in `log.py` that installs fresh jaxqtl-owned console and optional disk handlers for one command invocation, tags ownership, and closes/removes only those handlers on exit. Use it for all CLI commands so established commands retain their `<out>.log` sidecars while `state-factor` passes no disk path and can never create one. Do not let handler state leak across repeated in-process calls or remove unrelated externally installed handlers.
- Keep the handler thin: load/select data, construct a single or streaming LOCO result iterator, write the atomic artifact, log the final path/diagnostics, and return `0`. Translate structural I/O/factorization/artifact failures to an actionable error log and return `1`; let argparse usage failures remain exit `2`.
- Change `main` to return the selected handler's integer status instead of discarding it. Emit the established `Finished! Thank you!` message only when status is `0`; on status `1`, preserve the handler's actionable error and emit no success completion text. Existing handlers return `0`; preserve their parsing, output, and success behavior.
- Use console-only logging for directory-producing `state-factor` so `get_logger` does not create `<output-directory>.log` beside an artifact before atomic publication. Preserve the existing disk-log naming for all established commands.

**Testing:**
- Write help/parser/dispatch tests first and confirm they fail before implementation. Use `redirect_stdout`/`redirect_stderr` or direct parser invocation; capture fixtures are unavailable because tests run with `-p no:capture`.
- Verify the exact effective-option/help table, required arguments, mutual exclusions, strict alpha parsing, solver-specific option validation, one-chromosome and `--loco` dispatch, single/mixed cell-type behavior, CPU/seed/config replay in the manifest, success `0`, runtime `1` without the success message, and argparse usage `2`.
- Invoke established and new commands repeatedly in both orders within one process. Assert established sidecars remain unchanged, new commands never create a sidecar, owned streams/handlers are closed and removed after every call, stderr has no duplicate lines, and unrelated logger handlers survive.
- Run a deterministic small end-to-end fixture that writes and reloads a valid artifact. Also verify existing `cis`, `nominal`, `trans`, and `compute-pcs` help/dispatch behavior remains unchanged.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cli/test_state_cli.py -x`
- Expected red: `state-factor` parser/handler cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cli/test_state_cli.py`
- Expected green: all new CLI and regression cases pass.

**Commit:** `feat: add state-factor command`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Run Phase 2 regression and quality gates

**Verifies:** cis-state.AC1.1, cis-state.AC1.2, cis-state.AC1.3, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6, cis-state.AC1.7, cis-state.AC5.1, cis-state.AC5.2

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Confirm every application source file retains its functional-core or imperative-shell classification and that public state/artifact results expose no SciPy solver objects.
- Audit the final diff for dense transformed count allocation, weighted-factor leakage, partial artifact visibility, and accidental changes to pseudobulk APIs.
- Inspect `git status --short`; keep `/Users/nicholas/Projects/jaxqtl/state.md` untracked and unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_io/test_single_cell.py tests/test_state/test_pflog.py tests/test_state/test_factor.py tests/test_io/test_state_artifact.py tests/test_cli/test_state_cli.py`
- Expected: all Phase 1-2 tests pass.
- Run: `uv run pytest -p no:capture`
- Expected: the full suite passes with no new warnings or regressions.
- Run: `uv run ruff check src tests scripts`
- Expected: no lint errors.
- Run: `uv run ruff format --check src tests scripts`
- Expected: all files already formatted.
- Run: `uv run ty check src tests scripts`
- Expected: no type-check errors.
- Run: `rg -n '^# pattern: Imperative Shell$' src/jaxqtl/cli.py src/jaxqtl/log.py`
- Expected: both modified shell files have an explicit FCIS classification.
- Run: `git diff --check`
- Expected: no whitespace errors.
- Verifies: cis-state.AC1.1, cis-state.AC1.2, cis-state.AC1.3, cis-state.AC1.4, cis-state.AC1.5, cis-state.AC1.6, cis-state.AC1.7, cis-state.AC5.1, cis-state.AC5.2.

**Commit:** None unless verification reveals a required fix; fix it with the owning task's tests and a narrowly scoped commit.
<!-- END_TASK_7 -->
