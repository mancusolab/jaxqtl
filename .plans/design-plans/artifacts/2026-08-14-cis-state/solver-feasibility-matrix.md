# Solver feasibility matrix

## Context
- Plan slug: `cis-state`
- Generated date: `2026-08-14`

| Candidate | Problem-form fit | AD compatibility | Constraint handling | Status/error mapping | Feasible | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| SciPy `LinearOperator` + `svds` | Matrix-free partial SVD of the donor-balanced state operator | Not JAX-transformable; AD is not required for preprocessing | Rank/tolerance/iteration controls; input alpha validation remains outside solver | Python exception plus explicit residual/convergence diagnostics in the state result | yes | Initial CPU backend. Benchmark PROPACK and ARPACK before fixing the default. |
| Existing JAX QR/Cholesky plus checked adapter | Small dense weighted least-squares in absorbed IRLS | JAX JIT/AD compatible | Host restricts the solver to `QRSolve`/`CholeskySolve`; kernel checks rank, finite output, scaled backward error, and `cond(X_w)^2*eps` before Cholesky | Array-valued kernel status translated by the host wrapper | yes | Reuses the two direct implementations in `src/jaxqtl/infer/_solve.py`; explicitly rejects `CGSolve`/unknown solvers because the existing interface discards convergence status. Never receives donor dummies. |
| Lineax Cholesky with `throw=False` | One SPD nuisance-information system with multiple right-hand sides in the efficient-score projection | JAX JIT/vmap compatible | Tag SPD only after rank plus `cond(J_XX)*eps` accuracy-budget checks; initialize once and require mapped statuses/scaled backward errors | Inspect mapped `Solution.result` arrays and translate on the host | yes | Used only by the Phase 4 projection adapter, not through `AbstractLinearSolve`. |
| Optimistix | Could express a generic nonlinear/profile solve | JAX JIT/AD compatible | Explicit tolerances and result codes | Optimistix result channel | conditional | Not selected initially because the existing NB update and a grouped IRLS loop already match the supplied deterministic update structure. Reconsider only if custom convergence behavior proves inadequate. |
| Existing `GeneralizedLinearModel` | Dense observation-level GLM fit and covariance | JAX JIT/AD compatible | Existing model checks | Existing result fields | no | Requires a dense design and covariance path; cannot absorb donor effects at target scale without changing established contracts. |
| Custom grouped IRLS shell | Alternating absorbed mean and profile-dispersion updates | JAX JIT compatible; component AD tests required | Boundary validation, fixed-shape loop state, checked direct reduced solves | Array status code/fixed diagnostics from the kernel; host-only `AbsorbedNBResult` enum/message; no valid p-value on failure | yes | Model-specific shell that reuses family/link/dispersion and the checked QR/Cholesky adapter. |
| Custom donor-contribution operator | Blocked `T`, `T^T`, norm, and trace actions | JAX JIT/vmap compatible; not an optimization surface | Fixed block sizes and guarded kernel scales | Domain status for nonfinite/invalid moments | yes | Necessary because materializing `N x (p*r)` is prohibited. |

## Decision
- Preferred solver path: SciPy matrix-free `svds` for state construction; checked existing JAX QR/Cholesky solves within a custom absorbed IRLS shell; checked Lineax Cholesky only for Phase 4's multi-RHS nuisance projection; and a custom blocked donor operator for calibration.
- Reason: Each solver is applied to the problem shape it supports. No chosen path creates dense donor or interaction designs, and existing pseudobulk solvers remain unchanged.
- Benchmark or validation requirement before implementation: compare ARPACK and PROPACK convergence, residuals, runtime, and peak memory at representative `M`, `q`, and `r`; compare reduced solves with explicit dense least squares; compare the donor operator and trace estimates with a dense `T` on small fixtures.
