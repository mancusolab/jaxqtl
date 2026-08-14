# Solver feasibility matrix

## Context
- Plan slug: `cis-state`
- Generated date: `2026-08-14`

| Candidate | Problem-form fit | AD compatibility | Constraint handling | Status/error mapping | Feasible | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| SciPy `LinearOperator` + `svds` | Matrix-free partial SVD of the donor-balanced state operator | Not JAX-transformable; AD is not required for preprocessing | Rank/tolerance/iteration controls; input alpha validation remains outside solver | Python exception plus explicit residual/convergence diagnostics in the state result | yes | Initial CPU backend. Benchmark PROPACK and ARPACK before fixing the default. |
| Lineax QR/Cholesky | Small dense weighted least-squares and nuisance-information solves | JAX JIT/AD compatible | Rank/conditioning handled through solver choice and result codes | `throw=False` internally with translated domain status | yes | Reuses `src/jaxqtl/infer/_solve.py`; never receives donor dummies. |
| Optimistix | Could express a generic nonlinear/profile solve | JAX JIT/AD compatible | Explicit tolerances and result codes | Optimistix result channel | conditional | Not selected initially because the existing NB update and a grouped IRLS loop already match the supplied deterministic update structure. Reconsider only if custom convergence behavior proves inadequate. |
| Existing `GeneralizedLinearModel` | Dense observation-level GLM fit and covariance | JAX JIT/AD compatible | Existing model checks | Existing result fields | no | Requires a dense design and covariance path; cannot absorb donor effects at target scale without changing established contracts. |
| Custom grouped IRLS shell | Alternating absorbed mean and profile-dispersion updates | JAX JIT compatible; component AD tests required | Boundary validation, fixed-shape loop state, reduced-solve result propagation | Stable `AbsorbedNBResult` status/message; no valid p-value on failure | yes | Model-specific shell that reuses family/link/dispersion and reduced solvers. |
| Custom donor-contribution operator | Blocked `T`, `T^T`, norm, and trace actions | JAX JIT/vmap compatible; not an optimization surface | Fixed block sizes and guarded kernel scales | Domain status for nonfinite/invalid moments | yes | Necessary because materializing `N x (p*r)` is prohibited. |

## Decision
- Preferred solver path: SciPy matrix-free `svds` for state construction; Lineax-backed reduced solves within a custom absorbed IRLS shell; a custom blocked donor operator for calibration.
- Reason: Each solver is applied to the problem shape it supports. No chosen path creates dense donor or interaction designs, and existing pseudobulk solvers remain unchanged.
- Benchmark or validation requirement before implementation: compare ARPACK and PROPACK convergence, residuals, runtime, and peak memory at representative `M`, `q`, and `r`; compare reduced solves with explicit dense least squares; compare the donor operator and trace estimates with a dense `T` on small fixtures.
