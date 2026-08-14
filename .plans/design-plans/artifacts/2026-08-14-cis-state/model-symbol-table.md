# Model symbol table

## Context
- Plan slug: `cis-state`
- Generated date: `2026-08-14`
- Source: `state.md` plus the confirmed decisions in `.scientific/kickoff.md`

The design uses `ell_c` for cell library size and `L` for the sparse shifted-log matrix. This removes the conflicting use of `L` in the source brief.

| Symbol | Meaning | Domain/support | Shape/type | Defined in source | Notes |
| --- | --- | --- | --- | --- | --- |
| `N` | Number of aligned donors | positive integer | scalar | `state.md` | Independent genetic units |
| `M` | Number of retained cells | positive integer | scalar | `state.md` | Cells are clustered within donors |
| `q` | Number of retained state-construction genes | positive integer | scalar | `state.md` | Chromosome-specific in LOCO mode |
| `p` | Number of valid standardized cis variants | positive integer | scalar | `state.md` | May vary by focal gene/block |
| `r` | State-factor rank | `1 <= r < min(M, q)` | scalar | `state.md` | Explicit rank is always supported |
| `d` | Number of small nuisance columns | nonnegative integer | scalar | design clarification | Includes state main effects and adjusted covariates |
| `i,c,g,j,k,h` | Donor, cell, gene, variant, state, and chromosome indices | integer index | scalar | `state.md` | Bounds follow corresponding dimensions |
| `X_count` | Raw state-construction counts | nonnegative integers | sparse `M x q` | `state.md` | Never CPM-normalized or densely transformed |
| `mu_g` | Mean raw count for gene `g` in selected cells | nonnegative real | scalar per gene | `state.md` | PFlog sufficient statistic |
| `v_g` | Raw-count variance for gene `g` | nonnegative real | scalar per gene | `state.md` | Uses the reference estimator convention |
| `A_PF`, `A_PF,h` | Total and chromosome-specific PFlog numerator sums | real | scalar | confirmed clarification | Sum of `mu_g^2 * (v_g - mu_g)` |
| `B_PF`, `B_PF,h` | Total and chromosome-specific PFlog denominator sums | nonnegative real | scalar | confirmed clarification | Sum of `mu_g^4` |
| `alpha_-h` | LOCO PFlog overdispersion | positive finite real | scalar | confirmed clarification | `(A_PF - A_PF,h) / (B_PF - B_PF,h)`; never silently clipped |
| `L^(-h)` | Sparse shifted-log matrix | real, structurally sparse | `M x q_-h` operator | `state.md` | `log1p(4 * alpha_-h * X_count^(-h))` |
| `J_q` | Feature CLR-centering operator | symmetric idempotent | `q x q` implicit operator | `state.md` | `I - 11^T/q` |
| `C_D` | Within-donor centering operator | symmetric idempotent | `M x M` implicit operator | `state.md` | Implemented by donor segment reductions, never materialized |
| `D_w` | Donor-balancing cell-weight diagonal | nonnegative diagonal | `M x M` implicit diagonal | `state.md` | Weight `1/[N(n_i-1)]` for donors with at least two cells |
| `Z` | Donor- and CLR-centered shifted-log expression | real | `M x q` implicit operator | `state.md` | `C_D L J_q` |
| `V_r` | Leading right singular vectors of `sqrt(D_w) Z` | orthonormal columns | `q x r` | `state.md` | Learned on donor-balanced operator |
| `S` | Cell-level state factors | real | `M x r` | `state.md` | `Z V_r`; not whitened or column-standardized |
| `Y`, `y` | Raw focal-gene UMI counts | nonnegative integers | `M` | `state.md` | Outcome for one focal gene |
| `ell` | Cell library size | positive real | `M` | `state.md` | Disambiguated from shifted-log `L` |
| `o` | Log offset | finite real | `M` or scalar | `state.md` | Normally `log(ell)` |
| `donor_index` | Dense cell-to-donor mapping | integer in `[0,N)` | `M` | `state.md` | Replaces donor incidence matrix |
| `alpha_i^donor` | Unrestricted donor fixed effect | real, boundary possible | `N` | `state.md` | Zero-total donors are removed before fitting |
| `C` | Optional cell-level nuisance covariates | finite real | `M x c_dim` | `state.md` | Does not include a global intercept |
| `D` | Required donor covariates | finite real | `N x d_D` | confirmed clarification | Includes genotype PCs, age, sex, and other donor characteristics |
| `X_nuis` | Small cell-level nuisance design after construction | finite real | `M x d` | `state.md` + clarification | Includes `S`, `C`, and `D x S`; donor-constant main effects are absent |
| `beta` | Non-donor nuisance coefficients | real | `d` | `state.md` | Fitted with absorbed IRLS |
| `phi` | NB2 dispersion | nonnegative finite real | scalar | `state.md` | `Var(Y)=mu+phi*mu^2`; distinct from PFlog `alpha` |
| `mu` | Fitted focal-gene mean | positive finite real | `M` | `state.md` | Log mean includes offset, donor effect, and nuisance design |
| `w` | NB IRLS weight | positive finite real | `M` | `state.md` | `mu/(1+phi*mu)` |
| `z` | IRLS working response after offset convention | finite real | `M` | `state.md` | Donor-centered within each iteration |
| `q_score` | GLM score residual | finite real | `M` | `state.md` | `(Y-mu)/(1+phi*mu)` |
| `G_raw` | Donor-level cis dosage | finite dosage | `N x p_raw` | `state.md` | Never expanded to cells |
| `G` | Valid standardized genotype matrix | finite, column mean zero and scale one | `N x p` | confirmed clarification | Monomorphic/invalid variants are removed and reported |
| `Gamma` | Cis-by-state interaction coefficients | real | `p x r` | `state.md` | `Gamma_jk ~ Normal(0,tau)`; no public feature weights |
| `H` | Conceptual interaction design | real | `M x (p*r)` | `state.md` | `H[c,(j,k)] = G[donor(c),j] * S[c,k]`; never materialized |
| `A_score` | Donor-state aggregation | real | `N x r` | `state.md` | Row `i` is `sum_{c in i} q_score[c] S[c,:]` |
| `U` | Full interaction-score matrix | real | `p x r` | `state.md` | `G^T A_score` |
| `Q` | Omnibus score statistic | nonnegative real | scalar | `state.md` | Frobenius norm squared of `U` after whole-kernel normalization |
| `X_0`, `S_0` | IRLS-weighted donor-centered nuisance/state arrays | real | `M x d`, `M x r` | `state.md` | Used for efficient nuisance projection |
| `J_XX` | Nuisance information | symmetric PSD | `d x d` | `state.md` | Solved through stable reduced linear algebra, not inverted explicitly |
| `J_HX` | Interaction-nuisance information | real | `(p*r) x d` conceptually | `state.md` | Constructed in variant blocks |
| `b_i` | Donor nuisance-score contribution | real | `d` per donor | `state.md` | `X_0,i^T q_i` |
| `u_i` | Raw donor interaction-score contribution | real | `p*r` conceptually | `state.md` | Outer product of standardized genotype and donor-state score |
| `t_i` | Efficient donor interaction-score contribution | real | `p*r` conceptually | `state.md` | `u_i - J_HX J_XX^{-1} b_i` |
| `T` | Matrix of efficient donor contributions | real | `N x (p*r)` conceptual operator | `state.md` | Never materialized at production scale |
| `xi_bi` | Donor multiplier | `-1` or `+1` | `B_boot x N` in blocks | `state.md` | Seeded Rademacher draw |
| `B_boot` | Number of multiplier replicates | positive integer | scalar | `state.md` | P-value resolution is `1/(B_boot+1)` |
| `B_cluster` | Donor-space covariance operator | symmetric PSD | `N x N` implicit operator | `state.md` | `T T^T` |
| `c_1,c_2,c_3` | First three spectral trace moments | nonnegative real | scalar | `state.md` | `c_1` exact; higher traces estimated with probes |
| `tau` | Interaction-variance scale under alternatives | nonnegative real | scalar | `state.md` | Zero under null, positive in power simulations |

## Checks
- [x] No undefined symbols in the design equations.
- [x] Conflicting library-size and shifted-log notation is disambiguated.
- [x] Support and domain constraints are explicit.
- [x] Conceptual prohibited matrices are labeled as implicit or never materialized.
