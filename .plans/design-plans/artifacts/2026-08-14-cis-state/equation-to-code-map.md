# Equation-to-code map

## Context
- Plan slug: `cis-state`
- Generated date: `2026-08-14`

Target names identify stable responsibility boundaries. Exact private filenames may consolidate during implementation.

| Equation ID | Equation | Intended computation | Target responsibility | Test/criterion target | Status |
| --- | --- | --- | --- | --- | --- |
| EQ-1 | `alpha_-h = (A_PF-A_PF,h)/(B_PF-B_PF,h)` | LOCO PFlog estimate from cached per-gene sufficient statistics | `state.estimate_pflog_alpha` | `cis-state.AC1.3` | planned |
| EQ-2 | `L = log1p(4*alpha_-h*X_count)` | Zero-preserving shifted-log transform of sparse values | PFlog sparse operator | `cis-state.AC1.4` | planned |
| EQ-3 | `Z = C_D L J_q` | Implicit CLR and within-donor centering | state composite operator | `cis-state.AC1.5` | planned |
| EQ-4 | `D_w[c,c] = 1/[N(n_i-1)]` | Equal donor influence in basis learning | donor-balanced state operator | `cis-state.AC1.6` | planned |
| EQ-5 | `sqrt(D_w)Z = U Sigma V^T`, `S=ZV_r` | Matrix-free right singular basis and unwhitened factors | state factorization | `cis-state.AC1.7` | planned |
| EQ-6 | `log(mu)=o+alpha_i^donor+X_nuis beta` | NB2 null mean with absorbed donor effects | absorbed cell-NB inference | `cis-state.AC2.2` | planned |
| EQ-7 | `X_0^T W X_0 beta = X_0^T W z_0` | Exact weighted FWL reduced solve | grouped absorbed WLS | `cis-state.AC2.3` | planned |
| EQ-8 | `w=mu/(1+phi*mu)` | NB2 log-link IRLS weights | existing family plus absorbed IRLS | `cis-state.AC2.4` | planned |
| EQ-9 | `q_score=(Y-mu)/(1+phi*mu)` | Mean score used by the omnibus test | absorbed-NB result | `cis-state.AC2.4` | planned |
| EQ-10 | `A_score[i,:]=sum_{c in i} q_score[c]S[c,:]` | Donor-state score aggregation | omnibus interaction score | `cis-state.AC3.3` | planned |
| EQ-11 | `U=G^T A_score`, `Q=||U||_F^2` | Standardized-genotype omnibus statistic | omnibus interaction score | `cis-state.AC3.3` | planned |
| EQ-12 | `t_i=u_i-J_HX J_XX^{-1}b_i` | Nuisance-efficient donor contribution | efficient cluster-score operator | `cis-state.AC3.4` | planned |
| EQ-13 | `U_b*=sum_i xi_bi t_i` | Blocked donor multiplier score | multiplier calibration | `cis-state.AC4.1` | planned |
| EQ-14 | `p=(1+sum_b I(Q_b*>=Q))/(B_boot+1)` | Finite-replicate Monte Carlo p-value | multiplier calibration | `cis-state.AC4.1` | planned |
| EQ-15 | `B_cluster=T T^T` | Matrix-free donor-space covariance action | moment calibration operator | `cis-state.AC4.4` | planned |
| EQ-16 | `c_l=tr(B_cluster^l)` | Exact/probed spectral moments with MC error | moment calibration | `cis-state.AC4.4` | planned |
| EQ-17 | `Gamma_jk ~ Normal(0,tau)` | Alternative interaction generator after fixed genotype scaling | cis-state simulator | `cis-state.AC7.2` | planned |

## Checks
- [x] Optimization direction and score-statistic sign match the supplied model.
- [x] Every update rule maps to a concrete computational responsibility.
- [x] Every mapped equation has a planned acceptance-criterion target.
- [x] No mapping requires a prohibited dense matrix.
