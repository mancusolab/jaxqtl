# Optimization routines

These lower-level routines support GLM fitting and permutation calibration. Most CLI and mapping users should
construct a model rather than call them directly.

## Iteratively reweighted least squares (IRLS)

::: jaxqtl.infer.irls

---

::: jaxqtl.infer.lstsq

---

::: jaxqtl.infer.SolveResult
    options:
        members: false

## Beta approximation for permutation p-values

For cis mapping, `jaxqtl` can fit a Beta approximation to the distribution of permutation p-values:

::: jaxqtl.infer.infer_beta_params

---

::: jaxqtl.infer.BetaParams
    options:
        members: false
