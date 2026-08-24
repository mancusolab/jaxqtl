# Linear and generalized linear models

`jaxqtl` implements linear models and generalized linear models (GLMs) used during variant-level association testing.

??? abstract "`jaxqtl.infer.AbstractLinearModel`"

    ::: jaxqtl.infer.AbstractLinearModel
        options:
            show_bases: true
            members:
                - fit

---

::: jaxqtl.infer.LinearModel
    options:
        show_bases: true
        members:
            - fit

---

::: jaxqtl.infer.GeneralizedLinearModel
    options:
        show_bases: true
        members:
            - fit

## Fitted result

::: jaxqtl.infer.ModelResult
    options:
        members: false

## Fitting process

Generalized linear model fitting consists of:

1. Selecting a response family and link.
2. Solving weighted least-squares subproblems with the configured solver.
3. Estimating dispersion when the family requires it.
4. Computing a coefficient covariance matrix and Wald statistics.
