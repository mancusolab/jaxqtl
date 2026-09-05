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

NB2 initialization fits Poisson means, subtracts the Poisson variance contribution from the moment estimate of
dispersion, and takes at most one accepted, backtracked dispersion step at those means. Subsequent IRLS iterations
update coefficients and dispersion jointly. Each iteration computes one weighted least-squares direction;
backtracking reuses that direction at successively halved step sizes.

## Convergence

`tol` bounds the absolute change in total negative log likelihood. Once this change is small, IRLS checks the
gradients at the accepted coefficients and dispersion. A small likelihood change alone is not sufficient.
Under `vmap`, JAX can evaluate both conditional branches, so batched fits may perform these checks more often.

For `n` observations and design-column RMS magnitudes `s_j`, the coefficient criterion is
`max_j(abs(dNLL/dbeta_j) / (n * s_j)) <= gtol`. This makes the check insensitive to multiplying a covariate by a
constant. It does not whiten correlated covariates or require another linear solve.

NB2 additionally requires `abs(projected(dNLL/dalpha)) / n <= gtol`. At the lower dispersion bound (`1e-9`), a
nonnegative gradient satisfies the constraint; at the upper bound (`1e9`), a nonpositive gradient does. Interior
gradients must be small in either direction. The log-alpha gradient is converted to the alpha-space gradient so
that small alpha alone cannot produce apparent convergence.

Both tolerances default to `1e-3` and are exposed as CLI options `--tol` and `--gtol`. NB2's initial Poisson fit uses
the same tolerances. `converged=False` indicates exhausted backtracking, an iteration limit, or an unchanged
nonstationary state; reaching a dispersion bound is not by itself convergence.
