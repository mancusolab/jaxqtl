# Models

`jaxqtl` fits linear models and generalized linear models (GLMs) to associate variants with molecular phenotypes.

## Summary

- Choose a family (Gaussian, Poisson, Negative Binomial) depending on the phenotype type.
- Provide covariates `X` (and optionally an offset).
- Fit the null (covariate-only) model once per phenotype, then test many variants efficiently.

See the API reference under `API → Statistical Models` for the concrete classes.
