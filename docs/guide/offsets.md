# Offsets

Poisson and Negative Binomial models use an offset to represent known exposure. For pseudobulk expression, the
usual exposure is library size:

$$
\log(\mu_i) = x_i^\mathsf{T}\beta + g_i\beta_g + \log(l_i).
$$

Here, `l_i` is the library size and the value supplied to the model is `log(l_i)`.

Every retained sample must have a positive library size. Omit donor–cell-type combinations with no retained cells or
zero total abundance before mapping.

!!! warning "Pass log exposure, not raw exposure"

    The offset is added to the linear predictor. Passing raw library sizes changes the model and can produce invalid
    estimates. Offset files and offset covariates must therefore contain log-scale values.

The mapping commands accept one of three mutually exclusive offset sources:

- `--offset PATH` reads an `iid` column and one log-offset column from a tab-delimited file.
- `--offset-name-from-covar NAME` uses an existing covariate column as the fixed offset.
- `--set-offset-from-libsize` computes log library size from the loaded phenotype matrix.

## When to compute library size

Compute library size before filtering phenotypes. Removing genes first changes the total exposure. The
`ExpressionData` loader follows this order when phenotype filters are supplied, but it cannot recover genes that were
removed before the file was written.

Use a precomputed offset when the phenotype input is already restricted to a chromosome, gene list, or expression
subset.
