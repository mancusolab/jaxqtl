# Statistical model

jaxQTL models a molecular phenotype `y_i` for individual `i` with covariates `x_i`, genotype `g_i`, and an optional
fixed offset `o_i`:

$$
g(\mu_i) = x_i^\mathsf{T}\beta + g_i\beta_g + o_i,
\qquad \mu_i = E(y_i \mid x_i, g_i).
$$

The link function `g` and response variance are determined by the selected family.

The CLI supports `gaussian` for continuous phenotypes, `poisson` for counts with variance equal to the mean,
and `nb` for overdispersed counts.

## Count models

For Poisson and Negative Binomial expression models, jaxQTL uses a log link. With library-size exposure, the model is

$$
\log(\mu_i) = x_i^\mathsf{T}\beta + g_i\beta_g + \log(l_i).
$$

The Negative Binomial model uses the NB2 variance

$$
\operatorname{Var}(y_i \mid x_i, g_i) = \mu_i + \alpha\mu_i^2,
$$

where `alpha` is the fitted overdispersion. Setting `alpha = 0` gives the Poisson variance.

## Effect interpretation

Under the log link, `exp(beta_g)` is the multiplicative change in the expected count for a one-unit increase in the
counted `a1` allele, holding covariates and exposure fixed.

!!! warning "Check the model and counted allele before interpreting effects"

    Do not apply the count-model rate-ratio interpretation to Gaussian models or to an allele other than the output
    `a1`. Always read the effect together with the selected family, link, and allele metadata.

See [Troubleshooting](troubleshooting.md) for NB2 estimation and solver controls,
[Tests and gene-level calibration](tests.md) for inference choices, and
[Models and distributions](../api/models/glm.md) for the Python API.
