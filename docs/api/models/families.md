# Response families

`jaxqtl` implements generalized linear model (GLM) families as exponential families with an associated link function.

??? abstract "`jaxqtl.distribution.ExponentialFamily`"

    ::: jaxqtl.distribution.ExponentialFamily
        options:
            show_bases: true
            members:
                - __init__
                - scale
                - negloglikelihood
                - variance
                - sample
                - calc_weight
                - init_eta
                - update_dispersion
                - estimate_dispersion

## Continuous families

::: jaxqtl.distribution.Gaussian
    options:
        show_bases: true
        members:
            - __init__

---

::: jaxqtl.distribution.Gamma
    options:
        show_bases: true
        members:
            - __init__

---

## Discrete families

::: jaxqtl.distribution.Poisson
    options:
        show_bases: true
        members:
            - __init__

---

::: jaxqtl.distribution.NegativeBinomial
    options:
        show_bases: true
        members:
            - __init__

---

::: jaxqtl.distribution.Binomial
    options:
        show_bases: true
        members:
            - __init__

## Distribution utilities

::: jaxqtl.distribution.t_cdf

---

::: jaxqtl.distribution.ncx2_sf
