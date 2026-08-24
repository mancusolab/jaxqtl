# Link functions

Links map the mean parameter $\mu$ of a GLM family to a linear predictor $\eta$:

- forward: $\eta = g(\mu)$
- inverse: $\mu = g^{-1}(\eta)$

Each response family defines its valid links. Constructing a family with an incompatible link raises `ValueError`.

??? abstract "`jaxqtl.distribution.AbstractLink`"

    ::: jaxqtl.distribution.AbstractLink
        options:
            show_bases: true
            members:
                - __call__
                - inverse
                - deriv
                - inverse_deriv

## Concrete links

::: jaxqtl.distribution.IdentityLink
    options:
        members:
            - __init__

---

::: jaxqtl.distribution.LogLink
    options:
        members:
            - __init__

---

::: jaxqtl.distribution.LogitLink
    options:
        members:
            - __init__

---

::: jaxqtl.distribution.InverseLink
    options:
        members:
            - __init__

---

::: jaxqtl.distribution.PowerLink
    options:
        members:
            - __init__

---

::: jaxqtl.distribution.NBLink
    options:
        members:
            - __init__
