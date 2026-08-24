# Linear solvers

GLM fitting in `jaxqtl` reduces each IRLS iteration to a (weighted) least-squares solve. The solver choice can impact
both speed and numerical stability.

??? abstract "`jaxqtl.infer.AbstractLinearSolve`"

    ::: jaxqtl.infer.AbstractLinearSolve
        options:
            members:
                - wgt_lstsq
                - lstsq

---

::: jaxqtl.infer.QRSolve
    options:
        members: false

---

::: jaxqtl.infer.CholeskySolve
    options:
        members: false

---

::: jaxqtl.infer.CGSolve
    options:
        members: false
