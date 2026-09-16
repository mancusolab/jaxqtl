# pattern: Imperative Shell

import csv
import gzip

from collections.abc import Collection
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import Literal

import numpy as np
import polars as pl

import equinox as eqx
import jax

from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ._normalization import edger_calcnormfactors, edger_cpm, inverse_normal_transform
from ._utils import validate_sample_ids, validate_user_columns


@dataclass
class ExpressionData:
    r"""Expression values, feature metadata, and library sizes aligned by sample.

    **Attributes:**

    - `pheno`: Sample-by-feature Polars frame. The first column is `iid`; all
      remaining columns contain expression values named by phenotype ID.
    - `pheno_meta`: Feature-level Polars frame with chromosome, start, end, and
      `phenotype_id` columns, in that order and matching the expression columns.
    - `libsize`: Polars frame with `iid` and `libsize` columns. Library sizes are
      computed before optional phenotype filtering.
    """

    pheno: pl.DataFrame
    pheno_meta: pl.DataFrame
    libsize: pl.DataFrame

    def __iter__(self):
        r"""Yield expression values and genomic metadata for each phenotype.

        **Returns:**

        An iterator of `(expression, phenotype_id, chrom, start, end)` tuples.
        `expression` is a floating-point JAX array with shape `(n,)`.
        """
        for chrom, start, end, gene in self.pheno_meta.iter_rows():
            expr = self.pheno.get_column(gene).to_jax().astype(float)  # ug i dont like this casting
            yield expr, gene, chrom, start, end

    def to_jax(self):
        r"""Convert expression values to a floating-point JAX array.

        **Returns:**

        An array with shape `(n, g)`, where rows are samples and columns are
        phenotypes. The `iid` column is excluded.
        """
        return self.pheno.select(pl.all().exclude("iid")).to_jax().astype(float)  # ug i dont like this casting

    def validate_values(self, *, require_nonnegative: bool = False) -> None:
        """Reject nonnumeric, missing, or nonfinite expression; optionally require counts >= 0."""
        values = self.pheno.select(pl.exclude("iid"))
        if not all(dtype.is_numeric() for dtype in values.dtypes):
            raise ValueError("Expression values must be numeric and finite")
        if not values.select(pl.all().is_finite().fill_null(False).all()).to_numpy().all():
            raise ValueError("Expression values must be finite")
        if require_nonnegative and not values.select((pl.all() >= 0).all()).to_numpy().all():
            raise ValueError("Log transforms require nonnegative expression values")

    def filter_genes_by_ids(self, *, keep: list[str] | None = None, drop: list[str] | None = None) -> "ExpressionData":
        """Select gene IDs in metadata order, preserving samples and original library sizes.

        `keep` and `drop` are mutually exclusive. Unknown IDs raise `ValueError`.
        With neither selection supplied, return this expression container unchanged.
        """
        if keep is not None and drop is not None:
            raise ValueError("Cannot specify both `keep` and `drop` gene IDs")
        if keep is None and drop is None:
            return self

        observed = self.pheno_meta.get_column("phenotype_id").to_list()
        selected = validate_user_columns(keep if keep is not None else drop, observed)
        matches = pl.col("phenotype_id").is_in(selected)
        meta = self.pheno_meta.filter(matches if keep is not None else ~matches)
        pheno = self.pheno.select(["iid", *meta.get_column("phenotype_id").to_list()])
        return ExpressionData(pheno, meta, self.libsize)

    def filter_genes_by_chromosomes(self, chromosomes: Collection[str]) -> "ExpressionData":
        r"""Keep phenotypes whose chromosome label is in `chromosomes`.

        Expression columns and phenotype metadata remain in phenotype-metadata
        order. Library sizes are preserved from the original expression input.

        **Arguments:**

        - `chromosomes`: Exact chromosome labels to retain.

        **Returns:**

        A new `ExpressionData` containing only phenotypes on the requested
        chromosomes.
        """
        chromosome_labels = list(chromosomes)
        meta = self.pheno_meta.filter(pl.col("chrom").cast(pl.Utf8).is_in(chromosome_labels))
        names = meta.get_column("phenotype_id").to_list()
        pheno = self.pheno.select(["iid", *names])
        return ExpressionData(pheno=pheno, pheno_meta=meta, libsize=self.libsize)

    @property
    def offset_from_libsize(self) -> pl.DataFrame:
        r"""Compute log-library-size offsets.

        **Returns:**

        A Polars frame with `iid` and `offset` columns in the same sample order as
        `libsize`. The offset is `log(libsize)`.
        """
        return self.libsize.with_columns(pl.col("libsize").log().alias("offset")).select(["iid", "offset"])

    def filter_genes_by_percentage(self, express_percent: float) -> "ExpressionData":
        r"""Keep phenotypes expressed in more than a fraction of samples.

        Expression is defined as a value greater than zero. Library sizes are
        preserved from the unfiltered expression matrix.

        **Arguments:**

        - `express_percent`: Exclusive lower bound on the expressed-sample fraction,
          between 0 and 1.

        **Returns:**

        A new `ExpressionData` containing only phenotypes above the threshold.

        **Raises:**

        - `ValueError`: If `express_percent` is outside `[0, 1]`.
        """
        if not (0 <= express_percent <= 1):
            raise ValueError("`express_percent` must be between 0 and and 1")
        col_means = (
            self.pheno.select((pl.all().exclude("iid") > 0).mean())
            .to_numpy()
            .ravel()  # compute mean for all non-iid columns
        )
        keep = col_means > express_percent
        names = np.array(self.pheno.columns[1:])[keep].tolist()

        pheno = self.pheno.select(["iid"] + names)
        meta = self.pheno_meta.filter(pl.col("phenotype_id").is_in(names))
        return ExpressionData(pheno=pheno, pheno_meta=meta, libsize=self.libsize)

    def filter_individuals_by_percentage(self, express_percent: float) -> "ExpressionData":
        r"""Keep samples expressing more than a fraction of phenotypes.

        Expression is defined as a value greater than zero. The returned expression
        and library-size frames contain the same retained samples and order.

        **Arguments:**

        - `express_percent`: Exclusive lower bound on the expressed-phenotype
          fraction, between 0 and 1.

        **Returns:**

        A new `ExpressionData` containing only samples above the threshold.

        **Raises:**

        - `ValueError`: If `express_percent` is outside `[0, 1]`.
        """
        if not (0 <= express_percent <= 1):
            raise ValueError("`express_percent` must be between 0 and and 1")

        pheno = (
            self.pheno.with_columns(pl.mean_horizontal(pl.all().exclude("iid") > 0).alias("prop"))
            .filter(pl.col("prop") > express_percent)
            .drop("prop")
        )
        libsize = self.libsize.join(pheno, on="iid", how="semi", maintain_order="right")
        return ExpressionData(pheno=pheno, pheno_meta=self.pheno_meta, libsize=libsize)

    def normalize(self, normalization: Literal["none", "library-size", "tmm"] = "library-size") -> "ExpressionData":
        """Scale expression to the median library size, without a log transform.

        `library-size` uses stored totals. `tmm` multiplies totals by TMM factors
        estimated from this container's current samples and nonzero genes.
        Estimate normalization before selecting PCA genes. `none` returns this
        container unchanged. Original library sizes and metadata are preserved.
        Invalid counts, sample IDs, or retained library sizes raise `ValueError`.
        """
        if normalization not in {"none", "library-size", "tmm"}:
            raise ValueError(f"Unknown PCA normalization: {normalization!r}")
        if normalization == "none":
            return self
        validate_sample_ids(self.pheno, "Expression")
        self.validate_values(require_nonnegative=True)
        validate_sample_ids(self.libsize, "Library sizes")
        if "libsize" not in self.libsize.columns:
            raise ValueError("Normalization requires library sizes")
        library_sizes = (
            self.pheno.select("iid")
            .join(self.libsize, on="iid", how="left", validate="1:1", maintain_order="left")
            .get_column("libsize")
            .cast(pl.Float64, strict=False)
            .to_numpy()
        )
        if library_sizes.size == 0 or not np.all(np.isfinite(library_sizes) & (library_sizes > 0)):
            raise ValueError("Normalization requires finite, positive library sizes for every sample")
        effective_sizes = library_sizes
        if normalization == "tmm":
            values = self.pheno.select(pl.exclude("iid"))
            nonzero = values.select((pl.all() > 0).any()).to_numpy().ravel()
            genes = [name for name, keep in zip(values.columns, nonzero, strict=True) if keep]
            if not genes:
                raise ValueError("TMM requires genes with positive counts")
            counts = values.select(genes).to_jax(dtype=pl.Float64 if jax.config.read("jax_enable_x64") else pl.Float32)
            factors = np.asarray(edger_calcnormfactors(counts.T, library_sizes=jnp.asarray(library_sizes)))
            if not np.all(np.isfinite(factors) & (factors > 0)):
                raise ValueError("TMM produced invalid normalization factors")
            effective_sizes = library_sizes * factors
        size_factors = effective_sizes / np.median(effective_sizes)
        pheno = self.pheno.with_columns(pl.exclude("iid") / pl.Series("size_factor", size_factors))
        return ExpressionData(pheno, self.pheno_meta, self.libsize)

    def compute_pcs(
        self,
        num_pcs: int,
        rng_key: PRNGKeyArray,
        *,
        normalization: Literal["none", "library-size", "tmm"] = "library-size",
        transform: Literal["none", "log1p"] = "log1p",
    ) -> tuple[pl.DataFrame, np.ndarray]:
        r"""Compute probabilistic-PCA scores from the expression matrix.

        Phenotypes are normalized and optionally log-transformed, constant genes are removed, and
        remaining genes are standardized across samples before fitting.
        The randomized initialization is determined by `rng_key`.
        A final projected SVD orders the unit-norm sample directions by decreasing
        explained variance within the fitted subspace.

        **Arguments:**

        - `num_pcs`: Number of expression principal components to return. It must
          not exceed `min(n_samples - 1, n_variable_genes)` after transformation.
        - `rng_key`: JAX PRNG key controlling the probabilistic-PCA initialization.
        - `normalization`: `"library-size"` (default) divides counts by library
          sizes relative to their median. `"tmm"` first adjusts sizes using TMM
          factors estimated from current genes. `"none"` skips normalization.
        - `transform`: `"log1p"` (default) applies `log(1 + y)` after normalization;
          `"none"` skips the log transform. For previously normalized/log-transformed
          inputs, explicitly disable the stages already applied.

        **Returns:**

        A tuple `(pcs, explained_variance_ratio)`. `pcs` is a Polars frame with `iid` followed by
        `ExprPC1` through `ExprPC{num_pcs}`, in decreasing variance order.
        Rows preserve the input sample order. Components have unit norm rather
        than being scaled by their singular values.
        `explained_variance_ratio` is a NumPy array of shape `(num_pcs,)` in
        the same component order. Each value is the squared projected singular
        value divided by the total sum of squares of the transformed,
        standardized matrix, including variance outside the fitted subspace.

        **Raises:**

        - `ValueError`: If there are fewer than two samples, no variable genes,
          invalid sample IDs, nonnumeric or nonfinite expression, an unsupported
          transform, negative expression for a log transform, or an invalid
          component count. Also raised if normalization library sizes are missing,
          nonnumeric, nonfinite, or nonpositive.
        """
        if num_pcs < 1:
            raise ValueError("`num_pcs` must be greater than 0")

        validate_sample_ids(self.pheno, "Expression")
        values = self.pheno.select(pl.exclude("iid"))
        if self.pheno.height < 2:
            raise ValueError("PCA requires at least two samples")
        if not values.width:
            raise ValueError("PCA requires variable genes after filtering")
        if transform not in {"none", "log1p"}:
            raise ValueError(f"Unknown PCA transform: {transform!r}")
        self.validate_values(require_nonnegative=normalization != "none" or transform == "log1p")
        normalized = self.normalize(normalization)
        values = normalized.pheno.select(pl.exclude("iid"))
        pheno = values.to_jax(dtype=pl.Float64 if jax.config.read("jax_enable_x64") else pl.Float32)
        if transform == "log1p":
            pheno = jnp.log1p(pheno)

        if not bool(jnp.all(jnp.isfinite(pheno))):
            raise ValueError("PCA transformed expression values must be finite")
        std = pheno.std(axis=0)
        variable = np.asarray(pheno.max(axis=0) != pheno.min(axis=0))
        pheno = pheno[:, variable]
        if not pheno.shape[1]:
            raise ValueError("PCA requires variable genes after transformation")
        max_pcs = min(pheno.shape[0] - 1, pheno.shape[1])
        if num_pcs > max_pcs:
            raise ValueError(f"`num_pcs` must not exceed {max_pcs} for the retained samples and variable genes")
        pheno = (pheno - pheno.mean(axis=0)) / std[variable]  # standardize genes
        if not bool(jnp.all(jnp.isfinite(pheno))):
            raise ValueError("PCA standardized expression values must be finite; check the expression scale")
        U, singular_values = _prob_pca(rng_key, pheno, num_pcs)
        explained_variance_ratio = singular_values**2 / jnp.sum(pheno**2)
        data = {"iid": self.pheno.get_column("iid").to_numpy()}
        for i, eigvec in enumerate(U.T, start=1):
            data[f"ExprPC{i}"] = np.asarray(eigvec)

        df_pcs = pl.DataFrame(data=data)

        return df_pcs, np.asarray(explained_variance_ratio)

    @classmethod
    def from_bedfile(
        cls,
        path_or_filename: str | PathLike,
        keep_individuals: list[str] | None = None,
        drop_individuals: list[str] | None = None,
        keep_pheno: list[str] | None = None,
        drop_pheno: list[str] | None = None,
    ):
        r"""Load expression data from a BED-like or Parquet table.

        The first four columns must be chromosome, start, end, and phenotype ID,
        using one of the accepted case-insensitive aliases. Remaining columns are
        sample IDs. Library sizes are computed from all loaded phenotypes before
        applying `keep_pheno` or `drop_pheno`.

        **Arguments:**

        - `path_or_filename`: `.bed`, `.bed.gz`, `.parquet`, or `.parquet.gz` input.
        - `keep_individuals`: Optional sample IDs to retain.
        - `drop_individuals`: Optional sample IDs to remove.
        - `keep_pheno`: Optional phenotype IDs to retain.
        - `drop_pheno`: Optional phenotype IDs to remove.

        **Returns:**

        An `ExpressionData` with samples in rows and phenotypes in columns.

        **Raises:**

        - `ValueError`: If keep and drop filters are both supplied for the same
          axis, requested names are missing, required metadata columns are invalid,
          or the file suffix is unsupported.
        """
        if keep_individuals is not None and drop_individuals is not None:
            raise ValueError("Cannot specify both `keep_individuals` and `drop_individuals`")
        if keep_pheno is not None and drop_pheno is not None:
            raise ValueError("Cannot specify both `keep_pheno` and `drop_pheno`")
        if not isinstance(path_or_filename, str | PathLike):
            raise ValueError(f"`path_or_filename` must be `str` or `PathLike`, not {type(path_or_filename)}")

        # load using a lazy frame to speed things up in Rust-based parsing before moving into Python space
        name = str(path_or_filename)
        if name.endswith((".bed", ".bed.gz")):
            open_file = gzip.open if name.endswith(".gz") else open
            with open_file(name, "rt") as stream:
                sample_ids = next(csv.reader(stream, delimiter="\t"), [])[4:]
            if any(not iid for iid in sample_ids) or len(set(sample_ids)) != len(sample_ids):
                raise ValueError("Expression must have nonempty, unique sample IDs")
            phenotype_lf = pl.scan_csv(name, separator="\t", has_header=True)
        elif name.endswith((".parquet", ".parquet.gz")):
            phenotype_lf = pl.scan_parquet(name)
        else:
            raise ValueError(f"File {path_or_filename} is unsupported for bed-style phenotype data.")

        schema = phenotype_lf.collect_schema()
        colnames = list(schema.keys())

        # Allowed options for each of the first four positions
        # this is messy and likely to be brittle if we want to analyze other molecular types
        # would be simpler to be strict about column names or make user specify the name of
        # the 4th phenotype name column
        resolved = []
        allowed = [
            {"chrom", "#chrom", "chr", "#chr"},
            {"start"},
            {"end"},
            {"pheno_id", "pheno", "gene_id", "geneid", "gene"},
        ]
        for i, options in enumerate(allowed):
            name = colnames[i]
            if name.lower() not in options:
                opts_str = ", ".join(sorted(options))
                raise ValueError(f"Column {i} expected to be one of `{opts_str}`, got `{name!r}`")
            resolved.append(name)

        samples = colnames[4:]
        if keep_individuals is not None:
            selected = set(validate_user_columns(keep_individuals, samples))
            samples = [sample for sample in samples if sample in selected]
        elif drop_individuals is not None:
            excluded = set(validate_user_columns(drop_individuals, samples))
            samples = [sample for sample in samples if sample not in excluded]
        if not samples:
            raise ValueError("No samples remain after sample selection")
        columns = resolved + samples

        phenotype_lf = phenotype_lf.select(columns)

        # Compute library sizes before filtering genes.
        libsize = (
            phenotype_lf.select(samples)
            .sum()
            .collect()
            .transpose(include_header=True, header_name="iid", column_names=["libsize"])
        )

        # recast chrom col to str
        meta_lf = phenotype_lf.select(resolved).with_columns(pl.col(resolved[0]).cast(pl.Utf8))

        # Canonicalize all accepted metadata aliases.
        normalized_chrom = "chrom"
        normalized_pheno = "phenotype_id"
        meta_lf = meta_lf.rename(dict(zip(resolved, [normalized_chrom, "start", "end", normalized_pheno], strict=True)))

        meta = meta_lf.collect()
        ids = meta.get_column(normalized_pheno)
        if ids.null_count() or ids.is_duplicated().any():
            raise ValueError("Expression phenotype IDs must be non-null and unique")
        genes = ids.to_list()

        pheno = (
            phenotype_lf.select(samples)
            .collect()
            .transpose(
                include_header=True,
                header_name="iid",
                column_names=genes,
            )
        )
        return cls(pheno, meta, libsize).filter_genes_by_ids(keep=keep_pheno, drop=drop_pheno)


def bed_transform_y(pheno_path: str | PathLike[str], method: str = "log1p"):
    r"""Transform expression values in a BED-style count matrix.

    The input's first four columns are preserved as feature metadata. Remaining
    columns are sample counts. Genes with zero counts across every sample are
    removed before transformation.

    **Arguments:**

    - `pheno_path`: Path to a tab-delimited BED-style matrix with four metadata
      columns followed by one count column per sample.
    - `method`: `"log1p"` applies `log(1 + count)` independently to each count.
      `"tmm"` applies TMM-normalized CPM followed by a row-wise inverse-normal
      transform across samples.

    **Returns:**

    A Polars frame containing the retained metadata columns and transformed
    sample-expression columns.

    **Raises:**

    - `ValueError`: If `method` is not `"log1p"` or `"tmm"`.
    """
    count_df = pl.read_csv(
        str(pheno_path),
        separator="\t",
        infer_schema_length=10000,
    )
    expr_cols = count_df.columns[4:]

    if "#chr" in count_df.columns:
        count_df = count_df.with_columns(pl.col("#chr").cast(pl.Utf8))
    if "#Chr" in count_df.columns:
        count_df = count_df.with_columns(pl.col("#Chr").cast(pl.Utf8))

    # filter genes with zero expression (first step of edger_cpm)
    # (must be done before transforms to keep row counts correct)
    count_df = count_df.filter(pl.sum_horizontal(pl.col(expr_cols)) > 0)

    if method == "log1p":
        count_df = count_df.with_columns([pl.col(name).log1p().alias(name) for name in expr_cols])
    elif method == "tmm":
        tmm_counts = edger_cpm(count_df.select(pl.col(expr_cols)).to_numpy())
        norm_df = np.asarray(inverse_normal_transform(tmm_counts))
        if norm_df.shape[0] == count_df.height:
            count_df = count_df.with_columns([pl.Series(name, norm_df[:, i]) for i, name in enumerate(expr_cols)])
        else:
            raise ValueError("row number doesn't match")
    else:
        raise ValueError(f"Unsupported mode {method}")

    return count_df


@partial(jax.jit, static_argnums=(2, 3, 4))
def _prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3) -> tuple[Array, Array]:
    import jax.lax as lax
    import jax.random as rdm
    import lineax as lx

    n_dim, p_dim = X.shape

    # initial guess for W
    w_key, z_key = rdm.split(rng_key, 2)

    # good enough for initialization
    solver = lx.Cholesky()

    multi_linear_solve = eqx.filter_vmap(lx.linear_solve, in_axes=(None, 1, None))

    # check if reach the max_iter, or met the norm criterion every 100 iteration
    def _condition(carry):
        i, _, Z, old_Z = carry
        iter_check = i < max_iter
        tol_check = jnp.linalg.norm(Z - old_Z) > tol
        # scaled_tol_check = tol_check / n_dim > tol
        return iter_check & tol_check

    # EM algorithm for PPCA
    def _step(carry):
        i, W, Z, _ = carry

        # E step
        W_op = lx.MatrixLinearOperator(W @ W.T, tags=lx.positive_semidefinite_tag)
        Z_new = multi_linear_solve(W_op, W @ X.T, solver).value

        # M step
        Z_op = lx.MatrixLinearOperator(Z_new.T @ Z_new, tags=lx.positive_semidefinite_tag)
        W = multi_linear_solve(Z_op, Z_new.T @ X, solver).value.T

        return i + 1, W, Z_new, Z

    W = rdm.normal(w_key, shape=(k, p_dim))
    Z = rdm.normal(z_key, shape=(n_dim, k))
    Z_zero = jnp.zeros_like(Z)
    initial_carry = 0, W, Z, Z_zero

    _, W, Z, _ = lax.while_loop(_condition, _step, initial_carry)
    Q, _ = jnp.linalg.qr(Z)
    # Rotate the EM subspace into variance-ordered principal directions using a k-by-p SVD.
    rotation, singular_values, _ = jnp.linalg.svd(Q.T @ X, full_matrices=False)
    return Q @ rotation, singular_values
