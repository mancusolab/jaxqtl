from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import Literal

import numpy as np
import polars as pl
import qtl.io
import qtl.norm

import equinox as eqx
import jax

from jax import numpy as jnp
from jax.scipy import stats as jsp_stats
from jaxtyping import Array, PRNGKeyArray

from ._utils import validate_user_columns


@dataclass
class ExpressionData:
    """Phenotype matrix plus gene-level metadata and library sizes."""

    pheno: pl.DataFrame
    pheno_meta: pl.DataFrame
    libsize: pl.DataFrame

    def __iter__(self):
        """Yield expression values and genomic metadata for each phenotype."""
        for chrom, start, end, gene in self.pheno_meta.iter_rows():
            expr = self.pheno.get_column(gene).to_jax().astype(float)  # ug i dont like this casting
            yield expr, gene, chrom, start, end

    def to_jax(self):
        """Return expression values as a JAX array (samples x phenotypes)."""
        return self.pheno.select(pl.all().exclude("iid")).to_jax().astype(float)  # ug i dont like this casting

    @property
    def offset_from_libsize(self) -> pl.DataFrame:
        """Compute log library size offsets."""
        return self.libsize.with_columns(pl.col("libsize").log().alias("offset")).select(["iid", "offset"])

    def filter_genes_by_percentage(self, express_percent: float) -> "ExpressionData":
        """Keep genes expressed in more than `express_percent` of samples."""
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
        """Keep samples expressing more than `express_percent` of genes."""
        if not (0 <= express_percent <= 1):
            raise ValueError("`express_percent` must be between 0 and and 1")

        pheno = (
            self.pheno.with_columns(pl.mean_horizontal(pl.all().exclude("iid") > 0).alias("prop"))
            .filter(pl.col("prop") > express_percent)
            .drop("prop")
        )
        libsize = self.libsize.join(pheno, on="iid", how="semi", maintain_order="right")
        return ExpressionData(pheno=pheno, pheno_meta=self.pheno_meta, libsize=libsize)

    def compute_pcs(
        self,
        num_pcs: int,
        rng_key: PRNGKeyArray,
        transform: Literal["log1p", "tmm"] | None = None,
    ) -> pl.DataFrame:
        """Compute PCA scores from expression data with optional transformation."""
        if num_pcs < 1:
            raise ValueError("`num_pcs` must be greater than 0")

        num = pl.all().exclude("iid")
        pheno = self.pheno.select(num).to_jax()

        if transform == "tmm":
            raise NotImplementedError("'tmm' transform not implemented yet.")
            tmm_counts_df = edger_cpm(pheno, normalized_lib_sizes=True)
            pheno = inverse_normal_transform(tmm_counts_df)
        elif transform == "log1p":
            pheno = jnp.log1p(pheno)  # prevent log(0)

        pheno = (pheno - pheno.mean(axis=0)) / pheno.std(axis=0)  # standardize genes
        n, _ = pheno.shape
        U = _prob_pca(rng_key, pheno, num_pcs)
        data = {"iid": self.pheno.get_column("iid").to_numpy()}
        for i, eigvec in enumerate(U.T):
            data[f"ExprPC{i}"] = np.asarray(eigvec)

        df_pcs = pl.DataFrame(data=data)

        return df_pcs

    @classmethod
    def from_bedfile(
        cls,
        path_or_filename: str | PathLike,
        keep_individuals: list[str] | None = None,
        drop_individuals: list[str] | None = None,
        keep_pheno: list[str] | None = None,
        drop_pheno: list[str] | None = None,
    ):
        """Load expression, metadata, and library size information from a BED-like file."""
        if keep_individuals and drop_individuals:
            raise ValueError("Cannot specify both `keep_individuals` and `drop_individuals`")
        if keep_pheno and drop_pheno:
            raise ValueError("Cannot specify both `keep_pheno` and `drop_pheno`")
        if not isinstance(path_or_filename, (str, PathLike)):
            raise ValueError(f"`path_or_filename` must be `str` or `PathLike`, not {type(path_or_filename)}")

        # load using a lazy frame to speed things up in Rust-based parsing before moving into Python space
        name = str(path_or_filename)
        if name.endswith((".bed", ".bed.gz")):
            phenotype_lf = pl.scan_csv(path_or_filename, separator="\t", has_header=True)
        elif name.endswith((".parquet", ".parquet.gz")):
            phenotype_lf = pl.scan_parquet(path_or_filename)
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

        if keep_individuals is not None:
            keep_individuals = validate_user_columns(keep_individuals, colnames)
            # make sure for some weird reason individual ids passed in don't also have the 4 required column names
            keep_individuals = list(set(keep_individuals) - set(resolved))

            # restrict to the 4 required + individuals specified
            columns = resolved + keep_individuals
        elif drop_individuals is not None:
            drop_individuals = validate_user_columns(drop_individuals, colnames)
            # restrict to the required columns - individuals specified; ensure that the 4 required columns stay if for
            # some weird reason the user specified them as individuals to drop
            columns = list(set(colnames) - (set(drop_individuals) - set(resolved)))
        else:
            columns = colnames

        phenotype_lf = phenotype_lf.select(columns)

        # compute library size from entire counts, before filting out genes
        libsize = (
            phenotype_lf.select(columns[4:])
            .sum()
            .collect()
            .transpose(include_header=True, header_name="iid", column_names=["libsize"])
        )

        # recast chrom col to str
        meta_lf = phenotype_lf.select(resolved).with_columns(pl.col(resolved[0]).cast(pl.Utf8))

        # drop '#' from col-name if its there
        normalized_chrom = "chrom"
        normalized_pheno = "phenotype_id"
        if colnames[0][0] == "#":
            meta_lf = meta_lf.rename({colnames[0]: normalized_chrom, colnames[3]: normalized_pheno})
        else:
            meta_lf = meta_lf.rename({colnames[3]: normalized_pheno})

        # go eager to pull out pheno names
        meta_lf = meta_lf.collect()
        genes = meta_lf.get_column(normalized_pheno).to_list()
        if keep_pheno is not None:
            keep_pheno = validate_user_columns(keep_pheno, genes)
            keep_genes = pl.Series("genes", keep_pheno)
            meta_lf = meta_lf.filter(pl.col(normalized_pheno).is_in(keep_genes))
        elif drop_pheno is not None:
            drop_pheno = validate_user_columns(drop_pheno, genes)
            keep_genes = pl.Series("genes", list(set(genes) - set(drop_pheno)))
            meta_lf = meta_lf.filter(pl.col(normalized_pheno).is_in(keep_genes))
        else:
            keep_genes = pl.Series("genes", genes)

        # everything after 4th col is expression data
        # go eager mode, then transpose and relabel everything
        # then restrict to either all genes, or user-specified genes
        phenotype_lf = (
            phenotype_lf.select(columns[4:])
            .collect()
            .transpose(
                include_header=True,
                header_name="iid",
                column_names=genes,
            )
            .select(["iid"] + keep_genes.to_list())
        )

        # join libsize based on final filtered samples
        libsize = libsize.join(phenotype_lf, on="iid", how="semi", maintain_order="right")

        return cls(phenotype_lf, meta_lf, libsize)


def bed_transform_y(pheno_path: str | PathLike[str], method: str = "log1p"):
    """Perform transformation on gene expression count matrix
    count_df: rows are genes, columns are individual ID
    """
    count_df = pl.read_csv(
        pheno_path,
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
        # use edger TMM method to calculate size factor and convert to counts per million
        tmm_counts = qtl.norm.edger_cpm(
            count_df.select(pl.col(expr_cols)).to_numpy(),
            normalized_lib_sizes=True,
        )
        # inverse normal transformation on each gene (row)
        norm_df = np.asarray(qtl.norm.inverse_normal_transform(tmm_counts))
        if norm_df.shape[0] == count_df.height:
            count_df = count_df.with_columns([pl.Series(name, norm_df[:, i]) for i, name in enumerate(expr_cols)])
        else:
            raise ValueError("row number doesn't match")
    else:
        raise ValueError(f"Unsupported mode {method}")

    return count_df


def edger_cpm(counts_df, tmm=None, normalized_lib_sizes=True):
    """
    Return edgeR normalized/rescaled CPM (counts per million)

    Reproduces edgeR::cpm.DGEList
    """
    lib_size = counts_df.sum(axis=0)
    if normalized_lib_sizes:
        if tmm is None:
            tmm = edger_calcnormfactors(counts_df)
        lib_size = lib_size * tmm
    return counts_df / lib_size * 1e6


def edger_calcnormfactors(
    counts,
    ref=None,
    logratio_trim=0.3,
    sum_trim=0.05,
    acutoff=-1e10,
):
    """
    JAX version of edgeR::calcNormFactors.default (TMM normalization).

    Parameters
    ----------
    counts : array-like, shape (G, S)
        Count matrix (genes x samples). Typically counts_df.values.
    ref : int or None
        Reference sample index. If None, chosen as in edgeR.
    logratio_trim : float
        Proportion to trim from M-values (log fold change).
    sum_trim : float
        Proportion to trim from A-values (average log expression).
    acutoff : float
        Minimum A-value.
    verbose : bool
        If True, print reference index (host-side, not jitted).

    Returns
    -------
    tmm : jax.numpy.ndarray, shape (S,)
        TMM normalization factors.
    """
    Y = jnp.asarray(counts, dtype=jnp.float32)  # shape (G, S)
    G, ns = Y.shape

    # library sizes
    N = jnp.sum(Y, axis=0)  # shape (S,)

    # select reference sample if not given
    if ref is None:
        Y_norm_tmp = Y / N
        f75 = jnp.percentile(Y_norm_tmp, 75.0, axis=0)  # shape (S,)
        dev = jnp.abs(f75 - jnp.mean(f75))
        ref = int(jnp.argmin(dev))

    # normalized counts and reference column
    Y_norm = Y / N
    ref_profile = Y[:, ref] / N[ref]  # shape (G,)

    # log fold change (M) and average log expression (A)
    logR = jnp.log2(Y_norm / ref_profile[:, None])  # shape (G, S)
    logYnorm = jnp.log2(Y_norm)
    log_ref = jnp.log2(ref_profile)
    absE = 0.5 * (logYnorm + log_ref[:, None])  # shape (G, S)

    # weights v (w in paper)
    v = (N - Y) / (N * Y)  # shape (G, S)
    v = v + v[:, ref][:, None]  # v_i + v_ref

    def tmm_for_sample(logR_col, absE_col, v_col):
        """
        Compute TMM factor for a single sample (column).
        logR_col, absE_col, v_col: shape (G,)
        """
        # finite & above A cutoff
        fin = jnp.isfinite(logR_col) & jnp.isfinite(absE_col) & (absE_col > acutoff)

        n = jnp.sum(fin)  # number of "valid" genes

        def nonempty_case(args):
            logR_col, absE_col, v_col, fin, n = args

            # Use NaNs to tell rankdata which entries to ignore
            logR_for_rank = jnp.where(fin, logR_col, jnp.nan)
            absE_for_rank = jnp.where(fin, absE_col, jnp.nan)

            rankR = jsp_stats.rankdata(
                logR_for_rank,
                method="average",
                axis=None,
                nan_policy="omit",
            )
            rankE = jsp_stats.rankdata(
                absE_for_rank,
                method="average",
                axis=None,
                nan_policy="omit",
            )

            n_f = n.astype(jnp.float32)

            loL = jnp.floor(n_f * logratio_trim) + 1.0
            hiL = n_f + 1.0 - loL
            loS = jnp.floor(n_f * sum_trim) + 1.0
            hiS = n_f + 1.0 - loS

            keep = fin & (rankR >= loL) & (rankR <= hiL) & (rankE >= loS) & (rankE <= hiS)

            w = v_col  # variance term; paper has a known typo about 1/v

            num = jnp.nansum(jnp.where(keep, logR_col / w, 0.0))
            den = jnp.nansum(jnp.where(keep, 1.0 / w, 0.0))

            # guard against den == 0
            tmm_val = jnp.where(den > 0.0, 2.0 ** (num / den), 1.0)
            return tmm_val

        # if no valid genes for this column, just return 1.0
        tmm_val = jax.lax.cond(
            n > 0,
            nonempty_case,
            lambda _: jnp.array(1.0, dtype=jnp.float32),
            (logR_col, absE_col, v_col, fin, n),
        )
        return tmm_val

    # vectorize over columns (samples)
    tmm = jax.vmap(tmm_for_sample, in_axes=1)(logR, absE, v)  # shape (S,)

    # center normalization factors to have geometric mean 1
    tmm = tmm / jnp.exp(jnp.mean(jnp.log(tmm)))
    return tmm


def inverse_normal_transform(pheno):
    """Apply inverse normal transform across observations."""
    r = jsp_stats.rankdata(pheno)
    return jsp_stats.norm.ppf(r / (pheno.shape[0] + 1))


@partial(jax.jit, static_argnums=(2, 3, 4))
def _prob_pca(rng_key, X, k, max_iter=1000, tol=1e-3) -> Array:
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
    Z, _ = jnp.linalg.qr(Z)

    return Z
