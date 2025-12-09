import gzip
import os
import re

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import Any, Literal, Optional, Union

import decoupler as dc
import numpy as np
import pandas as pd
import polars as pl
import qtl.io
import qtl.norm
import scanpy as sc

from anndata import AnnData
from scipy.sparse import diags

import equinox as eqx
import jax

from jax import numpy as jnp
from jax.scipy import stats as jsp_stats
from jaxtyping import Array, PRNGKeyArray

from .utils import validate_user_columns


@dataclass
class ExpressionData:
    pheno: pl.DataFrame
    pheno_meta: pl.DataFrame
    # libsize: Array

    def __iter__(self):
        for chrom, start, end, gene in self.pheno_meta.iter_rows():
            expr = self.pheno.get_column(gene).to_jax().astype(float)  # ug i dont like this casting
            yield expr, gene, chrom, start, end

    def to_jax(self):
        return self.pheno.select(pl.all().exclude("iid")).to_jax().astype(float)  # ug i dont like this casting

    @property
    def offset_from_libsize(self) -> pl.DataFrame:
        return self.pheno.select(
            pl.col("iid"),
            pl.sum_horizontal(pl.exclude("iid")).log().alias("offset"),
        )

    def filter_genes_by_percentage(self, express_percent: float) -> "ExpressionData":
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
        return ExpressionData(pheno=pheno, pheno_meta=meta)

    def filter_individuals_by_percentage(self, express_percent: float) -> "ExpressionData":
        if not (0 <= express_percent <= 1):
            raise ValueError("`express_percent` must be between 0 and and 1")

        pheno = (
            self.pheno.with_columns(pl.mean_horizontal(pl.all().exclude("iid") > 0).alias("prop"))
            .filter(pl.col("prop") > express_percent)
            .drop("prop")
        )
        return ExpressionData(pheno=pheno, pheno_meta=self.pheno_meta)

    def compute_pcs(
        self,
        num_pcs: int,
        rng_key: PRNGKeyArray,
        transform: Optional[Literal["log1p", "tmm"]] = None,
    ) -> pl.DataFrame:
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
        path_or_filename: Union[str, PathLike],
        keep_individuals: Optional[list[str]] = None,
        drop_individuals: Optional[list[str]] = None,
        keep_pheno: Optional[list[str]] = None,
        drop_pheno: Optional[list[str]] = None,
    ):
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

        return cls(phenotype_lf, meta_lf)


# TODO: need find out commonly used parameters
@dataclass
class SingleCellFilter:
    """Filtering metric for single cell data"""

    id_col: str = "donor_id"
    celltype_col: str = "cell_type"
    mt_col: str = "percent.mt"
    geneid_col: str = "ensemble_id"
    layer: Optional[str] = None  # which layer to perform
    min_cells: int = 3
    min_genes: int = 200
    max_genes: int = 2500  # can decide this based on plotting
    percent_mt: int = 5  # 5 means 5%
    norm_target_sum: float = 1e5  # not recommended
    bulk_method: str = "mean"
    bulk_min_prop: float = 0.0  # Minimum proportion of cells that express a gene in a sample.
    bulk_min_smpls: int = 0  # Minimum number of samples with >= proportion of cells with expression than min_prop
    bulk_min_cells: int = 0
    bulk_min_count: int = 0


class PhenoIO(eqx.Module):
    """Read genotype or count data from different file format"""

    @abstractmethod
    def __call__(self, pheno_path: str):
        pass

    @abstractmethod
    def process(self, dat: Any, filter_opt=SingleCellFilter) -> pd.DataFrame:
        pass


class H5AD(PhenoIO):
    def __call__(self, pheno_path: str):
        """Read raw count file in H5AD format"""
        return sc.read_h5ad(pheno_path)

    def process(
        self,
        dat: AnnData,
        filter_opt=SingleCellFilter,
        divide_size_factor: bool = False,
        norm_fix_L: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter single cell data and create pseudo-bulk
        dat.X: n_obs (cell) x n_vars (genes)
        dat.var_name = 'ensembl_id'
        ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

        :param dat: AnnData
        :param filter_opt: SingleCellFilter metrics
        :param divide_size_factor: `TRUE` if normalize read counts between individuals
        :param norm_fix_L: specify if normalize read counts to a fixed total amount
        :return: pseudo bulk RNA seq data for all cell types, index by ['donor_id', 'cell_type']

        """
        # TODO: check these result, make col names consistent
        # filter cells by min number of genes expressed (in place)
        sc.pp.filter_cells(dat, min_genes=filter_opt.min_genes)

        #  filter cells with too many genes expressed (in place)
        sc.pp.filter_cells(dat, max_genes=filter_opt.max_genes)

        # filter genes by min number of cells expressed (in place)
        sc.pp.filter_genes(dat, min_cells=filter_opt.min_cells)

        # filter cells that have >5% mitochondrial counts
        # here return the actual sparse matrix instead of View for shifted_transformation_nolog()
        # dat = dat[dat.obs[filter_opt.mt_col] < filter_opt.percent_mt, :].copy()

        if filter_opt.mt_col in dat.obs.columns:
            dat = dat[dat.obs[filter_opt.mt_col] < filter_opt.percent_mt, :]

        # normalize total
        if norm_fix_L is not None:
            sc.pp.normalize_total(dat, target_sum=norm_fix_L)  # fixed L
        if divide_size_factor:
            dat = adjust_size_factor(dat)

        # mean count for given cell type within individual and create a view
        # first compute and then filter
        dat.bulk = dc.get_pseudobulk(
            dat,
            layer=filter_opt.layer,
            sample_col=filter_opt.id_col,
            groups_col=filter_opt.celltype_col,
            mode=filter_opt.bulk_method,  # take mean across cells for each individual
            min_cells=filter_opt.bulk_min_cells,  # exclude sample with < min cells from calc
            min_counts=filter_opt.bulk_min_count,  # exclude sample < min # summed count from calc
            min_prop=filter_opt.bulk_min_prop,  # selects genes that expressed across > % cells in each sample
            min_smpls=filter_opt.bulk_min_smpls,  # this condition is met across a minimum number of samples
        )

        # create pd.Dataframe
        count = pd.DataFrame(dat.bulk.X)  # sample_cell x gene
        count = count.set_index([dat.bulk.obs[filter_opt.id_col], dat.bulk.obs[filter_opt.celltype_col]])
        count.columns = dat.bulk.var.index  # use var.index as gene names

        return count

    @staticmethod
    def write_bed(
        pheno: pd.DataFrame,
        filter_opt=SingleCellFilter,
        gtf_bed_path: str = "../example/data/Homo_sapiens.GRCh37.87.bed.gz",
        out_dir: str = "../example/local/phe_bed",
        celltype_path: Optional[str] = None,
        suffix: Optional[str] = None,
        autosomal_only: bool = True,
    ):
        """After creating pseudo-bulk using process(), create bed file for each cell type"""

        if celltype_path is None:
            cell_type_list = pheno.index.to_frame()[filter_opt.celltype_col].unique().tolist()
        else:
            cell_type_list = pd.read_csv(celltype_path, sep="\t", header=None).iloc[:, 0].to_list()

        for cell_type in cell_type_list:
            pheno_onetype = pheno[pheno.index.get_level_values(filter_opt.celltype_col) == cell_type]

            # remove cell type index
            pheno_onetype = pheno_onetype.reset_index(level=filter_opt.celltype_col, drop=True)

            # transpose s.t samples on columns, put ensembl_id back to column
            bed = pheno_onetype.T
            bed = bed.reset_index()

            # load gtf file for locating tss
            gene_map = load_gene_gft_bed(gtf_bed_path)

            # remove "chr" in prefix if there is any
            gene_map["chr"] = [s.removeprefix("chr") for s in gene_map["chr"]]

            if autosomal_only:
                gene_map = gene_map.loc[gene_map.chr.isin([str(i) for i in range(1, 23)])]

            # inner join
            out = pd.merge(gene_map, bed, left_on="ensemble_id", right_on=filter_opt.geneid_col)
            out = out.drop("ensemble_id", axis=1)
            out = out.rename(columns={filter_opt.geneid_col: "phenotype_id", "chr": "#Chr"})

            cell_type_outname = re.sub("[^0-9a-zA-Z]+", "_", cell_type)

            if suffix is None:
                outname = os.path.join(out_dir, f"{cell_type_outname}.bed.gz")
            else:
                outname = os.path.join(out_dir, f"{cell_type_outname}.{suffix}.bed.gz")

            out.to_csv(
                outname,
                index=False,
                sep="\t",
            )


def load_gene_gft_bed(gtf_bed_path: str) -> pd.DataFrame:
    """Read gft bed file"""
    gene_map = pd.read_csv(gtf_bed_path, delimiter="\t")
    gene_map.columns = [
        "chr",
        "start",
        "end",
        "ensemble_id",
    ]

    return gene_map


def adjust_size_factor(adata: AnnData):
    """Suggested by AE & Huber 2023 paper
    size factor = (sum_g Y_gc) / L
    where L = (sum_gc Y_gc) / (number of cells)

    adapt code from: https://github.com/mousepixels/sanbomics_scripts/blob/main/shifted_transformation.ipynb
    """
    # TODO: need do this by cell type? right now this divide by average across all cells all cell type
    # X: cell x gene
    size_factors = adata.X.sum(axis=1) / np.mean(adata.X.sum(axis=1))  # (num cell x 1)

    # array.A1 returns self as a flattened array, same as array.ravel()
    adata.X = diags(1.0 / size_factors.A1).dot(adata.X)
    # adata.X = adata.X.toarray()  # convert to dense array
    # adata.X = adata.X + y0
    # adata.X.data = (
    #     adata.X.data + y0
    # )  # !!! add y0 to non-sparse values, not sure if need add y0 to zeros raw count

    return adata


def bed_transform_y(pheno_path: str, method: str = "log1p"):
    """Perform transformation on gene expression count matrix
    count_df: rows are genes, columns are individual ID
    """
    count_df = pd.read_csv(pheno_path, sep="\t", dtype={"#chr": str, "#Chr": str})
    # filter genes with zero expression (first step of edger_cpm);
    # Note: do this firstly here to avoid incorrect row numbers
    count_df = count_df[count_df.iloc[:, 4:].sum(axis=1) > 0]

    if method == "log1p":
        count_df.iloc[:, 4:] = np.log1p(count_df.iloc[:, 4:])  # prevent log(0)
    elif method == "tmm":
        # use edger TMM method to calculate size factor and convert to counts per million
        tmm_counts_df = qtl.norm.edger_cpm(count_df.iloc[:, 4:], normalized_lib_sizes=True)
        # # mask is filter by gene
        # inverse normal transformation on each gene (row)
        norm_df = qtl.norm.inverse_normal_transform(tmm_counts_df)
        if count_df.shape[0] == norm_df.shape[0]:
            count_df.iloc[:, 4:] = norm_df
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
    r = jsp_stats.rankdata(pheno)
    return jsp_stats.norm.ppf(r / (pheno.shape[0] + 1))


def gtf_to_tss_bed(annotation_gtf, feature="gene", exclude_chrs=[], phenotype_id="gene_id"):
    """Parse genes and TSSs from GTF and return DataFrame for BED output
    This function is from: https://github.com/broadinstitute/pyqtl/blob/master/qtl/io.py
    """

    chrom = []
    start = []
    end = []
    gene_id = []
    gene_name = []

    if annotation_gtf.endswith(".gz"):
        opener = gzip.open(annotation_gtf, "rt")
    else:
        opener = open(annotation_gtf)

    with opener as gtf:
        for row in gtf:
            row = row.strip().split("\t")
            if row[0][0] == "#" or row[2] != feature:
                continue  # skip header
            chrom.append(row[0])

            # TSS: gene start (0-based coordinates for BED)
            if row[6] == "+":
                start.append(np.int64(row[3]) - 1)
                end.append(np.int64(row[3]))
            elif row[6] == "-":
                start.append(np.int64(row[4]) - 1)  # last base of gene
                end.append(np.int64(row[4]))
            else:
                raise ValueError("Strand not specified.")

            attributes = defaultdict()
            for a in row[8].replace('"', "").split(";")[:-1]:
                kv = a.strip().split(" ")
                if kv[0] != "tag":
                    attributes[kv[0]] = kv[1]
                else:
                    attributes.setdefault("tags", []).append(kv[1])

            gene_id.append(attributes["gene_id"])
            gene_name.append(attributes["gene_name"])

    if phenotype_id == "gene_id":
        bed_df = pd.DataFrame(
            data={"chr": chrom, "start": start, "end": end, "gene_id": gene_id},
            columns=["chr", "start", "end", "gene_id"],
            index=gene_id,
        )
    elif phenotype_id == "gene_name":
        bed_df = pd.DataFrame(
            data={"chr": chrom, "start": start, "end": end, "gene_id": gene_name},
            columns=["chr", "start", "end", "gene_id"],
            index=gene_name,
        )
    # drop rows corresponding to excluded chromosomes
    mask = np.ones(len(chrom), dtype=bool)
    for k in exclude_chrs:
        mask = mask & (bed_df["chr"] != k)
    bed_df = bed_df[mask]

    # sort by start position
    bed_df = bed_df.groupby("chr", sort=False, group_keys=False).apply(lambda x: x.sort_values("start"))

    return bed_df


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
