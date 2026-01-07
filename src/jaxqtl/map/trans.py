import itertools as itt

from collections.abc import Iterator
from logging import Logger

import numpy as np
import polars as pl

import equinox as eqx

from jax import numpy as jnp

from ..families.distribution import NegativeBinomial
from ..infer.utils import HypothesisTest
from ..io.data import ReadyDataState
from ..log import get_log


def map_trans(
    data: ReadyDataState,
    test: HypothesisTest,
    chunk_size: int = 5000,
    verbose: bool = True,
    log: Logger | None = None,
    seed: int = 123,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    r"""Perform trans-eQTL mapping in genotype chunks and stream results.

    **Arguments:**
    - `data`: Genotype/expression/covariate bundle aligned on IID.
    - `test`: Hypothesis test to apply per variant (score or Wald).
    - `chunk_size`: Number of variants per block to stream from genotype storage.
    - `verbose`: Whether to emit progress logging.
    - `log`: Optional logger to use; defaults to module logger.
    - `seed`: PRNG seed reserved for future stochastic operations.

    **Returns:**
    - `Iterator[tuple[pl.DataFrame, pl.DataFrame]]`: `(variant_df, sumstat_df)` pairs per genotype block.
    """
    if log is None:
        log = get_log()

    # we don't use this at the moment, but keep it passed in, for futureproofing i suppose
    # key = rdm.key(seed)

    # we may as well do trans scans in bulk to save computational time from streaming genotypes
    pheno = data.expression.to_jax()
    y_var = jnp.var(pheno, axis=0)
    keep = ~jnp.isnan(y_var) & (y_var > 0)
    pheno = pheno[:, keep]
    # this is a code smell, this entire block should get pushed into a method
    pheno_ids = data.expression.pheno_meta.filter(np.asarray(keep)).get_column("phenotype_id").to_list()

    if jnp.sum(keep) == 0:
        log.warning("No variation found in phenotype data. Exiting trans mapping/scan.")
        return None

    for i, (geno_chunk, variant_info) in enumerate(data.iter_geno(chunk_size)):
        pos = variant_info.get_column("pos").to_jax()
        chrom = variant_info.get_column("chrom").unique()
        lstart = jnp.min(pos)
        rend = jnp.max(pos)

        if verbose:
            log.info(f"Performing trans-qtl scan for over region {chrom}:{lstart}-{rend}")

        # we can vmap our test function across the batch of phenotypes
        test_result = eqx.filter_jit(eqx.filter_vmap(test, in_axes=(None, None, 1, None)))(
            data.covar, geno_chunk, pheno, data.offset
        )
        variant_df, sumstat_df = _process_result(geno_chunk, variant_info, test_result, pheno_ids)

        # if we didn't fit a negative binomial, just drop the alpha column as its const 0
        # its usually a code-smell to refer to chained attributes (ie something.something.something), but w/e
        if not isinstance(test.model.family, NegativeBinomial):
            sumstat_df = sumstat_df.drop("nb_alpha")

        yield variant_df, sumstat_df

        if verbose:
            log.info(f"Finished trans-qtl scan for over region {chrom}:{lstart}-{rend}")

    return None


def _process_result(geno_chunk, region_df, test_result, pheno_ids):
    """Format per-chunk test results into variant- and summary-level frames."""
    n, p = geno_chunk.shape
    counts = jnp.sum(geno_chunk, axis=0)
    af = counts / (2.0 * n)
    flag = af <= 0.5
    ma_counts = jnp.where(flag, counts, 2 * n - counts)
    if jnp.ndim(test_result.disp) == 1:
        nb_alpha = jnp.repeat(test_result.disp, p)
    else:
        nb_alpha = test_result.disp.flatten()

    if jnp.ndim(test_result.converged) == 1:
        glm_converged = jnp.repeat(test_result.converged, p)
    else:
        glm_converged = test_result.converged.flatten()

    k = len(pheno_ids)
    assert k == test_result.beta.shape[0], "we're not passing in the same number of phenotypes"

    pheno_ids = list(itt.chain.from_iterable(itt.repeat(pheno_id, p) for pheno_id in pheno_ids))
    snp_ids = region_df.get_column("snp").to_list()
    snp_ids = list(itt.chain.from_iterable(snp_ids for _ in range(k)))
    variant_df = region_df.with_columns(
        pl.Series("af", np.array(af)),
        pl.Series("ma_count", np.array(ma_counts, dtype=int)),
    ).select(["chrom", "snp", "pos", "a1", "a0", "af", "ma_count"])
    sumstats_df = pl.DataFrame(
        {
            "phenotype": pheno_ids,
            "snp": snp_ids,
            "beta": np.asarray(test_result.beta.flatten()),
            "se": np.asarray(test_result.se.flatten()),
            "pvalue": np.asarray(test_result.p.flatten()),
            "nb_alpha": np.asarray(nb_alpha),
            "model_converged": np.asarray(glm_converged),
        }
    )
    # put pheno id in front
    return variant_df, sumstats_df


def get_trans_schemas() -> tuple[dict[str, type], dict[str, type]]:
    """Return expected Polars schema dictionaries for trans mapping outputs.

    **Returns:**
    - `tuple[dict[str, type], dict[str, type]]`: Variant schema and summary-stat schema.
    """
    var_schema = {
        "chrom": str,
        "snp": str,
        "pos": int,
        "a1": str,
        "a0": str,
        "af": float,
        "ma_count": int,
    }
    stats_schema = {
        "phenotype": str,
        "snp": str,
        "beta": float,
        "se": float,
        "pvalue": float,
        "nb_alpha": float,
        "model_converged": bool,
    }
    return var_schema, stats_schema
