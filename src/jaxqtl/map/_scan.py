# pattern: Functional Core
"""Execute association classes with stable compiled fitting and block kernels.

NumPy is confined to variable-width host buffers: JAX slicing, padding, or
assembling those windows would specialize on each gene's variant count. All
statistical calculations and fixed-size reductions use JAX. Host buffers are
transferred only at kernel ingress and output assembly, never through JIT.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm

from jax import lax
from jaxtyping import Array, PRNGKeyArray

from ..hypothesis._aggregate import AbstractAggregateTest, ACAT, BetaCalibration, BetaPermutation
from ..hypothesis._base import AbstractHypothesisTest, TestResult


StateT = TypeVar("StateT")


@eqx.filter_jit
def full_scan(X, G, y, offset, snp_test, gene_test, key):
    """Transformable full-window composition for direct array callers."""
    result = snp_test(X, G, y, offset)
    return result, gene_test(X, G, y, offset, result, snp_test, key)


@eqx.filter_jit
def _initialize(test: AbstractHypothesisTest[StateT], X, y, offset) -> StateT:
    return test.init(X, y, offset)


@eqx.filter_jit
def _test_block(test, X, G, state, width, total, reduction):
    result = test.test(X, G, state)
    components = None
    if reduction is not None:
        valid = jnp.arange(G.shape[1]) < width
        components = reduction.components(result.p, valid, jnp.asarray(1.0, dtype=result.p.dtype) / total)
    return result, components


@eqx.filter_jit
def _initialize_permutations(test: AbstractHypothesisTest[StateT], X, y, offset, key, size: int):
    def initialize_one(carry, _):
        carry, p_key = rdm.split(carry)
        indices = rdm.permutation(p_key, jnp.arange(len(y)))
        perm_offset = offset[indices] if offset.ndim > 0 else offset
        return carry, test.init(X, y[indices], perm_offset)

    return lax.scan(initialize_one, key, xs=None, length=size)


@eqx.filter_jit
def _permutation_block_maxima(test, X, G, states, width):
    def score_one(_, state):
        result = test.test(X, G, state)
        valid = jnp.arange(G.shape[1]) < width
        return None, jnp.nanmax(jnp.where(valid, jnp.abs(result.z), jnp.nan))

    return lax.scan(score_one, None, states)[1]


_merge_maxima = eqx.filter_jit(jnp.fmax)


@eqx.filter_jit
def _update_acat(aggregation, state, block):
    return aggregation.update(state, block)


@eqx.filter_jit
def _finish_acat(aggregation, state):
    return aggregation.finalize(state)


@eqx.filter_jit
def _fit_calibration(aggregation, maxima, dof):
    return aggregation.fit_calibration(maxima, dof)


@eqx.filter_jit
def _adjust(aggregation, z, calibration):
    return aggregation.adjust(z, calibration)


@eqx.filter_jit
def _allele_summary_block(G):
    counts = jnp.sum(G, axis=0)  # genoio returns counts for the a1 allele.
    af = counts / (2.0 * G.shape[0])
    ma_counts = jnp.where(af <= 0.5, counts, 2 * G.shape[0] - counts)
    return af, ma_counts


@eqx.filter_jit
def _lead_candidates(pvalues, width):
    valid = (jnp.arange(pvalues.shape[0]) < width) & jnp.isfinite(pvalues)
    minimum = jnp.min(jnp.where(valid, pvalues, jnp.inf))
    return minimum, valid & (pvalues == minimum)


def select_lead_variant(pvalues: Array, key: PRNGKeyArray) -> int | None:
    """Select the minimum finite p-value with the existing random tie policy.

    Reduce fixed blocks in JAX; collect only host indices for output selection.
    Tie counts can still introduce a small random-choice compilation signature.
    """
    minimum = float("inf")
    indices = []
    for start, stop, block in _HostBuffers(pvalues).blocks(2048):
        value, mask = jax.device_get(_lead_candidates(block, jnp.asarray(stop - start)))
        value = float(value)
        if value < minimum:
            minimum, indices = value, []
        if value == minimum:
            indices.extend(start + i for i, selected in enumerate(mask.tolist()) if selected)
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    return int(rdm.choice(key, jnp.asarray(indices), replace=False))


class _HostBuffers:
    """Pack an unchanged host input; statistical operations belong to kernels.

    Full blocks are host views. Only the final partial block needs padding; its
    device array is reused across passes, including permutation batches.
    """

    def __init__(self, values):
        self.values = np.asarray(jax.device_get(values))
        self._tail_size = None
        self._tail = None

    def blocks(self, block_size):
        m = self.values.shape[-1]
        for start in range(0, m, block_size):
            stop = min(start + block_size, m)
            values = self.values[..., start:stop]
            if stop - start == block_size:
                yield start, stop, jax.device_put(values)
            else:
                if self._tail_size != block_size:
                    block = np.empty((*self.values.shape[:-1], block_size), dtype=self.values.dtype)
                    block[..., : stop - start] = values
                    # Duplicate a real column to avoid synthetic zero-variance failures.
                    block[..., stop - start :] = values[..., :1]
                    self._tail = jax.device_put(block)
                    self._tail_size = block_size
                yield start, stop, self._tail

    @staticmethod
    def allocate(size, example):
        return np.empty(size, dtype=example.dtype)

    @staticmethod
    def append(output, values, start, stop):
        output[start:stop] = jax.device_get(values)[: stop - start]


def allele_summaries(G) -> tuple[np.ndarray, np.ndarray]:
    """Return host AF/MAC columns, computing statistics in fixed JAX blocks."""
    buffers = _HostBuffers(G)
    m = buffers.values.shape[1]
    if m == 0:
        # Preserve JAX's reduction and division dtype promotion for empty windows.
        return jax.device_get(_allele_summary_block(jax.device_put(buffers.values)))
    af_output, mac_output = None, None
    for start, stop, block in buffers.blocks(2048):
        af, ma_counts = _allele_summary_block(block)
        if af_output is None:
            af_output = buffers.allocate(m, af)
            mac_output = buffers.allocate(m, ma_counts)
        buffers.append(af_output, af, start, stop)
        buffers.append(mac_output, ma_counts, start, stop)
    assert af_output is not None and mac_output is not None
    return af_output, mac_output


@dataclass(frozen=True)
class AssociationScan(Generic[StateT]):
    """One configured execution policy, independent of gene metadata and I/O.

    Score, SPA, and Wald share fixed genotype blocks. Aggregators own their
    statistical workflow; omitting aggregation selects nominal testing only.
    """

    test: AbstractHypothesisTest[StateT]
    aggregation: AbstractAggregateTest | None = None
    block_size: int | None = None
    batch_size: int = 32

    def __post_init__(self):
        if self.block_size is None:
            default_size = self.aggregation.block_size if self.aggregation is not None else 2048
            object.__setattr__(self, "block_size", default_size)
        for name in ("block_size", "batch_size"):
            value = getattr(self, name)
            if value is None and name == "block_size":
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def blocked(self) -> bool:
        return self.block_size is not None

    @property
    def cache_clear_interval(self) -> int | None:
        return None if self.blocked else 20

    def run(self, X, G, y, offset, key: PRNGKeyArray):
        if self.aggregation is None:
            raise ValueError("gene-level scans require an aggregation")
        if not self.blocked:
            return full_scan(X, G, y, offset, self.test, self.aggregation, key)
        return self.aggregation.scan(self, X, G, y, offset, key)

    def observed(self, X, G, y, offset, *, reduction: ACAT | None = None):
        X, y, offset = jnp.asarray(X), jnp.asarray(y), jnp.asarray(offset)
        buffers = _HostBuffers(G)
        m = buffers.values.shape[1]
        if m == 0:
            raise ValueError("association testing requires at least one variant")
        state = _initialize(self.test, X, y, offset)
        columns, scalars = {}, {}
        accumulator = None
        for start, stop, block in buffers.blocks(self.block_size):
            result, components = _test_block(
                self.test, X, block, state, jnp.asarray(stop - start), jnp.asarray(m), reduction
            )
            if reduction is not None:
                if accumulator is None:
                    accumulator = reduction.init(components.statistic.dtype)
                accumulator = _update_acat(reduction, accumulator, components)
            result = jax.device_get(result)
            for name, value in zip(result._fields, result, strict=True):
                if getattr(value, "ndim", 0) == 0:
                    scalars[name] = value
                else:
                    if name not in columns:
                        columns[name] = buffers.allocate(m, value)
                    buffers.append(columns[name], value, start, stop)
        return TestResult(**jax.device_put(columns | scalars)), accumulator

    def permutation_maxima(self, X, G, y, offset, key):
        permutations = self.aggregation
        if not isinstance(permutations, BetaPermutation):
            raise TypeError("permutation maxima require BetaPermutation")
        count = permutations.max_perm_direct
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("max_perm_direct must be a positive integer")
        X, y, offset = jnp.asarray(X), jnp.asarray(y), jnp.asarray(offset)
        buffers = _HostBuffers(G)
        if buffers.values.shape[1] == 0:
            raise ValueError("permutation testing requires at least one variant")
        maxima = []
        for start in range(0, count, self.batch_size):
            size = min(self.batch_size, count - start)
            key, states = _initialize_permutations(self.test, X, y, offset, key, size)
            batch_maxima = None
            for first, last, block in buffers.blocks(self.block_size):
                values = _permutation_block_maxima(self.test, X, block, states, jnp.asarray(last - first))
                batch_maxima = _merge_maxima(values if batch_maxima is None else batch_maxima, values)
                # Finish this block before transferring the next one; asynchronous
                # dispatch must not retain a whole window of genotype buffers.
                batch_maxima = jax.block_until_ready(batch_maxima)
            assert batch_maxima is not None  # The nonempty window guarantees at least one block.
            maxima.append(batch_maxima)
        return jnp.concatenate(maxima)

    def fit_calibration(self, maxima, dof) -> BetaCalibration:
        return _fit_calibration(self.aggregation, maxima, dof)

    def adjust(self, z, calibration: BetaCalibration) -> Array:
        buffers = _HostBuffers(z)
        adjusted = None
        for start, stop, block in buffers.blocks(self.block_size):
            values = _adjust(self.aggregation, block, calibration)
            if adjusted is None:
                adjusted = buffers.allocate(len(buffers.values), values)
            buffers.append(adjusted, values, start, stop)
        return jax.device_put(adjusted)

    def finish_acat(self, state) -> Array:
        return _finish_acat(self.aggregation, state)
