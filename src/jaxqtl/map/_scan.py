# pattern: Imperative Shell
"""Execute association classes with stable compiled fitting and block kernels.

NumPy is confined to variable-width host buffers: JAX slicing, padding, or
assembling those windows would specialize on each gene's variant count. All
statistical calculations and fixed-size reductions use JAX. Host buffers are
transferred only at kernel ingress and output assembly, never through JIT.
"""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm

from jax import lax
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..hypothesis._aggregate import AbstractAggregateTest, BetaPermutation, PermutationReference
from ..hypothesis._base import AbstractHypothesisTest, TestResult


StateT = TypeVar("StateT")


@eqx.filter_jit
def full_scan(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    snp_test: AbstractHypothesisTest,
    gene_test: AbstractAggregateTest,
    key: PRNGKeyArray,
):
    """Compute a full window without choosing a lead or finalizing its gene p-value.

    Inputs are covariates `X (n, p)`, genotypes `G (n, m)`, outcome `y (n,)`,
    and a scalar or length-`n` offset. `snp_test` fits the hypothesis test;
    `gene_test` supplies the reduction and, for BetaPermutation, the permutation count.

    Returns `(test_result, reduction_state, reference)`. BetaPermutation supplies
    a reference and no observed reduction; observed-only methods supply a reduction
    and no reference. These full-window kernels specialize on the input shapes.
    """
    X, G, y, offset = map(jnp.asarray, (X, G, y, offset))
    result = snp_test(X, G, y, offset)
    if isinstance(gene_test, BetaPermutation):
        maxima = full_permutation_maxima(X, G, y, offset, snp_test, gene_test, key)
        reference = PermutationReference(maxima, X.shape[0] - X.shape[1] - 1)
        return result, None, reference
    values = gene_test.statistic(result)
    state = gene_test.init(values.dtype, num_variants=G.shape[1])
    state = gene_test.update(state, values, jnp.ones_like(values, dtype=bool))
    return result, state, None


@eqx.filter_jit
def full_permutation_maxima(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike,
    test: AbstractHypothesisTest,
    aggregation: BetaPermutation,
    key: PRNGKeyArray,
) -> Array:
    """Return one maximum absolute statistic per permutation using full-window kernels.

    Inputs follow `full_scan`; `aggregation.max_perm_direct` sets the number of
    permutations. Outcome rows and vector offsets are shuffled together, while
    covariates and genotypes remain fixed. Key splitting matches the blocked path.
    """
    X, G, y, offset = map(jnp.asarray, (X, G, y, offset))

    def permute_one(key, _):
        key, p_key = rdm.split(key)
        indices = rdm.permutation(p_key, jnp.arange(len(y)))
        perm_offset = offset[indices] if offset.ndim > 0 else offset
        result = test(X, G, y[indices], perm_offset)
        values = aggregation.statistic(result)
        state = aggregation.init(values.dtype, num_variants=G.shape[1])
        return key, aggregation.update(state, values, jnp.ones_like(values, dtype=bool))

    return lax.scan(permute_one, key, xs=None, length=aggregation.max_perm_direct)[1]


@eqx.filter_jit
def _initialize(test: AbstractHypothesisTest[StateT], X, y, offset) -> StateT:
    return test.init(X, y, offset)


@eqx.filter_jit
def _test_block(test, X, G, state, aggregation=None):
    result = test.test(X, G, state)
    return result, None if aggregation is None else aggregation.statistic(result)


@eqx.filter_jit
def _update_reduction(aggregation, state, values, width):
    if aggregation is None:
        return state
    valid = jnp.arange(values.shape[-1]) < width
    return aggregation.update(state, values, valid)


@eqx.filter_jit
def _init_reduction(aggregation, example, total):
    return None if aggregation is None else aggregation.init(example.dtype, num_variants=total)


@eqx.filter_jit
def _initialize_permutations(test: AbstractHypothesisTest[StateT], X, y, offset, key, size: int):
    def initialize_one(carry, _):
        carry, p_key = rdm.split(carry)
        indices = rdm.permutation(p_key, jnp.arange(len(y)))
        perm_offset = offset[indices] if offset.ndim > 0 else offset
        return carry, test.init(X, y[indices], perm_offset)

    return lax.scan(initialize_one, key, xs=None, length=size)


@eqx.filter_jit
def _permutation_test_block(test, aggregation, X, G, states):
    # Selecting statistics inside JIT lets XLA discard unused SPA tail calculations.
    def score_one(_, state):
        return None, aggregation.statistic(test.test(X, G, state))

    return lax.scan(score_one, None, states)[1]


@eqx.filter_jit
def _update_permutation_reductions(aggregation, states, statistics, width):
    valid = jnp.arange(statistics.shape[-1]) < width
    return eqx.filter_vmap(aggregation.update, in_axes=(0, 0, None))(states, statistics, valid)


@eqx.filter_jit
def _init_permutation_reductions(aggregation, example, total):
    state = aggregation.init(example.dtype, num_variants=total)
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (example.shape[0], *value.shape)), state)


@eqx.filter_jit
def _allele_summary_block(G):
    counts = jnp.sum(G, axis=0)  # genoio returns counts for the a1 allele.
    af = counts / (2.0 * G.shape[0])
    ma_counts = jnp.where(af <= 0.5, counts, 2 * G.shape[0] - counts)
    return af, ma_counts


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
        """Yield (start, stop, device_block), padding only the final block."""
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


class _ResultBuffer:
    """Assemble results in owned NumPy columns, preserving shared scalar metadata.

    Each column is allocated once. Appending transfers one result tree to the host;
    finishing transfers the assembled tree back to JAX for the scan return contract.
    """

    def __init__(self, num_variants):
        self.num_variants = num_variants
        self.values = {}

    def append(self, start, stop, result: TestResult):
        """Copy real variants into their output slice and discard padded entries."""
        result = jax.device_get(result)
        for name, value in result._asdict().items():
            if np.ndim(value) == 0:
                self.values[name] = value
            else:
                if name not in self.values:
                    self.values[name] = np.empty(self.num_variants, dtype=value.dtype)
                self.values[name][start:stop] = value[: stop - start]

    def finish(self) -> TestResult:
        """Return the completed result as JAX arrays."""
        return TestResult(**jax.device_put(self.values))


def allele_summaries(G: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return host allele-frequency and minor-allele-count arrays for `G (n, m)`.

    Genotypes count the a1 allele. Calculations use fixed JAX blocks; assembly
    preserves the resulting dtypes and returns two empty arrays when `m == 0`.
    """
    buffers = _HostBuffers(G)
    m = buffers.values.shape[1]
    if m == 0:
        # Preserve JAX's reduction and division dtype promotion for empty windows.
        return jax.device_get(_allele_summary_block(jax.device_put(buffers.values)))
    af_output: np.ndarray
    mac_output: np.ndarray
    for start, stop, block in buffers.blocks(2048):
        af, ma_counts = jax.device_get(_allele_summary_block(block))
        if start == 0:
            af_output = np.empty(m, dtype=af.dtype)
            mac_output = np.empty(m, dtype=ma_counts.dtype)
        af_output[start:stop] = af[: stop - start]
        mac_output[start:stop] = ma_counts[: stop - start]
    return af_output, mac_output


@dataclass
class AssociationScan(Generic[StateT]):
    """Execute observed tests and permutation reductions in fixed genotype blocks.

    Cis orchestration owns lead selection and gene-level finalization. This class
    owns hypothesis fitting, block transfers, and permutation batching.

    **Attributes:**

    - `test`: Score, SPA, or Wald hypothesis test.
    - `aggregation`: Optional numerical reducer; BetaPermutation enables permutation scans.
    - `block_size`: Positive genotype-block width. Defaults to the aggregation's
      preference, or 2048 for nominal scans. A remaining `None` selects full-window
      execution in cis orchestration rather than this class's block methods.
    - `batch_size`: Positive number of permutations fitted together; defaults to 32.
    """

    test: AbstractHypothesisTest[StateT]
    aggregation: AbstractAggregateTest | None = None
    block_size: int | None = None
    batch_size: int = 32

    def __post_init__(self):
        if self.block_size is None:
            self.block_size = self.aggregation.block_size if self.aggregation is not None else 2048
        for name, value in (("block_size", self.block_size), ("batch_size", self.batch_size)):
            if value is None and name == "block_size":
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def blocked(self) -> bool:
        """Whether a genotype-block width is configured."""
        return self.block_size is not None

    def observed(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        *,
        reduction: AbstractAggregateTest | None = None,
    ) -> tuple[TestResult, Any]:
        """Test observed variants and optionally accumulate an unfinalized reduction.

        `X`, `G`, and `y` have shapes `(n, p)`, `(n, m)`, and `(n,)`;
        `offset` is scalar or length `n`. The hypothesis test is initialized once.
        `reduction=None` requests variant results only, regardless of the configured
        aggregation.

        Returns `(TestResult, reduction_state)`, with `None` state when no reduction
        is requested. Results are assembled on the host and returned as JAX arrays.
        Requires a configured block width and at least one variant.
        """
        X, y, offset = jnp.asarray(X), jnp.asarray(y), jnp.asarray(offset)
        buffers = _HostBuffers(G)
        m = buffers.values.shape[1]
        if m == 0:
            raise ValueError("association testing requires at least one variant")
        state = _initialize(self.test, X, y, offset)
        output = _ResultBuffer(m)
        accumulator = None
        for start, stop, block in buffers.blocks(self.block_size):
            result, values = _test_block(self.test, X, block, state, reduction)
            if start == 0:
                accumulator = _init_reduction(reduction, values, jnp.asarray(m))
            accumulator = _update_reduction(reduction, accumulator, values, jnp.asarray(stop - start))
            output.append(start, stop, result)
        return output.finish(), accumulator

    def permutation_maxima(
        self,
        X: ArrayLike,
        G: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        key: PRNGKeyArray,
    ) -> Array:
        """Return a length-`max_perm_direct` array of permutation maxima.

        Inputs follow `observed`; `key` determines permutation order independently
        of block and batch sizes. Each permuted outcome is initialized once and
        reused across genotype blocks. Vector offsets follow the outcome shuffle.

        Requires BetaPermutation, a configured block width, a positive permutation
        count, and at least one variant. Only completed maxima are retained between batches.
        """
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
                statistics = _permutation_test_block(self.test, permutations, X, block, states)
                if first == 0:
                    batch_maxima = _init_permutation_reductions(
                        permutations, statistics, jnp.asarray(buffers.values.shape[1])
                    )
                batch_maxima = _update_permutation_reductions(
                    permutations, batch_maxima, statistics, jnp.asarray(last - first)
                )
                # Finish this block before transferring the next one; asynchronous
                # dispatch must not retain a whole window of genotype buffers.
                batch_maxima = jax.block_until_ready(batch_maxima)
            assert batch_maxima is not None  # Nonempty windows always initialize the accumulator.
            maxima.append(batch_maxima)
        return jnp.concatenate(maxima)
