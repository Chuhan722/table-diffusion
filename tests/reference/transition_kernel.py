"""Finite-population transition-probability gradient, research reference implementation.

COUNT units throughout: q(S) = sum_i a(x_i), L = (y-q)^T W (y-q)/2.
No proposal-loss test, rejection, retry, or best-of-candidate operation occurs.
Every block draws one final outcome; different blocks draw independently.

The caller supplies a finite list of LEGAL record states and allowable block outcomes.
Enumerating every outcome is an oracle-sized reference, not a scalable implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import itertools
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BlockSupport:
    rows: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]
    mobility: tuple[float, ...] | None = None


@dataclass
class BlockTransition:
    rows: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]  # current block first
    probabilities: NDArray[np.float64]
    gains: NDArray[np.float64]
    rates: NDArray[np.float64]


@dataclass
class KernelResult:
    blocks: list[BlockTransition]
    row_marginals: NDArray[np.float64]
    old_loss: float
    dissipation: float
    cross_curvature: float
    step: float
    expected_loss: float
    certified_upper_bound: float

    def sample(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Draw each BLOCK once. Do not sample row_marginals independently!"""
        out = np.empty(self.row_marginals.shape[0], dtype=np.int64)
        for block in self.blocks:
            j = int(rng.choice(len(block.outcomes), p=block.probabilities))
            out[list(block.rows)] = block.outcomes[j]
        return out


def complete_supports(n: int, m: int, block_size: int = 1) -> list[BlockSupport]:
    """Exhaustive oracle supports; cost is exponential in block_size."""
    if n < 1 or m < 1 or block_size < 1:
        raise ValueError("n, m, and block_size must be positive")
    result = []
    for start in range(0, n, block_size):
        rows = tuple(range(start, min(n, start + block_size)))
        outcomes = tuple(itertools.product(range(m), repeat=len(rows)))
        result.append(BlockSupport(rows, outcomes))
    return result


def construct_kernel(
    current: Sequence[int],
    features: ArrayLike,
    target: ArrayLike,
    weights: ArrayLike,
    supports: Sequence[BlockSupport] | None = None,
    damping: float = 1.0,
    gain_rtol: float = 1e-12,
) -> KernelResult:
    """Construct a no-rejection kernel with exact quadratic expected-loss identity.

    damping in (0,1] multiplies the analytically optimal feasible line-search step.
    It is fixed before sampling, not selected using a sampled table's loss.
    Formulas are exact in real arithmetic; this implementation uses float64.
    gain_rtol suppresses numerical sign noise near zero gain BEFORE sampling;
    it is not an empirical-loss accept/reject test. This can omit extremely small
    true improvements. For exact rational certification use verify.exact_kernel.
    """
    raw_current = np.asarray(current)
    if raw_current.ndim != 1 or not np.issubdtype(raw_current.dtype, np.integer):
        raise ValueError("current must be a one-dimensional sequence of integer state ids")
    s = raw_current.astype(np.int64, copy=True)
    a = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("features must have shape (number_of_states, number_of_queries)")
    m, queries = a.shape
    n = len(s)
    if n == 0 or np.any(s < 0) or np.any(s >= m):
        raise ValueError("current state ids must refer to features")
    if y.shape != (queries,) or w.shape != (queries,):
        raise ValueError("target and weights must have one value per query")
    if not (np.isfinite(a).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("all numeric inputs must be finite")
    if np.any(w <= 0) or not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("weights must be positive and damping must be in (0,1]")
    if not np.isfinite(gain_rtol) or gain_rtol < 0:
        raise ValueError("gain_rtol must be finite and nonnegative")
    if supports is None:
        supports = complete_supports(n, m)
    flat_rows = [i for spec in supports for i in spec.rows]
    if sorted(flat_rows) != list(range(n)) or any(not spec.rows for spec in supports):
        raise ValueError("blocks must form a disjoint partition of all row indices")

    residual = y - a[s].sum(axis=0)
    old_loss = float(0.5 * np.dot(w * residual, residual))
    transitions: list[BlockTransition] = []
    drifts: list[NDArray[np.float64]] = []
    D = 0.0
    max_rate = 0.0

    for spec in supports:
        source = tuple(int(s[i]) for i in spec.rows)
        if len(set(spec.outcomes)) != len(spec.outcomes):
            raise ValueError("duplicate block outcomes: aggregate them before construction")
        mob = spec.mobility or tuple(1.0 for _ in spec.outcomes)
        if len(mob) != len(spec.outcomes):
            raise ValueError("mobility and outcomes have different lengths")
        entries: list[tuple[tuple[int, ...], float]] = []
        for outcome, c in zip(spec.outcomes, mob):
            if len(outcome) != len(spec.rows) or any(not isinstance(u, (int, np.integer)) or u < 0 or u >= m for u in outcome):
                raise ValueError("invalid state tuple in block support")
            if not np.isfinite(c) or c < 0:
                raise ValueError("mobility must be finite and nonnegative")
            if outcome != source:
                entries.append((outcome, float(c)))
        outcomes = (source,) + tuple(outcome for outcome, _ in entries)
        deltas = np.zeros((len(outcomes), queries), dtype=np.float64)
        c = np.zeros(len(outcomes), dtype=np.float64)
        source_sum = a[list(source)].sum(axis=0)
        for j, (outcome, mobility) in enumerate(entries, start=1):
            deltas[j] = a[list(outcome)].sum(axis=0) - source_sum
            c[j] = mobility
        linear = deltas @ (w * residual)
        quadratic = 0.5 * np.sum(deltas**2 * w, axis=1)
        gains = linear - quadratic
        threshold = gain_rtol * np.maximum(1.0, np.abs(linear) + quadratic)
        positive = np.where(gains > threshold, gains, 0.0)
        rates = c * positive
        D += float(np.dot(c, positive**2))
        max_rate = max(max_rate, float(rates.sum()))
        drifts.append(rates @ deltas)
        transitions.append(BlockTransition(spec.rows, outcomes, np.zeros(len(outcomes)), gains, rates))

    v = np.stack(drifts)
    total_v = v.sum(axis=0)
    C = float(0.5 * (np.dot(total_v * w, total_v) - np.sum(v**2 * w)))
    if D == 0.0:
        h = 0.0
    else:
        if max_rate <= 0 or not np.isfinite(D) or not np.isfinite(C):
            raise FloatingPointError("non-finite or inconsistent rate moments")
        feasible_limit = 1.0 / max_rate
        optimal = min(feasible_limit, D / (2.0 * C)) if C > 0 else feasible_limit
        h = damping * optimal

    marginals = np.zeros((n, m), dtype=np.float64)
    for block in transitions:
        block.probabilities = h * block.rates
        block.probabilities[0] = 1.0 - float(block.probabilities[1:].sum())
        # Only roundoff cleanup: not an objective-based acceptance or modification.
        if block.probabilities[0] < -1e-12:
            raise FloatingPointError("step violates the stochasticity constraint")
        if block.probabilities[0] < 0:
            block.probabilities[0] = 0.0
            block.probabilities /= block.probabilities.sum()
        for outcome, probability in zip(block.outcomes, block.probabilities):
            for i, state in zip(block.rows, outcome):
                marginals[i, state] += probability
    expected_loss = old_loss - h * D + h * h * C
    upper = old_loss - 0.5 * h * D
    return KernelResult(transitions, marginals, old_loss, D, C, h, expected_loss, upper)
