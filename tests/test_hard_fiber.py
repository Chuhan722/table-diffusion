"""硬纤维不变量、随机流、原子移动及有限目标分布的回归。"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, permutations, product
from math import factorial, prod

import numpy as np
import pytest

from resevo import hard_fiber
from resevo.hard_fiber import hard_fiber_sample


def _counts(table, fields):
    projection = np.asarray(table)[:, fields]
    return Counter(map(tuple, projection.tolist()))


def _hist(table):
    return Counter(map(tuple, np.asarray(table).tolist()))


def _cycle(scale=1):
    x = np.array([((b + d) % 3, b, (b + 2 * d) % 3, d) for b, d in product(range(3), repeat=2)])
    return np.repeat(x, np.array([5] + [1] * 8) * scale, axis=0)


def _prepare(table, doms, sets):
    original, kept = hard_fiber._validated_input(table, doms, sets, 1, np.random.default_rng(1))
    by_field = tuple(
        tuple(i for i, fields in enumerate(kept) if len(fields) > 1 and j in fields)
        for j in range(original.shape[1])
    )
    return original, kept, by_field


def _accounting(info):
    assert info.executed_steps == info.lazy_steps + info.single_proposals + info.paired_proposals
    assert info.single_proposals + info.paired_proposals == (
        info.rejected_constraints + info.identity_proposals + info.relabel_proposals + info.histogram_moves
    )
    assert info.invariants_verified and not info.mixing_certified and info.backend == "cpu"


def test_fixed_seed_ownership_and_complete_marginals():
    x = _cycle()
    before = x.copy()
    sets = [(0, 1), (1, 2), (2, 3), (0, 3)]
    source_sets = deepcopy(sets)
    a = hard_fiber_sample(x, [3] * 4, sets, 20000, np.random.default_rng(7301))
    b = hard_fiber_sample(x, [3] * 4, sets, 20000, np.random.default_rng(7301))
    assert np.array_equal(x, before) and sets == source_sets
    assert not np.shares_memory(a.codes, x)
    assert np.array_equal(a.codes, b.codes) and a.diagnostics == b.diagnostics
    assert a.diagnostics.histogram_moves > 0
    assert _hist(a.codes) != _hist(x)
    for fields in [*(tuple([j]) for j in range(4)), *sets]:
        assert _counts(a.codes, fields) == _counts(x, fields)
    _accounting(a.diagnostics)


def test_atomic_four_row_witness_and_rejected_half():
    x = np.array([[0, 0, 0, 0], [0, 2, 1, 1], [1, 1, 1, 0], [2, 1, 0, 1]])
    original, kept, by_field = _prepare(x, [3] * 4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    half = original.copy()
    assert hard_fiber._apply_exchange(half, np.array([0, 1]), (3,), kept, by_field)[0] == "rejected_constraints"
    assert np.array_equal(half, original)
    full = original.copy()
    outcome, moved, checked = hard_fiber._apply_exchange(full, np.arange(4), (3, 2), kept, by_field)
    assert outcome == "histogram_moves" and moved == 4 and checked > 0
    assert full.tolist() == [[0, 0, 0, 1], [0, 2, 1, 0], [1, 1, 0, 0], [2, 1, 1, 1]]
    for fields in kept:
        assert _counts(original, fields) == _counts(full, fields)
    assert hard_fiber._apply_exchange(full, np.arange(4), (3, 2), kept, by_field)[0] == "histogram_moves"
    assert np.array_equal(full, original)


def test_complete_pair_can_change_higher_order_but_not_pair_counts():
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    original, kept, by_field = _prepare(x, [2] * 3, [(0, 1), (0, 2), (1, 2)])
    after = original.copy()
    outcome, _, _ = hard_fiber._apply_exchange(after, np.arange(4), (2, 2), kept, by_field)
    assert outcome == "histogram_moves"
    assert _hist(after) != _hist(original)
    for fields in kept:
        assert _counts(after, fields) == _counts(original, fields)


def test_higher_order_constraint_is_not_reduced_to_pairs():
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    original, kept, by_field = _prepare(x, [2] * 3, [(0, 1, 2)])
    after = original.copy()
    assert hard_fiber._apply_exchange(after, np.arange(4), (2, 2), kept, by_field)[0] == "rejected_constraints"
    assert np.array_equal(after, original)


def test_zero_steps_canonicalize_and_preserve_rng():
    rng = np.random.default_rng(5)
    state = deepcopy(rng.bit_generator.state)
    x = np.arange(12, dtype=np.int16).reshape(4, 3)
    out = hard_fiber_sample(x, [12] * 3, [(), (2, 0, 2), (0, 2)], 0, rng)
    assert out.diagnostics.preserved_sets == ((0,), (0, 2), (1,), (2,))
    assert out.diagnostics.stop_reason == "disabled"
    assert out.diagnostics.executed_steps == 0
    assert np.array_equal(x, out.codes) and not np.shares_memory(x, out.codes)
    assert rng.bit_generator.state == state
    _accounting(out.diagnostics)


def test_single_row_reports_no_pairs_without_consuming_rng():
    rng = np.random.default_rng(8)
    before = deepcopy(rng.bit_generator.state)
    out = hard_fiber_sample(np.array([[1, 0]]), [2, 2], [(0, 1)], 20, rng)
    assert out.codes.tolist() == [[1, 0]]
    assert out.diagnostics.requested_steps == 20 and out.diagnostics.executed_steps == 0
    assert out.diagnostics.stop_reason == "no_row_pairs" and rng.bit_generator.state == before
    _accounting(out.diagnostics)


@pytest.mark.parametrize("size", [2, 3])
def test_small_tables_use_only_single_proposals(size):
    x = np.array([[i, i] for i in range(size)])
    out = hard_fiber_sample(x, [size, size], [], 100, np.random.default_rng(9))
    assert out.diagnostics.paired_proposals == 0 and out.diagnostics.single_proposals > 0
    assert 0 < out.diagnostics.lazy_steps < 100 and out.diagnostics.executed_steps == 100
    for field in (0, 1):
        assert _counts(x, (field,)) == _counts(out.codes, (field,))
    _accounting(out.diagnostics)


def test_sparse_large_domains_and_read_only_strided_input():
    high = np.iinfo(np.int64).max
    storage = np.array([[high - 1, high - 2, 0, 1], [1, 0, high - 2, high - 1]], dtype=np.int64)
    x = storage.T
    x.setflags(write=False)
    out = hard_fiber_sample(x, [high, high], [(0, 1)], 50, np.random.default_rng(3))
    assert _hist(out.codes) == _hist(x) and not np.shares_memory(out.codes, storage)
    _accounting(out.diagnostics)


@pytest.mark.parametrize("steps", [-1, True, np.bool_(False), 1.5, None, "5", np.nan])
def test_invalid_steps_fail_before_randomness(steps):
    rng = np.random.default_rng(11)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="非负整数"):
        hard_fiber_sample(np.zeros((4, 2), dtype=np.int64), [2, 2], [], steps, rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("table,domains,sets", [
    (np.zeros((0, 2), dtype=int), [2, 2], []),
    (np.zeros((4, 0), dtype=int), [], []),
    (np.zeros(4, dtype=int), [2], []),
    (np.zeros((4, 2), dtype=float), [2, 2], []),
    (np.zeros((4, 2), dtype=bool), [2, 2], []),
    (np.array([[np.nan]]), [2], []),
    (np.array([[-1]]), [2], []),
    (np.array([[2]]), [2], []),
    (np.array([[2**63]], dtype=np.uint64), [2], []),
    (np.zeros((4, 2), dtype=int), [2], []),
    (np.zeros((4, 2), dtype=int), [2, 0], []),
    (np.zeros((4, 2), dtype=int), [True, False], []),
    (np.zeros((4, 2), dtype=int), [2, True], []),
    (np.zeros((4, 2), dtype=int), [2.0, 2.0], []),
    (np.zeros((4, 2), dtype=int), [2, 2**63], []),
    (np.zeros((4, 2), dtype=int), [np.int64(2**63 - 1), np.uint64(2**63)], []),
    (np.zeros((4, 2), dtype=int), [2, 2], None),
    (np.zeros((4, 2), dtype=int), [2, 2], [(0, -1)]),
    (np.zeros((4, 2), dtype=int), [2, 2], [(2,)]),
    (np.zeros((4, 2), dtype=int), [2, 2], [(True,)]),
    (np.zeros((4, 2), dtype=int), [2, 2], [(0, True)]),
    (np.zeros((4, 2), dtype=int), [2, 2], [("0",)]),
    (np.zeros((4, 2), dtype=int), [2, 2], [[[0, 1]]]),
])
def test_invalid_inputs_are_rejected_even_for_zero_steps(table, domains, sets):
    rng = np.random.default_rng(11)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        hard_fiber_sample(table, domains, sets, 0, rng)
    assert rng.bit_generator.state == before


def test_invalid_rng_is_explicit():
    with pytest.raises(ValueError, match="Generator"):
        hard_fiber_sample(np.zeros((4, 2), dtype=int), [2, 2], [], 0, None)


def test_failed_final_invariant_never_returns_success(monkeypatch):
    original = np.zeros((4, 2), dtype=int)

    def broken(current, kept, by_field, rng):
        current[0, 0] = 1
        return "single_proposals", "histogram_moves", 1, 0

    monkeypatch.setattr(hard_fiber, "_step", broken)
    with pytest.raises(RuntimeError, match="完整边缘复核失败"):
        hard_fiber_sample(original, [2, 2], [(0, 1)], 1, np.random.default_rng(1))
    assert not original.any()


def _row_pair_actions(n, width, paired):
    for first in combinations(range(n), 2):
        if not paired:
            for field in range(width):
                yield np.array(first), (field,)
            continue
        rest = [i for i in range(n) if i not in first]
        for second in combinations(rest, 2):
            for fields in product(range(width), repeat=2):
                yield np.array(first + second), fields


def test_actual_chain_kernel_has_exact_inverse_factorial_stationarity():
    states = np.array(list(product(range(2), repeat=3)))
    starting = np.repeat(np.array([(a, b, a) for a, b in product(range(2), repeat=2)]), 2, axis=0)
    _, kept, by_field = _prepare(starting, [2] * 3, [(0, 1), (1, 2)])
    target = [_counts(starting, fields) for fields in kept]
    tables = []
    histograms = []
    for ids in combinations_with_replacement(range(8), 8):
        table = states[list(ids)]
        if all(_counts(table, fields) == count for fields, count in zip(kept, target)):
            tables.append(table)
            histograms.append(tuple(np.bincount(ids, minlength=8).tolist()))
    assert len(tables) == 9
    lookup = {hist: i for i, hist in enumerate(histograms)}
    matrix = [[Fraction(0) for _ in tables] for _ in tables]
    actions = [list(_row_pair_actions(8, 3, paired)) for paired in (False, True)]
    for i, table in enumerate(tables):
        matrix[i][i] = Fraction(1, 2)
        for choices in actions:
            visits = Counter()
            for rows, fields in choices:
                after = table.copy()
                hard_fiber._apply_exchange(after, rows, fields, kept, by_field)
                ids = np.ravel_multi_index(after.T, [2] * 3)
                visits[lookup[tuple(np.bincount(ids, minlength=8).tolist())]] += 1
            for j, count in visits.items():
                matrix[i][j] += Fraction(count, 4 * len(choices))
    weights = [factorial(8) // prod(factorial(n) for n in hist) for hist in histograms]
    pi = [Fraction(w, sum(weights)) for w in weights]
    assert pi[lookup[(1,) * 8]] == Fraction(4, 9)
    assert all(sum(row) == 1 for row in matrix)
    assert all(matrix[i][i] >= Fraction(1, 2) for i in range(9))
    assert all(sum(pi[i] * matrix[i][j] for i in range(9)) == pi[j] for j in range(9))
    assert all(pi[i] * matrix[i][j] == pi[j] * matrix[j][i] for i in range(9) for j in range(9))
    assert any(sum(matrix[i][j] for i in range(9)) != 1 for j in range(9))


def test_single_swaps_remove_paired_only_support_barrier():
    tables = [np.column_stack([np.arange(4), permutation]) for permutation in permutations(range(4))]
    _, kept, by_field = _prepare(tables[0], [4, 4], [])
    lookup = {tuple(table[:, 1]): i for i, table in enumerate(tables)}
    adjacency = {False: [set() for _ in tables], True: [set() for _ in tables]}
    for i, table in enumerate(tables):
        for paired in (False, True):
            for rows, fields in _row_pair_actions(4, 2, paired):
                after = table.copy()
                hard_fiber._apply_exchange(after, rows, fields, kept, by_field)
                after = after[np.argsort(after[:, 0])]
                adjacency[paired][i].add(lookup[tuple(after[:, 1])])

    def reachable(paired_only):
        seen = {0}
        while True:
            following = seen | {j for i in seen for j in adjacency[True][i]}
            if not paired_only:
                following |= {j for i in seen for j in adjacency[False][i]}
            if following == seen:
                return len(seen)
            seen = following

    # 两个不交换位从恒等排列只生成Klein四元群；奇偶性只是更弱的必要限制。
    assert reachable(True) == 4
    assert reachable(False) == 24


def test_diagnostics_are_finite_serializable_integer_counts():
    import json

    result = hard_fiber_sample(_cycle(), [3] * 4, [(0, 1), (1, 2), (2, 3), (0, 3)], 100, np.random.default_rng(4))
    payload = json.loads(json.dumps(asdict(result.diagnostics), allow_nan=False))
    assert payload["kernel_contract"] == "lazy-symmetric-exchanges-v1"
    assert payload["executed_steps"] == 100
    _accounting(result.diagnostics)
