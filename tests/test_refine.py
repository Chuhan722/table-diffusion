"""零空间精修模块测试，边缘不变性、门槛、停止、确定性与自检器本身。"""
import numpy as np
import pytest

from resevo.refine import (
    embed_kurtosis,
    refine_rows,
    verify_null_space,
)


def make_flat_table(seed: int = 5, n_ctx: int = 60) -> tuple[np.ndarray, list[int]]:
    """构造欠聚簇小表，若干上下文桶，桶内 (j,k) 值对铺满对角与交叉图案。

    前四列为上下文（桶内相同），后两列为字段对 (j,k)，
    每桶放 (0,0)(1,1)(0,1)(1,0) 各若干行，保证存在大量合法零空间移动。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_ctx):
        ctx = rng.integers(0, 3, size=4)
        for vj, vk in ((0, 0), (1, 1), (0, 1), (1, 0)):
            for _ in range(int(rng.integers(1, 4))):
                rows.append(list(ctx) + [vj, vk])
    X = np.array(rows, dtype=np.int16)
    domain_sizes = [3, 3, 3, 3, 2, 2]
    return X, domain_sizes


def make_clustered_table(n: int = 500) -> tuple[np.ndarray, list[int]]:
    """构造重尾表，绝大多数行同一模式少数散点，嵌入峰度显著为正。"""
    rng = np.random.default_rng(9)
    X = np.zeros((n, 5), dtype=np.int16)
    scatter = rng.integers(0, 4, size=(n // 10, 5))
    X[: n // 10] = scatter
    domain_sizes = [4] * 5
    return X, domain_sizes


def test_refine_preserves_all_first_second_marginals():
    """强制施药后全部一阶二阶计数逐格不变且确有移动发生。"""
    X, doms = make_flat_table()
    X1, report = refine_rows(X, doms, seed=1, force=True, max_sweeps=8)
    assert report.treated
    assert report.accepted_total > 0, "构造表应存在合法移动"
    assert verify_null_space(X, X1, doms)
    assert not np.array_equal(X, X1), "有接受时表应发生变化"


def test_refine_does_not_mutate_input():
    X, doms = make_flat_table()
    before = X.copy()
    refine_rows(X, doms, seed=1, force=True, max_sweeps=4)
    assert np.array_equal(X, before)


def test_gate_skips_clustered_table():
    """峰度为正的重尾表不过门槛，原样输出零改动。"""
    X, doms = make_clustered_table()
    assert embed_kurtosis(X, doms) > 0
    X1, report = refine_rows(X, doms, seed=1)
    assert not report.treated
    assert report.accepted_total == 0
    assert np.array_equal(X, X1)


def test_gate_admits_flat_table():
    """构造欠聚簇表峰度为负，过默认门槛。"""
    X, doms = make_flat_table()
    kurt = embed_kurtosis(X, doms)
    assert kurt < -0.05
    _, report = refine_rows(X, doms, seed=1, max_sweeps=2)
    assert report.treated


def test_stop_rule_halts_before_max_sweeps():
    """移动耗尽或接受率跌破阈值时提前收手。"""
    X, doms = make_flat_table(n_ctx=20)
    _, report = refine_rows(X, doms, seed=1, force=True, max_sweeps=64)
    assert 0 < len(report.sweeps) < 64
    last = report.sweeps[-1]
    assert last.accepted == 0 or last.rate < 0.006


def test_refine_deterministic_same_seed():
    X, doms = make_flat_table()
    X1, r1 = refine_rows(X, doms, seed=7, force=True, max_sweeps=4)
    X2, r2 = refine_rows(X, doms, seed=7, force=True, max_sweeps=4)
    assert np.array_equal(X1, X2)
    assert r1.accepted_total == r2.accepted_total


def test_no_legal_moves_returns_input_unchanged():
    """全行上下文互异的高域小表无候选，施药也零接受零改动。"""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 50, size=(40, 4)).astype(np.int16)
    doms = [50] * 4
    X1, report = refine_rows(X, doms, seed=1, force=True, max_sweeps=3)
    assert report.accepted_total == 0
    assert np.array_equal(X, X1)


def test_verify_null_space_detects_first_order_change():
    X, doms = make_flat_table()
    Y = X.copy()
    Y[0, 0] = (Y[0, 0] + 1) % doms[0]
    assert not verify_null_space(X, Y, doms)


def test_verify_null_space_detects_second_order_change():
    """一阶守恒二阶破坏的改动必须被识破，两行同字段对换不同值。"""
    X, doms = make_flat_table()
    Y = X.copy()
    r1, r2 = None, None
    for a in range(len(Y)):
        for b in range(a + 1, len(Y)):
            if Y[a, 4] != Y[b, 4] and not np.array_equal(Y[a], Y[b]):
                r1, r2 = a, b
                break
        if r1 is not None:
            break
    assert r1 is not None
    Y[r1, 4], Y[r2, 4] = Y[r2, 4], Y[r1, 4]
    if verify_null_space(X, Y, doms):
        pytest.skip("找到的两行恰为合法交换，构造不适用")
    assert not verify_null_space(X, Y, doms)


def test_simpson_monotone_nondecreasing():
    """贪心只接受正增量，辛普森集中度逐扫不降。"""
    X, doms = make_flat_table()
    _, report = refine_rows(X, doms, seed=2, force=True, max_sweeps=8)
    values = [s.simpson for s in report.sweeps]
    assert all(b >= a for a, b in zip(values, values[1:]))
