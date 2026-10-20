"""与复制进仓的参考实现 entropic_kernel.py 做随机小例对拍。

参考实现与本仓实现运算顺序刻意保持一致，随机小例上应当逐位或近逐位一致。
参考文件从参考资料复制，不 import 任何旧仓路径。
"""
import itertools

import numpy as np
import pytest

from entropic_kernel import construct_entropic_kernel
from transition_kernel import BlockSupport as RefBlockSupport

from resevo.candidates import BlockSupport
from resevo.engine import build_kernel
from resevo.state import make_workload


def random_case(rng, max_states=6, max_queries=4, max_rows=8):
    m = int(rng.integers(2, max_states + 1))
    q = int(rng.integers(1, max_queries + 1))
    n = int(rng.integers(1, max_rows + 1))
    features = rng.integers(0, 2, size=(m, q)).astype(float)
    target = rng.uniform(-2.0, n + 2.0, size=q).round(3)
    weights = rng.uniform(0.5, 2.0, size=q).round(3)
    ids = rng.integers(0, m, size=n)
    return features, target, weights, ids


def assert_same_kernel(mine, ref, atol=1e-13):
    assert mine.status in ("ok", "no_positive_direction")
    assert abs(mine.old_loss - ref.old_loss) <= atol
    assert abs(mine.beta - ref.beta) <= atol * max(1.0, abs(ref.beta))
    assert abs(mine.direction_gain - ref.dissipation) <= atol
    assert abs(mine.interaction - ref.cross_curvature) <= atol
    assert abs(mine.step - ref.step) <= atol * max(1.0, abs(ref.step))
    assert abs(mine.expected_loss - ref.expected_loss) <= atol
    assert abs(mine.analytic_upper_bound - ref.certified_upper_bound) <= atol
    assert mine.row_marginals.shape == ref.row_marginals.shape
    assert np.allclose(mine.row_marginals, ref.row_marginals, atol=atol)


def test_parity_random_single_row_full_support():
    # 20 个随机小例，全集单行支持，与参考实现对拍
    rng = np.random.default_rng(20260911)
    for k in range(20):
        features, target, weights, ids = random_case(rng)
        workload = make_workload(features, target, weights)
        mine = build_kernel(workload, ids)
        ref = construct_entropic_kernel(
            ids, features, target, weights,
            stay_probability=0.9, progress_fraction=0.5,
        )
        assert_same_kernel(mine, ref)


def test_parity_four_record_anchor(four_record):
    workload, ids = four_record
    mine = build_kernel(workload, ids)
    ref = construct_entropic_kernel(
        ids, np.asarray(workload.features), np.asarray(workload.target),
        np.asarray(workload.weights),
        stay_probability=0.9, progress_fraction=0.5,
    )
    assert_same_kernel(mine, ref)
    # 参考实现自身也应复现文档 beta 锚点
    assert abs(ref.beta - 3.981402641214968) < 1e-9


def test_parity_frozen_case():
    # 完美表两边都必须冻结，期望损失等于旧损失
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([1.0, 1.0])
    weights = np.ones(2)
    ids = np.array([1, 2])
    workload = make_workload(features, target, weights)
    mine = build_kernel(workload, ids)
    ref = construct_entropic_kernel(ids, features, target, weights)
    assert mine.status == "no_positive_direction"
    assert ref.step == 0.0 and mine.step == 0.0
    assert abs(mine.expected_loss - ref.expected_loss) < 1e-15


def test_parity_two_row_blocks():
    # 双行记录块的联合支持对拍，含文档补偿例子
    features = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    target = np.array([1.0, 1.0, 0.0])
    weights = np.array([2.0, 2.0, 1.0])
    ids = np.array([0, 3])
    outcomes = tuple(itertools.product(range(4), repeat=2))
    workload = make_workload(features, target, weights)
    mine = build_kernel(workload, ids, supports=[BlockSupport((0, 1), outcomes)])
    ref = construct_entropic_kernel(
        ids, features, target, weights,
        supports=[RefBlockSupport((0, 1), outcomes)],
        stay_probability=0.9, progress_fraction=0.5,
    )
    assert_same_kernel(mine, ref)
    assert abs(mine.expected_loss - 0.0040808634) < 1e-9


def test_parity_random_two_row_blocks():
    # 随机两行块划分对拍
    rng = np.random.default_rng(20260912)
    for _ in range(10):
        features, target, weights, ids = random_case(rng, max_states=4, max_rows=6)
        n = len(ids)
        workload = make_workload(features, target, weights)
        m = features.shape[0]
        my_supports, ref_supports = [], []
        for start in range(0, n, 2):
            rows = tuple(range(start, min(n, start + 2)))
            outs = tuple(itertools.product(range(m), repeat=len(rows)))
            my_supports.append(BlockSupport(rows, outs))
            ref_supports.append(RefBlockSupport(rows, outs))
        mine = build_kernel(workload, ids, supports=my_supports)
        ref = construct_entropic_kernel(
            ids, features, target, weights, supports=ref_supports,
            stay_probability=0.9, progress_fraction=0.5,
        )
        assert_same_kernel(mine, ref)


def test_parity_nondefault_parameters():
    # 非默认保持比例 alpha 阻尼的随机例对拍，堵住只在默认参数对拍的盲区
    rng = np.random.default_rng(20260914)
    for stay, alpha, damping in [(0.7, 0.3, 1.0), (0.5, 0.8, 0.5), (0.95, 0.2, 0.25)]:
        for _ in range(5):
            features, target, weights, ids = random_case(rng)
            workload = make_workload(features, target, weights)
            mine = build_kernel(
                workload, ids,
                stay_probability=stay, alpha=alpha, damping=damping,
            )
            ref = construct_entropic_kernel(
                ids, features, target, weights,
                stay_probability=stay, progress_fraction=alpha, damping=damping,
            )
            assert_same_kernel(mine, ref)


def test_parity_with_mobility_weights():
    # 非均匀迁移率也要一致
    rng = np.random.default_rng(20260913)
    features, target, weights, ids = random_case(rng, max_states=4, max_rows=4)
    m = features.shape[0]
    workload = make_workload(features, target, weights)
    my_supports, ref_supports = [], []
    for i in range(len(ids)):
        outs = tuple((u,) for u in range(m))
        mob = tuple(float(c) for c in rng.uniform(0.1, 3.0, size=m).round(2))
        my_supports.append(BlockSupport((i,), outs, mob))
        ref_supports.append(RefBlockSupport((i,), outs, mob))
    mine = build_kernel(workload, ids, supports=my_supports)
    ref = construct_entropic_kernel(
        ids, features, target, weights, supports=ref_supports,
        stay_probability=0.9, progress_fraction=0.5,
    )
    assert_same_kernel(mine, ref)
