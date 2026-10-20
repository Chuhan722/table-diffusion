"""引擎组装的锚点测试，含最终核矩阵、精确恒等式枚举与双行块端到端。"""
import itertools

import numpy as np
import pytest

from conftest import FOUR_FEATURES
from resevo.candidates import BlockSupport
from resevo.engine import (
    attribute_marginal,
    build_kernel,
    evolve,
    sample_next,
    two_stage_view,
)
from resevo.state import make_workload, table_loss

# 文档最终核矩阵锚点，行=当前行，列=下一状态
K_ANCHOR = np.array(
    [
        [0.693510, 0.000107, 0.306277, 0.000107],
        [0.118242, 0.0, 0.865607, 0.016152],
        [0.000136, 3.4552e-7, 0.999846, 0.0000185],
        [0.000131, 0.0000179, 0.051284, 0.948568],
    ]
)
EXPECTED_LOSS_ANCHOR = 0.242254152907815


def enumerate_next_generation(result, workload, ids):
    """完整枚举全部下一代表，返回 (概率, 新表损失) 列表，按块联合概率展开。"""
    combos = []
    index_ranges = [range(len(b.outcomes)) for b in result.blocks]
    for picks in itertools.product(*index_ranges):
        p = 1.0
        new_ids = np.asarray(ids).astype(np.int64, copy=True)
        for block, j in zip(result.blocks, picks):
            p *= float(block.probabilities[j])
            new_ids[list(block.rows)] = block.outcomes[j]
        combos.append((p, table_loss(workload, new_ids)))
    return combos


def test_kernel_matrix_anchor(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids)
    assert result.status == "ok"
    assert np.allclose(result.row_marginals, K_ANCHOR, atol=1e-6)
    # 最小概率格与文档一致，双精度下全支撑没有下溢为零
    assert abs(result.row_marginals[2, 1] - 3.4552e-7) / 3.4552e-7 < 1e-3
    # 全支撑指非保持后继不因负增益被删除，行 2 的保持概率本来就是 0
    for block in result.blocks:
        assert np.all(block.probabilities[1:] > 0.0)
        assert block.probabilities[0] >= 0.0


def test_headline_numbers(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids)
    assert abs(result.beta - 3.981402641214968) < 1e-9
    assert abs(result.direction_gain - 0.75) < 1e-12
    assert abs(result.interaction - 0.1548345932) < 1e-9
    assert abs(result.step - 1.4360970925) < 1e-9
    assert abs(result.expected_loss - EXPECTED_LOSS_ANCHOR) < 1e-12
    upper = 1.0 - result.step * 0.75 / 2
    assert abs(result.analytic_upper_bound - upper) < 1e-12
    assert result.expected_loss <= result.analytic_upper_bound + 1e-15


def test_row_change_probabilities(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids)
    rho = np.array([float(b.probabilities[1:].sum()) for b in result.blocks])
    assert np.allclose(rho, [0.306490, 1.0, 0.0001544, 0.0514324], atol=1e-6)
    assert abs(rho[1] - 1.0) < 1e-12
    assert abs(result.expected_changed_rows - 1.358077) < 1e-6


def test_enumeration_identity(four_record):
    # 256 种下一代完整枚举，平均损失精确等于 L-hD+h^2C，上升概率对上文档
    workload, ids = four_record
    result = build_kernel(workload, ids)
    combos = enumerate_next_generation(result, workload, ids)
    assert len(combos) == 256
    total_p = sum(p for p, _ in combos)
    mean_loss = sum(p * l for p, l in combos)
    rise_p = sum(p for p, l in combos if l > result.old_loss)
    assert abs(total_p - 1.0) < 1e-12
    assert abs(mean_loss - result.expected_loss) < 1e-12
    assert abs(rise_p - 0.0227002) < 1e-6


def test_cross_check_expected_loss_formula(four_record):
    # 用最终概率 K 的另一个表达式交叉验证期望损失
    workload, ids = four_record
    result = build_kernel(workload, ids)
    w = workload.weights
    e = workload.target - FOUR_FEATURES[ids].sum(axis=0)
    mu_sum = np.zeros_like(e)
    spread = 0.0
    for block in result.blocks:
        mu = block.probabilities @ block.deltas
        s2 = float(block.probabilities @ np.sum(block.deltas**2 * w, axis=1))
        mu_sum += mu
        spread += s2 - float(np.dot(mu * w, mu))
    alt = float(np.dot((e - mu_sum) * w, e - mu_sum)) / 2 + spread / 2
    assert abs(alt - result.expected_loss) < 1e-12


def test_row_budget_variant_anchor(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids, max_expected_rows=0.25)
    assert abs(result.step - 0.2643622352) < 1e-9
    assert abs(result.expected_changed_rows - 0.25) < 1e-12
    assert abs(result.expected_loss - 0.8125493094) < 1e-9


def test_perfect_table_freezes():
    # 目标恰好等于当前答案时没有正方向，明确报告停滞而不是静默收敛
    workload = make_workload(FOUR_FEATURES, np.array([2.0, 2.0, 1.0]), np.ones(3))
    ids = np.array([0, 1, 2, 3])
    result = build_kernel(workload, ids)
    assert result.status == "no_positive_direction"
    assert result.step == 0.0
    assert result.expected_loss == result.old_loss
    assert np.array_equal(result.row_marginals[np.arange(4), ids], np.ones(4))


def test_probabilities_valid_and_two_stage(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids)
    for i, block in enumerate(result.blocks):
        p = block.probabilities
        assert np.all(p >= 0.0)
        assert abs(p.sum() - 1.0) < 1e-12
        rho, T = two_stage_view(block)
        assert 0.0 <= rho <= 1.0 + 1e-12
        # 两阶段重组必须还原原核，rho*T(u) 等于非保持概率
        assert np.allclose(rho * T[1:], p[1:], atol=1e-12)
        assert abs((1.0 - rho) - p[0]) < 1e-12


def test_attribute_marginal_from_kernel(four_record):
    # 属性边缘从同一个核取，状态编号 2,3 表示 A=1，1,3 表示 B=1
    workload, ids = four_record
    result = build_kernel(workload, ids)
    a_of = lambda x: x // 2
    b_of = lambda x: x % 2
    for i in range(4):
        pa = attribute_marginal(result, i, a_of, 1)
        assert abs(pa - (result.row_marginals[i, 2] + result.row_marginals[i, 3])) < 1e-15
        pb = attribute_marginal(result, i, b_of, 1)
        assert abs(pb - (result.row_marginals[i, 1] + result.row_marginals[i, 3])) < 1e-15


def test_joint_kernel_not_independent_columns():
    # 两行表支持只含 00 与 11，跨行交互截住步长后每行核为 P(00)=P(11)=1/2，
    # 独立拼接两列均匀边缘会产生 01 与 10，联合核不会
    workload = make_workload(FOUR_FEATURES, np.array([1.0, 1.0, 1.0]), np.ones(3))
    ids = np.array([0, 0])
    supports = [
        BlockSupport((0,), ((0,), (3,))),
        BlockSupport((1,), ((0,), (3,))),
    ]
    result = build_kernel(workload, ids, supports=supports)
    assert abs(result.interaction - 0.75) < 1e-12
    assert abs(result.step - 1.0) < 1e-12
    K = result.row_marginals[0]
    assert K[1] == 0.0 and K[2] == 0.0
    assert abs(K[0] - 0.5) < 1e-12 and abs(K[3] - 0.5) < 1e-12
    a_of = lambda x: x // 2
    b_of = lambda x: x % 2
    pa1 = attribute_marginal(result, 0, a_of, 1)
    pb1 = attribute_marginal(result, 0, b_of, 1)
    joint_11 = K[3]
    # 两个属性边缘都是 1/2，但边缘乘积 1/4 不等于联合概率 1/2
    assert abs(pa1 - joint_11) < 1e-12
    assert abs(pa1 * pb1 - joint_11) > 0.2


def test_sampling_reproducible_and_readonly(four_record):
    workload, ids = four_record
    before = ids.copy()
    result = build_kernel(workload, ids)
    s1 = sample_next(result, ids, np.random.default_rng(20260911))
    s2 = sample_next(result, ids, np.random.default_rng(20260911))
    assert np.array_equal(s1, s2)
    assert np.array_equal(ids, before)
    assert len(s1) == len(ids)


def test_sampling_frequency_matches_kernel(four_record):
    workload, ids = four_record
    result = build_kernel(workload, ids)
    rng = np.random.default_rng(20260911)
    counts = np.zeros((4, 4))
    n = 20000
    for _ in range(n):
        nxt = sample_next(result, ids, rng)
        for i, x in enumerate(nxt):
            counts[i, x] += 1
    assert np.allclose(counts / n, result.row_marginals, atol=0.02)


def test_two_row_block_end_to_end_anchor():
    # 双行块补偿例子，单行支持停滞，两行完整支持块 E[L'] 对上文档
    workload = make_workload(
        FOUR_FEATURES, np.array([1.0, 1.0, 0.0]), np.array([2.0, 2.0, 1.0])
    )
    ids = np.array([0, 3])
    single = build_kernel(workload, ids)
    assert single.status == "no_positive_direction"

    joint_support = [
        BlockSupport((0, 1), tuple(itertools.product(range(4), repeat=2)))
    ]
    result = build_kernel(workload, ids, supports=joint_support)
    assert result.status == "ok"
    assert abs(result.old_loss - 0.5) < 1e-15
    assert abs(result.expected_loss - 0.0040808634) < 1e-9
    combos = enumerate_next_generation(result, workload, ids)
    assert len(combos) == 16
    mean_loss = sum(p * l for p, l in combos)
    assert abs(mean_loss - result.expected_loss) < 1e-12


def test_evolve_smoke(four_record):
    workload, ids = four_record
    out = evolve(workload, ids, 6, np.random.default_rng(20260911))
    assert out.stop_reason in ("no_positive_direction", "round_limit")
    assert 1 <= len(out.records) <= 6
    assert all(r.old_loss >= 0.0 for r in out.records)
    # 轮内共享旧残差由构造保证，这里检查日志与终态自洽
    final_loss = table_loss(workload, out.state_ids)
    assert final_loss >= 0.0
    if out.stop_reason == "no_positive_direction":
        assert out.records[-1].status == "no_positive_direction"


def _pure_stay_supports(n):
    """空路径菜单，规范化后纯保持，必然冻结。"""
    return [BlockSupport((i,), ()) for i in range(n)]


def test_provider_mode_matches_fixed_supports(four_record):
    """固定菜单提供器与旧接口逐轮一致，接口改动不改变任何数学。"""
    workload, ids = four_record
    supports = None  # 全集由 build_kernel 内部构造，这里提供器显式给出同样的全集
    from resevo.candidates import full_single_row_supports

    fixed = full_single_row_supports(len(ids), workload.num_states)
    calls = []

    def provider(state_ids, round_index, frozen_streak):
        calls.append(round_index)
        return workload, fixed

    out_a = evolve(workload, ids, 5, np.random.default_rng(99), supports=fixed)
    out_b = evolve(
        None, ids, 5, np.random.default_rng(99), support_provider=provider
    )
    assert calls == list(range(len(out_b.records)))
    assert out_a.stop_reason == out_b.stop_reason
    assert len(out_a.records) == len(out_b.records)
    for ra, rb in zip(out_a.records, out_b.records):
        assert ra.old_loss == rb.old_loss
        assert ra.beta == rb.beta
        assert ra.step == rb.step
        assert ra.status == rb.status
    np.testing.assert_array_equal(out_a.state_ids, out_b.state_ids)


def test_provider_and_supports_mutually_exclusive(four_record):
    workload, ids = four_record
    from resevo.candidates import full_single_row_supports

    fixed = full_single_row_supports(len(ids), workload.num_states)
    with pytest.raises(ValueError):
        evolve(
            workload, ids, 2, np.random.default_rng(0),
            supports=fixed, support_provider=lambda s, k, f: (workload, fixed),
        )
    with pytest.raises(ValueError):
        evolve(None, ids, 2, np.random.default_rng(0))


def test_frozen_retry_then_recover(four_record):
    """前两轮无增益菜单不停机，第三轮好菜单继续下降，重试预算内恢复。"""
    workload, ids = four_record
    from resevo.candidates import full_single_row_supports

    good = full_single_row_supports(len(ids), workload.num_states)

    def provider(state_ids, round_index, frozen_streak):
        if round_index < 2:
            return workload, _pure_stay_supports(len(state_ids))
        return workload, good

    out = evolve(
        None, ids, 6, np.random.default_rng(5),
        support_provider=provider, max_frozen_retries=3,
    )
    assert [r.status for r in out.records[:2]] == ["no_positive_direction"] * 2
    assert out.records[2].status == "ok"
    assert out.records[2].old_loss == out.records[0].old_loss  # 冻结轮表没动
    assert table_loss(workload, out.state_ids) <= out.records[0].old_loss


def test_frozen_streak_exhausts_retries(four_record):
    """连续冻结超过重试预算即停，且停止原因明确报告。"""
    workload, ids = four_record

    def provider(state_ids, round_index, frozen_streak):
        return workload, _pure_stay_supports(len(state_ids))

    out = evolve(
        None, ids, 50, np.random.default_rng(1),
        support_provider=provider, max_frozen_retries=2,
    )
    assert out.stop_reason == "no_positive_direction"
    assert len(out.records) == 3  # 首次冻结加两次重试
    np.testing.assert_array_equal(out.state_ids, ids)
    out_default = evolve(
        None, ids, 50, np.random.default_rng(1), support_provider=provider
    )
    assert len(out_default.records) == 1  # 默认零重试保持旧行为


def test_frozen_streak_resets_after_progress(four_record):
    """冻结计数在成功轮清零，间歇冻结不会被累计成停止条件。"""
    workload, ids = four_record
    from resevo.candidates import full_single_row_supports

    good = full_single_row_supports(len(ids), workload.num_states)

    def provider(state_ids, round_index, frozen_streak):
        if round_index % 2 == 0:
            return workload, _pure_stay_supports(len(state_ids))
        return workload, good

    out = evolve(
        None, ids, 8, np.random.default_rng(3),
        support_provider=provider, max_frozen_retries=1,
    )
    # 奇数轮都是好菜单，冻结从未连续两次，应跑满或提前把损失打到零
    assert out.stop_reason in ("round_limit", "no_positive_direction")
    if out.stop_reason == "no_positive_direction":
        assert [r.status for r in out.records[-2:]] == ["no_positive_direction"] * 2
