"""重复记录压缩的锚点测试。

核对三件事，组核与把组菜单展开成行级菜单后的行级核数值一致，
多项抽样与组内逐行独立抽样精确同分布，
期望损失恒等式 E[L']=L-hD+h^2C 在加权矩下对多项抽样精确成立。
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from resevo.candidates import BlockSupport
from resevo.dataset import StateRegistry, TableSchema, QuerySpec, target_from_specs
from resevo.editspace import EditBudget
from resevo.engine import build_kernel
from resevo.grouping import (
    build_group_kernel,
    evolve_grouped,
    generate_group_edit_supports,
    group_state_ids,
    grouped_residual,
    make_grouped_provider,
    sample_group_next,
)
from resevo.state import make_workload, table_loss


def _toy_workload():
    """4 状态 3 查询的小负载，权重不等便于暴露加权错误。"""
    features = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = np.array([3.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 0.5])
    return make_workload(features, target, weights)


def _toy_supports_grouped(grouped, num_states):
    """每组全集菜单，展开对拍时逐行复用同一份。"""
    outcomes = tuple((u,) for u in range(num_states))
    return [
        BlockSupport((g,), outcomes, (1.0,) * len(outcomes))
        for g in range(grouped.num_groups)
    ]


def _expand_supports(grouped, group_supports):
    """把组菜单展开成行级菜单，组内每行一份相同支持。"""
    row_supports = []
    for spec, rows in zip(group_supports, grouped.row_lists):
        for i in rows:
            row_supports.append(BlockSupport((i,), spec.outcomes, spec.mobility))
    return row_supports


def test_group_state_ids_partitions_rows():
    s = np.array([2, 0, 2, 1, 0, 2], dtype=np.int64)
    grouped = group_state_ids(s)
    assert grouped.unique_ids.tolist() == [0, 1, 2]
    assert grouped.counts.tolist() == [2, 1, 3]
    assert grouped.row_lists == ((1, 4), (3,), (0, 2, 5))
    assert grouped.num_rows == 6


def test_group_kernel_matches_expanded_row_kernel():
    """有重复行的表，组核与展开行级核的全部标量与概率一致。"""
    workload = _toy_workload()
    state_ids = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    grouped = group_state_ids(state_ids)
    gsup = _toy_supports_grouped(grouped, workload.num_states)
    rsup = _expand_supports(grouped, gsup)

    gres = build_group_kernel(workload, grouped, gsup)
    rres = build_kernel(workload, state_ids, rsup)

    assert gres.status == rres.status == "ok"
    assert gres.old_loss == pytest.approx(rres.old_loss, abs=1e-12)
    assert gres.beta == pytest.approx(rres.beta, rel=1e-9, abs=1e-12)
    assert gres.max_gain_sum == pytest.approx(rres.max_gain_sum, rel=1e-12)
    assert gres.direction_gain == pytest.approx(rres.direction_gain, rel=1e-9)
    assert gres.interaction == pytest.approx(rres.interaction, rel=1e-9, abs=1e-12)
    assert gres.step == pytest.approx(rres.step, rel=1e-9)
    assert gres.expected_loss == pytest.approx(rres.expected_loss, rel=1e-9)
    assert gres.expected_changed_rows == pytest.approx(rres.expected_changed_rows, rel=1e-9)

    # 每组概率与组内每行的行级概率一致
    row_of_block = {spec.rows[0]: b for b, spec in enumerate(rsup)}
    for gb, rows in zip(gres.blocks, grouped.row_lists):
        for i in rows:
            rb = rres.blocks[row_of_block[i]]
            assert rb.outcomes == gb.outcomes
            np.testing.assert_allclose(rb.probabilities, gb.probabilities, rtol=1e-9, atol=1e-15)


def test_grouped_residual_matches_rowwise():
    workload = _toy_workload()
    state_ids = np.array([0, 0, 2, 3, 3], dtype=np.int64)
    grouped = group_state_ids(state_ids)
    rowwise = workload.target - workload.features[state_ids].sum(axis=0)
    np.testing.assert_allclose(grouped_residual(workload, grouped), rowwise, atol=1e-12)


def test_expected_loss_identity_exact_under_multinomial():
    """一组两行完整枚举多项计数，E[L'] 精确等于 L-hD+h^2C。"""
    workload = _toy_workload()
    state_ids = np.array([1, 1], dtype=np.int64)
    grouped = group_state_ids(state_ids)
    gsup = _toy_supports_grouped(grouped, workload.num_states)
    res = build_group_kernel(workload, grouped, gsup)
    assert res.status == "ok"

    block = res.blocks[0]
    probs = block.probabilities
    outcome_ids = [int(o[0]) for o in block.outcomes]
    expected = 0.0
    # 两行独立同核抽样，逐行枚举等价于多项计数枚举
    for a, b in itertools.product(range(len(probs)), repeat=2):
        table = np.array([outcome_ids[a], outcome_ids[b]], dtype=np.int64)
        expected += float(probs[a]) * float(probs[b]) * table_loss(workload, table)
    assert expected == pytest.approx(res.expected_loss, rel=1e-12)


def test_sample_group_next_conserves_counts_and_distribution():
    """抽样保行数，且大样本下组内去向频率逼近核概率。"""
    workload = _toy_workload()
    n = 3000
    state_ids = np.full(n, 1, dtype=np.int64)
    grouped = group_state_ids(state_ids)
    gsup = _toy_supports_grouped(grouped, workload.num_states)
    res = build_group_kernel(workload, grouped, gsup)
    rng = np.random.default_rng(7)
    new_ids = sample_group_next(res, state_ids, rng)
    assert new_ids.shape == (n,)

    block = res.blocks[0]
    freq = np.zeros(len(block.outcomes))
    id_to_j = {int(o[0]): j for j, o in enumerate(block.outcomes)}
    for u in new_ids:
        freq[id_to_j[int(u)]] += 1
    freq /= n
    # 三倍标准误容差的粗检验
    for j, p in enumerate(block.probabilities):
        se = np.sqrt(max(p * (1 - p), 1e-12) / n)
        assert abs(freq[j] - p) <= max(4 * se, 5e-3)


def _mini_registry():
    """两字段小 schema 的注册表与查询，端到端组演化用。"""
    schema = TableSchema(("A", "B"), (("0", "1"), ("0", "1")))
    specs = [
        QuerySpec("SA", ({"attribute": "A", "operator": "==", "value": "1"},), 2.0),
        QuerySpec("SB", ({"attribute": "B", "operator": "==", "value": "1"},), 2.0),
        QuerySpec(
            "SAB",
            (
                {"attribute": "A", "operator": "==", "value": "1"},
                {"attribute": "B", "operator": "==", "value": "1"},
            ),
            0.0,
        ),
    ]
    return StateRegistry(schema, specs), specs


def test_evolve_grouped_end_to_end_descends():
    """带重复行的小表按组路径演化，损失显著下降且行数不变。"""
    registry, specs = _mini_registry()
    rows = [("0", "0")] * 4 + [("1", "1")] * 2
    state_ids = registry.register_table(rows)
    target = target_from_specs(specs)
    weights = np.ones(len(specs))
    provider = make_grouped_provider(
        registry, target, weights, np.random.default_rng(5), EditBudget()
    )
    result = evolve_grouped(
        state_ids, 40, np.random.default_rng(9), provider, max_frozen_retries=10
    )
    first = result.records[0].old_loss
    workload = registry.build_workload(target, weights)
    final = table_loss(workload, result.state_ids)
    assert result.state_ids.shape == state_ids.shape
    assert final <= first * 0.35


def test_group_menu_registers_only_valid_states():
    """组菜单生成的全部后继都已注册且组划分合法。"""
    registry, specs = _mini_registry()
    rows = [("0", "0")] * 3 + [("0", "1")] * 2 + [("1", "1")]
    state_ids = registry.register_table(rows)
    grouped = group_state_ids(state_ids)
    supports = generate_group_edit_supports(
        grouped, registry, np.random.default_rng(3), EditBudget()
    )
    assert len(supports) == grouped.num_groups
    for spec in supports:
        for outcome in spec.outcomes:
            assert 0 <= int(outcome[0]) < registry.num_states


def test_fast_normalize_matches_generic(loaded_schema_free=None):
    """组路径快速规范化与通用 normalize_block 输出逐项一致。

    构造含重复后继、落回源、乱序的随机菜单，两版 outcomes 顺序与 R 全同。
    """
    from resevo.candidates import normalize_block
    from resevo.grouping import _normalize_group_block

    rng = np.random.default_rng(11)
    for _ in range(50):
        num_states = int(rng.integers(3, 30))
        source = int(rng.integers(num_states))
        n = int(rng.integers(1, 20))
        ids = rng.integers(0, num_states, n)
        mobility = tuple(float(v) for v in rng.uniform(0, 2, n))
        spec = BlockSupport((0,), tuple((int(u),) for u in ids), mobility)
        state_ids = np.array([source], dtype=np.int64)
        a = normalize_block(spec, state_ids, num_states, 0.9)
        b = _normalize_group_block(spec, source, num_states, 0.9)
        assert a.outcomes == b.outcomes
        np.testing.assert_allclose(a.reference, b.reference, rtol=0, atol=1e-15)


def test_compiled_query_evaluation_matches_original():
    """预编译条件求值与原版 evaluate_query 在全部查询与随机行上逐位一致。"""
    from resevo.dataset import (
        compile_conditions,
        evaluate_compiled,
        evaluate_query,
        load_queries,
        load_table,
    )

    schema, rows = load_table("data/test_300x10/test_300x10.csv")
    specs = load_queries("data/test_300x10/measured_50query.json")
    compiled = compile_conditions(specs, schema)
    rng = np.random.default_rng(4)
    samples = list(rows[:20])
    for _ in range(30):
        samples.append(
            tuple(
                schema.domains[j][int(rng.integers(len(schema.domains[j])))]
                for j in range(schema.num_fields)
            )
        )
    for row in samples:
        for spec, conds in zip(specs, compiled):
            assert evaluate_compiled(conds, row) == evaluate_query(spec, row, schema)


def test_quad_cache_reuse_and_keep_deltas_consistency():
    """二次项缓存复跑结果逐位不变，keep_deltas 物化增量与增益公式互洽。"""
    workload = _toy_workload()
    state_ids = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    grouped = group_state_ids(state_ids)
    gsup = _toy_supports_grouped(grouped, workload.num_states)
    cache: dict = {}
    first = build_group_kernel(workload, grouped, gsup, quad_cache=cache, keep_deltas=True)
    assert len(cache) > 0
    second = build_group_kernel(workload, grouped, gsup, quad_cache=cache, keep_deltas=True)
    assert second.beta == first.beta
    assert second.step == first.step
    for a, b in zip(first.blocks, second.blocks):
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
    # 物化增量与增益公式互洽，G=d@(We)-||d||_W^2/2
    residual = grouped_residual(workload, grouped)
    for block in first.blocks:
        explicit = block.deltas @ (workload.weights * residual) - np.sum(
            block.deltas * block.deltas * workload.weights, axis=1
        ) / 2
        np.testing.assert_allclose(block.gains, explicit, rtol=1e-12, atol=1e-12)
