"""有限编辑候选生成器测试，预算、复现、红线签名与规范化合并。"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from resevo.candidates import normalize_block
from resevo.dataset import (
    QuerySpec,
    StateRegistry,
    TableSchema,
    load_queries,
    load_table,
    query_field_sets,
)
from resevo.editspace import (
    EditBudget,
    generate_edit_supports,
    tuples_from_ids,
)


def toy_schema() -> tuple[TableSchema, list[QuerySpec]]:
    """两字段 0/1 玩具 schema，查询为 A、B、A 且 B。"""
    schema = TableSchema(("A", "B"), (("0", "1"), ("0", "1")))
    specs = [
        QuerySpec("qa", ({"attribute": "A", "operator": "==", "value": "1"},), 1.0),
        QuerySpec("qb", ({"attribute": "B", "operator": "==", "value": "1"},), 1.0),
        QuerySpec(
            "qab",
            (
                {"attribute": "A", "operator": "==", "value": "1"},
                {"attribute": "B", "operator": "==", "value": "1"},
            ),
            0.0,
        ),
    ]
    return schema, specs


@pytest.fixture()
def real_setup():
    schema, rows = load_table("data/test_300x10/test_300x10.csv")
    specs = load_queries("data/test_300x10/measured_50query.json")
    registry = StateRegistry(schema, specs)
    return schema, rows, specs, registry


def test_signature_excludes_residual_and_target():
    """结构红线，生成器签名不接收残差损失与目标，杜绝残差参与候选生成。"""
    params = set(inspect.signature(generate_edit_supports).parameters)
    banned = {"residual", "loss", "target", "workload", "gains", "weights"}
    assert params.isdisjoint(banned)


def test_budget_cap_and_registration(real_setup):
    schema, rows, specs, registry = real_setup
    rng = np.random.default_rng(7)
    table = rows[:40]
    supports = generate_edit_supports(table, schema, registry, rng)
    assert len(supports) == 40
    cap = EditBudget().max_nonstay_paths()
    assert cap == 32
    for i, spec in enumerate(supports):
        assert spec.rows == (i,)
        assert len(spec.outcomes) <= cap
        assert len(spec.mobility) == len(spec.outcomes)
        assert all(m == 1.0 for m in spec.mobility)
        for (state_id,) in spec.outcomes:
            row = registry.state_tuple(state_id)
            assert all(row[j] in schema.domains[j] for j in range(schema.num_fields))


def test_same_seed_same_menu(real_setup):
    schema, rows, specs, registry = real_setup
    table = rows[:15]
    menu_a = generate_edit_supports(table, schema, registry, np.random.default_rng(42))
    menu_b = generate_edit_supports(table, schema, registry, np.random.default_rng(42))
    for sa, sb in zip(menu_a, menu_b):
        assert sa.outcomes == sb.outcomes
        assert sa.mobility == sb.mobility


def test_reference_mass_merging_toy():
    """小值域下重复路径必然出现，规范化后质量守恒且按路径数合并。"""
    schema, specs = toy_schema()
    registry = StateRegistry(schema, specs)
    table = [("0", "0"), ("1", "1")]
    ids = registry.register_table(table)
    rng = np.random.default_rng(0)
    supports = generate_edit_supports(
        table, schema, registry, rng, joint_field_sets=[(0, 1)]
    )
    for spec, source_id in zip(supports, ids):
        nb = normalize_block(spec, ids, registry.num_states)
        assert nb.outcomes[0] == (int(source_id),)
        assert nb.reference[0] == pytest.approx(0.9)
        assert nb.reference.sum() == pytest.approx(1.0)
        # 两字段 0/1 值域下非保持后继至多 3 个，重复路径的质量必须被合并
        assert len(nb.outcomes) <= 4


def test_joint_edit_reaches_diagonal():
    """联合修改让 00 一步到 11，这是单字段修改覆盖不了的方向。"""
    schema, specs = toy_schema()
    registry = StateRegistry(schema, specs)
    table = [("0", "0")]
    budget = EditBudget(
        max_edit_fields=0, max_values_per_field=0, donor_copies=0,
        joint_edits=8, explore_edits=0,
    )
    seen = set()
    for seed in range(10):
        supports = generate_edit_supports(
            table, schema, registry, np.random.default_rng(seed),
            budget=budget, joint_field_sets=[(0, 1)],
        )
        for (state_id,) in supports[0].outcomes:
            seen.add(registry.state_tuple(state_id))
    assert ("1", "1") in seen


def test_explore_can_create_novel_states(real_setup):
    """探索来源允许产生当前种群中不存在的状态，注册表随之增长。"""
    schema, rows, specs, registry = real_setup
    table = rows[:10]
    registry.register_table(table)
    before = registry.num_states
    budget = EditBudget(
        max_edit_fields=0, max_values_per_field=0, donor_copies=0,
        joint_edits=0, explore_edits=4,
    )
    generate_edit_supports(table, schema, registry, np.random.default_rng(3), budget=budget)
    assert registry.num_states > before


def test_donor_only_recombines_table_values():
    """供体复制只重组当前表已有的字段值，不发明新值。"""
    schema, specs = toy_schema()
    registry = StateRegistry(schema, specs)
    table = [("0", "0"), ("0", "1")]
    registry.register_table(table)
    budget = EditBudget(
        max_edit_fields=0, max_values_per_field=0, donor_copies=8,
        joint_edits=0, explore_edits=0,
    )
    supports = generate_edit_supports(
        table, schema, registry, np.random.default_rng(11), budget=budget
    )
    column_values = [{row[j] for row in table} for j in range(2)]
    for spec in supports:
        for (state_id,) in spec.outcomes:
            row = registry.state_tuple(state_id)
            assert all(row[j] in column_values[j] for j in range(2))


def test_degenerate_domain_pure_stay():
    """值域全退化时没有可生成路径，规范化后是纯保持块。"""
    schema = TableSchema(("A",), (("x",),))
    specs = [QuerySpec("q", ({"attribute": "A", "operator": "==", "value": "x"},), 1.0)]
    registry = StateRegistry(schema, specs)
    table = [("x",)]
    ids = registry.register_table(table)
    supports = generate_edit_supports(
        table, schema, registry, np.random.default_rng(0)
    )
    nb = normalize_block(supports[0], ids, registry.num_states)
    assert len(nb.outcomes) == 1
    np.testing.assert_array_equal(nb.reference, [1.0])


def test_tuples_from_ids_roundtrip(real_setup):
    schema, rows, specs, registry = real_setup
    ids = registry.register_table(rows[:5])
    assert tuples_from_ids(registry, ids) == rows[:5]


def test_edit_provider_descends_on_toy_table():
    """编辑候选加提供器端到端，无死角玩具问题损失必须精确到零。"""
    from resevo.editspace import make_edit_provider
    from resevo.engine import evolve
    from resevo.state import table_loss

    # 只留 A B 两个单字段查询，任何状态都有单行下降路径，无补偿死角
    schema = TableSchema(("A", "B"), (("0", "1"), ("0", "1")))
    specs = [
        QuerySpec("qa", ({"attribute": "A", "operator": "==", "value": "1"},), 1.0),
        QuerySpec("qb", ({"attribute": "B", "operator": "==", "value": "1"},), 1.0),
    ]
    registry = StateRegistry(schema, specs)
    table = [("0", "0"), ("0", "0")]
    ids = registry.register_table(table)
    y = np.array([1.0, 1.0])
    w = np.array([2.0, 2.0])
    provider = make_edit_provider(
        registry, y, w, np.random.default_rng(2026), joint_field_sets=[(0, 1)]
    )
    out = evolve(
        None, ids, 60, np.random.default_rng(7),
        support_provider=provider, max_frozen_retries=10,
    )
    final_workload = registry.build_workload(y, w)
    initial_workload_loss = table_loss(final_workload, ids)
    final_loss = table_loss(final_workload, out.state_ids)
    assert final_loss < initial_workload_loss
    assert final_loss == 0.0  # 无死角玩具问题可精确解出


def test_compensation_deadlock_reported_as_frozen():
    """文档的补偿例子，单行菜单在死角必须明确报告冻结，不许伪装收敛。

    表 11 加 00 在查询 A B 与 A 且 B 下所有单行动作增益全负，
    这正是配对板块存在的理由，第一版单行菜单应停在冻结状态。
    """
    from resevo.editspace import make_edit_provider
    from resevo.engine import evolve
    from resevo.state import table_loss

    schema, specs = toy_schema()
    registry = StateRegistry(schema, specs)
    table = [("1", "1"), ("0", "0")]
    ids = registry.register_table(table)
    y = np.array([1.0, 1.0, 0.0])
    w = np.array([2.0, 2.0, 1.0])
    provider = make_edit_provider(
        registry, y, w, np.random.default_rng(9), joint_field_sets=[(0, 1)]
    )
    out = evolve(
        None, ids, 30, np.random.default_rng(3),
        support_provider=provider, max_frozen_retries=5,
    )
    wl = registry.build_workload(y, w)
    assert out.stop_reason == "no_positive_direction"
    assert table_loss(wl, out.state_ids) == 0.5  # 死角损失原样保留
    np.testing.assert_array_equal(out.state_ids, ids)  # 表一步都没动


def test_edit_provider_descends_on_real_slice():
    """真实数据前 60 行切片，编辑候选多轮演化损失显著下降。"""
    from resevo.dataset import target_from_specs
    from resevo.editspace import make_edit_provider
    from resevo.engine import evolve
    from resevo.state import table_loss

    schema, rows = load_table("data/test_300x10/test_300x10.csv")
    specs = load_queries("data/test_300x10/measured_50query.json")
    registry = StateRegistry(schema, specs)
    # 目标按 60/300 缩放，避免小切片追整表计数
    y = target_from_specs(specs) * (60 / 300)
    w = np.ones(len(specs))
    field_sets = [fs for fs in query_field_sets(specs, schema) if len(fs) >= 2]
    menu_rng = np.random.default_rng(11)
    init_rng = np.random.default_rng(5)
    init_rows = [
        tuple(schema.domains[j][int(init_rng.integers(len(schema.domains[j])))]
              for j in range(schema.num_fields))
        for _ in range(60)
    ]
    ids = registry.register_table(init_rows)
    provider = make_edit_provider(
        registry, y, w, menu_rng, joint_field_sets=field_sets
    )
    out = evolve(
        None, ids, 40, np.random.default_rng(17),
        support_provider=provider, max_frozen_retries=10,
    )
    wl = registry.build_workload(y, w)
    initial = table_loss(wl, ids)
    final = table_loss(wl, out.state_ids)
    assert final < 0.5 * initial  # 四十轮至少砍半，宽松界防随机波动
