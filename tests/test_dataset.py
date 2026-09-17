"""数据加载模块测试，锚点是真实表上 50 查询结果与 JSON 全对。"""
from __future__ import annotations

import numpy as np
import pytest

from resevo.dataset import (
    StateRegistry,
    evaluate_condition,
    load_queries,
    load_table,
    query_field_sets,
    target_from_specs,
)
from resevo.state import table_answers

DATA_DIR = "data/test_300x10"
CSV_PATH = f"{DATA_DIR}/test_300x10.csv"
QUERY_PATH = f"{DATA_DIR}/measured_50query.json"


@pytest.fixture(scope="module")
def loaded():
    schema, rows = load_table(CSV_PATH)
    specs = load_queries(QUERY_PATH)
    return schema, rows, specs


def test_table_shape_and_domains(loaded):
    schema, rows, _ = loaded
    assert len(rows) == 300
    assert schema.num_fields == 10
    assert schema.fields[0] == "age"  # BOM 必须被剥掉
    assert len(schema.domains[schema.field_index("age")]) == 64
    assert schema.domains[schema.field_index("income")] == ("high", "low", "middle")


def test_fifty_queries_loaded(loaded):
    _, _, specs = loaded
    assert len(specs) == 50
    y = target_from_specs(specs)
    assert y.shape == (50,)
    assert y.min() >= 0


def test_condition_operators():
    assert evaluate_condition({"operator": "==", "value": "rent"}, "rent")
    assert not evaluate_condition({"operator": "==", "value": "rent"}, "owned")
    cond = {"operator": "between", "lower": 18, "upper": 24}
    assert evaluate_condition(cond, "24")
    assert not evaluate_condition(cond, "25")
    assert evaluate_condition({"operator": ">=", "value": 65}, "65")
    assert not evaluate_condition({"operator": ">=", "value": 65}, "64")
    with pytest.raises(ValueError):
        evaluate_condition({"operator": "!="}, "x")


def test_anchor_real_table_matches_all_results(loaded):
    """强锚点，用我们的求值器在真实表上重算 50 个查询，必须 50 对 50 全对。"""
    schema, rows, specs = loaded
    registry = StateRegistry(schema, specs)
    state_ids = registry.register_table(rows)
    workload = registry.build_workload(target_from_specs(specs), np.ones(len(specs)))
    q = table_answers(workload, state_ids)
    np.testing.assert_array_equal(q, target_from_specs(specs))


def test_registry_dedup_and_lazy_growth(loaded):
    schema, rows, specs = loaded
    registry = StateRegistry(schema, specs)
    a = registry.register(rows[0])
    b = registry.register(rows[0])
    assert a == b == 0
    assert registry.num_states == 1
    c = registry.register(rows[1])
    assert c == 1
    assert registry.state_tuple(0) == rows[0]
    # 表里不存在的新组合状态也能注册并求值
    novel = list(rows[0])
    novel[schema.field_index("income")] = "high"
    novel[schema.field_index("region")] = "rural"
    d = registry.register(tuple(novel))
    assert d == registry.num_states - 1
    with pytest.raises(ValueError):
        registry.register(("bad",))


def test_registry_feature_prefix_stable(loaded):
    """扩表后旧状态贡献行前缀逐位不变，跨轮损失才可比。"""
    schema, rows, specs = loaded
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    registry = StateRegistry(schema, specs)
    ids = registry.register_table(rows[:20])
    before = registry.build_workload(y, w).features.copy()
    registry.register_table(rows[20:40])
    after = registry.build_workload(y, w).features
    np.testing.assert_array_equal(after[: before.shape[0]], before)
    assert after.shape[0] >= before.shape[0]
    assert ids.dtype == np.int64


def test_query_field_sets(loaded):
    schema, _, specs = loaded
    sets = query_field_sets(specs, schema)
    assert len(sets) == 50
    sizes = {len(s) for s in sets}
    assert sizes == {1, 2, 3}  # 单条件 双条件 三条件都有


def test_registry_growth_matches_manual_recompute(loaded):
    """预分配扩容跨界后，视图矩阵与逐状态手工求值逐位一致。"""
    from resevo.dataset import evaluate_query

    schema, rows, specs = loaded
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    registry = StateRegistry(schema, specs)
    registry._INITIAL_CAPACITY = 3  # 实例覆盖类属性，强制多次扩容
    ids = registry.register_table(rows[:11])
    feats = registry.build_workload(y, w).features
    assert feats.shape[0] == registry.num_states
    for sid in range(registry.num_states):
        manual = np.array(
            [evaluate_query(spec, registry.state_tuple(sid), schema) for spec in specs]
        )
        np.testing.assert_array_equal(feats[sid], manual)
    assert ids.max() == registry.num_states - 1


def test_workload_view_is_readonly_and_stable_after_growth(loaded):
    """负载视图只读不可写，且扩容后旧负载持有的数值逐位不变。"""
    schema, rows, specs = loaded
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    registry = StateRegistry(schema, specs)
    registry._INITIAL_CAPACITY = 4
    registry.register_table(rows[:4])
    old = registry.build_workload(y, w)
    snapshot = old.features.copy()
    with pytest.raises(ValueError):
        old.features[0, 0] = 99.0
    registry.register_table(rows[4:40])  # 触发多次翻倍扩容
    np.testing.assert_array_equal(old.features, snapshot)
    new = registry.build_workload(y, w)
    np.testing.assert_array_equal(new.features[:4], snapshot)


def test_register_edit_matches_full_register_bitwise(loaded):
    """增量注册与全量注册的贡献行逐位一致，随机编辑两百例交叉对拍。

    两个独立注册表灌同一批编辑，一个走 register_edit 增量路径，
    一个走 register 全量求值，特征矩阵与编号序列必须完全相同。
    """
    schema, rows, specs = loaded
    rng = np.random.default_rng(20260916)
    reg_inc = StateRegistry(schema, specs)
    reg_full = StateRegistry(schema, specs)
    base_inc = reg_inc.register_table(rows[:40])
    base_full = reg_full.register_table(rows[:40])
    np.testing.assert_array_equal(base_inc, base_full)
    for _ in range(200):
        b = int(rng.integers(reg_inc.num_states))
        source = reg_inc.state_tuple(b)
        k = int(rng.integers(1, 4))
        fields = rng.permutation(schema.num_fields)[:k]
        edited = list(source)
        for j in fields:
            j = int(j)
            alternatives = [v for v in schema.domains[j] if v != edited[j]]
            if alternatives:
                edited[j] = alternatives[int(rng.integers(len(alternatives)))]
        got_inc = reg_inc.register_edit(b, tuple(edited))
        got_full = reg_full.register(tuple(edited))
        assert got_inc == got_full
    assert reg_inc.num_states == reg_full.num_states
    view_inc = reg_inc.build_workload(
        target_from_specs(specs), np.ones(len(specs))
    ).features
    view_full = reg_full.build_workload(
        target_from_specs(specs), np.ones(len(specs))
    ).features
    np.testing.assert_array_equal(view_inc, view_full)


def test_register_edit_rejects_bad_base(loaded):
    schema, rows, specs = loaded
    registry = StateRegistry(schema, specs)
    registry.register_table(rows[:3])
    with pytest.raises(ValueError):
        registry.register_edit(99, rows[0])


def test_halfspace_query_semantics():
    """半空间查询全路径同值，直接求值，编译求值，注册与增量注册。"""
    import numpy as np

    from resevo.dataset import (
        QuerySpec,
        StateRegistry,
        TableSchema,
        compile_conditions,
        evaluate_compiled,
        evaluate_query,
        query_field_sets,
    )

    schema = TableSchema(
        ("x", "y", "z"),
        (("0", "1"), ("a", "b", "c"), ("0", "1")),
    )
    hs = {"operator": "halfspace",
          "scores": {"x": {"0": -2, "1": 3}, "y": {"a": 1, "c": -4}},
          "threshold": 2}
    specs = [
        QuerySpec("h0", (hs,), 0.0),
        QuerySpec("e0", ({"attribute": "z", "operator": "==", "value": "1"},), 0.0),
    ]
    assert query_field_sets(specs, schema)[0] == (0, 1)
    compiled = compile_conditions(specs, schema)
    rng = np.random.default_rng(0)
    registry = StateRegistry(schema, specs)
    for _ in range(60):
        row = tuple(dom[int(rng.integers(len(dom)))] for dom in schema.domains)
        total = {"0": -2, "1": 3}[row[0]] + {"a": 1, "b": 0, "c": -4}[row[1]]
        want = 1.0 if total >= 2 else 0.0
        assert evaluate_query(specs[0], row, schema) == want
        assert evaluate_compiled(compiled[0], row) == want
        base_id = registry.register(row)
        # 增量注册路径改半空间涉及字段，与全量注册特征逐位一致
        edited = list(row)
        j = int(rng.integers(3))
        dom = schema.domains[j]
        edited[j] = dom[(dom.index(row[j]) + 1) % len(dom)]
        eid = registry.register_edit(base_id, tuple(edited))
        full = np.array(
            [evaluate_query(s, tuple(edited), schema) for s in specs]
        )
        np.testing.assert_array_equal(
            registry.build_workload(np.zeros(2), np.ones(2)).features[eid], full
        )


def test_halfspace_must_be_single_condition():
    """半空间混搭其他条件在编译期拒绝。"""
    import pytest as _pytest

    from resevo.dataset import QuerySpec, TableSchema, compile_conditions

    schema = TableSchema(("x",), (("0", "1"),))
    bad = QuerySpec("b0", (
        {"operator": "halfspace", "scores": {"x": {"0": 1}}, "threshold": 1},
        {"attribute": "x", "operator": "==", "value": "1"},
    ), 0.0)
    with _pytest.raises(ValueError):
        compile_conditions([bad], schema)
