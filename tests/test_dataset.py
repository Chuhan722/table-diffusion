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
