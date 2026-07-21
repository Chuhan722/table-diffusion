"""
测试查询评价器

验证：
1. 查询评价器能正确解析查询定义
2. 单个条件评价正确（== 和 between 算子）
3. 多条件合取（AND）正确
4. 在原始数据上的 50 个查询结果与预期完全一致
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.queries import (
    load_queries,
    load_data,
    eval_condition,
    eval_query,
    eval_query_mask,
    evaluate_table,
    verify_evaluator
)


def test_load_queries():
    """测试查询文件加载"""
    queries = load_queries("configs/test_300x10/measured_50query.json")

    # 应该有 50 个查询
    assert len(queries) == 50

    # 每个查询必须有的字段
    for q in queries:
        assert "id" in q
        assert "conditions" in q
        assert "result" in q
        assert isinstance(q["conditions"], list)
        assert len(q["conditions"]) >= 1


def test_load_data():
    """测试数据加载"""
    df = load_data("data/test_300x10/test_300x10.csv")

    # 应该有 300 行 10 列
    assert df.shape == (300, 10)

    # 必须包含这些列
    expected_cols = [
        "age", "education", "employment", "income", "marital",
        "children", "housing", "vehicle", "health", "region"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_eval_condition_equality():
    """测试等值条件（== 算子）"""
    # 构造测试数据
    df = pd.DataFrame({
        "education": ["bachelor", "high_school", "bachelor", "vocational"]
    })

    condition = {
        "attribute": "education",
        "operator": "==",
        "value": "bachelor"
    }

    mask = eval_condition(df, condition)

    # 应该有 2 条记录满足
    assert mask.sum() == 2
    assert mask.tolist() == [True, False, True, False]


def test_eval_condition_between():
    """测试区间条件（between 算子）"""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60]
    })

    condition = {
        "attribute": "age",
        "operator": "between",
        "lower": 25,
        "upper": 50
    }

    mask = eval_condition(df, condition)

    # 30, 40, 50 在区间内
    assert mask.sum() == 3
    assert mask.tolist() == [False, True, True, True, False]


def test_eval_query_single_condition():
    """测试单条件查询"""
    df = pd.DataFrame({
        "education": ["bachelor", "high_school", "bachelor", "vocational"],
        "age": [25, 30, 35, 40]
    })

    query = {
        "id": "test",
        "conditions": [
            {
                "attribute": "education",
                "operator": "==",
                "value": "bachelor"
            }
        ]
    }

    count = eval_query(df, query)
    assert count == 2


def test_eval_query_multiple_conditions():
    """测试多条件查询（AND 逻辑）"""
    df = pd.DataFrame({
        "education": ["bachelor", "bachelor", "high_school", "bachelor"],
        "age": [25, 35, 30, 45]
    })

    query = {
        "id": "test",
        "conditions": [
            {
                "attribute": "age",
                "operator": "between",
                "lower": 30,
                "upper": 40
            },
            {
                "attribute": "education",
                "operator": "==",
                "value": "bachelor"
            }
        ]
    }

    # 只有第二条记录同时满足：age=35 在 [30,40] 且 education=bachelor
    count = eval_query(df, query)
    assert count == 1


def test_evaluate_table():
    """测试整表评价（多个查询）"""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "education": ["bachelor"] * 4
    })

    queries = [
        {
            "id": "q1",
            "conditions": [
                {"attribute": "age", "operator": "between", "lower": 25, "upper": 45}
            ]
        },
        {
            "id": "q2",
            "conditions": [
                {"attribute": "education", "operator": "==", "value": "bachelor"}
            ]
        }
    ]

    counts = evaluate_table(df, queries)

    assert isinstance(counts, np.ndarray)
    assert len(counts) == 2
    assert counts[0] == 2  # q1: age in [25,45] → 30, 40
    assert counts[1] == 4  # q2: education=bachelor → 全部


def test_verify_evaluator_on_real_data():
    """
    【关键测试】在原始数据上验证评价器正确性

    将评价器计算的 50 个查询结果与预期的 result 字段逐一对比。
    如果全部匹配，说明查询评价器逻辑正确。
    """
    result = verify_evaluator(
        data_path="data/test_300x10/test_300x10.csv",
        query_path="configs/test_300x10/measured_50query.json",
        verbose=True  # 打印详细对比信息
    )

    assert result is True, "查询评价器在原始数据上的结果与预期不符，需要检查逻辑"


def test_value_type_handling():
    """
    测试 value 类型处理（数字 vs 字符串）

    确保 JSON 里的数字 value 被正确转成字符串，以匹配数据。
    """
    df = pd.DataFrame({
        "children": ["0", "1", "2_plus", "0"]
    })

    # 模拟 JSON 解析后 value 可能是数字的情况
    queries_raw = [
        {
            "id": "test",
            "conditions": [
                {"attribute": "children", "operator": "==", "value": 0}  # 数字
            ]
        }
    ]

    # load_queries 会把 value 转成字符串
    for q in queries_raw:
        for cond in q["conditions"]:
            if cond["operator"] == "==" and "value" in cond:
                cond["value"] = str(cond["value"])

    count = eval_query(df, queries_raw[0])
    assert count == 2  # 两个 "0"


def test_eval_query_mask_basic():
    """测试 eval_query_mask 返回布尔掩码"""
    df = pd.DataFrame({
        "education": ["bachelor", "high_school", "bachelor", "vocational"]
    })

    query = {
        "id": "test",
        "conditions": [
            {
                "attribute": "education",
                "operator": "==",
                "value": "bachelor"
            }
        ]
    }

    mask = eval_query_mask(df, query)

    # 应该返回布尔数组
    assert mask.dtype == bool
    # 形状应该是 (N,)
    assert mask.shape == (4,)
    # 应该有 2 个 True（两个 bachelor）
    assert mask.sum() == 2
    assert mask.tolist() == [True, False, True, False]


def test_eval_query_mask_multiple_conditions():
    """测试 eval_query_mask 处理多条件（AND）"""
    df = pd.DataFrame({
        "education": ["bachelor", "bachelor", "high_school", "bachelor"],
        "age": [25, 35, 30, 45]
    })

    query = {
        "id": "test",
        "conditions": [
            {
                "attribute": "age",
                "operator": "between",
                "lower": 30,
                "upper": 40
            },
            {
                "attribute": "education",
                "operator": "==",
                "value": "bachelor"
            }
        ]
    }

    mask = eval_query_mask(df, query)

    # 只有第二条记录（age=35, edu=bachelor）同时满足
    assert mask.sum() == 1
    assert mask.tolist() == [False, True, False, False]


def test_eval_query_uses_eval_query_mask():
    """测试 eval_query 内部调用 eval_query_mask（重构验证）"""
    df = pd.DataFrame({
        "A": ["0", "1", "0", "1"]
    })

    query = {
        "id": "test",
        "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]
    }

    # 从掩码算计数
    mask = eval_query_mask(df, query)
    count_from_mask = int(mask.sum())

    # 直接调用 eval_query
    count_direct = eval_query(df, query)

    # 两者应该一致
    assert count_from_mask == count_direct == 2
