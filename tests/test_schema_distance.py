"""
测试 schema 和 distance

验证：
1. schema 加载正确
2. 距离计算的基本性质（自己和自己=0、对称性）
3. age（数值）距离计算
4. 类别属性距离计算
5. 全对全 vs 小池子的接口
6. 与真实数据集成
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import load_schema, Schema, AttributeBlock
from table_diffevo.distance import pairwise_block_distance


def test_load_schema():
    """测试 schema 加载"""
    schema = load_schema("configs/test_300x10/schema.yaml")

    assert schema.n_blocks() == 10
    assert len(schema.attribute_names()) == 10

    # age 是数值块
    age_block = schema.get_block("age")
    assert age_block.is_numeric()
    assert age_block.range == [18, 100]

    # education 是类别块
    edu_block = schema.get_block("education")
    assert edu_block.is_categorical()
    assert set(edu_block.values) == {"high_school", "vocational", "bachelor", "postgraduate"}


def test_schema_queries():
    """测试 schema 查询功能"""
    schema = load_schema("configs/test_300x10/schema.yaml")

    numeric = schema.get_numeric_blocks()
    assert len(numeric) == 1
    assert numeric[0].name == "age"

    categorical = schema.get_categorical_blocks()
    assert len(categorical) == 9


def test_distance_self_is_zero():
    """测试记录和自己的距离为 0"""
    df = pd.DataFrame({
        "age": [25, 30],
        "education": ["bachelor", "high_school"],
        "employment": ["employed", "student"],
        "income": ["high", "low"],
        "marital": ["single", "married"],
        "children": ["0", "1"],
        "housing": ["rent", "owned"],
        "vehicle": ["1", "0"],
        "health": ["good", "fair"],
        "region": ["urban", "rural"]
    })

    schema = load_schema("configs/test_300x10/schema.yaml")
    distances = pairwise_block_distance(df, df, schema)

    # 对角线应该全为 0
    np.testing.assert_array_almost_equal(np.diag(distances), 0.0)


def test_distance_symmetry():
    """测试距离对称性：d(x, z) = d(z, x)"""
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "education": ["bachelor", "high_school", "postgraduate"],
        "employment": ["employed", "student", "retired"],
        "income": ["high", "low", "middle"],
        "marital": ["single", "married", "separated"],
        "children": ["0", "1", "2_plus"],
        "housing": ["rent", "owned", "mortgage"],
        "vehicle": ["1", "0", "2_plus"],
        "health": ["good", "fair", "poor"],
        "region": ["urban", "rural", "suburban"]
    })

    schema = load_schema("configs/test_300x10/schema.yaml")
    distances = pairwise_block_distance(df, df, schema)

    # 矩阵应该对称
    np.testing.assert_array_almost_equal(distances, distances.T)


def test_numeric_block_distance():
    """测试数值块（age）的距离计算"""
    df_current = pd.DataFrame({
        "age": [18, 50],  # 最小值和中间值
        "education": ["bachelor"] * 2,
        "employment": ["employed"] * 2,
        "income": ["high"] * 2,
        "marital": ["single"] * 2,
        "children": ["0"] * 2,
        "housing": ["rent"] * 2,
        "vehicle": ["1"] * 2,
        "health": ["good"] * 2,
        "region": ["urban"] * 2
    })

    df_donors = pd.DataFrame({
        "age": [100],  # 最大值
        "education": ["bachelor"],
        "employment": ["employed"],
        "income": ["high"],
        "marital": ["single"],
        "children": ["0"],
        "housing": ["rent"],
        "vehicle": ["1"],
        "health": ["good"],
        "region": ["urban"]
    })

    schema = load_schema("configs/test_300x10/schema.yaml")
    distances = pairwise_block_distance(df_current, df_donors, schema)

    # age 是唯一不同的块，其余 9 个块都相同
    # 记录0（age=18）vs donor（age=100）：age 差 = (100-18)/(100-18) = 1.0
    # 总距离 = (1.0 + 0*9) / 10 = 0.1
    assert distances[0, 0] == pytest.approx(0.1)

    # 记录1（age=50）vs donor（age=100）：age 差 = (100-50)/(100-18) = 50/82 ≈ 0.61
    # 总距离 = (0.61 + 0*9) / 10 ≈ 0.061
    assert distances[1, 0] == pytest.approx(50 / 82 / 10)


def test_categorical_block_distance():
    """测试类别块的距离计算"""
    df_current = pd.DataFrame({
        "age": [25, 25],
        "education": ["high_school", "bachelor"],  # 第0条不同，第1条相同
        "employment": ["employed"] * 2,
        "income": ["high"] * 2,
        "marital": ["single"] * 2,
        "children": ["0"] * 2,
        "housing": ["rent"] * 2,
        "vehicle": ["1"] * 2,
        "health": ["good"] * 2,
        "region": ["urban"] * 2
    })

    df_donors = pd.DataFrame({
        "age": [25],
        "education": ["bachelor"],
        "employment": ["employed"],
        "income": ["high"],
        "marital": ["single"],
        "children": ["0"],
        "housing": ["rent"],
        "vehicle": ["1"],
        "health": ["good"],
        "region": ["urban"]
    })

    schema = load_schema("configs/test_300x10/schema.yaml")
    distances = pairwise_block_distance(df_current, df_donors, schema)

    # 记录0：education 不同（1个块），总距离 = 1/10 = 0.1
    assert distances[0, 0] == pytest.approx(0.1)

    # 记录1：完全相同，总距离 = 0
    assert distances[1, 0] == pytest.approx(0.0)


def test_distance_range():
    """测试距离落在 [0, 1] 区间"""
    from table_diffevo.queries import load_data

    df = load_data("data/test_300x10/test_300x10.csv")
    schema = load_schema("configs/test_300x10/schema.yaml")

    distances = pairwise_block_distance(df, df, schema)

    assert np.all(distances >= 0.0)
    assert np.all(distances <= 1.0)


def test_pairwise_different_sizes():
    """测试当前表和参考表大小不同（小池子场景）"""
    df_current = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "education": ["bachelor"] * 5,
        "employment": ["employed"] * 5,
        "income": ["high"] * 5,
        "marital": ["single"] * 5,
        "children": ["0"] * 5,
        "housing": ["rent"] * 5,
        "vehicle": ["1"] * 5,
        "health": ["good"] * 5,
        "region": ["urban"] * 5
    })

    df_donors = pd.DataFrame({
        "age": [25, 35],  # 只有 2 条
        "education": ["bachelor"] * 2,
        "employment": ["employed"] * 2,
        "income": ["high"] * 2,
        "marital": ["single"] * 2,
        "children": ["0"] * 2,
        "housing": ["rent"] * 2,
        "vehicle": ["1"] * 2,
        "health": ["good"] * 2,
        "region": ["urban"] * 2
    })

    schema = load_schema("configs/test_300x10/schema.yaml")
    distances = pairwise_block_distance(df_current, df_donors, schema)

    # 应该返回 (5, 2) 矩阵
    assert distances.shape == (5, 2)


def test_integration_with_real_data():
    """集成测试：在真实数据上计算距离"""
    from table_diffevo.queries import load_data

    df = load_data("data/test_300x10/test_300x10.csv")
    schema = load_schema("configs/test_300x10/schema.yaml")

    # 全对全
    distances_full = pairwise_block_distance(df, df, schema)
    assert distances_full.shape == (300, 300)

    # 小池子
    pool = df.sample(50, random_state=42)
    distances_pool = pairwise_block_distance(df, pool, schema)
    assert distances_pool.shape == (300, 50)


def test_input_validation():
    """测试输入校验"""
    schema = load_schema("configs/test_300x10/schema.yaml")

    # 缺少属性
    df_incomplete = pd.DataFrame({
        "age": [25],
        "education": ["bachelor"]
        # 缺少其余属性
    })

    df_complete = pd.DataFrame({
        "age": [25],
        "education": ["bachelor"],
        "employment": ["employed"],
        "income": ["high"],
        "marital": ["single"],
        "children": ["0"],
        "housing": ["rent"],
        "vehicle": ["1"],
        "health": ["good"],
        "region": ["urban"]
    })

    with pytest.raises(ValueError, match="缺少属性"):
        pairwise_block_distance(df_incomplete, df_complete, schema)

    with pytest.raises(ValueError, match="缺少属性"):
        pairwise_block_distance(df_complete, df_incomplete, schema)
