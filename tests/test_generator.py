"""
测试合成表初始化

锚定 S_0 的结构、合法性、复现性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.generator import init_synthetic_table


def make_toy_schema():
    """1 个数值块 + 2 个类别块"""
    return Schema([
        AttributeBlock(name="age", type="numeric", description="年龄", range=[18, 100]),
        AttributeBlock(name="edu", type="categorical", description="学历",
                       values=["low", "mid", "high"]),
        AttributeBlock(name="job", type="categorical", description="职业",
                       values=["a", "b", "c"]),
    ])


class TestStructure:
    """结构：条数、列名、列顺序"""

    def test_record_count(self):
        """记录条数与请求一致"""
        schema = make_toy_schema()
        s0 = init_synthetic_table(300, schema, np.random.default_rng(0))
        assert len(s0) == 300

    def test_columns_match_schema(self):
        """列名和列顺序与 schema 一致"""
        schema = make_toy_schema()
        s0 = init_synthetic_table(10, schema, np.random.default_rng(0))
        assert list(s0.columns) == schema.attribute_names()

    def test_column_count(self):
        """列数与 schema 块数一致"""
        schema = make_toy_schema()
        s0 = init_synthetic_table(10, schema, np.random.default_rng(0))
        assert s0.shape[1] == schema.n_blocks()


class TestLegality:
    """合法性：所有取值落在 schema 合法域内"""

    def test_numeric_within_range(self):
        """数值列在合法范围内（含端点）"""
        schema = make_toy_schema()
        s0 = init_synthetic_table(1000, schema, np.random.default_rng(1))
        assert s0["age"].between(18, 100).all()

    def test_categorical_in_legal_values(self):
        """类别列取值都在合法集合内"""
        schema = make_toy_schema()
        s0 = init_synthetic_table(1000, schema, np.random.default_rng(2))
        assert s0["edu"].isin(["low", "mid", "high"]).all()
        assert s0["job"].isin(["a", "b", "c"]).all()

    def test_can_reach_range_endpoints(self):
        """足够多样本时，数值端点应可被抽到（含端点验证）"""
        schema = Schema([
            AttributeBlock(name="x", type="numeric", description="小范围", range=[0, 2]),
        ])
        s0 = init_synthetic_table(2000, schema, np.random.default_rng(3))
        vals = set(s0["x"].unique())
        # 范围 [0,2] 含端点，应能抽到 0、1、2
        assert vals == {0, 1, 2}


class TestReproducibility:
    """复现性"""

    def test_same_seed_same_table(self):
        """相同种子 → 相同表"""
        schema = make_toy_schema()
        s1 = init_synthetic_table(100, schema, np.random.default_rng(42))
        s2 = init_synthetic_table(100, schema, np.random.default_rng(42))
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seed_different_table(self):
        """不同种子 → 不同表"""
        schema = make_toy_schema()
        s1 = init_synthetic_table(100, schema, np.random.default_rng(1))
        s2 = init_synthetic_table(100, schema, np.random.default_rng(2))
        assert not s1.equals(s2)


class TestValidation:
    """参数校验"""

    def test_zero_records(self):
        schema = make_toy_schema()
        with pytest.raises(ValueError, match="n_records 必须 > 0"):
            init_synthetic_table(0, schema)

    def test_negative_records(self):
        schema = make_toy_schema()
        with pytest.raises(ValueError, match="n_records 必须 > 0"):
            init_synthetic_table(-5, schema)


class TestIntegration:
    """与真实 schema 集成"""

    def test_with_real_schema(self):
        """真实 schema：条数、列、合法性"""
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_data

        schema = load_schema("configs/schema.yaml")
        df = load_data("data/test_300x10.csv")

        # 与源数据条数、列一致
        s0 = init_synthetic_table(len(df), schema, np.random.default_rng(2024))
        assert len(s0) == len(df)
        assert list(s0.columns) == list(df.columns)

        # age 合法范围
        assert s0["age"].between(18, 100).all()

    def test_can_be_evaluated_by_queries(self):
        """S_0 可被查询评价器处理（下游可用性）"""
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_queries, evaluate_table

        schema = load_schema("configs/schema.yaml")
        queries = load_queries("configs/measured_50query.json")

        s0 = init_synthetic_table(300, schema, np.random.default_rng(7))
        counts = evaluate_table(s0, queries)
        # 能算出计数向量，长度与查询数一致
        assert len(counts) == len(queries)
        assert (counts >= 0).all()
