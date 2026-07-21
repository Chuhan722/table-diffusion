"""
测试向参考记录靠近一步

锚定记录参与、属性块复制、变异三个动作的正确性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.update import evolve_step


def make_toy_schema():
    """构造一个小 schema：1 个数值块 + 2 个类别块"""
    return Schema([
        AttributeBlock(name="age", type="numeric", description="年龄", range=[18, 100]),
        AttributeBlock(name="edu", type="categorical", description="学历",
                       values=["low", "mid", "high"]),
        AttributeBlock(name="job", type="categorical", description="职业",
                       values=["a", "b", "c"]),
    ])


def make_tables(n=5):
    """构造对齐的当前表和参考表"""
    current = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "edu": ["low", "low", "mid", "mid", "high"],
        "job": ["a", "b", "a", "b", "c"],
    })
    donors = pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "edu": ["high", "high", "high", "high", "high"],
        "job": ["c", "c", "c", "c", "c"],
    })
    return current.head(n), donors.head(n)


class TestEvolveStepBasics:
    """基本行为"""

    def test_output_shape_and_columns(self):
        """输出 shape 和列与输入一致"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(0)
        result = evolve_step(current, donors, schema, rng=rng)
        assert result.shape == current.shape
        assert list(result.columns) == list(current.columns)

    def test_does_not_mutate_input(self):
        """不修改输入表"""
        schema = make_toy_schema()
        current, donors = make_tables()
        current_copy = current.copy()
        donors_copy = donors.copy()
        rng = np.random.default_rng(1)
        evolve_step(current, donors, schema, rng=rng)
        pd.testing.assert_frame_equal(current, current_copy)
        pd.testing.assert_frame_equal(donors, donors_copy)

    def test_reproducible_with_same_seed(self):
        """固定种子可复现"""
        schema = make_toy_schema()
        current, donors = make_tables()
        r1 = evolve_step(current, donors, schema, rng=np.random.default_rng(42))
        r2 = evolve_step(current, donors, schema, rng=np.random.default_rng(42))
        pd.testing.assert_frame_equal(r1, r2)


class TestParticipation:
    """记录参与概率 rho"""

    def test_rho_zero_keeps_all_unchanged(self):
        """rho=0 时全表保持不变"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(7)
        result = evolve_step(current, donors, schema, rho=0.0, rng=rng)
        pd.testing.assert_frame_equal(
            result, current.reset_index(drop=True)
        )

    def test_rho_one_eta_one_copies_all_diff_blocks(self):
        """rho=1, eta=1, mu=0 时，所有不同的块都被复制"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(3)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=1.0, mu=0.0, rng=rng)
        # eta=1 全复制 → 结果应等于 donors
        pd.testing.assert_frame_equal(
            result, donors.reset_index(drop=True)
        )


class TestBlockCopy:
    """属性块复制概率 eta"""

    def test_eta_zero_no_copy(self):
        """eta=0, mu=0 时不复制任何块（即使参与）"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(5)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=0.0, mu=0.0, rng=rng)
        # 没有复制、没有变异 → 保持原样
        pd.testing.assert_frame_equal(
            result, current.reset_index(drop=True)
        )

    def test_same_block_not_changed(self):
        """当前记录与参考记录相同的块保持不变"""
        schema = make_toy_schema()
        # 构造 edu 块完全相同的情况
        current = pd.DataFrame({
            "age": [20, 30],
            "edu": ["mid", "mid"],
            "job": ["a", "b"],
        })
        donors = pd.DataFrame({
            "age": [25, 35],
            "edu": ["mid", "mid"],  # 与 current 相同
            "job": ["c", "c"],
        })
        rng = np.random.default_rng(9)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=1.0, mu=0.0, rng=rng)
        # edu 块相同，无论 eta 多大都不变
        assert list(result["edu"]) == ["mid", "mid"]


class TestMutation:
    """变异概率 mu"""

    def test_mu_zero_no_mutation(self):
        """mu=0 时不发生变异（结果只可能来自复制）"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(11)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=1.0, mu=0.0, rng=rng)
        # mu=0 + eta=1 → 结果等于 donors，不会出现 donors 之外的值
        pd.testing.assert_frame_equal(
            result, donors.reset_index(drop=True)
        )

    def test_mutation_produces_legal_values(self):
        """变异产生的值都在合法范围内"""
        schema = make_toy_schema()
        # 大表 + 高变异率，逼出变异
        n = 200
        current = pd.DataFrame({
            "age": [30] * n,
            "edu": ["mid"] * n,
            "job": ["a"] * n,
        })
        donors = current.copy()  # donor 与 current 完全相同 → 无复制，只可能变异
        rng = np.random.default_rng(13)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=1.0, mu=1.0, rng=rng)
        # 所有值必须合法
        assert result["age"].between(18, 100).all()
        assert result["edu"].isin(["low", "mid", "high"]).all()
        assert result["job"].isin(["a", "b", "c"]).all()

    def test_mutation_happens_with_high_mu(self):
        """donor=current 时，高 mu 下应观察到变异（值偏离原值）"""
        schema = make_toy_schema()
        n = 200
        current = pd.DataFrame({
            "age": [30] * n,
            "edu": ["mid"] * n,
            "job": ["a"] * n,
        })
        donors = current.copy()  # 无复制来源
        rng = np.random.default_rng(17)
        result = evolve_step(current, donors, schema,
                             rho=1.0, eta=1.0, mu=1.0, rng=rng)
        # 至少有一些记录的某个块发生了变化（变异）
        changed = (result != current.reset_index(drop=True)).any(axis=1)
        assert changed.sum() > 0


class TestValidation:
    """参数校验"""

    def test_rho_out_of_range(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="rho 必须在"):
            evolve_step(current, donors, schema, rho=1.5)

    def test_eta_out_of_range(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="eta 必须在"):
            evolve_step(current, donors, schema, eta=-0.1)

    def test_mu_out_of_range(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="mu 必须在"):
            evolve_step(current, donors, schema, mu=2.0)

    def test_length_mismatch(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="行数.*不一致"):
            evolve_step(current, donors.head(2), schema)


class TestIntegration:
    """与上游模块的集成"""

    def test_with_real_schema_and_sampling(self):
        """真实 schema + 抽样索引 → 靠近一步"""
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_data
        from table_diffevo.distance import pairwise_block_distance
        from table_diffevo.sampling import compute_sampling_probs, sample_donors

        df = load_data("data/test_300x10.csv")
        schema = load_schema("configs/schema.yaml")

        # 构造随机适应度和距离，走完整抽样流程
        rng = np.random.default_rng(2024)
        fitness = rng.random(len(df))
        distances = pairwise_block_distance(df, df, schema)
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8)
        donor_idx = sample_donors(probs, rng)
        donors = df.iloc[donor_idx].reset_index(drop=True)

        result = evolve_step(df, donors, schema, rng=rng)
        # 形状不变，列不变
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
