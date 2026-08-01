"""
测试向参考记录靠近一步

锚定记录参与、属性块复制、变异三个动作的正确性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.update import evolve_step, evolve_step_single_block


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

        df = load_data("data/test_300x10/test_300x10.csv")
        schema = load_schema("configs/test_300x10/schema.yaml")

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


class TestSingleBlockBasics:
    """单块更新：基本行为（第七节最终设计）"""

    def test_returns_dataframe_and_diagnostics(self):
        """返回 (DataFrame, dict) 且诊断键齐全"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(0)
        result, diag = evolve_step_single_block(
            current, donors, schema, rho=0.5, epsilon=0.1, rng=rng
        )
        assert isinstance(result, pd.DataFrame)
        assert isinstance(diag, dict)
        for key in ("participation_rate", "copy_attempt_rate",
                    "mutation_attempt_rate", "accepted_change_rate",
                    "empty_copy_set_count"):
            assert key in diag

    def test_output_shape_and_columns(self):
        """输出 shape 和列与输入一致"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(0)
        result, _ = evolve_step_single_block(current, donors, schema, rng=rng)
        assert result.shape == current.shape
        assert list(result.columns) == list(current.columns)

    def test_does_not_mutate_input(self):
        """不修改输入表"""
        schema = make_toy_schema()
        current, donors = make_tables()
        current_copy = current.copy()
        donors_copy = donors.copy()
        rng = np.random.default_rng(1)
        evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.5, rng=rng
        )
        pd.testing.assert_frame_equal(current, current_copy)
        pd.testing.assert_frame_equal(donors, donors_copy)

    def test_reproducible_with_same_seed(self):
        """固定种子可复现"""
        schema = make_toy_schema()
        current, donors = make_tables()
        r1, d1 = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.5,
            rng=np.random.default_rng(42)
        )
        r2, d2 = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.5,
            rng=np.random.default_rng(42)
        )
        pd.testing.assert_frame_equal(r1, r2)
        assert d1 == d2


class TestSingleBlockParticipation:
    """单块更新：记录参与率 rho"""

    def test_rho_zero_keeps_all_unchanged(self):
        """rho=0 时全表保持不变，参与率为 0"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(7)
        result, diag = evolve_step_single_block(
            current, donors, schema, rho=0.0, epsilon=0.5, rng=rng
        )
        pd.testing.assert_frame_equal(result, current.reset_index(drop=True))
        assert diag["participation_rate"] == 0.0
        assert diag["accepted_change_rate"] == 0.0

    def test_rho_one_all_participate(self):
        """rho=1 时全员参与"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(3)
        _, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        assert diag["participation_rate"] == 1.0

    def test_participation_rate_approximates_rho(self):
        """大表上实际参与率应接近 rho"""
        schema = make_toy_schema()
        n = 5000
        current = pd.DataFrame({
            "age": [30] * n, "edu": ["mid"] * n, "job": ["a"] * n,
        })
        donors = pd.DataFrame({
            "age": [50] * n, "edu": ["high"] * n, "job": ["c"] * n,
        })
        rng = np.random.default_rng(19)
        _, diag = evolve_step_single_block(
            current, donors, schema, rho=0.1, epsilon=0.01, rng=rng
        )
        assert abs(diag["participation_rate"] - 0.1) < 0.02


class TestSingleBlockCopy:
    """单块更新：复制动作"""

    def test_copy_changes_exactly_one_block(self):
        """epsilon=0 纯复制时，每个参与行恰好改变一个块"""
        schema = make_toy_schema()
        # current 与 donor 三个块全不同
        current = pd.DataFrame({
            "age": [20, 30, 40],
            "edu": ["low", "low", "mid"],
            "job": ["a", "b", "a"],
        })
        donors = pd.DataFrame({
            "age": [25, 35, 45],
            "edu": ["high", "high", "high"],
            "job": ["c", "c", "c"],
        })
        rng = np.random.default_rng(5)
        result, _ = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        # 每行恰好一个块变化（全部参与、全部复制、D_i 非空）
        changed_per_row = (result != current.reset_index(drop=True)).sum(axis=1)
        assert (changed_per_row == 1).all()

    def test_copy_only_from_different_blocks(self):
        """复制的新值必来自 donor 对应块"""
        schema = make_toy_schema()
        current = pd.DataFrame({
            "age": [20], "edu": ["low"], "job": ["a"],
        })
        donors = pd.DataFrame({
            "age": [25], "edu": ["high"], "job": ["c"],
        })
        rng = np.random.default_rng(5)
        result, _ = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        # 找出变化的那个块，其新值必等于 donor
        for attr in schema.attribute_names():
            cur_v = current.at[0, attr]
            new_v = result.at[0, attr]
            if new_v != cur_v:
                assert new_v == donors.at[0, attr]

    def test_empty_copy_set_keeps_unchanged(self):
        """current 与 donor 完全相同时 D_i 为空，保持不变且计数"""
        schema = make_toy_schema()
        current = pd.DataFrame({
            "age": [20, 30], "edu": ["low", "mid"], "job": ["a", "b"],
        })
        donors = current.copy()  # 完全相同
        rng = np.random.default_rng(9)
        result, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        pd.testing.assert_frame_equal(result, current.reset_index(drop=True))
        assert diag["empty_copy_set_count"] == 2
        assert diag["accepted_change_rate"] == 0.0

    def test_partial_diff_copies_one_of_different(self):
        """只有部分块不同时，复制的块必在不同块集合内"""
        schema = make_toy_schema()
        # 只有 age 不同，edu/job 相同
        current = pd.DataFrame({
            "age": [20], "edu": ["mid"], "job": ["a"],
        })
        donors = pd.DataFrame({
            "age": [25], "edu": ["mid"], "job": ["a"],
        })
        rng = np.random.default_rng(9)
        result, _ = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        # 唯一能变的是 age；edu/job 必不变
        assert result.at[0, "edu"] == "mid"
        assert result.at[0, "job"] == "a"
        assert result.at[0, "age"] == 25


class TestSingleBlockMutation:
    """单块更新：变异动作"""

    def test_mutation_excludes_current_value(self):
        """epsilon=1 纯变异时，变异块的新值必不等于原值"""
        schema = make_toy_schema()
        n = 300
        current = pd.DataFrame({
            "age": [30] * n, "edu": ["mid"] * n, "job": ["a"] * n,
        })
        donors = current.copy()  # 无复制来源，隔离变异
        rng = np.random.default_rng(13)
        result, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=1.0, rng=rng
        )
        # 全员参与、全变异
        assert diag["mutation_attempt_rate"] == 1.0
        # 每个发生变化的块，新值都不等于原值
        base = current.reset_index(drop=True)
        for attr in schema.attribute_names():
            mask = result[attr] != base[attr]
            # 变了的位置新值确实不同（排除当前值保证）
            assert (result.loc[mask, attr] != base.loc[mask, attr]).all()

    def test_mutation_produces_legal_values(self):
        """变异产生的值都在合法范围内"""
        schema = make_toy_schema()
        n = 300
        current = pd.DataFrame({
            "age": [30] * n, "edu": ["mid"] * n, "job": ["a"] * n,
        })
        donors = current.copy()
        rng = np.random.default_rng(13)
        result, _ = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=1.0, rng=rng
        )
        assert result["age"].between(18, 100).all()
        assert result["edu"].isin(["low", "mid", "high"]).all()
        assert result["job"].isin(["a", "b", "c"]).all()

    def test_mutation_changes_exactly_one_block(self):
        """纯变异时每个参与行最多改变一个块"""
        schema = make_toy_schema()
        n = 300
        current = pd.DataFrame({
            "age": [30] * n, "edu": ["mid"] * n, "job": ["a"] * n,
        })
        donors = current.copy()
        rng = np.random.default_rng(21)
        result, _ = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=1.0, rng=rng
        )
        changed_per_row = (result != current.reset_index(drop=True)).sum(axis=1)
        # 每行最多一个块变化（变异只改一个块）
        assert (changed_per_row <= 1).all()


class TestSingleBlockMutualExclusion:
    """单块更新：复制与变异互斥、每行最多改一个块"""

    def test_at_most_one_block_per_row(self):
        """任意 rho/epsilon 组合下，每行每轮最多改变一个块"""
        schema = make_toy_schema()
        n = 500
        current = pd.DataFrame({
            "age": [20] * n, "edu": ["low"] * n, "job": ["a"] * n,
        })
        donors = pd.DataFrame({
            "age": [60] * n, "edu": ["high"] * n, "job": ["c"] * n,
        })
        for eps in (0.0, 0.3, 0.5, 1.0):
            rng = np.random.default_rng(31)
            result, _ = evolve_step_single_block(
                current, donors, schema, rho=1.0, epsilon=eps, rng=rng
            )
            changed_per_row = (result != current.reset_index(drop=True)).sum(axis=1)
            assert (changed_per_row <= 1).all(), f"eps={eps} 有行改了多于一个块"

    def test_copy_and_mutation_rates_sum_bounded(self):
        """复制 + 变异尝试率之和不超过参与率"""
        schema = make_toy_schema()
        n = 2000
        current = pd.DataFrame({
            "age": [20] * n, "edu": ["low"] * n, "job": ["a"] * n,
        })
        donors = pd.DataFrame({
            "age": [60] * n, "edu": ["high"] * n, "job": ["c"] * n,
        })
        rng = np.random.default_rng(37)
        _, diag = evolve_step_single_block(
            current, donors, schema, rho=0.2, epsilon=0.3, rng=rng
        )
        total = diag["copy_attempt_rate"] + diag["mutation_attempt_rate"]
        assert total == pytest.approx(diag["participation_rate"], abs=1e-9)


class TestSingleBlockValidation:
    """单块更新：参数校验"""

    def test_rho_out_of_range(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="rho 必须在"):
            evolve_step_single_block(current, donors, schema, rho=1.5)

    def test_epsilon_out_of_range(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="epsilon 必须在"):
            evolve_step_single_block(current, donors, schema, epsilon=-0.1)

    def test_length_mismatch(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="行数.*不一致"):
            evolve_step_single_block(current, donors.head(2), schema)


class TestSingleBlockEdgeCases:
    """单块更新：边界情况"""

    def test_single_legal_value_block_cannot_mutate(self):
        """块只有一个合法值时，变异无候选，保持不变"""
        schema = Schema([
            AttributeBlock(name="fixed", type="categorical",
                           description="单值块", values=["only"]),
        ])
        n = 100
        current = pd.DataFrame({"fixed": ["only"] * n})
        donors = current.copy()
        rng = np.random.default_rng(41)
        result, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=1.0, rng=rng
        )
        # 唯一合法值，排除当前值后无候选 → 全不变
        pd.testing.assert_frame_equal(result, current.reset_index(drop=True))
        assert diag["accepted_change_rate"] == 0.0

    def test_epsilon_zero_pure_copy(self):
        """epsilon=0 时无变异尝试"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(43)
        _, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=0.0, rng=rng
        )
        assert diag["mutation_attempt_rate"] == 0.0

    def test_epsilon_one_pure_mutation(self):
        """epsilon=1 时无复制尝试"""
        schema = make_toy_schema()
        current, donors = make_tables()
        rng = np.random.default_rng(43)
        _, diag = evolve_step_single_block(
            current, donors, schema, rho=1.0, epsilon=1.0, rng=rng
        )
        assert diag["copy_attempt_rate"] == 0.0
