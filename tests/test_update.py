"""
测试向参考记录靠近一步

锚定记录参与、属性块复制、变异三个动作的正确性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.directional_diffusion import tilted_copy_probabilities
from table_diffevo.update import (
    _sample_legal_value,
    _sample_mutation_block,
    apply_update_random_plan,
    evolve_step,
    sample_update_random_plan,
)


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


def legacy_valid_evolve_step(
    current,
    donors,
    schema,
    *,
    rho,
    eta,
    mu,
    rng,
    direction_scores=None,
    direction_strength=0.0,
):
    """重放拆分前的有效输入路径，锁定历史随机数消费顺序。"""

    n_records = len(current)
    attr_names = schema.attribute_names()
    next_table = current.reset_index(drop=True).copy()
    donors = donors.reset_index(drop=True)
    participate = rng.random(n_records) < rho
    for attr_idx, attr in enumerate(attr_names):
        current_values = current[attr].reset_index(drop=True).to_numpy()
        donor_values = donors[attr].to_numpy()
        differ = current_values != donor_values
        if direction_scores is None or direction_strength == 0.0:
            copy_roll = rng.random(n_records) < eta
        else:
            copy_probability = tilted_copy_probabilities(
                eta,
                direction_scores[:, attr_idx],
                direction_strength,
            )
            copy_roll = rng.random(n_records) < copy_probability
        copy_mask = participate & differ & copy_roll
        if copy_mask.any():
            new_values = next_table[attr].to_numpy().copy()
            new_values[copy_mask] = donor_values[copy_mask]
            next_table[attr] = new_values

    mutate_mask = participate & (rng.random(n_records) < mu)
    for row_index in np.nonzero(mutate_mask)[0]:
        attribute = _sample_mutation_block(schema, rng)
        value = _sample_legal_value(schema.get_block(attribute), rng)
        next_table.at[row_index, attribute] = value
    return next_table


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

    @pytest.mark.parametrize("value", [0.0, np.inf, True, "30"])
    def test_direction_logit_clip_is_validated(self, value):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="logit_clip"):
            evolve_step(
                current,
                donors,
                schema,
                direction_logit_clip=value,
            )

    def test_return_diagnostics_requires_boolean(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        with pytest.raises(ValueError, match="return_diagnostics"):
            evolve_step(
                current,
                donors,
                schema,
                return_diagnostics="yes",
            )


class TestTransitionDiagnostics:
    def test_observation_preserves_table_and_rng_stream(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        plain_rng = np.random.default_rng(20260814)
        observed_rng = np.random.default_rng(20260814)

        plain = evolve_step(
            current,
            donors,
            schema,
            rho=1.0,
            eta=0.4,
            mu=0.0,
            rng=plain_rng,
        )
        observed, diagnostics = evolve_step(
            current,
            donors,
            schema,
            rho=1.0,
            eta=0.4,
            mu=0.0,
            rng=observed_rng,
            return_diagnostics=True,
        )

        pd.testing.assert_frame_equal(observed, plain)
        np.testing.assert_array_equal(
            observed_rng.random(20), plain_rng.random(20)
        )
        assert diagnostics == {
            "participating_rows": len(current),
            "mutated_rows": 0,
        }


class TestUpdateRandomPlan:
    @pytest.mark.parametrize(
        (
            "seed",
            "rho",
            "eta",
            "mu",
            "direction_strength",
        ),
        [
            (2026082601, 0.6, 0.4, 0.7, 0.0),
            (2026082602, 0.0, 0.5, 1.0, 0.0),
            (2026082603, 1.0, 0.0, 1.0, 0.0),
            (2026082604, 0.8, 0.5, 0.6, 1.7),
        ],
    )
    def test_evolve_step_matches_pre_split_table_and_rng_endpoint(
        self,
        seed,
        rho,
        eta,
        mu,
        direction_strength,
    ):
        schema = make_toy_schema()
        current, donors = make_tables()
        direction_scores = np.asarray([
            [-1.5, 0.0, 1.2],
            [0.2, -0.9, 0.0],
            [1.0, 0.3, -0.4],
            [-0.1, 1.6, -1.1],
            [0.7, -0.2, 0.5],
        ])
        use_direction = direction_strength != 0.0
        legacy_rng = np.random.default_rng(seed)
        split_rng = np.random.default_rng(seed)

        expected = legacy_valid_evolve_step(
            current,
            donors,
            schema,
            rho=rho,
            eta=eta,
            mu=mu,
            rng=legacy_rng,
            direction_scores=(direction_scores if use_direction else None),
            direction_strength=direction_strength,
        )
        observed = evolve_step(
            current,
            donors,
            schema,
            rho=rho,
            eta=eta,
            mu=mu,
            rng=split_rng,
            copy_direction_scores=(
                direction_scores if use_direction else None
            ),
            copy_direction_strength=direction_strength,
        )

        pd.testing.assert_frame_equal(observed, expected)
        assert split_rng.bit_generator.state == legacy_rng.bit_generator.state

    def test_sample_then_apply_is_the_same_public_transition(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        plan_rng = np.random.default_rng(2026082605)
        evolve_rng = np.random.default_rng(2026082605)

        plan = sample_update_random_plan(
            current,
            donors,
            schema,
            rho=0.8,
            eta=0.35,
            mu=0.6,
            rng=plan_rng,
        )
        materialized = apply_update_random_plan(
            current, donors, schema, plan
        )
        expected = evolve_step(
            current,
            donors,
            schema,
            rho=0.8,
            eta=0.35,
            mu=0.6,
            rng=evolve_rng,
        )

        pd.testing.assert_frame_equal(materialized, expected)
        assert plan_rng.bit_generator.state == evolve_rng.bit_generator.state
        assert plan.participate.dtype == np.bool_
        assert plan.initial_copy_mask.shape == (len(current), schema.n_blocks())
        assert plan.initial_copy_mask.dtype == np.bool_

    def test_final_copy_override_reuses_the_same_mutations(self):
        schema = make_toy_schema()
        current, donors = make_tables()
        plan = sample_update_random_plan(
            current,
            donors,
            schema,
            rho=1.0,
            eta=1.0,
            mu=1.0,
            rng=np.random.default_rng(2026082606),
        )
        no_copy = np.zeros_like(plan.initial_copy_mask)

        observed = apply_update_random_plan(
            current,
            donors,
            schema,
            plan,
            final_copy_mask=no_copy,
        )
        expected = current.reset_index(drop=True).copy()
        for event in plan.mutation_events:
            expected.at[event.row_index, event.attribute] = event.value

        pd.testing.assert_frame_equal(observed, expected)
        assert plan.initial_copy_mask.any()
        assert len(plan.mutation_events) == len(current)


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
