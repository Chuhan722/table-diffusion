"""
测试扩散演化主循环

锚定主循环的结构、终止条件、复现性、方向正确性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.evolution import run_evolution


def make_toy_schema():
    """1 个数值块 + 2 个类别块"""
    return Schema([
        AttributeBlock(name="age", type="numeric", description="年龄", range=[18, 100]),
        AttributeBlock(name="edu", type="categorical", description="学历",
                       values=["low", "mid", "high"]),
        AttributeBlock(name="job", type="categorical", description="职业",
                       values=["a", "b", "c"]),
    ])


def make_toy_queries():
    """几个简单查询"""
    return [
        {"conditions": [{"attribute": "edu", "operator": "==", "value": "high"}]},
        {"conditions": [{"attribute": "job", "operator": "==", "value": "a"}]},
        {"conditions": [{"attribute": "age", "operator": ">=", "value": 50}]},
    ]


class TestBasics:
    """基本结构与返回"""

    def test_output_shape(self):
        """best_S 形状正确"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        best_S, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=0
        )
        assert best_S.shape == (100, 3)
        assert list(best_S.columns) == schema.attribute_names()

    def test_diagnostics_keys(self):
        """诊断信息包含约定字段"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=0
        )
        assert "loss_history" in diag
        assert "best_loss" in diag
        assert "rounds_run" in diag
        assert "stopped_early" in diag
        assert "accept_history" in diag

    def test_target_length_mismatch(self):
        """target 长度与查询数不一致报错"""
        schema = make_toy_schema()
        queries = make_toy_queries()  # 3 个查询
        target = np.array([30, 40])  # 只有 2
        with pytest.raises(ValueError, match="target 长度.*与查询数.*不一致"):
            run_evolution(target, queries, schema, n_records=100)


class TestTermination:
    """终止条件"""

    def test_runs_full_rounds_when_not_converged(self):
        """未达标时跑满 n_rounds"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=15, seed=0
        )
        if not diag["stopped_early"]:
            assert diag["rounds_run"] == 15

    def test_max_rounds_respected(self):
        """rounds_run 不超过 n_rounds"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=10, seed=0
        )
        assert diag["rounds_run"] <= 10


class TestReproducibility:
    """复现性"""

    def test_same_seed_same_result(self):
        """相同种子 → 相同结果"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        s1, d1 = run_evolution(target, queries, schema, n_records=100,
                               n_rounds=20, seed=42)
        s2, d2 = run_evolution(target, queries, schema, n_records=100,
                               n_rounds=20, seed=42)
        pd.testing.assert_frame_equal(s1, s2)
        assert d1["loss_history"] == d2["loss_history"]

    def test_different_seed_different_result(self):
        """不同种子 → 结果不同（loss 轨迹不同）"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, d1 = run_evolution(target, queries, schema, n_records=100,
                              n_rounds=20, seed=1)
        _, d2 = run_evolution(target, queries, schema, n_records=100,
                              n_rounds=20, seed=2)
        assert d1["loss_history"] != d2["loss_history"]


class TestCorrectness:
    """方向正确性：核心验证"""

    def test_best_loss_not_worse_than_initial(self):
        """best_loss 不会比初始轮更差"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=50, seed=0
        )
        assert diag["best_loss"] <= diag["loss_history"][0]

    def test_loss_decreases_over_time(self):
        """演化应降低 loss：最终 best_loss 明显小于初始 loss"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=100, seed=0
        )
        initial_loss = diag["loss_history"][0]
        assert diag["best_loss"] < initial_loss

    def test_accepted_steps_never_increase_loss(self):
        """整代检查保证：loss_history 单调不增"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=50, seed=3
        )
        losses = diag["loss_history"]
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i-1] + 1e-9


class TestIntegration:
    """真实数据端到端"""

    def test_real_data_end_to_end(self):
        """真实 schema + 50 查询，跑几轮，验证 loss 下降"""
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_queries

        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([q["result"] for q in queries])

        best_S, diag = run_evolution(
            target, queries, schema, n_records=300, n_rounds=30, seed=2024
        )
        assert best_S.shape == (300, 10)
        assert diag["best_loss"] <= diag["loss_history"][0]
