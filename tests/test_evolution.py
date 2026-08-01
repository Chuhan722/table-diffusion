"""
测试扩散演化主循环

锚定主循环的结构、终止条件、复现性、方向正确性。
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.evolution import run_evolution
import table_diffevo.evolution as evolution_module
import table_diffevo.update as update_module


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
        # 计时字段（扫描时估时/对比用）
        assert "elapsed_sec" in diag
        assert "sec_per_round" in diag
        assert diag["elapsed_sec"] > 0
        assert diag["sec_per_round"] > 0

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


class TestStateCache:
    """当前表不变时复用评价与距离，接受提案后正确失效。"""

    @staticmethod
    def _schema_and_queries():
        schema = Schema([
            AttributeBlock(
                name="x", type="categorical", description="x", values=["a", "b"]
            )
        ])
        queries = [
            {"conditions": [{"attribute": "x", "operator": "==", "value": "a"}]}
        ]
        return schema, queries

    def test_selected_distance_mean_numpy(self):
        """NumPy 路径只取每行选中的距离。"""
        distances = np.array([[0.0, 0.25], [0.75, 0.0]])
        donor_idx = np.array([1, 0])
        result = evolution_module._mean_selected_distance(distances, donor_idx)
        assert result == pytest.approx(0.5)

    def test_selected_distance_mean_torch(self):
        """torch 路径在设备上 gather，数值与 NumPy 路径一致。"""
        torch = pytest.importorskip("torch")
        distances = torch.tensor([[0.0, 0.25], [0.75, 0.0]])
        donor_idx = np.array([1, 0])
        result = evolution_module._mean_selected_distance(
            distances, donor_idx, use_torch=True
        )
        assert result == pytest.approx(0.5)

    def test_zero_rounds_evaluates_initial_state_without_distance(self, monkeypatch):
        """零轮仍返回有效初始 best，但不应构造未使用的距离矩阵。"""
        schema, queries = self._schema_and_queries()
        initial = pd.DataFrame({"x": ["a", "b", "b", "b"]})
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        _, diag = run_evolution(
            np.array([0]), queries, schema,
            n_records=4, n_rounds=0, seed=0, device="numpy",
        )

        assert diag["rounds_run"] == 0
        assert diag["loss_history"] == []
        assert diag["best_loss"] == pytest.approx(0.5)
        assert diag["state_evaluation_count"] == 1
        assert diag["distance_evaluation_count"] == 0
        assert diag["fitness_dominance_rate_history"] == []
        assert diag["fitness_copy_participation_scale_history"] == []

    def test_initially_converged_stops_before_distance(self, monkeypatch):
        """初始表已达标时只用状态缓存完成终止检查，不计算距离。"""
        schema, queries = self._schema_and_queries()
        initial = pd.DataFrame({"x": ["a", "b", "b", "b"]})
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        _, diag = run_evolution(
            np.array([1]), queries, schema,
            n_records=4, n_rounds=5, seed=0, device="numpy", log_every=100,
        )

        assert diag["rounds_run"] == 1
        assert diag["stopped_early"] is True
        assert diag["best_loss"] == 0.0
        assert diag["state_evaluation_count"] == 1
        assert diag["distance_evaluation_count"] == 0
        assert diag["fitness_dominance_rate_history"] == []
        assert diag["fitness_copy_participation_scale_history"] == []

    def test_rejected_rounds_reuse_state_and_distance(self, monkeypatch):
        """连续拒绝时只评价一次当前表和一次距离，但每轮仍重新抽样。"""
        schema, queries = self._schema_and_queries()
        initial = pd.DataFrame({"x": ["a", "b", "b", "b"]})
        calls = {"full_eval": 0, "proposal_eval": 0, "distance": 0, "probs": 0}

        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        original_eval = evolution_module.evaluate_vectorized

        def counted_eval(*args, **kwargs):
            if kwargs.get("want_fitness", True):
                calls["full_eval"] += 1
            else:
                calls["proposal_eval"] += 1
            return original_eval(*args, **kwargs)

        monkeypatch.setattr(evolution_module, "evaluate_vectorized", counted_eval)

        original_distance = evolution_module.pairwise_block_distance

        def counted_distance(*args, **kwargs):
            calls["distance"] += 1
            return original_distance(*args, **kwargs)

        monkeypatch.setattr(
            evolution_module, "pairwise_block_distance", counted_distance
        )

        original_probs = evolution_module.compute_sampling_probs

        def counted_probs(*args, **kwargs):
            calls["probs"] += 1
            return original_probs(*args, **kwargs)

        monkeypatch.setattr(evolution_module, "compute_sampling_probs", counted_probs)
        monkeypatch.setattr(
            evolution_module, "evolve_step",
            lambda current, donors, schema, rho, eta, mu, rng:
                pd.DataFrame({"x": ["a"] * len(current)}),
        )

        _, diag = run_evolution(
            np.array([0]), queries, schema,
            n_records=4, n_rounds=3, seed=0, device="numpy", log_every=100,
        )

        assert diag["accept_history"] == [False, False, False]
        assert diag["state_evaluation_count"] == 1
        assert diag["distance_evaluation_count"] == 1
        assert calls == {
            "full_eval": 1,
            "proposal_eval": 3,
            "distance": 1,
            "probs": 3,
        }

    def test_accepted_round_invalidates_state_and_distance(self, monkeypatch):
        """接受提案后，下一轮必须重新评价新表并重算距离。"""
        schema, queries = self._schema_and_queries()
        initial = pd.DataFrame({"x": ["a", "b", "b", "b"]})
        proposals = [
            pd.DataFrame({"x": ["a", "a", "b", "b"]}),
            initial,
        ]
        proposal_position = 0

        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        def fake_evolve(current, donors, schema, rho, eta, mu, rng):
            nonlocal proposal_position
            proposal = proposals[proposal_position]
            proposal_position += 1
            return proposal.copy()

        monkeypatch.setattr(evolution_module, "evolve_step", fake_evolve)

        _, diag = run_evolution(
            np.array([3]), queries, schema,
            n_records=4, n_rounds=2, seed=0, device="numpy", log_every=100,
        )

        assert diag["accept_history"] == [True, False]
        assert diag["state_evaluation_count"] == 2
        assert diag["distance_evaluation_count"] == 2
        assert diag["best_loss"] == pytest.approx(0.5)


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


class TestExcludeSelf:
    """对角线屏蔽在主循环层面的行为 + 自身抽样率诊断字段"""

    def _setup(self):
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        return schema, queries, target

    def test_self_rate_history_present_and_length(self):
        """诊断含 donor_self_rate_history，长度=实际轮数。"""
        schema, queries, target = self._setup()
        _, diag = run_evolution(
            target, queries, schema, n_records=80, n_rounds=15, seed=0,
            distance_mode='geometric',
        )
        assert "donor_self_rate_history" in diag
        assert len(diag["donor_self_rate_history"]) == diag["rounds_run"]

    def test_default_excludes_self_rate_zero(self):
        """默认 exclude_self=True → 每轮自身抽样率恒为 0。"""
        schema, queries, target = self._setup()
        _, diag = run_evolution(
            target, queries, schema, n_records=80, n_rounds=15, seed=0,
            distance_mode='geometric',
        )
        assert diag["params"]["exclude_self"] is True
        assert all(r == 0.0 for r in diag["donor_self_rate_history"])

    def test_disabled_allows_self_sampling(self):
        """exclude_self=False → 全对全候选池允许抽到自己，自身率应出现非零。"""
        schema, queries, target = self._setup()
        _, diag = run_evolution(
            target, queries, schema, n_records=80, n_rounds=30, seed=0,
            distance_mode='geometric', alpha_min=6.0, alpha_max=10.0,
            exclude_self=False,
        )
        assert diag["params"]["exclude_self"] is False
        assert any(r > 0.0 for r in diag["donor_self_rate_history"])


class TestFitnessDominanceGate:
    """donor 复制偏向适应度支配 pair，同时保留可控探索。"""

    @staticmethod
    def _setup():
        schema = Schema([
            AttributeBlock(
                name="x", type="categorical", description="x",
                values=["a", "b"],
            )
        ])
        queries = [
            {"conditions": [{"attribute": "x", "operator": "==", "value": "a"}]}
        ]
        initial = pd.DataFrame({"x": ["a", "b"]})
        return schema, queries, initial

    def test_gate_keeps_better_recipient_and_updates_worse_one(self, monkeypatch):
        schema, queries, initial = self._setup()
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        # a 的 donor 是较差的 b；b 的 donor 是较好的 a。
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.array([1, 0]),
        )

        best, diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=1, seed=0,
            rho=1.0, eta=1.0, mu=0.0,
            fitness_dominance_gate=True,
            fitness_dominance_exploration_rate=0.0,
        )

        assert best["x"].tolist() == ["a", "a"]
        assert diag["best_loss"] == 0.0
        assert diag["fitness_dominance_rate_history"] == [0.5]
        assert diag["fitness_copy_participation_scale_history"] == [0.5]
        assert diag["params"]["fitness_dominance_gate"] is True
        assert diag["params"]["fitness_dominance_exploration_rate"] == 0.0

    def test_default_gate_keeps_soft_copy_support(self, monkeypatch):
        schema, queries, initial = self._setup()
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.array([1, 0]),
        )
        seen_scales = []

        def fake_evolve(
            current, donors, schema, rho, eta, mu, rng,
            copy_participation_scale,
        ):
            seen_scales.append(copy_participation_scale.copy())
            return current.copy()

        monkeypatch.setattr(evolution_module, "evolve_step", fake_evolve)
        _, diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=1, seed=0,
            fitness_dominance_gate=True,
        )

        assert len(seen_scales) == 1
        assert np.array_equal(seen_scales[0], np.array([0.02, 1.0]))
        assert diag["fitness_copy_participation_scale_history"] == [0.51]
        assert diag["params"]["fitness_dominance_exploration_rate"] == 0.02

    def test_without_gate_preserves_original_swap(self, monkeypatch):
        schema, queries, initial = self._setup()
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.array([1, 0]),
        )

        best, diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=1, seed=0,
            rho=1.0, eta=1.0, mu=0.0,
            fitness_dominance_gate=False,
        )

        assert best["x"].tolist() == initial["x"].tolist()
        assert diag["best_loss"] == pytest.approx(0.5)
        assert diag["params"]["fitness_dominance_gate"] is False
        assert diag["fitness_copy_participation_scale_history"] == [1.0]

    def test_copy_scale_is_reused_across_retries(self, monkeypatch):
        schema, queries, initial = self._setup()
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.array([1, 0]),
        )
        seen_scales = []

        def fake_evolve(
            current, donors, schema, rho, eta, mu, rng,
            copy_participation_scale,
        ):
            seen_scales.append(copy_participation_scale.copy())
            value = "b" if len(seen_scales) == 1 else "a"
            return pd.DataFrame({"x": [value] * len(current)})

        monkeypatch.setattr(evolution_module, "evolve_step", fake_evolve)
        _, diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=1, seed=0,
            rho=1.0, eta=1.0, mu=0.0,
            max_retries=1, retry_rho_decay=0.5,
            fitness_dominance_gate=True,
            fitness_dominance_exploration_rate=0.7,
        )

        assert len(seen_scales) == 2
        assert np.array_equal(seen_scales[0], np.array([0.7, 1.0]))
        assert np.array_equal(seen_scales[1], seen_scales[0])
        assert diag["accepted_attempt_history"] == [2]
        assert diag["best_loss"] == 0.0

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_gate_parameter_requires_bool(self, value):
        schema, queries, _ = self._setup()
        with pytest.raises(ValueError, match="fitness_dominance_gate"):
            run_evolution(
                np.array([2]), queries, schema,
                n_records=2, n_rounds=1,
                fitness_dominance_gate=value,
            )

    @pytest.mark.parametrize(
        "value", [-0.01, 1.01, np.inf, np.nan, "0.1", True]
    )
    def test_exploration_rate_bounds(self, value):
        schema, queries, _ = self._setup()
        with pytest.raises(
            ValueError, match="fitness_dominance_exploration_rate"
        ):
            run_evolution(
                np.array([2]), queries, schema,
                n_records=2, n_rounds=1,
                fitness_dominance_exploration_rate=value,
            )

    def test_exploration_one_matches_gate_off_exactly(self, monkeypatch):
        schema, queries, initial = self._setup()
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.array([1, 0]),
        )

        baseline, baseline_diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=3, seed=19,
            rho=0.6, eta=0.4, mu=0.3,
            fitness_dominance_gate=False,
        )
        explored, explored_diag = run_evolution(
            np.array([2]), queries, schema,
            n_records=2, n_rounds=3, seed=19,
            rho=0.6, eta=0.4, mu=0.3,
            fitness_dominance_gate=True,
            fitness_dominance_exploration_rate=1.0,
        )

        pd.testing.assert_frame_equal(explored, baseline)
        for key in (
            "best_loss",
            "rounds_run",
            "stopped_early",
            "loss_history",
            "accept_history",
            "donor_fitness_history",
            "donor_distance_history",
            "donor_self_rate_history",
            "fitness_dominance_rate_history",
            "fitness_copy_participation_scale_history",
            "alpha_history",
            "proposal_attempts_history",
            "accepted_attempt_history",
            "accepted_rho_history",
            "state_evaluation_count",
            "distance_evaluation_count",
            "normalized_l1_error",
            "normalized_l1_median",
            "normalized_l1_p90",
            "normalized_l1_max",
            "initialization",
        ):
            assert explored_diag[key] == baseline_diag[key]
        assert all(
            rate == 1.0
            for rate in explored_diag[
                "fitness_copy_participation_scale_history"
            ]
        )

    def test_equal_fitness_can_escape_through_mutation(self, monkeypatch):
        n_records = 4
        schema = Schema([
            AttributeBlock(
                name="x", type="categorical", description="x",
                values=["a", "b"],
            )
        ])
        queries = [
            {
                "conditions": [
                    {"attribute": "x", "operator": "==", "value": "b"}
                ]
            }
        ]
        initial = pd.DataFrame({"x": ["a"] * n_records})
        monkeypatch.setattr(
            evolution_module, "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )
        monkeypatch.setattr(
            evolution_module, "sample_donors",
            lambda *args, **kwargs: np.roll(np.arange(n_records), 1),
        )
        monkeypatch.setattr(
            update_module, "_sample_mutation_block", lambda *args: "x"
        )
        monkeypatch.setattr(
            update_module, "_sample_legal_value", lambda *args: "b"
        )

        best, diag = run_evolution(
            np.array([n_records]), queries, schema,
            n_records=n_records, n_rounds=1, seed=0,
            rho=1.0, eta=1.0, mu=1.0,
            fitness_dominance_gate=True,
            fitness_dominance_exploration_rate=0.0,
        )

        assert best["x"].tolist() == ["b"] * n_records
        assert diag["best_loss"] == 0.0
        assert diag["fitness_dominance_rate_history"] == [0.0]
        assert diag["fitness_copy_participation_scale_history"] == [0.0]


class TestProposalRetries:
    """提案被拒后缩小 rho 重试。"""

    def test_default_is_single_attempt(self):
        """默认不重试，每轮只评估一个提案。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])

        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=12, seed=7,
        )

        assert diag["params"]["max_retries"] == 0
        assert all(n == 1 for n in diag["proposal_attempts_history"])
        assert all(a in (0, 1) for a in diag["accepted_attempt_history"])

    def test_rejected_proposal_retries_with_smaller_rho(self, monkeypatch):
        """首次故意变差，第二次改到目标：应在缩小 rho 后接受。"""
        schema = Schema([
            AttributeBlock(
                name="x", type="categorical", description="x", values=["a", "b"]
            )
        ])
        queries = [
            {"conditions": [{"attribute": "x", "operator": "==", "value": "a"}]}
        ]
        seen_rhos = []

        def fake_evolve(current, donors, schema, rho, eta, mu, rng):
            seen_rhos.append(rho)
            value = "a" if len(seen_rhos) == 1 else "b"
            return pd.DataFrame({"x": [value] * len(current)})

        monkeypatch.setattr(evolution_module, "evolve_step", fake_evolve)
        _, diag = run_evolution(
            np.array([0]), queries, schema,
            n_records=20, n_rounds=1, seed=0,
            rho=0.2, max_retries=1, retry_rho_decay=0.25,
        )

        assert seen_rhos == pytest.approx([0.2, 0.05])
        assert diag["proposal_attempts_history"] == [2]
        assert diag["accepted_attempt_history"] == [2]
        assert diag["accepted_rho_history"] == pytest.approx([0.05])
        assert diag["best_loss"] == 0.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"max_retries": -1}, "max_retries"),
            ({"max_retries": 1.5}, "max_retries"),
            ({"retry_rho_decay": 0.0}, "retry_rho_decay"),
            ({"retry_rho_decay": 1.0}, "retry_rho_decay"),
        ],
    )
    def test_invalid_retry_parameters(self, kwargs, message):
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        with pytest.raises(ValueError, match=message):
            run_evolution(target, queries, schema, n_records=100, **kwargs)
