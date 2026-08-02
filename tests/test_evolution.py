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


class TestUpdateMode:
    """update_mode 开关：legacy（旧机制）vs single_block（单块复制/变异）。"""

    def test_default_is_legacy(self):
        """默认 update_mode='legacy'，params 如实记录。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=10, seed=0
        )
        assert diag["params"]["update_mode"] == "legacy"
        assert diag["params"]["epsilon"] == 0.01

    def test_legacy_identical_to_before(self):
        """legacy 模式与不传 update_mode 完全一致（回归保护）。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        s1, d1 = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=42
        )
        s2, d2 = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=42,
            update_mode="legacy",
        )
        pd.testing.assert_frame_equal(s1, s2)
        assert d1["loss_history"] == d2["loss_history"]

    def test_legacy_no_single_block_diag(self):
        """legacy 模式不产生 single_block 诊断（历史列表为空）。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=10, seed=0,
            update_mode="legacy",
        )
        assert diag["participation_rate_history"] == []
        assert diag["copy_attempt_rate_history"] == []
        assert diag["mutation_attempt_rate_history"] == []
        assert diag["accepted_change_rate_history"] == []
        assert diag["empty_copy_set_count_history"] == []

    def test_single_block_runs(self):
        """single_block 模式能跑通并返回正确形状。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        best_S, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=0,
            update_mode="single_block", epsilon=0.1,
        )
        assert best_S.shape == (100, 3)
        assert list(best_S.columns) == schema.attribute_names()
        assert diag["params"]["update_mode"] == "single_block"
        assert diag["params"]["epsilon"] == 0.1

    def test_single_block_records_diagnostics(self):
        """single_block 模式逐轮记录五项诊断，长度与轮数一致。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=15, seed=3,
            update_mode="single_block", epsilon=0.1,
        )
        rounds = diag["rounds_run"]
        assert len(diag["participation_rate_history"]) == rounds
        assert len(diag["copy_attempt_rate_history"]) == rounds
        assert len(diag["mutation_attempt_rate_history"]) == rounds
        assert len(diag["accepted_change_rate_history"]) == rounds
        assert len(diag["empty_copy_set_count_history"]) == rounds
        # 诊断取值合理
        for r in diag["participation_rate_history"]:
            assert 0.0 <= r <= 1.0

    def test_single_block_reproducible(self):
        """single_block 模式相同种子 → 相同结果。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        s1, d1 = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=9,
            update_mode="single_block", epsilon=0.1,
        )
        s2, d2 = run_evolution(
            target, queries, schema, n_records=100, n_rounds=20, seed=9,
            update_mode="single_block", epsilon=0.1,
        )
        pd.testing.assert_frame_equal(s1, s2)
        assert d1["loss_history"] == d2["loss_history"]

    def test_single_block_monotonic_loss(self):
        """single_block 模式 loss 单调不增（世代验收保证方向正确）。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=30, seed=1,
            update_mode="single_block", epsilon=0.1,
        )
        losses = diag["loss_history"]
        for prev, cur in zip(losses, losses[1:]):
            assert cur <= prev + 1e-9

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"update_mode": "bogus"}, "update_mode"),
            ({"epsilon": -0.1}, "epsilon"),
            ({"epsilon": 1.5}, "epsilon"),
        ],
    )
    def test_invalid_update_mode_parameters(self, kwargs, message):
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        with pytest.raises(ValueError, match=message):
            run_evolution(target, queries, schema, n_records=100, **kwargs)


class TestSingleBlockDirectionGuard:
    """single_block 与残差方向核联合语义未定义，应互斥拒绝。"""

    def test_single_block_with_direction_raises(self):
        """single_block + residual_directed_diffusion 同开直接报错。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        with pytest.raises(ValueError, match="联合算子"):
            run_evolution(
                target, queries, schema, n_records=100, n_rounds=10, seed=0,
                update_mode="single_block", residual_directed_diffusion=True,
            )

    def test_guard_raises_before_direction_computation(self):
        """护栏在方向矩阵计算前抛出：未产生任何方向评价。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        # 用极小 n_records/n_rounds，若护栏失效则会进入主循环并计算方向
        with pytest.raises(ValueError, match="联合算子"):
            run_evolution(
                target, queries, schema, n_records=10, n_rounds=1, seed=0,
                update_mode="single_block", residual_directed_diffusion=True,
                diffusion_direction_strength=1.0,
            )

    def test_single_block_alone_has_no_copy_kernel_entropy(self):
        """single_block 不使用 eta 复制核，熵历史应全为 None。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=10, seed=0,
            update_mode="single_block", epsilon=0.1,
        )
        entropy_hist = diag["copy_probability_entropy_history"]
        assert len(entropy_hist) == 10
        assert all(e is None for e in entropy_hist)
        # 方向核未启用，方向评价次数为 0
        assert diag["direction_evaluation_count"] == 0

    def test_legacy_alone_still_records_eta_entropy(self):
        """回归：legacy 单独运行仍按 eta 记录复制核熵（非 None）。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=10, seed=0,
            update_mode="legacy",
        )
        entropy_hist = diag["copy_probability_entropy_history"]
        assert all(e is not None for e in entropy_hist)


class TestSingleBlockEmptyCopySetRetrySemantics:
    """empty_copy_set_count_history 口径：每轮只记被接受 / 最后一次失败尝试。

    坐实 PROJECT_STATUS 的口径说明——max_retries>0 时该历史不含被丢弃的中间尝试。
    用 monkeypatch 控制每次 attempt 的 proposal 与 empty_copy_set_count。
    """

    def _schema_queries(self):
        schema = Schema([
            AttributeBlock(
                name="x", type="categorical", description="x", values=["a", "b"]
            )
        ])
        queries = [
            {"conditions": [{"attribute": "x", "operator": "==", "value": "a"}]}
        ]
        return schema, queries

    def test_first_rejected_second_accepted_records_accepted_attempt(self):
        """首试拒、二试接受：历史只记被接受那次的 empty_copy_set_count。"""
        schema, queries = self._schema_queries()
        # target=0 个 "a"。首试全 "a"(count_a=20，loss 变差被拒，empty=7)，
        # 二试全 "b"(count_a=0 命中 target 被接受，empty=2)。
        attempts = [("a", 7), ("b", 2)]
        calls = {"i": 0}

        def fake(current, donors, schema, rho, epsilon, rng):
            value, empty = attempts[calls["i"]]
            calls["i"] += 1
            proposal = pd.DataFrame({"x": [value] * len(current)})
            return proposal, {
                "participation_rate": 1.0, "copy_attempt_rate": 1.0,
                "mutation_attempt_rate": 0.0, "accepted_change_rate": 1.0,
                "empty_copy_set_count": empty,
            }

        orig = evolution_module.evolve_step_single_block
        evolution_module.evolve_step_single_block = fake
        try:
            _, diag = run_evolution(
                np.array([0]), queries, schema,
                n_records=20, n_rounds=1, seed=0,
                rho=0.5, max_retries=1, retry_rho_decay=0.5,
                update_mode="single_block", epsilon=0.0,
            )
        finally:
            evolution_module.evolve_step_single_block = orig

        assert diag["accept_history"] == [True]
        assert diag["accepted_attempt_history"] == [2]  # 第 2 次尝试被接受
        # 只记被接受那次(2)，不是两次相加(9)也不是首次(7)
        assert diag["empty_copy_set_count_history"] == [2]

    def test_all_rejected_records_last_attempt(self):
        """全部拒绝：历史记最后一次失败尝试的 empty_copy_set_count。"""
        schema, queries = self._schema_queries()
        # target=20 个 "a"，但每次都产出全 "b"(count_a=0) → 恒不改善被拒。
        empties = [5, 4, 3]  # 三次尝试(初试+2重试)各自的 empty_copy_set_count
        calls = {"i": 0}

        def fake(current, donors, schema, rho, epsilon, rng):
            empty = empties[calls["i"]]
            calls["i"] += 1
            proposal = pd.DataFrame({"x": ["b"] * len(current)})
            return proposal, {
                "participation_rate": 1.0, "copy_attempt_rate": 1.0,
                "mutation_attempt_rate": 0.0, "accepted_change_rate": 1.0,
                "empty_copy_set_count": empty,
            }

        orig = evolution_module.evolve_step_single_block
        evolution_module.evolve_step_single_block = fake
        try:
            _, diag = run_evolution(
                np.array([20]), queries, schema,
                n_records=20, n_rounds=1, seed=0,
                rho=0.5, max_retries=2, retry_rho_decay=0.5,
                update_mode="single_block", epsilon=0.0,
            )
        finally:
            evolution_module.evolve_step_single_block = orig

        assert diag["accept_history"] == [False]
        assert diag["accepted_attempt_history"] == [0]  # 全拒
        assert diag["proposal_attempts_history"] == [3]
        # 记最后一次失败尝试(3)，不是首次(5)也不是三次相加(12)
        assert diag["empty_copy_set_count_history"] == [3]
