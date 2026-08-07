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

    def test_candidate_budget_triggers_early_stop(self):
        """给一个远小于 n_rounds 所需评估数的预算，应在跑满 n_rounds 前提前停止。

        锚定 candidate_budget 的实际停止行为（不只是配置校验）。
        注意实现语义：预算检查在提案被接受时跳过（接受即 break），
        因此实际评估次数会略微越过预算才停，这里断言 >= budget 而非精确相等。
        """
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        budget = 5
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=200, seed=0,
            candidate_budget=budget,
        )
        assert diag["candidate_budget_exhausted"] is True
        assert diag["candidate_evaluation_count"] >= budget
        # 因预算而非 n_rounds 停止：轮数应远少于 n_rounds
        assert diag["rounds_run"] < 200

    def test_no_candidate_budget_runs_full_rounds(self):
        """不设预算（None）时不应因预算提前停止。"""
        schema = make_toy_schema()
        queries = make_toy_queries()
        target = np.array([30, 40, 50])
        _, diag = run_evolution(
            target, queries, schema, n_records=100, n_rounds=15, seed=0,
            candidate_budget=None,
        )
        assert diag["candidate_budget_exhausted"] is False
        # 未达标时应跑满（与 test_runs_full_rounds_when_not_converged 同前提）
        if not diag["stopped_early"]:
            assert diag["rounds_run"] == 15


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


class TestFactorizedGibbsClosedLoop:
    """低阶因子 Gibbs 接入标准接受闭环后的语义与随机流。"""

    @staticmethod
    def _schema_queries_target():
        schema = Schema([
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b", "c")
        ])
        queries = [
            {"conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
            ]},
            {"conditions": [
                {"attribute": "b", "operator": "==", "value": 1},
            ]},
            {"conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]},
            {"conditions": [
                {"attribute": "a", "operator": "==", "value": 0},
                {"attribute": "b", "operator": "==", "value": 0},
                {"attribute": "c", "operator": "==", "value": 1},
            ]},
        ]
        # 半整数 target 不可能被整数计数精确命中，避免测试意外提前停止。
        target = np.array([8.5, 10.5, 4.5, 2.5])
        return schema, queries, target

    @staticmethod
    def _run_kwargs():
        return {
            "n_records": 20,
            "n_rounds": 5,
            "seed": 17,
            "rho": 0.8,
            "eta": 0.5,
            "mu": 0.1,
            "device": "numpy",
            "distance_mode": "geometric",
            "residual_directed_diffusion": True,
            "diffusion_direction_strength": 2.0,
            "diffusion_direction_normalization": "initial_rms",
            "log_every": 100,
        }

    def test_zero_sweeps_matches_existing_closed_loop(self):
        schema, queries, target = self._schema_queries_target()
        implicit, implicit_diag = run_evolution(
            target, queries, schema, **self._run_kwargs()
        )
        explicit, explicit_diag = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=0,
            **self._run_kwargs(),
        )

        pd.testing.assert_frame_equal(explicit, implicit)
        for key in (
            "loss_history",
            "accept_history",
            "donor_fitness_history",
            "donor_distance_history",
            "raw_proposal_gain_history",
            "accepted_attempt_history",
            "initial_table_sha256",
            "primary_rng_post_initialization_state_sha256",
            "primary_rng_state_sha256",
        ):
            assert explicit_diag[key] == implicit_diag[key]
        assert explicit_diag[
            "factorized_gibbs_attempt_diagnostics_history"
        ] == [[] for _ in explicit_diag["accept_history"]]
        assert explicit_diag["factorized_gibbs_microsteps"] == 0
        assert explicit_diag["factorized_gibbs_rng_state_sha256"] is None

    def test_nonzero_sweeps_are_reproducible_and_preserve_primary_rng(self):
        schema, queries, target = self._schema_queries_target()
        baseline, baseline_diag = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=0,
            **self._run_kwargs(),
        )
        candidate, candidate_diag = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=2,
            **self._run_kwargs(),
        )
        repeated, repeated_diag = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=2,
            **self._run_kwargs(),
        )

        assert len(baseline) == len(candidate)
        pd.testing.assert_frame_equal(repeated, candidate)
        assert candidate_diag["loss_history"] == repeated_diag["loss_history"]
        assert (
            candidate_diag["initial_table_sha256"]
            == baseline_diag["initial_table_sha256"]
        )
        assert (
            candidate_diag["primary_rng_post_initialization_state_sha256"]
            == baseline_diag["primary_rng_post_initialization_state_sha256"]
        )
        assert (
            candidate_diag["primary_rng_state_sha256"]
            == baseline_diag["primary_rng_state_sha256"]
        )
        assert (
            candidate_diag["factorized_gibbs_rng_state_sha256"]
            == repeated_diag["factorized_gibbs_rng_state_sha256"]
        )
        assert (
            candidate_diag["factorized_gibbs_initial_rng_state_sha256"]
            != candidate_diag["factorized_gibbs_rng_state_sha256"]
        )
        assert candidate_diag["factorized_gibbs_active_rows"] > 0
        assert candidate_diag["factorized_gibbs_factor_count"] > 0
        assert candidate_diag["factorized_gibbs_microsteps"] > 0
        assert len(candidate_diag[
            "factorized_gibbs_attempt_diagnostics_history"
        ]) == len(candidate_diag["accept_history"])
        assert all(
            len(attempts) == 1
            for attempts in candidate_diag[
                "factorized_gibbs_attempt_diagnostics_history"
            ]
        )

    def test_factorized_retry_records_every_attempt(self, monkeypatch):
        schema = Schema([
            AttributeBlock(
                name="x",
                type="categorical",
                description="x",
                values=["a", "b"],
            )
        ])
        queries = [
            {"conditions": [
                {"attribute": "x", "operator": "==", "value": "a"},
            ]}
        ]
        initial = pd.DataFrame({"x": ["a", "b", "b", "b"]})
        seen_rhos = []
        seen_logit_clips = []

        monkeypatch.setattr(
            evolution_module,
            "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        def fake_factorized(
            current, donors, schema, queries, residual, **kwargs
        ):
            seen_rhos.append(kwargs["rho"])
            seen_logit_clips.append(kwargs["gibbs_logit_clip"])
            value = "a" if len(seen_rhos) == 1 else "b"
            proposal = pd.DataFrame({"x": [value] * len(current)})
            diagnostics = {
                "participating_rows": 4,
                "active_gibbs_rows": 3,
                "active_blocks": 3,
                "factor_count": 2,
                "factor_table_entries": 4,
                "gibbs_microsteps": 6,
                "factor_build_elapsed_sec": 0.25,
                "gibbs_sample_elapsed_sec": 0.125,
            }
            return proposal, diagnostics

        monkeypatch.setattr(
            evolution_module,
            "evolve_step_factorized_gibbs",
            fake_factorized,
        )
        _, diagnostics = run_evolution(
            np.array([0]),
            queries,
            schema,
            n_records=4,
            n_rounds=1,
            seed=0,
            rho=0.2,
            max_retries=1,
            retry_rho_decay=0.25,
            residual_directed_diffusion=True,
            factorized_gibbs_sweeps=2,
            factorized_gibbs_logit_clip=17.0,
            device="numpy",
            log_every=100,
        )

        assert seen_rhos == pytest.approx([0.2, 0.05])
        assert seen_logit_clips == [17.0, 17.0]
        assert diagnostics["proposal_attempts_history"] == [2]
        assert diagnostics["accepted_attempt_history"] == [2]
        attempts = diagnostics[
            "factorized_gibbs_attempt_diagnostics_history"
        ]
        assert len(attempts) == 1
        assert len(attempts[0]) == 2
        assert diagnostics["factorized_gibbs_active_rows"] == 6
        assert diagnostics["factorized_gibbs_factor_count"] == 4
        assert diagnostics["factorized_gibbs_microsteps"] == 12
        assert diagnostics[
            "factorized_gibbs_factor_build_elapsed_sec"
        ] == pytest.approx(0.5)

    def test_zero_rounds_do_not_consume_factorized_rng(self):
        schema, queries, target = self._schema_queries_target()
        _, diagnostics = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=2,
            **{**self._run_kwargs(), "n_rounds": 0},
        )

        assert diagnostics["rounds_run"] == 0
        assert diagnostics[
            "factorized_gibbs_attempt_diagnostics_history"
        ] == []
        assert diagnostics["factorized_gibbs_microsteps"] == 0
        assert (
            diagnostics["factorized_gibbs_initial_rng_state_sha256"]
            == diagnostics["factorized_gibbs_rng_state_sha256"]
        )

    def test_initial_convergence_does_not_consume_factorized_rng(
        self, monkeypatch
    ):
        schema, queries, _ = self._schema_queries_target()
        initial = pd.DataFrame({
            "a": [1, 0, 1, 0],
            "b": [1, 1, 0, 0],
            "c": [1, 0, 1, 1],
        })
        target = np.asarray([
            2.0,
            2.0,
            1.0,
            1.0,
        ])
        monkeypatch.setattr(
            evolution_module,
            "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        result, diagnostics = run_evolution(
            target,
            queries,
            schema,
            n_records=4,
            n_rounds=5,
            seed=17,
            rho=0.8,
            eta=0.5,
            mu=0.1,
            device="numpy",
            residual_directed_diffusion=True,
            diffusion_direction_strength=2.0,
            factorized_gibbs_sweeps=2,
            log_every=100,
        )

        pd.testing.assert_frame_equal(result, initial)
        assert diagnostics["rounds_run"] == 1
        assert diagnostics["stopped_early"] is True
        assert diagnostics["accept_history"] == []
        assert diagnostics[
            "factorized_gibbs_attempt_diagnostics_history"
        ] == []
        assert (
            diagnostics["factorized_gibbs_initial_rng_state_sha256"]
            == diagnostics["factorized_gibbs_rng_state_sha256"]
        )

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"factorized_gibbs_sweeps": -1}, "sweeps"),
            ({"factorized_gibbs_sweeps": 1.5}, "sweeps"),
            ({"factorized_gibbs_sweeps": True}, "sweeps"),
            ({"factorized_gibbs_max_order": -1}, "max_order"),
            (
                {"factorized_gibbs_sweeps": 1,
                 "factorized_gibbs_max_order": 0},
                "max_order",
            ),
            ({"factorized_gibbs_max_order": 1.5}, "max_order"),
            ({"factorized_gibbs_max_order": 9}, "max_order"),
            (
                {"factorized_gibbs_sweeps": 1,
                 "residual_directed_diffusion": False},
                "residual_directed_diffusion",
            ),
            ({"factorized_gibbs_sweeps": 1, "eta": 0.0}, "eta"),
            ({"factorized_gibbs_sweeps": 1, "eta": 1.0}, "eta"),
            ({"factorized_gibbs_sweeps": 1, "seed": None}, "seed"),
            ({"factorized_gibbs_logit_clip": 0.0}, "logit_clip"),
            ({"factorized_gibbs_logit_clip": np.inf}, "logit_clip"),
            ({"factorized_gibbs_logit_clip": True}, "logit_clip"),
        ],
    )
    def test_rejects_invalid_factorized_parameters(self, kwargs, message):
        schema, queries, target = self._schema_queries_target()
        parameters = self._run_kwargs()
        parameters.update(kwargs)
        with pytest.raises(ValueError, match=message):
            run_evolution(target, queries, schema, **parameters)

    def test_cuda_smoke(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")
        schema, queries, target = self._schema_queries_target()
        result, diagnostics = run_evolution(
            target,
            queries,
            schema,
            factorized_gibbs_sweeps=1,
            **{
                **self._run_kwargs(),
                "device": "cuda",
                "n_rounds": 2,
            },
        )

        assert result.shape == (20, 3)
        assert diagnostics["factorized_gibbs_microsteps"] > 0


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
