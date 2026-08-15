"""残差自冷却机制（Issue #44）的单元与集成测试。"""

import hashlib

import numpy as np
import pandas as pd
import pytest

import table_diffevo.evolution as evolution_module
from table_diffevo.evolution import _self_cooling_factor, run_evolution
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


def _tiny_problem():
    schema = Schema([
        AttributeBlock(name="a", type="categorical", description="a",
                       values=["0", "1"]),
        AttributeBlock(name="b", type="categorical", description="b",
                       values=["0", "1"]),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"}]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": "1"}]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "b", "operator": "==", "value": "1"}]},
    ]
    target = np.asarray([6.0, 2.0, 2.0])
    return schema, queries, target


def _table_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _run(**overrides):
    schema, queries, target = _tiny_problem()
    kwargs = dict(
        target=target,
        queries=queries,
        schema=schema,
        n_records=8,
        n_rounds=12,
        seed=3,
        log_every=-1,
    )
    kwargs.update(overrides)
    return run_evolution(**kwargs)


class TestSelfCoolingFactor:
    def test_initial_round_ratio_is_one(self):
        ratio, factor = _self_cooling_factor(10.0, 10.0, 2.0)
        assert ratio == 1.0
        assert factor == 1.0

    def test_ratio_clipped_when_residual_exceeds_initial(self):
        ratio, factor = _self_cooling_factor(15.0, 10.0, 1.0)
        assert ratio == 1.0
        assert factor == 1.0

    def test_exponent_controls_steepness(self):
        _, linear = _self_cooling_factor(2.5, 10.0, 1.0)
        _, quadratic = _self_cooling_factor(2.5, 10.0, 2.0)
        _, sqrt = _self_cooling_factor(2.5, 10.0, 0.5)
        assert linear == pytest.approx(0.25)
        assert quadratic == pytest.approx(0.0625)
        assert sqrt == pytest.approx(0.5)
        assert quadratic < linear < sqrt

    def test_zero_initial_residual_freezes(self):
        ratio, factor = _self_cooling_factor(0.0, 0.0, 1.0)
        assert ratio == 0.0
        assert factor == 0.0


class TestSelfCoolingIntegration:
    def test_default_none_keeps_trajectory_and_records_unit_factor(self):
        best_default, diag_default = _run()
        best_explicit, diag_explicit = _run(residual_self_cooling=None)

        assert diag_default["loss_history"] == diag_explicit["loss_history"]
        assert _table_hash(best_default) == _table_hash(best_explicit)
        assert diag_default["self_cooling_history"] == (
            [1.0] * len(diag_default["loss_history"])
        )
        assert diag_default["self_cooling_stopped"] is False
        assert diag_default["params"]["residual_self_cooling"] is None
        assert diag_default["params"]["self_cooling_stop_ratio"] is None

    def test_cooling_history_aligned_bounded_and_starts_at_one(self):
        _, diag = _run(residual_self_cooling=1.0, n_rounds=20)

        history = diag["self_cooling_history"]
        assert len(history) == len(diag["loss_history"])
        assert history[0] == 1.0
        assert all(0.0 <= value <= 1.0 for value in history)
        assert diag["params"]["residual_self_cooling"] == 1.0

    def test_cooling_changes_trajectory_once_residual_drops(self):
        # 无门 + 30 行、有明确下降空间的 target：残差下降后冷却缩小 rho，
        # 轨迹相对关闭冷却的对照必然分叉。
        common = dict(
            n_rounds=60, rho=0.3, mu=0.02, seed=9, n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]), tol=float("inf"),
        )
        _, diag_off = _run(**common)
        _, diag_on = _run(residual_self_cooling=1.0, **common)

        assert min(diag_on["self_cooling_history"]) < 1.0
        assert diag_off["loss_history"] != diag_on["loss_history"]

    def test_stop_ratio_triggers_intrinsic_stop(self):
        _, diag = _run(
            residual_self_cooling=1.0,
            self_cooling_stop_ratio=0.999,
            n_rounds=50,
            max_retries=2,
        )

        assert diag["self_cooling_stopped"] is True
        assert diag["stopped_early"] is True
        assert diag["rounds_run"] < 50
        assert diag["params"]["self_cooling_stop_ratio"] == 0.999

    def test_stop_ratio_not_triggered_stays_false(self):
        _, diag = _run(
            residual_self_cooling=1.0,
            self_cooling_stop_ratio=1e-9,
            n_rounds=5,
        )

        assert diag["self_cooling_stopped"] is False

    @pytest.mark.parametrize("bad", [True, 0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_exponent_rejected(self, bad):
        with pytest.raises(ValueError, match="residual_self_cooling"):
            _run(residual_self_cooling=bad)

    @pytest.mark.parametrize("bad", [True, 0.0, 1.0, -0.5, float("nan")])
    def test_invalid_stop_ratio_rejected(self, bad):
        with pytest.raises(ValueError, match="self_cooling_stop_ratio"):
            _run(residual_self_cooling=1.0, self_cooling_stop_ratio=bad)

    def test_stop_ratio_requires_cooling_enabled(self):
        with pytest.raises(ValueError, match="需要同时启用"):
            _run(self_cooling_stop_ratio=0.5)

    def test_gate_free_configuration_runs(self):
        best, diag = _run(
            residual_self_cooling=1.0,
            tol=float("inf"),
            n_rounds=15,
        )

        assert all(diag["accept_history"])
        assert len(best) == 8

    def test_return_final_table_exposes_terminal_state(self):
        common = dict(
            n_rounds=40, rho=0.3, mu=0.02, seed=9, n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]), tol=float("inf"),
            return_final_table=True,
        )
        best, diag = _run(**common)

        final = diag["final_table"]
        assert isinstance(final, pd.DataFrame)
        assert len(final) == 30
        assert list(final.columns) == list(best.columns)
        # 无门配置下最终表与 best 表一般不同（终点回漂）。
        assert not final.equals(best) or (
            diag["loss_history"][-1] == diag["best_loss"]
        )

    def test_current_state_contract_preserves_frozen_gate_free_trajectory(
        self, monkeypatch
    ):
        """Stage 0 仪表不能改变任何一轮表、RNG 或旧诊断。"""
        original_evolve_step = evolution_module.evolve_step
        post_round_sha256 = []

        def observed_evolve_step(*args, **kwargs):
            proposal = original_evolve_step(*args, **kwargs)
            post_round_sha256.append(_table_hash(proposal))
            return proposal

        monkeypatch.setattr(
            evolution_module, "evolve_step", observed_evolve_step
        )
        best, diag = _run(
            n_rounds=12,
            rho=0.3,
            mu=0.05,
            seed=0,
            n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]),
            tol=float("inf"),
            return_final_table=True,
        )

        assert post_round_sha256 == [
            "c1f7ece7e56f41ad64494a62a4d7bd6711649f640488c890e7caf8dcd0f6dfea",
            "cd6ca7f7e87752e0dbc1c1d90b371327596874cc761abff76b349dc4b1482802",
            "81c1533e77c455474ce568f1403c856216a1d546bf74823daf1ec33b6ca4d457",
            "7ec0bf79a09a6ef1901b9fc6f0677c5cf964dcde23933f2f7bb307d76017870f",
            "d5fb12f2e1724ac95a92c94296c6a8c6c4312c16529de42b994b44720d1c48a4",
            "26885c30bc6920ac45775bef73290e670364d2954a21ec522fca8239ba14032d",
            "82a1ba5016780fba50c6c72b4ed85096543129c44f5dc3f6fceb7b0ee1e9a1b9",
            "426eaf75e168e27cc0654ff81e0b09f4fe82e16add07ffee555ae0957627764c",
            "1f6c2ae5972a98990a4ee6b167c8cf76a9d461b784b3fd5bdaddaa19fe41f3fb",
            "1f6c2ae5972a98990a4ee6b167c8cf76a9d461b784b3fd5bdaddaa19fe41f3fb",
            "194d623a3bd1c851655d0c6a62bfbb7f11fbc7159a04bb428474e1975c3bd09f",
            "71a5760dad836b3cd749beebe8472c6716f8d54cfcf235addf8c52a7baab48b0",
        ]
        assert _table_hash(best) == (
            "d5fb12f2e1724ac95a92c94296c6a8c6c4312c16529de42b994b44720d1c48a4"
        )
        assert _table_hash(diag["final_table"]) == (
            "71a5760dad836b3cd749beebe8472c6716f8d54cfcf235addf8c52a7baab48b0"
        )
        assert diag["loss_history"] == [
            49.0, 39.0, 39.0, 17.5, 10.5, 1.0,
            1.5, 6.0, 3.0, 1.0, 1.0, 3.0,
        ]
        assert diag["accept_history"] == [True] * 12
        assert diag["primary_rng_state_sha256"] == (
            "81de95ec548a6dd03c5d6d20594c7a8767a06038aae79f7bd17d9ee847a9d499"
        )
        assert diag["state_evaluation_count"] == 12
        assert diag["candidate_evaluation_count"] == 12
        assert diag["best_loss"] == 1.0
        assert diag["best_loss_diagnostic_only"] == 1.0

        history = diag["current_state_metrics_history"]
        assert [row["state_index"] for row in history] == list(range(13))
        assert [row["round"] for row in history] == list(range(13))
        assert [row["phase"] for row in history] == [
            "initial", *(["post_round"] * 12)
        ]
        assert [row["current_squared_loss"] for row in history] == (
            pytest.approx([
                49.0, 39.0, 39.0, 17.5, 10.5, 1.0, 1.5,
                6.0, 3.0, 1.0, 1.0, 3.0, 2.5,
            ])
        )

        _, queries, _ = _tiny_problem()
        target = np.asarray([25.0, 6.0, 5.0])
        final_q = evaluate_table(diag["final_table"], queries)
        expected_l1 = compute_normalized_l1(target, final_q, 30)
        expected_loss = compute_loss(target, final_q)
        assert history[-1]["current_normalized_l1"] == expected_l1
        assert history[-1]["current_squared_loss"] == expected_loss
        assert diag["final_current_normalized_l1"] == expected_l1
        assert diag["final_current_squared_loss"] == expected_loss

    def test_final_table_absent_by_default(self):
        _, diag = _run()

        assert "final_table" not in diag

    def test_final_table_loss_is_authoritative_terminal_metric(self):
        # loss_history[-1] 记录的是最后一次 proposal 之前的状态；最后一轮
        # 接受后最终表已变化。终态统计必须从 final_table 重算（审查意见）。
        _, diag = _run(
            n_rounds=12, rho=0.3, mu=0.05, seed=0, n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]), tol=float("inf"),
            return_final_table=True,
        )

        schema, queries, _ = _tiny_problem()
        target = np.asarray([25.0, 6.0, 5.0])
        final_loss = compute_loss(
            target, evaluate_table(diag["final_table"], queries)
        )
        assert final_loss == pytest.approx(2.5)
        assert diag["loss_history"][-1] == pytest.approx(3.0)
        assert final_loss != diag["loss_history"][-1]

    def test_monotone_cooling_never_reheats(self):
        common = dict(
            n_rounds=60, rho=0.3, mu=0.02, seed=9, n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]), tol=float("inf"),
        )
        _, diag = _run(
            residual_self_cooling=1.0, self_cooling_monotone=True, **common
        )

        history = diag["self_cooling_history"]
        assert all(b <= a + 1e-12 for a, b in zip(history, history[1:]))
        assert diag["params"]["self_cooling_monotone"] is True

    def test_non_monotone_ablation_can_reheat(self):
        # 较大 mu 持续注入噪声：残差下降后回升，非单调冷却因子随之复燃。
        common = dict(
            n_rounds=80, rho=0.4, mu=0.2, seed=3, n_records=30,
            target=np.asarray([25.0, 6.0, 5.0]), tol=float("inf"),
        )
        _, diag = _run(residual_self_cooling=1.0, **common)

        history = diag["self_cooling_history"]
        assert any(b > a for a, b in zip(history, history[1:]))
        assert diag["params"]["self_cooling_monotone"] is False

    def test_invalid_monotone_flag_rejected(self):
        with pytest.raises(ValueError, match="self_cooling_monotone"):
            _run(residual_self_cooling=1.0, self_cooling_monotone=1)
