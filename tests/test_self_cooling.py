"""残差自冷却机制（Issue #44）的单元与集成测试。"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from table_diffevo.evolution import _self_cooling_factor, run_evolution
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

    def test_final_table_absent_by_default(self):
        _, diag = _run()

        assert "final_table" not in diag

    def test_final_table_loss_is_authoritative_terminal_metric(self):
        # loss_history[-1] 记录的是最后一次 proposal 之前的状态；最后一轮
        # 接受后最终表已变化。终态统计必须从 final_table 重算（审查意见）。
        from table_diffevo.objective import compute_loss
        from table_diffevo.queries import evaluate_table

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
