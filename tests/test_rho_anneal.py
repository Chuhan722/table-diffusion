"""时间驱动几何 rho 退火（Issue #44 机制迭代）的单元与集成测试。"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from table_diffevo.evolution import run_evolution
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


class TestRhoAnnealDefaultEquivalence:
    def test_default_none_keeps_trajectory_and_schedule_constant(self):
        best_default, diag_default = _run()
        best_explicit, diag_explicit = _run(rho_anneal_end=None)

        assert diag_default["loss_history"] == diag_explicit["loss_history"]
        assert _table_hash(best_default) == _table_hash(best_explicit)
        assert diag_default["rho_schedule_history"] == (
            [0.1] * len(diag_default["loss_history"])
        )
        assert diag_default["params"]["rho_anneal_end"] is None

    def test_end_equal_to_rho_matches_constant_trajectory(self):
        _, diag_constant = _run(rho=0.1)
        _, diag_anneal = _run(rho=0.1, rho_anneal_end=0.1)

        assert diag_anneal["loss_history"] == diag_constant["loss_history"]
        assert diag_anneal["rho_schedule_history"] == pytest.approx(
            [0.1] * len(diag_anneal["loss_history"])
        )


class TestRhoAnnealSchedule:
    def test_geometric_interpolation_endpoints_and_monotone(self):
        rounds = 10
        _, diag = _run(rho=0.1, rho_anneal_end=0.01, n_rounds=rounds)

        schedule = diag["rho_schedule_history"]
        assert len(schedule) == len(diag["loss_history"])
        assert schedule[0] == pytest.approx(0.1)
        if len(schedule) == rounds:  # 无早停时终点精确到达 end
            assert schedule[-1] == pytest.approx(0.01)
        assert all(
            later <= earlier + 1e-12
            for earlier, later in zip(schedule, schedule[1:])
        )

    def test_geometric_shape_matches_closed_form(self):
        rounds = 8
        _, diag = _run(
            rho=0.08, rho_anneal_end=0.005, n_rounds=rounds, tol=float("inf")
        )
        schedule = diag["rho_schedule_history"]
        expected = [
            0.08 * (0.005 / 0.08) ** (t / (rounds - 1))
            for t in range(len(schedule))
        ]
        assert schedule == pytest.approx(expected)

    def test_schedule_recorded_in_params(self):
        _, diag = _run(rho=0.1, rho_anneal_end=0.02)
        assert diag["params"]["rho_anneal_end"] == pytest.approx(0.02)

    def test_single_round_uses_endpoint(self):
        # n_rounds=1 时 progress=1.0，调度立即位于终点。
        _, diag = _run(rho=0.1, rho_anneal_end=0.01, n_rounds=1)
        schedule = diag["rho_schedule_history"]
        if schedule:
            assert schedule[0] == pytest.approx(0.01)

    def test_combines_with_self_cooling_multiplicatively(self):
        # 组合时冷却因子乘在 rho_t 上：accepted_rho <= rho_t 逐轮成立。
        _, diag = _run(
            rho=0.1,
            rho_anneal_end=0.01,
            residual_self_cooling=1.0,
            tol=float("inf"),
            n_rounds=15,
        )
        schedule = diag["rho_schedule_history"]
        cooling = diag["self_cooling_history"]
        accepted = diag["accepted_rho_history"]
        for rho_t, c_t, acc in zip(schedule, cooling, accepted):
            if acc is not None:
                assert acc == pytest.approx(rho_t * c_t)


class TestRhoAnnealValidation:
    @pytest.mark.parametrize("bad", [
        0.0, -0.01, 0.2, float("nan"), float("inf"), True, "0.01",
    ])
    def test_rejects_invalid_values(self, bad):
        with pytest.raises((ValueError, TypeError)):
            _run(rho=0.1, rho_anneal_end=bad)

    def test_rejects_end_above_rho(self):
        with pytest.raises(ValueError, match="rho_anneal_end"):
            _run(rho=0.01, rho_anneal_end=0.05)
