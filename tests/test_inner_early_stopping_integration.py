"""Wiring contracts for terminal-current A/B/C stopping in run_evolution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

import table_diffevo.evolution as evolution_module
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema


def _binary_problem() -> tuple[Schema, list[dict]]:
    schema = Schema(
        [
            AttributeBlock(
                name="x",
                type="categorical",
                description="x",
                values=[0, 1],
            )
        ]
    )
    queries = [{"conditions": [{"attribute": "x", "operator": "==", "value": 1}]}]
    return schema, queries


def _table_with_one_count(count: int, *, n_records: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"x": [1] * count + [0] * (n_records - count)})


def _install_scripted_kernel(
    monkeypatch,
    *,
    initial_count: int,
    proposal_counts: Sequence[int],
) -> dict[str, int]:
    initial = _table_with_one_count(initial_count)
    proposals = iter(proposal_counts)
    calls = {"step": 0}
    monkeypatch.setattr(
        evolution_module,
        "init_synthetic_table",
        lambda *args, **kwargs: initial.copy(deep=True),
    )

    def scripted_step(S, *args, return_diagnostics=False, **kwargs):
        calls["step"] += 1
        proposal = _table_with_one_count(next(proposals), n_records=len(S))
        if return_diagnostics:
            return proposal, {"participating_rows": len(S)}
        return proposal

    monkeypatch.setattr(evolution_module, "evolve_step", scripted_step)
    return calls


def _run_scripted(
    *,
    n_rounds: int,
    patience_ticks: int,
    candidate_budget: int | None = None,
):
    schema, queries = _binary_problem()
    return run_evolution(
        np.array([0.0]),
        queries,
        schema,
        n_records=4,
        n_rounds=n_rounds,
        seed=53,
        tol=float("inf"),
        max_retries=0,
        inner_early_stopping_patience_ticks=patience_ticks,
        candidate_budget=candidate_budget,
        return_final_table=True,
        log_every=100_000,
        device="numpy",
    )


def test_initial_A_returns_S0_before_any_proposal(monkeypatch) -> None:
    calls = _install_scripted_kernel(
        monkeypatch,
        initial_count=0,
        proposal_counts=(),
    )
    schema, queries = _binary_problem()

    output, diagnostics = run_evolution(
        np.array([0.0]),
        queries,
        schema,
        n_records=4,
        n_rounds=5,
        tol=float("inf"),
        inner_early_stopping_patience_ticks=6,
        return_final_table=True,
        log_every=100_000,
        device="numpy",
    )

    assert calls["step"] == 0
    assert diagnostics["rounds_run"] == 0
    assert diagnostics["termination_reason"] == "fit_target_reached"
    assert diagnostics["inner_complete"] is True
    assert diagnostics["output_table_identity"] == "terminal_current"
    pd.testing.assert_frame_equal(output, diagnostics["final_table"])
    pd.testing.assert_frame_equal(output, _table_with_one_count(0))


def test_B_observes_applied_current_and_beats_C_on_same_state(monkeypatch) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=2,
        proposal_counts=(1, 3, 2),
    )

    output, diagnostics = _run_scripted(
        n_rounds=3,
        patience_ticks=2,
    )

    assert diagnostics["termination_reason"] == "early_stopped"
    assert diagnostics["rounds_run"] == 3
    assert diagnostics["inner_complete"] is True
    assert diagnostics["best_loss_diagnostic_only"] == pytest.approx(0.5)
    assert diagnostics["final_current_squared_loss"] == pytest.approx(2.0)
    assert diagnostics["output_squared_loss"] == pytest.approx(2.0)
    decision = diagnostics["inner_early_stopping"]["last_decision"]
    assert decision["external_resource_cap_reached"] is True
    assert decision["termination_reason"] == "early_stopped"
    assert decision["best_loss_diagnostic_only"] == pytest.approx(0.5)
    assert decision["terminal_output_loss"] == pytest.approx(2.0)
    pd.testing.assert_frame_equal(output, _table_with_one_count(2))
    pd.testing.assert_frame_equal(output, diagnostics["final_table"])


def test_max_round_C_returns_terminal_current_instead_of_best(monkeypatch) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=2,
        proposal_counts=(1, 3),
    )

    output, diagnostics = _run_scripted(
        n_rounds=2,
        patience_ticks=6,
    )

    assert diagnostics["termination_reason"] == "resource_cap_reached"
    assert diagnostics["inner_complete"] is False
    assert diagnostics["best_loss_diagnostic_only"] == pytest.approx(0.5)
    assert diagnostics["output_squared_loss"] == pytest.approx(4.5)
    assert diagnostics["normalized_l1_error"] == pytest.approx(0.75)
    assert diagnostics[
        "normalized_l1_at_best_squared_loss_diagnostic_only"
    ] == pytest.approx(0.25)
    assert (
        diagnostics["inner_early_stopping"]["resource_cap_source_diagnostic_only"]
        == "max_rounds"
    )
    pd.testing.assert_frame_equal(output, _table_with_one_count(3))
    pd.testing.assert_frame_equal(output, diagnostics["final_table"])


def test_candidate_budget_is_forwarded_as_external_C_after_application(
    monkeypatch,
) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=2,
        proposal_counts=(3,),
    )

    output, diagnostics = _run_scripted(
        n_rounds=10,
        patience_ticks=6,
        candidate_budget=1,
    )

    assert diagnostics["candidate_budget_exhausted"] is True
    assert diagnostics["candidate_evaluation_count"] == 1
    assert diagnostics["termination_reason"] == "resource_cap_reached"
    assert (
        diagnostics["inner_early_stopping"]["resource_cap_source_diagnostic_only"]
        == "candidate_budget"
    )
    pd.testing.assert_frame_equal(output, _table_with_one_count(3))


def test_zero_round_budget_returns_initial_current_through_C(monkeypatch) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=2,
        proposal_counts=(),
    )
    schema, queries = _binary_problem()

    output, diagnostics = run_evolution(
        np.array([0.0]),
        queries,
        schema,
        n_records=4,
        n_rounds=0,
        tol=float("inf"),
        inner_early_stopping_patience_ticks=6,
        return_final_table=True,
        device="numpy",
    )

    assert diagnostics["rounds_run"] == 0
    assert diagnostics["termination_reason"] == "resource_cap_reached"
    assert diagnostics["inner_complete"] is False
    assert (
        diagnostics["inner_early_stopping"]["resource_cap_source_diagnostic_only"]
        == "max_rounds"
    )
    pd.testing.assert_frame_equal(output, _table_with_one_count(2))


@pytest.mark.parametrize(
    (
        "initial_count",
        "proposal_counts",
        "target_count",
        "n_rounds",
        "patience_ticks",
        "expected_reason",
    ),
    [
        (0, (), 0.0, 5, 6, "fit_target_reached"),
        (2, (1, 3, 2), 0.0, 3, 2, "early_stopped"),
        (2, (1, 3), 0.0, 2, 6, "resource_cap_reached"),
    ],
)
def test_a_b_c_termination_reasons_are_valid_stationarity_trace_values(
    monkeypatch,
    initial_count,
    proposal_counts,
    target_count,
    n_rounds,
    patience_ticks,
    expected_reason,
) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=initial_count,
        proposal_counts=proposal_counts,
    )
    schema, queries = _binary_problem()

    _, diagnostics = run_evolution(
        np.array([target_count]),
        queries,
        schema,
        n_records=4,
        n_rounds=n_rounds,
        seed=53,
        tol=float("inf"),
        max_retries=0,
        inner_early_stopping_patience_ticks=patience_ticks,
        record_stationarity_trace=True,
        log_every=100_000,
        device="numpy",
    )

    trace = diagnostics["stationarity_trace"]
    assert diagnostics["termination_reason"] == expected_reason
    assert trace.termination_reason == expected_reason
    assert trace.state_count == diagnostics["rounds_run"] + 1
    trace.validate()


def test_observer_does_not_change_the_gate_free_random_trajectory() -> None:
    schema, queries = _binary_problem()
    parameters = {
        "target": np.array([1.5]),
        "queries": queries,
        "schema": schema,
        "n_records": 4,
        "n_rounds": 4,
        "seed": 20260817,
        "rho": 0.5,
        "tol": float("inf"),
        "max_retries": 0,
        "return_final_table": True,
        "record_transition_clocks": True,
        "log_every": 100_000,
        "device": "numpy",
    }

    _, legacy = run_evolution(**parameters)
    output, observed = run_evolution(
        **parameters,
        inner_early_stopping_patience_ticks=100,
    )

    pd.testing.assert_frame_equal(legacy["final_table"], observed["final_table"])
    pd.testing.assert_frame_equal(output, observed["final_table"])
    assert (
        legacy["current_state_metrics_history"]
        == observed["current_state_metrics_history"]
    )
    assert legacy["transition_clock_history"] == observed["transition_clock_history"]
    assert legacy["accept_history"] == observed["accept_history"]
    assert legacy["primary_rng_state_sha256"] == observed["primary_rng_state_sha256"]
    assert legacy["termination_reason"] == "max_rounds"
    assert observed["termination_reason"] == "resource_cap_reached"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"inner_early_stopping_patience_ticks": 0}, "正整数"),
        ({"inner_early_stopping_patience_ticks": True}, "正整数"),
        (
            {
                "inner_early_stopping_patience_ticks": 6,
                "stop_on_exact_residual": False,
            },
            "A 必须开启",
        ),
        (
            {
                "inner_early_stopping_patience_ticks": 6,
                "residual_self_cooling": 1.0,
                "self_cooling_stop_ratio": 0.1,
            },
            "不与 self_cooling_stop_ratio 混用",
        ),
        (
            {
                "inner_early_stopping_patience_ticks": 6,
                "tol": 0.0,
            },
            "tol=\\+inf",
        ),
        (
            {
                "inner_early_stopping_patience_ticks": 6,
                "max_retries": 1,
            },
            "max_retries 必须为 0",
        ),
    ],
)
def test_rejects_invalid_or_conflicting_wiring(overrides, message) -> None:
    schema, queries = _binary_problem()
    parameters = {
        "target": np.array([0.0]),
        "queries": queries,
        "schema": schema,
        "n_records": 4,
        "n_rounds": 1,
        "tol": float("inf"),
        "device": "numpy",
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_evolution(**parameters)
