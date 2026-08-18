"""Main-loop wiring contracts for the frozen two-level alpha schedule."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

import table_diffevo.evolution as evolution_module
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.stall_escape_alpha import (
    STALL_ESCAPE_ALPHA_SCHEDULE_MODE,
)


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
    queries = [
        {"conditions": [{"attribute": "x", "operator": "==", "value": 1}]}
    ]
    return schema, queries


def _table_with_one_count(count: int, *, n_records: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"x": [1] * count + [0] * (n_records - count)})


def _install_scripted_kernel(
    monkeypatch,
    *,
    initial_count: int,
    proposal_counts: Sequence[int],
) -> None:
    initial = _table_with_one_count(initial_count)
    proposals = iter(proposal_counts)
    monkeypatch.setattr(
        evolution_module,
        "init_synthetic_table",
        lambda *args, **kwargs: initial.copy(deep=True),
    )

    def scripted_step(S, *args, return_diagnostics=False, **kwargs):
        proposal = _table_with_one_count(next(proposals), n_records=len(S))
        if return_diagnostics:
            return proposal, {"participating_rows": len(S)}
        return proposal

    monkeypatch.setattr(evolution_module, "evolve_step", scripted_step)


def _adaptive_parameters(*, n_rounds: int) -> dict:
    schema, queries = _binary_problem()
    return {
        "target": np.array([0.0]),
        "queries": queries,
        "schema": schema,
        "n_records": 4,
        "n_rounds": n_rounds,
        "seed": 53,
        "tol": float("inf"),
        "max_retries": 0,
        "distance_mode": "geometric",
        "alpha_schedule_mode": STALL_ESCAPE_ALPHA_SCHEDULE_MODE,
        "inner_early_stopping_patience_ticks": 6,
        "record_transition_clocks": True,
        "log_every": 100_000,
        "device": "numpy",
    }


def test_main_loop_uses_frozen_16_16_12_12_16_16_timeline(
    monkeypatch,
) -> None:
    _install_scripted_kernel(
        monkeypatch,
        initial_count=2,
        proposal_counts=(3, 3, 3, 3, 3, 3),
    )

    _, diagnostics = run_evolution(**_adaptive_parameters(n_rounds=10))

    assert diagnostics["alpha_history"] == [
        16.0,
        16.0,
        12.0,
        12.0,
        16.0,
        16.0,
    ]
    assert diagnostics["termination_reason"] == "early_stopped"
    adaptive = diagnostics["adaptive_alpha"]
    assert adaptive["enabled"] is True
    assert adaptive["schedule_mode"] == STALL_ESCAPE_ALPHA_SCHEDULE_MODE
    assert adaptive["config"] == {
        "normal_alpha": 16.0,
        "escape_alpha": 12.0,
        "stall_trigger_ticks": 2,
        "escape_duration_ticks": 2,
    }
    assert adaptive["escape_count"] == 1
    assert len(adaptive["observation_history"]) == 6
    assert adaptive["observation_history"][1]["events"] == (
        "escape_started",
    )
    assert adaptive["observation_history"][3]["events"] == (
        "escape_tick_completed",
        "escape_completed",
    )
    assert adaptive["last_observation"]["events"] == (
        "termination_observed",
    )
    assert diagnostics["params"]["fixed_alpha"] is None
    assert diagnostics["params"]["adaptive_alpha_config"] == adaptive[
        "config"
    ]


def test_before_trigger_matches_fixed_alpha_16_without_extra_rng() -> None:
    schema, queries = _binary_problem()
    common = {
        "target": np.array([1.5]),
        "queries": queries,
        "schema": schema,
        "n_records": 4,
        "n_rounds": 1,
        "seed": 20260818,
        "rho": 0.5,
        "tol": float("inf"),
        "max_retries": 0,
        "distance_mode": "geometric",
        "inner_early_stopping_patience_ticks": 6,
        "record_transition_clocks": True,
        "return_final_table": True,
        "log_every": 100_000,
        "device": "numpy",
    }

    fixed_table, fixed = run_evolution(
        **common,
        alpha_schedule_mode="fixed",
        fixed_alpha=16.0,
    )
    adaptive_table, adaptive = run_evolution(
        **common,
        alpha_schedule_mode=STALL_ESCAPE_ALPHA_SCHEDULE_MODE,
    )

    pd.testing.assert_frame_equal(fixed_table, adaptive_table)
    pd.testing.assert_frame_equal(fixed["final_table"], adaptive["final_table"])
    assert fixed["initial_table_sha256"] == adaptive["initial_table_sha256"]
    assert fixed["primary_rng_post_initialization_state_sha256"] == adaptive[
        "primary_rng_post_initialization_state_sha256"
    ]
    assert fixed["primary_rng_state_sha256"] == adaptive[
        "primary_rng_state_sha256"
    ]
    assert fixed["transition_clock_history"] == adaptive[
        "transition_clock_history"
    ]
    assert fixed["current_state_metrics_history"] == adaptive[
        "current_state_metrics_history"
    ]
    assert fixed["alpha_history"] == adaptive["alpha_history"] == [16.0]


def test_total_round_budget_does_not_change_common_generated_prefix() -> None:
    common = _adaptive_parameters(n_rounds=4)
    common.update({
        "target": np.array([1.5]),
        "rho": 0.5,
        "return_final_table": True,
    })

    short_table, short = run_evolution(**common)
    common["n_rounds"] = 8
    _, long = run_evolution(**common)

    assert short["alpha_history"] == long["alpha_history"][:4]
    assert short["transition_clock_history"] == long[
        "transition_clock_history"
    ][:4]
    assert short["current_state_metrics_history"] == long[
        "current_state_metrics_history"
    ][:5]
    assert short["primary_rng_state_sha256"] == long[
        "transition_clock_history"
    ][3]["primary_rng_state_sha256"]
    assert short["adaptive_alpha"]["observation_history"][:3] == long[
        "adaptive_alpha"
    ]["observation_history"][:3]
    assert short["adaptive_alpha"]["observation_history"][3][
        "termination_reason"
    ] == "resource_cap_reached"
    assert long["adaptive_alpha"]["observation_history"][3][
        "termination_reason"
    ] == "in_progress"
    assert evolution_module._table_sha256(short_table) == short[
        "transition_clock_history"
    ][3]["post_current_table_sha256"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"inner_early_stopping_patience_ticks": None}, "要求启用"),
        ({"inner_early_stopping_patience_ticks": 5}, "patience_ticks=6"),
        ({"distance_mode": "linear"}, "geometric"),
        ({"fixed_alpha": 16.0}, "只允许"),
    ],
)
def test_main_loop_rejects_conflicting_schedule_wiring(
    overrides,
    message,
) -> None:
    parameters = _adaptive_parameters(n_rounds=1)
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_evolution(**parameters)


def test_horizon_invariant_guard_accepts_stall_escape_mode() -> None:
    parameters = _adaptive_parameters(n_rounds=1)
    parameters["horizon_invariant"] = True

    _, diagnostics = run_evolution(**parameters)

    assert diagnostics["params"]["horizon_invariant"] is True
    assert diagnostics["params"]["alpha_schedule_mode"] == (
        STALL_ESCAPE_ALPHA_SCHEDULE_MODE
    )
