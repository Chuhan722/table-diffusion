"""Read-only shadow replay against an unmodified gate-free generator run."""

import copy
import hashlib

import numpy as np
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.inner_stopping import (
    BestLossInnerStopper,
    BestLossStoppingConfig,
)
from table_diffevo.schema import AttributeBlock, Schema


def _artificial_binary_problem():
    schema = Schema(
        [
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b", "c")
        ]
    )
    queries = [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "c", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
        {
            "conditions": [
                {"attribute": "b", "operator": "==", "value": 1},
                {"attribute": "c", "operator": "==", "value": 1},
            ]
        },
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
                {"attribute": "c", "operator": "==", "value": 1},
            ]
        },
    ]
    target = np.array([8, 8, 8, 4, 4, 2], dtype=float)
    return schema, queries, target


def _table_sha256(frame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_unmodified_gate_free_run_can_be_replayed_in_loss_only_shadow() -> None:
    schema, queries, target = _artificial_binary_problem()
    n_records = 16
    best_table, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records=n_records,
        n_rounds=25,
        seed=20260816,
        rho=1.0,
        eta=0.45,
        mu=0.02,
        tol=float("inf"),
        max_retries=0,
        distance_mode="geometric",
        alpha_schedule_mode="fixed",
        fixed_alpha=6.0,
        residual_directed_diffusion=True,
        diffusion_direction_strength=0.8,
        diffusion_direction_normalization="fixed",
        diffusion_direction_reference_scale=1.25,
        diffusion_direction_logit_clip=9.0,
        horizon_invariant=True,
        stop_on_exact_residual=False,
        record_transition_clocks=True,
        device="numpy",
        log_every=100_000,
    )

    metrics = diagnostics["current_state_metrics_history"]
    clocks = diagnostics["transition_clock_history"]
    assert diagnostics["termination_reason"] == "max_rounds"
    assert diagnostics["rounds_run"] == 25
    assert diagnostics["accept_history"] == [True] * 25
    assert len(metrics) == 26
    assert len(clocks) == 25

    # The replay material deliberately excludes normalized L1 and every table.
    loss_only_states = [
        {
            "state_index": row["state_index"],
            "round": row["round"],
            "current_squared_loss": row["current_squared_loss"],
        }
        for row in metrics
    ]
    assert all("current_normalized_l1" not in row for row in loss_only_states)

    state_hashes = [diagnostics["initial_table_sha256"]] + [
        clock["post_current_table_sha256"] for clock in clocks
    ]
    assert all(state_hash is not None for state_hash in state_hashes)
    full_losses = [row["current_squared_loss"] for row in loss_only_states]
    full_best_index = min(
        range(len(full_losses)),
        key=lambda index: (full_losses[index], index),
    )
    assert full_best_index == 6
    assert diagnostics["best_loss"] == pytest.approx(full_losses[full_best_index])
    assert _table_sha256(best_table) == state_hashes[full_best_index]

    protected_diagnostics = copy.deepcopy(
        {
            "current_state_metrics_history": metrics,
            "transition_clock_history": clocks,
            "candidate_evaluation_count": diagnostics["candidate_evaluation_count"],
            "primary_rng_state_sha256": diagnostics["primary_rng_state_sha256"],
        }
    )

    stopper = BestLossInnerStopper(BestLossStoppingConfig(n_records=n_records))
    initial = stopper.observe_initial(loss_only_states[0]["current_squared_loss"])
    assert initial.termination_reason == "in_progress"

    cumulative_rows = 0
    shadow_best_index = 0
    shadow_decision = initial
    shadow_stop_state_index = None
    candidate_decision = None
    delayed_improvement_decision = None
    shadow_prefix_losses = [initial.current_loss]
    for state, clock in zip(loss_only_states[1:], clocks, strict=True):
        assert clock["state_index"] == state["state_index"]
        assert clock["round"] == state["round"]
        accepted_attempt = clock["accepted_attempt"]
        assert accepted_attempt == 1
        participating_rows = clock["attempts"][accepted_attempt - 1][
            "participating_rows"
        ]

        cumulative_rows += participating_rows
        shadow_prefix_losses.append(state["current_squared_loss"])
        shadow_decision = stopper.observe_post_round(
            current_loss=state["current_squared_loss"],
            participating_rows=participating_rows,
        )
        if shadow_decision.best_updated:
            shadow_best_index = state["state_index"]

        expected_best_index = min(
            range(len(shadow_prefix_losses)),
            key=lambda index: (shadow_prefix_losses[index], index),
        )
        assert shadow_best_index == expected_best_index
        assert shadow_decision.best_loss == pytest.approx(
            shadow_prefix_losses[expected_best_index]
        )
        assert shadow_decision.cumulative_participating_rows == (cumulative_rows)
        assert shadow_decision.normalized_work == pytest.approx(
            cumulative_rows / n_records
        )
        assert shadow_decision.completed_work_windows == (cumulative_rows // n_records)

        if state["state_index"] == 3:
            candidate_decision = shadow_decision
        if state["state_index"] == 6:
            delayed_improvement_decision = shadow_decision

        if shadow_decision.should_stop:
            shadow_stop_state_index = state["state_index"]
            break

    assert shadow_prefix_losses == [
        5.0,
        19.5,
        9.5,
        10.5,
        16.5,
        8.0,
        3.0,
        7.0,
        8.5,
        9.5,
        5.0,
        10.5,
        12.0,
    ]
    assert candidate_decision is not None
    assert candidate_decision.should_stop is False
    assert candidate_decision.best_loss == 5.0
    assert candidate_decision.consecutive_no_progress_windows == 3
    assert delayed_improvement_decision is not None
    assert delayed_improvement_decision.best_updated is True
    assert delayed_improvement_decision.best_loss == 3.0
    assert delayed_improvement_decision.consecutive_no_progress_windows == 0
    assert shadow_stop_state_index == 12
    assert diagnostics["rounds_run"] > shadow_stop_state_index
    assert shadow_decision.termination_reason == "optimization_stalled"
    assert shadow_decision.best_loss == 3.0
    assert shadow_decision.current_loss == 12.0
    assert shadow_decision.normalized_work == 12.0
    assert shadow_decision.consecutive_no_progress_windows == 6
    assert shadow_best_index == 6
    assert state_hashes[shadow_best_index] == _table_sha256(best_table)
    assert min(full_losses[13:]) >= shadow_decision.best_loss
    # Offline replay is read-only: it cannot mutate the completed run.
    assert metrics == protected_diagnostics["current_state_metrics_history"]
    assert clocks == protected_diagnostics["transition_clock_history"]
    assert (
        diagnostics["candidate_evaluation_count"]
        == (protected_diagnostics["candidate_evaluation_count"])
    )
    assert (
        diagnostics["primary_rng_state_sha256"]
        == (protected_diagnostics["primary_rng_state_sha256"])
    )
