"""Fixed artificial counterfactual-tail matrix for the 3+3 stopper.

The generator always runs its full horizon. The stopper sees the completed
trace only afterwards, so its predicted stop cannot change the trajectory or
consume randomness. Losses after that predicted stop form the counterfactual
tail used to detect a missed strict improvement.
"""

from __future__ import annotations

import numpy as np

from table_diffevo.evolution import run_evolution
from table_diffevo.inner_stopping import (
    BestLossInnerStopper,
    BestLossStoppingConfig,
)
from table_diffevo.schema import AttributeBlock, Schema

CASES = (
    # Forty expected normalized-work units per row, at two participation scales.
    (20260816, 1.0, 40),
    (20260817, 1.0, 40),
    (20260818, 1.0, 40),
    (20260816, 0.25, 160),
    (20260817, 0.25, 160),
    (20260818, 0.25, 160),
)


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


def _run_full_trace(*, seed: int, rho: float, n_rounds: int):
    schema, queries, target = _artificial_binary_problem()
    _, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records=16,
        n_rounds=n_rounds,
        seed=seed,
        rho=rho,
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
    assert diagnostics["termination_reason"] == "max_rounds"
    assert diagnostics["rounds_run"] == n_rounds
    return diagnostics


def _classify_trace(*, seed: int, rho: float, n_rounds: int) -> dict:
    diagnostics = _run_full_trace(seed=seed, rho=rho, n_rounds=n_rounds)
    metrics = diagnostics["current_state_metrics_history"]
    clocks = diagnostics["transition_clock_history"]
    losses = [float(row["current_squared_loss"]) for row in metrics]

    stopper = BestLossInnerStopper(BestLossStoppingConfig(n_records=16))
    decision = stopper.observe_initial(losses[0])
    stop_state_index = 0 if decision.should_stop else None
    cumulative_rows = 0
    cumulative_rows_at_stop = 0

    for state, clock in zip(metrics[1:], clocks, strict=True):
        accepted_attempt = clock["accepted_attempt"]
        assert accepted_attempt == 1
        participating_rows = int(
            clock["attempts"][accepted_attempt - 1]["participating_rows"]
        )
        cumulative_rows += participating_rows
        if decision.should_stop:
            continue
        decision = stopper.observe_post_round(
            current_loss=float(state["current_squared_loss"]),
            participating_rows=participating_rows,
        )
        if decision.should_stop:
            stop_state_index = int(state["state_index"])
            cumulative_rows_at_stop = cumulative_rows

    assert stop_state_index is not None
    stop_best_loss = min(losses[: stop_state_index + 1])
    tail_losses = losses[stop_state_index + 1 :]
    tail_best_loss = min(tail_losses, default=stop_best_loss)
    assert decision.best_loss == stop_best_loss

    return {
        "seed": seed,
        "rho": rho,
        "reason": decision.termination_reason,
        "stop_state": stop_state_index,
        "stop_work": cumulative_rows_at_stop / 16,
        "tail_work": (cumulative_rows - cumulative_rows_at_stop) / 16,
        "stop_best": stop_best_loss,
        "tail_best": tail_best_loss,
        "missed_strict_improvement": tail_best_loss < stop_best_loss,
    }


def _format_rows(rows: list[dict]) -> str:
    return "\n".join(
        (
            f"seed={row['seed']} rho={row['rho']} reason={row['reason']} "
            f"stop_state={row['stop_state']} stop_work={row['stop_work']:.3f} "
            f"tail_work={row['tail_work']:.3f} stop_best={row['stop_best']:.3f} "
            f"tail_best={row['tail_best']:.3f}"
        )
        for row in rows
    )


def test_fixed_counterfactual_tail_matrix_rejects_three_plus_three() -> None:
    rows = [
        _classify_trace(seed=seed, rho=rho, n_rounds=n_rounds)
        for seed, rho, n_rounds in CASES
    ]

    # Each stopped prefix must retain a substantive, independently generated tail.
    insufficient_tail = [row for row in rows if row["tail_work"] < 10.0]
    assert not insufficient_tail, _format_rows(insufficient_tail)

    # A fail-closed cap is not evidence that the optimizer completed normally.
    resource_limited = [row for row in rows if row["reason"] == "resource_cap_reached"]
    assert not resource_limited, _format_rows(resource_limited)

    # The pre-registered candidate gate failed. Preserve the exact counterexamples
    # so that a later edit cannot silently relabel 3+3 as safe.
    premature = [row for row in rows if row["missed_strict_improvement"]]
    observed_counterexamples = [
        (
            row["seed"],
            row["rho"],
            row["stop_state"],
            row["stop_work"],
            row["stop_best"],
            row["tail_best"],
        )
        for row in premature
    ]
    assert observed_counterexamples == [
        (20260816, 1.0, 12, 12.0, 3.0, 2.0),
        (20260817, 1.0, 20, 20.0, 3.0, 2.0),
        (20260818, 1.0, 8, 8.0, 3.0, 1.5),
        (20260816, 0.25, 27, 7.0625, 3.0, 1.0),
    ], _format_rows(rows)
