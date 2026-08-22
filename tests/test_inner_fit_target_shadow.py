"""Offline feasibility check for the one-count query-RMSE target."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema

N_RECORDS = 16
QUERY_COUNT_RMSE_TARGET = 1.0
MAX_NORMALIZED_WORK = 20.0
CASES = (
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


def _run_trace(*, seed: int, rho: float, n_rounds: int):
    schema, queries, target = _artificial_binary_problem()
    best_table, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records=N_RECORDS,
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
        log_every=0,
    )
    assert diagnostics["termination_reason"] == "max_rounds"
    assert diagnostics["rounds_run"] == n_rounds
    return queries, target, best_table, diagnostics


def _applied_participating_rows(clock: dict) -> int:
    accepted_attempt = clock["accepted_attempt"]
    assert accepted_attempt == 1
    return int(clock["attempts"][accepted_attempt - 1]["participating_rows"])


def _first_qualified(loss_only_states: list[dict], clocks: list[dict]) -> dict:
    query_count = 6
    qualifying_loss = 0.5 * query_count * QUERY_COUNT_RMSE_TARGET**2
    cumulative_rows = 0
    best_loss = math.inf

    for position, state in enumerate(loss_only_states):
        if position > 0:
            cumulative_rows += _applied_participating_rows(clocks[position - 1])
        best_loss = min(best_loss, float(state["current_squared_loss"]))
        count_rmse = math.sqrt(2.0 * best_loss / query_count)
        if count_rmse <= QUERY_COUNT_RMSE_TARGET:
            return {
                "state_index": int(state["state_index"]),
                "best_loss": best_loss,
                "count_rmse": count_rmse,
                "normalized_work": cumulative_rows / N_RECORDS,
                "qualifying_loss": qualifying_loss,
            }

    raise AssertionError("the fixed full trace never reached the fit target")


def _table_sha256(frame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _check_case(*, seed: int, rho: float, n_rounds: int) -> dict:
    queries, target, full_best_table, full = _run_trace(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
    )
    metrics = full["current_state_metrics_history"]
    clocks = full["transition_clock_history"]
    loss_only_states = [
        {
            "state_index": row["state_index"],
            "current_squared_loss": row["current_squared_loss"],
        }
        for row in metrics
    ]
    assert all("current_normalized_l1" not in row for row in loss_only_states)
    first = _first_qualified(loss_only_states, clocks)
    assert first["normalized_work"] <= MAX_NORMALIZED_WORK
    assert first["state_index"] > 0

    # Horizon invariance lets a prefix rerun materialize exactly the checkpoint
    # selected from loss alone, without adding an online stopping path.
    _, prefix_target, first_table, prefix = _run_trace(
        seed=seed,
        rho=rho,
        n_rounds=first["state_index"],
    )
    assert np.array_equal(prefix_target, target)
    assert (
        prefix["current_state_metrics_history"] == metrics[: first["state_index"] + 1]
    )
    assert prefix["transition_clock_history"] == clocks[: first["state_index"]]
    assert prefix["best_loss"] == pytest.approx(first["best_loss"])

    state_hashes = [full["initial_table_sha256"]] + [
        clock["post_current_table_sha256"] for clock in clocks
    ]
    assert _table_sha256(first_table) == state_hashes[first["state_index"]]

    # The checkpoint is fixed before either of these offline metrics is computed.
    first_answers = evaluate_table(first_table, queries)
    recomputed_loss = compute_squared_loss(target, first_answers)
    offline_l1 = compute_normalized_l1(target, first_answers, N_RECORDS)
    count_errors = target - first_answers
    count_rmse = float(np.sqrt(np.mean(np.square(count_errors))))
    residual_rmse = float(np.sqrt(np.mean(np.square(count_errors / N_RECORDS))))

    assert recomputed_loss == pytest.approx(first["best_loss"])
    assert count_rmse == pytest.approx(first["count_rmse"])
    assert residual_rmse == pytest.approx(count_rmse / N_RECORDS)
    assert count_rmse <= QUERY_COUNT_RMSE_TARGET
    assert offline_l1 <= QUERY_COUNT_RMSE_TARGET / N_RECORDS

    full_answers = evaluate_table(full_best_table, queries)
    full_offline_l1 = compute_normalized_l1(target, full_answers, N_RECORDS)
    return {
        "seed": seed,
        "rho": rho,
        "state": first["state_index"],
        "work": first["normalized_work"],
        "loss": recomputed_loss,
        "count_rmse": count_rmse,
        "offline_l1": offline_l1,
        "max_count_error": float(np.max(np.abs(count_errors))),
        "full_best_loss": float(full["best_loss"]),
        "full_offline_l1": full_offline_l1,
    }


def _format_rows(rows: list[dict]) -> str:
    return "\n".join(
        (
            f"seed={row['seed']} rho={row['rho']} state={row['state']} "
            f"work={row['work']:.4f} loss={row['loss']:.1f} "
            f"rmse={row['count_rmse']:.4f} l1={row['offline_l1']:.6f} "
            f"max_error={row['max_count_error']:.1f} "
            f"full_loss={row['full_best_loss']:.1f} "
            f"full_l1={row['full_offline_l1']:.6f}"
        )
        for row in rows
    )


def test_one_count_rmse_target_qualifies_all_fixed_traces_before_cap() -> None:
    rows = [
        _check_case(seed=seed, rho=rho, n_rounds=n_rounds)
        for seed, rho, n_rounds in CASES
    ]

    assert len(rows) == len(CASES)
    assert all(row["work"] <= MAX_NORMALIZED_WORK for row in rows), _format_rows(rows)
    assert all(row["count_rmse"] <= QUERY_COUNT_RMSE_TARGET for row in rows), (
        _format_rows(rows)
    )
    assert all(
        row["offline_l1"] <= QUERY_COUNT_RMSE_TARGET / N_RECORDS for row in rows
    ), _format_rows(rows)

    # Preserve the first observed fixed-matrix result, including the fact that
    # an average RMSE target does not bound every individual query by one count.
    observed = [
        (
            row["seed"],
            row["rho"],
            row["state"],
            row["work"],
            row["loss"],
            row["offline_l1"],
            row["max_count_error"],
            row["full_best_loss"],
            row["full_offline_l1"],
        )
        for row in rows
    ]
    expected = [
        (20260816, 1.0, 6, 6.0, 3.0, 1 / 24, 2.0, 2.0, 1 / 48),
        (20260817, 1.0, 14, 14.0, 3.0, 1 / 24, 2.0, 2.0, 1 / 24),
        (20260818, 1.0, 2, 2.0, 3.0, 1 / 24, 2.0, 1.5, 1 / 32),
        (20260816, 0.25, 2, 0.5625, 3.0, 1 / 24, 2.0, 1.0, 1 / 48),
        (20260817, 0.25, 2, 0.75, 3.0, 1 / 24, 2.0, 0.5, 1 / 96),
        (20260818, 0.25, 9, 2.5625, 1.5, 1 / 32, 1.0, 0.0, 0.0),
    ]
    assert len(observed) == len(expected)
    for actual, frozen in zip(observed, expected):
        assert actual[:3] == frozen[:3]
        assert actual[3:] == pytest.approx(frozen[3:])
