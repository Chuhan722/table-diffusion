"""Known-trajectory continuation diagnostic for terminal-output early stopping.

This is not an independent validation.  The six artificial traces and the
six-no-progress-tick rule were inspected in earlier Issue #53 work.  The sole
purpose here is to freeze the terminal current state at the fixed early-stop
boundary, then describe what happens at several relative-work checkpoints if
an offline shadow copy continues.  No single trace-horizon endpoint is treated
as a quality reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from statistics import mean, median
from typing import Any

import numpy as np

from table_diffevo.evolution import run_evolution
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema

ANALYSIS_ID = "issue53_terminal_early_stop_continuation_development_v2"
N_RECORDS = 16
NO_PROGRESS_PATIENCE_TICKS = 6
CONTINUATION_PATIENCE_MULTIPLES = (1, 2, 4)
CASES = (
    (20260816, 1.0, 40),
    (20260817, 1.0, 40),
    (20260818, 1.0, 40),
    (20260816, 0.25, 160),
    (20260817, 0.25, 160),
    (20260818, 0.25, 160),
)


@dataclass(frozen=True)
class EarlyStopDecision:
    reason: str
    state_index: int
    normalized_work: float
    completed_work_ticks: int
    consecutive_no_progress_ticks: int


@dataclass(frozen=True)
class ContinuationCheckpoint:
    patience_multiple: int
    requested_extra_work: float
    target_normalized_work: float
    status: str
    state_index: int | None
    actual_normalized_work: float | None
    actual_extra_work: float | None


def _require(condition: object, message: str) -> None:
    """Keep audit checks active even when Python runs with ``-O``."""

    if not condition:
        raise AssertionError(message)


def _artificial_binary_problem() -> tuple[Schema, list[dict], np.ndarray]:
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


def _trace_run_kwargs(*, seed: int, rho: float, n_rounds: int) -> dict[str, Any]:
    """Return the one frozen generator configuration shared by both paths."""

    return {
        "n_records": N_RECORDS,
        "n_rounds": n_rounds,
        "seed": seed,
        "rho": rho,
        "eta": 0.45,
        "mu": 0.02,
        "tol": float("inf"),
        "max_retries": 0,
        "distance_mode": "geometric",
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 6.0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 0.8,
        "diffusion_direction_normalization": "fixed",
        "diffusion_direction_reference_scale": 1.25,
        "diffusion_direction_logit_clip": 9.0,
        "horizon_invariant": True,
        "record_transition_clocks": True,
        "return_final_table": True,
        "device": "numpy",
        "log_every": 100_000,
    }


def _run_evolution_quietly(*args, **kwargs):
    """Keep the reproducible report on stdout as valid JSON."""

    with redirect_stdout(StringIO()):
        return run_evolution(*args, **kwargs)


def _run_trace(*, seed: int, rho: float, n_rounds: int):
    schema, queries, target = _artificial_binary_problem()
    _, diagnostics = _run_evolution_quietly(
        target,
        queries,
        schema,
        stop_on_exact_residual=False,
        **_trace_run_kwargs(seed=seed, rho=rho, n_rounds=n_rounds),
    )
    _require(
        diagnostics["termination_reason"] == "max_rounds",
        "full development trace must terminate at max_rounds",
    )
    _require(
        diagnostics["rounds_run"] == n_rounds,
        "full development trace must run the requested horizon",
    )
    return queries, target, diagnostics


def _run_online_early_stop(*, seed: int, rho: float, n_rounds: int):
    schema, queries, target = _artificial_binary_problem()
    output, diagnostics = _run_evolution_quietly(
        target,
        queries,
        schema,
        stop_on_exact_residual=True,
        inner_early_stopping_patience_ticks=NO_PROGRESS_PATIENCE_TICKS,
        **_trace_run_kwargs(seed=seed, rho=rho, n_rounds=n_rounds),
    )
    return queries, target, output, diagnostics


def replay_fixed_early_stop(
    losses: Sequence[float],
    participating_rows: Sequence[int],
    *,
    n_records: int = N_RECORDS,
    patience_ticks: int = NO_PROGRESS_PATIENCE_TICKS,
) -> EarlyStopDecision:
    """Replay A=zero residual and B=fixed no-new-best patience.

    Historical best is used only to decide whether a work tick contained
    progress.  The decision points to the terminal current state; it does not
    select or return the historical best state.
    """

    if len(losses) != len(participating_rows) + 1:
        raise ValueError("losses must contain the initial state plus one per round")
    if n_records <= 0 or patience_ticks <= 0:
        raise ValueError("n_records and patience_ticks must be positive")
    if not losses:
        raise ValueError("losses must not be empty")
    normalized_losses = [float(loss) for loss in losses]
    if any(not math.isfinite(loss) or loss < 0.0 for loss in normalized_losses):
        raise ValueError("losses must be finite and nonnegative")
    normalized_rows = [int(rows) for rows in participating_rows]
    if any(rows < 0 or rows > n_records for rows in normalized_rows):
        raise ValueError("participating rows must lie in [0, n_records]")

    best_loss = normalized_losses[0]
    if best_loss == 0.0:
        return EarlyStopDecision(
            reason="fit_target_reached",
            state_index=0,
            normalized_work=0.0,
            completed_work_ticks=0,
            consecutive_no_progress_ticks=0,
        )

    cumulative_rows = 0
    completed_ticks = 0
    consecutive_no_progress = 0
    tick_had_new_best = False

    for state_index, (current_loss, rows) in enumerate(
        zip(normalized_losses[1:], normalized_rows),
        start=1,
    ):
        if current_loss < best_loss:
            best_loss = current_loss
            tick_had_new_best = True

        cumulative_rows += rows
        if current_loss == 0.0:
            return EarlyStopDecision(
                reason="fit_target_reached",
                state_index=state_index,
                normalized_work=cumulative_rows / n_records,
                completed_work_ticks=cumulative_rows // n_records,
                consecutive_no_progress_ticks=consecutive_no_progress,
            )

        newly_completed_ticks = cumulative_rows // n_records - completed_ticks
        if newly_completed_ticks not in (0, 1):
            raise ValueError("one round cannot cross multiple natural work ticks")
        if newly_completed_ticks == 0:
            continue

        completed_ticks += 1
        if tick_had_new_best:
            consecutive_no_progress = 0
        else:
            consecutive_no_progress += 1
        tick_had_new_best = False

        if consecutive_no_progress >= patience_ticks:
            return EarlyStopDecision(
                reason="early_stopped",
                state_index=state_index,
                normalized_work=cumulative_rows / n_records,
                completed_work_ticks=completed_ticks,
                consecutive_no_progress_ticks=consecutive_no_progress,
            )

    return EarlyStopDecision(
        reason="reference_horizon_reached",
        state_index=len(normalized_losses) - 1,
        normalized_work=cumulative_rows / n_records,
        completed_work_ticks=completed_ticks,
        consecutive_no_progress_ticks=consecutive_no_progress,
    )


def locate_continuation_checkpoints(
    participating_rows: Sequence[int],
    *,
    stop_state_index: int,
    n_records: int = N_RECORDS,
    patience_ticks: int = NO_PROGRESS_PATIENCE_TICKS,
    patience_multiples: Sequence[int] = CONTINUATION_PATIENCE_MULTIPLES,
) -> list[ContinuationCheckpoint]:
    """Locate first real states at relative-work checkpoints after B.

    The helper only reads applied participating-row counts.  It never reads
    loss, L1, a historical-best table, or a trace-horizon terminal value.
    Missing checkpoints are right-censored instead of being replaced by the
    last available state.
    """

    if n_records <= 0 or patience_ticks <= 0:
        raise ValueError("n_records and patience_ticks must be positive")
    normalized_rows = [int(rows) for rows in participating_rows]
    if any(rows < 0 or rows > n_records for rows in normalized_rows):
        raise ValueError("participating rows must lie in [0, n_records]")
    if stop_state_index < 0 or stop_state_index > len(normalized_rows):
        raise ValueError("stop_state_index must identify an available state")

    normalized_multiples = [int(value) for value in patience_multiples]
    if any(value <= 0 for value in normalized_multiples):
        raise ValueError("patience multiples must be positive")
    if normalized_multiples != sorted(set(normalized_multiples)):
        raise ValueError("patience multiples must be strictly increasing")

    stop_cumulative_rows = sum(normalized_rows[:stop_state_index])
    cumulative_rows = stop_cumulative_rows
    search_state = stop_state_index
    checkpoints: list[ContinuationCheckpoint] = []
    for multiple in normalized_multiples:
        requested_extra_rows = multiple * patience_ticks * n_records
        target_cumulative_rows = stop_cumulative_rows + requested_extra_rows
        while (
            search_state < len(normalized_rows)
            and cumulative_rows < target_cumulative_rows
        ):
            cumulative_rows += normalized_rows[search_state]
            search_state += 1

        requested_extra_work = float(multiple * patience_ticks)
        target_normalized_work = target_cumulative_rows / n_records
        if cumulative_rows < target_cumulative_rows:
            checkpoints.append(
                ContinuationCheckpoint(
                    patience_multiple=multiple,
                    requested_extra_work=requested_extra_work,
                    target_normalized_work=target_normalized_work,
                    status="right_censored_by_known_trace_horizon",
                    state_index=None,
                    actual_normalized_work=None,
                    actual_extra_work=None,
                )
            )
            continue

        checkpoints.append(
            ContinuationCheckpoint(
                patience_multiple=multiple,
                requested_extra_work=requested_extra_work,
                target_normalized_work=target_normalized_work,
                status="observed",
                state_index=search_state,
                actual_normalized_work=cumulative_rows / n_records,
                actual_extra_work=(cumulative_rows - stop_cumulative_rows) / n_records,
            )
        )
    return checkpoints


def _table_sha256(frame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _participating_rows(clock: dict[str, Any]) -> int:
    accepted_attempt = int(clock["accepted_attempt"])
    if accepted_attempt <= 0:
        return 0
    return int(clock["attempts"][accepted_attempt - 1]["participating_rows"])


def _terminal_metrics(table, queries, target) -> tuple[float, float]:
    answers = evaluate_table(table, queries)
    return (
        float(compute_squared_loss(target, answers)),
        float(compute_normalized_l1(target, answers, N_RECORDS)),
    )


def _audit_online_wiring_against_full_trace(
    *,
    seed: int,
    rho: float,
    n_rounds: int,
    queries: Sequence[dict],
    target: np.ndarray,
    full: dict[str, Any],
    offline_decision: EarlyStopDecision,
) -> dict[str, Any]:
    """Require the online A/B output to equal the old offline replay prefix."""

    if offline_decision.reason not in {"fit_target_reached", "early_stopped"}:
        raise AssertionError("known trace must reach A or B before its old horizon")

    online_queries, online_target, online_output, online = _run_online_early_stop(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
    )
    _require(online_queries == queries, "online/offline query workloads differ")
    _require(
        np.array_equal(online_target, target),
        "online/offline targets differ",
    )

    stop_state = offline_decision.state_index
    full_metrics = full["current_state_metrics_history"]
    full_clocks = full["transition_clock_history"]
    expected_metrics = full_metrics[: stop_state + 1]
    expected_clocks = full_clocks[:stop_state]
    _require(
        online["current_state_metrics_history"] == expected_metrics,
        "online current metrics are not the frozen full-trace prefix",
    )
    _require(
        online["transition_clock_history"] == expected_clocks,
        "online transition clocks are not the frozen full-trace prefix",
    )
    _require(
        online["accept_history"] == full["accept_history"][:stop_state],
        "online accept history is not the frozen full-trace prefix",
    )

    expected_terminal_sha = (
        full["initial_table_sha256"]
        if stop_state == 0
        else full_clocks[stop_state - 1]["post_current_table_sha256"]
    )
    expected_rng_sha = (
        full["primary_rng_post_initialization_state_sha256"]
        if stop_state == 0
        else full_clocks[stop_state - 1]["primary_rng_state_sha256"]
    )
    expected_candidate_evaluations = (
        0
        if stop_state == 0
        else int(full_clocks[stop_state - 1]["candidate_evaluation_count_cumulative"])
    )
    output_sha = _table_sha256(online_output)
    final_table_sha = _table_sha256(online["final_table"])
    _require(
        output_sha == expected_terminal_sha,
        "main output table is not the expected terminal current state",
    )
    _require(
        final_table_sha == expected_terminal_sha,
        "diagnostic final table is not the expected terminal current state",
    )
    _require(
        online_output.reset_index(drop=True).equals(online["final_table"]),
        "main output and diagnostic final table differ",
    )
    _require(
        online["primary_rng_state_sha256"] == expected_rng_sha,
        "online primary RNG state is not the expected prefix state",
    )
    _require(
        online["candidate_evaluation_count"] == expected_candidate_evaluations,
        "online candidate count is not the expected prefix count",
    )

    expected_current_loss = float(full_metrics[stop_state]["current_squared_loss"])
    expected_current_l1 = float(full_metrics[stop_state]["current_normalized_l1"])
    output_loss, output_l1 = _terminal_metrics(
        online_output,
        queries,
        target,
    )
    _require(output_loss == expected_current_loss, "terminal loss replay differs")
    _require(output_l1 == expected_current_l1, "terminal L1 replay differs")
    _require(
        online["output_squared_loss"] == expected_current_loss,
        "output squared loss differs from the terminal current state",
    )
    _require(
        online["final_current_squared_loss"] == expected_current_loss,
        "final current squared loss differs from replay",
    )
    _require(
        online["normalized_l1_error"] == expected_current_l1,
        "reported normalized L1 differs from replay",
    )

    last_decision = online["inner_early_stopping"]["last_decision"]
    _require(
        online["termination_reason"] == offline_decision.reason,
        "online/offline stopping reasons differ",
    )
    _require(online["rounds_run"] == stop_state, "online stop state differs")
    _require(online["inner_complete"] is True, "A/B must complete the inner run")
    _require(
        online["fit_target_reached"]
        is (offline_decision.reason == "fit_target_reached"),
        "fit-target status differs from the stopping reason",
    )
    _require(
        online["output_table_identity"] == "terminal_current",
        "online output identity must be terminal_current",
    )
    _require(
        online["inner_early_stopping"]["resource_cap_source_diagnostic_only"]
        is None,
        "normal A/B termination must not report a resource-cap source",
    )
    _require(
        last_decision["external_resource_cap_reached"] is False,
        "normal A/B termination must precede external C",
    )
    _require(
        last_decision["state_index"] == stop_state,
        "last decision state differs from the stop state",
    )
    _require(
        last_decision["terminal_output_state_index"] == stop_state,
        "terminal output state differs from the stop state",
    )
    _require(
        last_decision["current_loss"] == expected_current_loss,
        "last decision current loss differs from replay",
    )
    _require(
        last_decision["terminal_output_loss"] == expected_current_loss,
        "terminal output loss differs from replay",
    )
    _require(
        last_decision["normalized_work"] == offline_decision.normalized_work,
        "online/offline normalized work differs",
    )
    _require(
        last_decision["completed_work_ticks"]
        == offline_decision.completed_work_ticks,
        "online/offline completed work ticks differ",
    )
    _require(
        last_decision["consecutive_no_progress_ticks"]
        == offline_decision.consecutive_no_progress_ticks,
        "online/offline no-progress ticks differ",
    )

    best_loss = min(float(row["current_squared_loss"]) for row in expected_metrics)
    _require(
        online["best_loss_diagnostic_only"] == best_loss,
        "online best-loss diagnostic differs from the prefix minimum",
    )
    return {
        "seed": seed,
        "rho": rho,
        "known_trace_horizon_rounds_not_C": n_rounds,
        "offline_replay_reason": offline_decision.reason,
        "online_reason": online["termination_reason"],
        "stop_state": stop_state,
        "stop_normalized_work": offline_decision.normalized_work,
        "terminal_current_loss": expected_current_loss,
        "historical_best_loss_diagnostic_only": best_loss,
        "terminal_current_above_best_diagnostic_only": (
            expected_current_loss > best_loss
        ),
        "terminal_current_table_sha256": expected_terminal_sha,
        "main_output_equals_terminal_current": True,
        "current_metrics_prefix_equal": True,
        "transition_clocks_prefix_equal": True,
        "accept_history_prefix_equal": True,
        "primary_rng_prefix_equal": True,
        "candidate_evaluations_prefix_equal": True,
        "resource_C_preempted_A_or_B": False,
    }


def audit_online_wiring_case(
    *,
    seed: int,
    rho: float,
    n_rounds: int,
) -> dict[str, Any]:
    """Reproduce one known-trajectory online/offline wiring audit."""

    queries, target, full = _run_trace(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
    )
    metrics = full["current_state_metrics_history"]
    clocks = full["transition_clock_history"]
    losses = [float(row["current_squared_loss"]) for row in metrics]
    participating_rows = [_participating_rows(clock) for clock in clocks]
    offline_decision = replay_fixed_early_stop(losses, participating_rows)
    return _audit_online_wiring_against_full_trace(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
        queries=queries,
        target=target,
        full=full,
        offline_decision=offline_decision,
    )


def _classify_case(*, seed: int, rho: float, n_rounds: int) -> dict[str, Any]:
    queries, target, full = _run_trace(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
    )
    metrics = full["current_state_metrics_history"]
    clocks = full["transition_clock_history"]
    losses = [float(row["current_squared_loss"]) for row in metrics]
    rows = [_participating_rows(clock) for clock in clocks]
    decision = replay_fixed_early_stop(losses, rows)
    online_wiring_audit = _audit_online_wiring_against_full_trace(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
        queries=queries,
        target=target,
        full=full,
        offline_decision=decision,
    )

    stop_state = decision.state_index
    _, prefix_target, prefix = _run_trace(
        seed=seed,
        rho=rho,
        n_rounds=stop_state,
    )
    _require(np.array_equal(prefix_target, target), "prefix target differs")
    _require(
        prefix["current_state_metrics_history"] == metrics[: stop_state + 1],
        "prefix current metrics differ from the full trace",
    )
    _require(
        prefix["transition_clock_history"] == clocks[:stop_state],
        "prefix transition clocks differ from the full trace",
    )

    stop_terminal_table = prefix["final_table"]
    state_hashes = [full["initial_table_sha256"]] + [
        clock["post_current_table_sha256"] for clock in clocks
    ]
    stop_table_sha = _table_sha256(stop_terminal_table)
    _require(
        stop_table_sha == state_hashes[stop_state],
        "stop table identity differs from the full trace",
    )

    stop_loss, stop_l1 = _terminal_metrics(stop_terminal_table, queries, target)
    _require(
        stop_loss == metrics[stop_state]["current_squared_loss"],
        "stop-table loss differs from the full trace",
    )
    _require(
        stop_l1 == metrics[stop_state]["current_normalized_l1"],
        "stop-table L1 differs from the full trace",
    )

    stop_best_loss = min(losses[: stop_state + 1])
    stop_candidate_evaluations = (
        int(clocks[stop_state - 1]["candidate_evaluation_count_cumulative"])
        if stop_state > 0
        else 0
    )

    first_a_after_stop = None
    if decision.reason == "early_stopped":
        first_a_after_stop = next(
            (
                {
                    "state_index": state_index,
                    "normalized_work": sum(rows[:state_index]) / N_RECORDS,
                }
                for state_index, loss in enumerate(
                    losses[stop_state + 1 :],
                    start=stop_state + 1,
                )
                if loss == 0.0
            ),
            None,
        )

    continuation_rows: list[dict[str, Any]] = []
    if decision.reason == "early_stopped":
        checkpoint_locations = locate_continuation_checkpoints(
            rows,
            stop_state_index=stop_state,
        )
        for location in checkpoint_locations:
            checkpoint: dict[str, Any] = {
                "patience_multiple": location.patience_multiple,
                "requested_extra_work": location.requested_extra_work,
                "target_normalized_work": location.target_normalized_work,
                "status": location.status,
                "state_index": location.state_index,
                "actual_normalized_work": location.actual_normalized_work,
                "actual_extra_work": location.actual_extra_work,
                "extra_raw_rounds": None,
                "extra_candidate_evaluations": None,
                "current_loss": None,
                "current_normalized_l1_offline": None,
                "current_loss_delta_continuation_minus_stop": None,
                "current_l1_delta_continuation_minus_stop_offline": None,
                "terminal_current_table_sha256": None,
                "historical_best_loss_at_checkpoint_diagnostic_only": None,
                "fit_target_reached_after_B_by_checkpoint": None,
            }
            if location.status != "observed":
                continuation_rows.append(checkpoint)
                continue

            checkpoint_state = location.state_index
            _require(
                checkpoint_state is not None,
                "observed continuation checkpoint must identify a state",
            )
            _, checkpoint_target, checkpoint_prefix = _run_trace(
                seed=seed,
                rho=rho,
                n_rounds=checkpoint_state,
            )
            _require(
                np.array_equal(checkpoint_target, target),
                "continuation checkpoint target differs",
            )
            _require(
                checkpoint_prefix["current_state_metrics_history"]
                == metrics[: checkpoint_state + 1],
                "continuation metrics are not the full-trace prefix",
            )
            _require(
                checkpoint_prefix["transition_clock_history"]
                == clocks[:checkpoint_state],
                "continuation clocks are not the full-trace prefix",
            )

            checkpoint_table = checkpoint_prefix["final_table"]
            checkpoint_sha = _table_sha256(checkpoint_table)
            _require(
                checkpoint_sha == state_hashes[checkpoint_state],
                "continuation table identity differs from the full trace",
            )
            checkpoint_loss, checkpoint_l1 = _terminal_metrics(
                checkpoint_table,
                queries,
                target,
            )
            _require(
                checkpoint_loss
                == metrics[checkpoint_state]["current_squared_loss"],
                "continuation loss differs from the full trace",
            )
            _require(
                checkpoint_l1
                == metrics[checkpoint_state]["current_normalized_l1"],
                "continuation L1 differs from the full trace",
            )
            checkpoint_candidates = int(
                clocks[checkpoint_state - 1]["candidate_evaluation_count_cumulative"]
            )
            checkpoint.update(
                {
                    "extra_raw_rounds": checkpoint_state - stop_state,
                    "extra_candidate_evaluations": (
                        checkpoint_candidates - stop_candidate_evaluations
                    ),
                    "current_loss": checkpoint_loss,
                    "current_normalized_l1_offline": checkpoint_l1,
                    "current_loss_delta_continuation_minus_stop": (
                        checkpoint_loss - stop_loss
                    ),
                    "current_l1_delta_continuation_minus_stop_offline": (
                        checkpoint_l1 - stop_l1
                    ),
                    "terminal_current_table_sha256": checkpoint_sha,
                    "historical_best_loss_at_checkpoint_diagnostic_only": min(
                        losses[: checkpoint_state + 1]
                    ),
                    "fit_target_reached_after_B_by_checkpoint": any(
                        loss == 0.0
                        for loss in losses[stop_state + 1 : checkpoint_state + 1]
                    ),
                }
            )
            continuation_rows.append(checkpoint)

    return {
        "seed": seed,
        "rho": rho,
        "known_trace_horizon_rounds_not_C": n_rounds,
        "known_trace_horizon_work_not_C": sum(rows) / N_RECORDS,
        "termination_reason": decision.reason,
        "stop_state": stop_state,
        "stop_work": decision.normalized_work,
        "stop_current_loss": stop_loss,
        "stop_current_normalized_l1": stop_l1,
        "stop_best_loss_diagnostic_only": stop_best_loss,
        "stop_current_minus_best_diagnostic_only": stop_loss - stop_best_loss,
        "stop_terminal_table_sha256": stop_table_sha,
        "candidate_evaluations_at_stop": stop_candidate_evaluations,
        "prefix_elapsed_sec_diagnostic_only": float(prefix["elapsed_sec"]),
        "first_fit_target_after_stop_diagnostic_only": first_a_after_stop,
        "online_wiring_audit": online_wiring_audit,
        "continuation_checkpoints": continuation_rows,
    }


def _continuation_comparison_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "continuation_better": sum(value < 0.0 for value in values),
        "equal": sum(value == 0.0 for value in values),
        "continuation_worse": sum(value > 0.0 for value in values),
    }


def build_report() -> dict[str, Any]:
    rows = [
        _classify_case(seed=seed, rho=rho, n_rounds=n_rounds)
        for seed, rho, n_rounds in CASES
    ]
    early_rows = [row for row in rows if row["termination_reason"] == "early_stopped"]
    online_wiring_rows = [row["online_wiring_audit"] for row in rows]
    current_minus_best = [
        row["stop_current_minus_best_diagnostic_only"] for row in early_rows
    ]
    continuation_curve = []
    for multiple in CONTINUATION_PATIENCE_MULTIPLES:
        checkpoints = [
            next(
                checkpoint
                for checkpoint in row["continuation_checkpoints"]
                if checkpoint["patience_multiple"] == multiple
            )
            for row in early_rows
        ]
        observed = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint["status"] == "observed"
        ]
        loss_deltas = [
            checkpoint["current_loss_delta_continuation_minus_stop"]
            for checkpoint in observed
        ]
        l1_deltas = [
            checkpoint["current_l1_delta_continuation_minus_stop_offline"]
            for checkpoint in observed
        ]
        continuation_curve.append(
            {
                "patience_multiple": multiple,
                "requested_extra_work": multiple * NO_PROGRESS_PATIENCE_TICKS,
                "eligible_early_stopped_cases": len(early_rows),
                "observed_count": len(observed),
                "right_censored_count": len(checkpoints) - len(observed),
                "current_loss_comparison": _continuation_comparison_counts(loss_deltas),
                "offline_l1_comparison": _continuation_comparison_counts(l1_deltas),
                "mean_current_loss_delta_continuation_minus_stop": mean(loss_deltas),
                "median_current_loss_delta_continuation_minus_stop": median(
                    loss_deltas
                ),
                "mean_offline_l1_delta_continuation_minus_stop": mean(l1_deltas),
                "median_offline_l1_delta_continuation_minus_stop": median(l1_deltas),
                "mean_actual_extra_work": mean(
                    checkpoint["actual_extra_work"] for checkpoint in observed
                ),
                "mean_extra_raw_rounds": mean(
                    checkpoint["extra_raw_rounds"] for checkpoint in observed
                ),
                "mean_extra_candidate_evaluations": mean(
                    checkpoint["extra_candidate_evaluations"] for checkpoint in observed
                ),
                "fit_target_reached_after_B_by_checkpoint_count": sum(
                    checkpoint["fit_target_reached_after_B_by_checkpoint"]
                    for checkpoint in observed
                ),
            }
        )
    return {
        "analysis_id": ANALYSIS_ID,
        "classification": "development_known_trajectories_not_validation",
        "online_inputs": [
            "current squared loss",
            "historical best-loss refresh event",
            "applied participating-row count",
        ],
        "forbidden_online_inputs": [
            "normalized L1",
            "shadow-continuation states",
            "historical best table as output",
        ],
        "candidate": {
            "natural_work_tick": "floor(cumulative participating rows / N)",
            "no_progress_patience_ticks": NO_PROGRESS_PATIENCE_TICKS,
            "progress": "at least one strict historical-best loss refresh in tick",
            "A_zero_noise": "current loss == 0",
            "B_output": "terminal current table and terminal current loss",
            "fixed_six_tick_role": "development baseline only",
        },
        "offline_continuation_audit": {
            "relative_checkpoint_patience_multiples": list(
                CONTINUATION_PATIENCE_MULTIPLES
            ),
            "checkpoint_identity": (
                "first real current state at or beyond tau + multiple*P"
            ),
            "missing_checkpoint": (
                "right_censored; never replaced by known trace terminal"
            ),
            "formal_B_output_frozen_before_shadow_continuation": True,
            "wall_clock_delta_available": False,
            "wall_clock_note": (
                "old traces lack cumulative per-state timing; no timing delta "
                "is fabricated from independent prefix replays"
            ),
        },
        "case_count": len(rows),
        "early_stopped_count": len(early_rows),
        "fit_target_reached_count": sum(
            row["termination_reason"] == "fit_target_reached" for row in rows
        ),
        "known_trace_horizon_reached_before_A_or_B_count": sum(
            row["termination_reason"] == "reference_horizon_reached" for row in rows
        ),
        "online_wiring_audit": {
            "audit_id": "issue53_online_early_stopping_wiring_known_traces_v1",
            "classification": (
                "development_known_trajectories_wiring_consistency_only"
            ),
            "case_count": len(online_wiring_rows),
            "all_online_reasons_equal_offline_replay": all(
                row["online_reason"] == row["offline_replay_reason"]
                for row in online_wiring_rows
            ),
            "fit_target_reached_count": sum(
                row["online_reason"] == "fit_target_reached"
                for row in online_wiring_rows
            ),
            "early_stopped_count": sum(
                row["online_reason"] == "early_stopped" for row in online_wiring_rows
            ),
            "resource_cap_reached_count": sum(
                row["online_reason"] == "resource_cap_reached"
                for row in online_wiring_rows
            ),
            "all_main_outputs_equal_terminal_current": all(
                row["main_output_equals_terminal_current"] for row in online_wiring_rows
            ),
            "all_current_metrics_prefixes_equal": all(
                row["current_metrics_prefix_equal"] for row in online_wiring_rows
            ),
            "all_transition_clock_prefixes_equal": all(
                row["transition_clocks_prefix_equal"] for row in online_wiring_rows
            ),
            "all_accept_history_prefixes_equal": all(
                row["accept_history_prefix_equal"] for row in online_wiring_rows
            ),
            "all_primary_rng_prefixes_equal": all(
                row["primary_rng_prefix_equal"] for row in online_wiring_rows
            ),
            "all_candidate_evaluation_prefixes_equal": all(
                row["candidate_evaluations_prefix_equal"] for row in online_wiring_rows
            ),
            "resource_C_preempted_A_or_B_count": sum(
                row["resource_C_preempted_A_or_B"] for row in online_wiring_rows
            ),
            "rows": online_wiring_rows,
            "interpretation_boundary": (
                "This only proves that the online wiring reproduces the old "
                "offline decision on six already-inspected artificial traces. "
                "It does not validate or tune P=6, C, or output quality."
            ),
        },
        "early_stop_aggregate": {
            "terminal_current_vs_historical_best_diagnostic_only": {
                "equal": sum(value == 0.0 for value in current_minus_best),
                "current_worse": sum(value > 0.0 for value in current_minus_best),
                "mean_current_minus_best_loss": mean(current_minus_best),
                "median_current_minus_best_loss": median(current_minus_best),
            },
            "continuation_gain_curve": continuation_curve,
        },
        "rows": rows,
        "interpretation_boundary": (
            "No known-trace terminal is a quality reference. Strict later "
            "improvement is not an automatic early-stop failure. These inspected "
            "trajectories only exercise the relative continuation protocol; they "
            "cannot validate or tune a production patience value."
        ),
    }


def main() -> None:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
