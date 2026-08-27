"""Contract tests for the frozen Issue #53 P=6 raw collector."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from scripts import collect_issue53_p6_unseen as collector
from scripts import issue53_p6_unseen_protocol as protocol
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.stationarity import (
    StationarityTrace,
    build_stationarity_observation,
)


def _all_zero_trace(workload, rounds=60):
    frame = pd.DataFrame(
        {
            attribute: [0] * workload.n_records
            for attribute in workload.schema.attribute_names()
        }
    )
    answers = np.asarray(evaluate_table(frame, workload.queries), dtype=float)
    loss = float(compute_squared_loss(workload.target, answers))
    table_sha = collector._frame_sha256(frame)
    observations = []
    metrics = []
    clocks = []
    for state_index in range(rounds + 1):
        initial = state_index == 0
        rng_sha = f"{state_index + 1:064x}"
        observation = build_stationarity_observation(
            frame=frame,
            target=workload.target,
            current_query_answers=answers,
            n_records=workload.n_records,
            squared_loss=loss,
            state_index=state_index,
            round_index=state_index,
            phase="initial" if initial else "post_round",
            proposal_attempt_count=0 if initial else 1,
            proposal_accepted=not initial,
            applied_attempt_index=0 if initial else 1,
            attempted_participating_row_count=(0 if initial else workload.n_records),
            applied_participating_row_count=(0 if initial else workload.n_records),
            actual_changed_row_count=0,
            actual_changed_cell_count=0,
            actual_changed_query_count=0,
            normalized_query_l1_movement_mean=0.0,
            gibbs_microstep_count_attempted=0,
            gibbs_microstep_count_applied=0,
            candidate_evaluation_count_cumulative=state_index,
            current_table_sha256=table_sha,
            primary_rng_state_sha256=rng_sha,
            factorized_gibbs_rng_state_sha256=None,
        )
        observations.append(observation)
        metrics.append(
            {
                "state_index": state_index,
                "round": state_index,
                "phase": "initial" if initial else "post_round",
                "current_normalized_l1": observation["current_normalized_l1"],
                "current_squared_loss": loss,
            }
        )
        if not initial:
            clocks.append(
                {
                    "state_index": state_index,
                    "round": state_index,
                    "attempts": [{"participating_rows": workload.n_records}],
                    "accepted_attempt": 1,
                    "candidate_evaluation_count_cumulative": state_index,
                    "post_current_table_sha256": table_sha,
                    "primary_rng_state_sha256": rng_sha,
                    "factorized_gibbs_rng_state_sha256": None,
                }
            )
    trace = StationarityTrace(
        n_records=workload.n_records,
        query_identity_sha256=workload.query_identity_sha256,
        target_identity_sha256=workload.target_identity_sha256,
        observations=observations,
        measured_query_answers=np.repeat(
            answers[np.newaxis, :],
            rounds + 1,
            axis=0,
        ),
        termination_reason="candidate_budget",
    )
    return frame, answers, loss, trace, metrics, clocks


def _fake_diagnostics(
    workload,
    frame,
    answers,
    loss,
    trace,
    metrics,
    clocks,
    *,
    rounds_run,
    online,
):
    normalized_l1 = compute_normalized_l1(
        workload.target,
        answers,
        workload.n_records,
    )
    per_query = np.abs(workload.target - answers) / workload.n_records
    reason = "early_stopped" if online else "candidate_budget"
    decision = (
        {
            "termination_reason": reason,
            "terminal_output_state_index": rounds_run,
            "normalized_work": float(rounds_run),
            "completed_work_ticks": rounds_run,
            "consecutive_no_progress_ticks": rounds_run,
        }
        if online
        else None
    )
    diagnostics = {
        "rounds_run": rounds_run,
        "termination_reason": reason,
        "stopped_early": online,
        "fit_target_reached": False if online else None,
        "inner_complete": True if online else None,
        "output_table_identity": (
            "terminal_current" if online else "historical_best_legacy"
        ),
        "output_squared_loss": loss,
        "final_current_squared_loss": loss,
        "final_current_normalized_l1": normalized_l1,
        "normalized_l1_error": normalized_l1,
        "normalized_l1_median": float(np.median(per_query)),
        "normalized_l1_p90": float(np.percentile(per_query, 90)),
        "normalized_l1_max": float(np.max(per_query)),
        "best_loss_diagnostic_only": loss,
        "normalized_l1_at_best_squared_loss_diagnostic_only": normalized_l1,
        "state_evaluation_count": max(1, rounds_run),
        "candidate_evaluation_count": rounds_run,
        "candidate_budget_exhausted": not online,
        "initial_table_sha256": trace.observations[0]["current_table_sha256"],
        "primary_rng_post_initialization_state_sha256": trace.observations[0][
            "primary_rng_state_sha256"
        ],
        "primary_rng_state_sha256": trace.observations[rounds_run][
            "primary_rng_state_sha256"
        ],
        "current_state_metrics_history": metrics[: rounds_run + 1],
        "transition_clock_history": clocks[:rounds_run],
        "accept_history": [True] * rounds_run,
        "proposal_attempts_history": [1] * rounds_run,
        "accepted_attempt_history": [1] * rounds_run,
        "inner_early_stopping": {
            "enabled": online,
            "patience_ticks": 6 if online else None,
            "last_decision": decision,
            "resource_cap_source_diagnostic_only": None,
        },
        "elapsed_sec": 0.01,
        "sec_per_round": 0.01 / rounds_run,
        "params": {
            "n_rounds": 60,
            "candidate_budget": 60,
            "tol": float("inf"),
        },
    }
    if online:
        diagnostics["final_table"] = frame.copy(deep=True)
    return diagnostics


def test_plan_is_exact_read_only_and_reports_worst_case_cost(monkeypatch):
    def forbidden_generator(*args, **kwargs):
        raise AssertionError("plan must not call generator")

    monkeypatch.setattr(collector, "run_evolution", forbidden_generator)
    plan = collector.build_collection_plan()

    assert plan["mode"] == "plan_only_no_generation_or_rng_instantiation"
    assert plan["protocol_sha256"] == protocol.FROZEN_PROTOCOL_SHA256
    assert plan["case_count"] == 12
    assert plan["online_round_cap"] == 1800
    assert plan["maximum_generator_call_count"] == 24
    assert plan["maximum_total_round_cap_if_all_B"] == 3600
    assert plan["shadow_policy"] == "B_only"
    assert plan["acceptance_evaluated_during_collection"] is False
    assert plan["generation_started"] is False
    assert set(inspect.signature(collector.run_primary_collection).parameters) == {
        "output_dir",
        "confirmed_protocol_sha256",
    }


def test_materialized_families_match_manifest_without_reference_table():
    observed = {}
    for name in ("binary_chain_4", "mixed_2x3x2"):
        workload = collector.materialize_family(name)
        observed[name] = (
            workload.n_records,
            workload.schema.attribute_names(),
            len(workload.queries),
            workload.target.tolist(),
        )
        assert not hasattr(workload, "reference_multiset")

    assert observed == {
        "binary_chain_4": (
            32,
            ["a", "b", "c", "d"],
            11,
            [16.0, 16.0, 16.0, 16.0, 12.0, 10.0, 8.0, 10.0, 8.0, 6.0, 6.0],
        ),
        "mixed_2x3x2": (
            36,
            ["x", "y", "z"],
            15,
            [
                18.0,
                14.0,
                10.0,
                12.0,
                19.0,
                6.0,
                6.0,
                6.0,
                6.0,
                6.0,
                7.0,
                11.0,
                4.0,
                5.0,
                2.0,
            ],
        ),
    }
    with pytest.raises(ValueError):
        collector.materialize_family("new_family")


def test_generator_kwargs_translate_protocol_without_open_overrides():
    case = protocol.primary_case_matrix()[0]
    online = collector._generator_kwargs(case, shadow=False)
    shadow = collector._generator_kwargs(case, shadow=True)

    assert online["n_records"] == 32
    assert online["n_rounds"] == online["candidate_budget"] == 60
    assert online["seed"] == 20260819
    assert online["rho"] == 1.0
    assert np.isposinf(online["tol"])
    assert online["max_retries"] == 0
    assert online["residual_self_cooling"] is None
    assert online["inner_early_stopping_patience_ticks"] == 6
    assert online["stop_on_exact_residual"] is True
    assert online["record_stationarity_trace"] is False
    assert shadow["inner_early_stopping_patience_ticks"] is None
    assert shadow["stop_on_exact_residual"] is False
    assert shadow["record_stationarity_trace"] is True
    differing = {key for key in online if online[key] != shadow[key]}
    assert differing == {
        "record_stationarity_trace",
        "stop_on_exact_residual",
        "inner_early_stopping_patience_ticks",
    }
    mutated = dict(case)
    mutated["seed"] += 1
    with pytest.raises(ValueError, match="冻结 primary case"):
        collector._generator_kwargs(mutated, shadow=False)


def test_wrong_sha_fails_before_environment_output_or_generator(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "must_not_exist"

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong SHA must fail first")

    monkeypatch.setattr(collector, "build_execution_manifest", forbidden)
    monkeypatch.setattr(collector, "run_evolution", forbidden)

    with pytest.raises(ValueError, match="显式确认"):
        collector.run_primary_collection(output, "0" * 64)
    assert not output.exists()


def test_dirty_tree_fails_before_source_hashing_or_rng(monkeypatch):
    git_calls = []

    def fake_git(_root, *arguments):
        git_calls.append(arguments)
        return "?? untracked.py"

    def forbidden_hash(_path):
        raise AssertionError("dirty tree must fail before source hashing")

    monkeypatch.setattr(collector, "_git_text", fake_git)
    monkeypatch.setattr(collector, "_sha256_file", forbidden_hash)

    with pytest.raises(RuntimeError, match="干净工作树"):
        collector.build_execution_manifest(collector._repo_root())
    assert git_calls == [("status", "--porcelain", "--untracked-files=all")]


def test_b_checkpoint_locator_uses_first_real_state_and_right_censors():
    workload = collector.materialize_family("binary_chain_4")
    _, _, _, trace, _, _ = _all_zero_trace(workload, rounds=15)

    checkpoints = collector.locate_b_shadow_checkpoints(trace, 2)

    assert checkpoints[0]["work_offset"] == 6
    assert checkpoints[0]["target_normalized_work"] == 8.0
    assert checkpoints[0]["status"] == "observed"
    assert checkpoints[0]["state_index"] == 8
    assert checkpoints[0]["actual_extra_work"] == 6.0
    assert checkpoints[1] == {
        "work_offset": 12,
        "target_normalized_work": 14.0,
        "status": "observed",
        "state_index": 14,
        "actual_normalized_work": 14.0,
        "actual_extra_work": 12.0,
        "extra_raw_rounds": 12,
        "extra_candidate_evaluations": 12,
        "current_table_sha256": trace.observations[14]["current_table_sha256"],
        "current_query_answers": trace.measured_query_answers[14].tolist(),
        "current_squared_loss": trace.observations[14]["current_squared_loss"],
        "current_normalized_l1": trace.observations[14]["current_normalized_l1"],
    }

    _, _, _, short_trace, _, _ = _all_zero_trace(workload, rounds=10)
    censored = collector.locate_b_shadow_checkpoints(short_trace, 2)
    assert censored[0]["status"] == "observed"
    assert censored[1]["status"] == "right_censored_by_resource_guard"
    assert censored[1]["state_index"] is None


def test_case_collector_persists_terminal_and_b_shadow_without_generator(
    tmp_path,
    monkeypatch,
):
    case = protocol.primary_case_matrix()[0]
    workload = collector.materialize_family(case["family"])
    frame, answers, loss, trace, metrics, clocks = _all_zero_trace(
        workload,
        rounds=60,
    )
    online = _fake_diagnostics(
        workload,
        frame,
        answers,
        loss,
        trace,
        metrics,
        clocks,
        rounds_run=6,
        online=True,
    )
    shadow = _fake_diagnostics(
        workload,
        frame,
        answers,
        loss,
        trace,
        metrics,
        clocks,
        rounds_run=60,
        online=False,
    )
    monkeypatch.setattr(
        collector,
        "_run_online",
        lambda _workload, _case: (frame.copy(deep=True), online),
    )
    monkeypatch.setattr(
        collector,
        "_run_shadow",
        lambda _workload, _case: (frame.copy(deep=True), shadow, trace),
    )
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    manifest_path = collector._collect_case(
        cases_dir,
        case,
        workload,
        "e" * 64,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["online"]["termination_reason"] == "early_stopped"
    assert manifest["online"]["stop_state_index"] == 6
    assert manifest["online"]["stop_normalized_work"] == 6.0
    assert manifest["shadow"]["collected"] is True
    assert manifest["shadow"]["prefix_audit"] == {
        "current_metrics_prefix_equal": True,
        "transition_clocks_prefix_equal": True,
        "accept_history_prefix_equal": True,
        "proposal_attempts_prefix_equal": True,
        "accepted_attempt_prefix_equal": True,
        "terminal_table_identity_equal": True,
        "terminal_query_vector_equal": True,
        "primary_rng_prefix_equal": True,
        "candidate_evaluations_prefix_equal": True,
    }
    assert [row["state_index"] for row in manifest["shadow"]["checkpoints"]] == [12, 18]
    assert manifest["acceptance_evaluated"] is False
    assert manifest["partial_matrix_classification_emitted"] is False
    assert (manifest_path.parent / "terminal_current.csv").is_file()
    assert (manifest_path.parent / "online_diagnostics.json").is_file()
    assert (manifest_path.parent / "shadow_trace" / "stationarity_trace.json").is_file()


def test_orchestrator_uses_all_twelve_frozen_cases_with_fake_collector(
    tmp_path,
    monkeypatch,
):
    observed = []
    fake_execution_manifest = {
        "contract_version": collector.COLLECTION_CONTRACT_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "execution_started": False,
        "formal_rng_instantiated": False,
    }

    def fake_collect(cases_dir, case, workload, execution_sha):
        observed.append(
            (
                case["case_id"],
                workload.name,
                execution_sha,
            )
        )
        destination = cases_dir / case["case_id"]
        destination.mkdir()
        path = destination / "case_manifest.json"
        collector._write_json_exclusive(
            path,
            {
                "case": case,
                "acceptance_evaluated": False,
            },
        )
        return path

    monkeypatch.setattr(
        collector,
        "build_execution_manifest",
        lambda _root: fake_execution_manifest,
    )
    monkeypatch.setattr(collector, "_collect_case", fake_collect)
    output = tmp_path / "formal"

    result_path = collector.run_primary_collection(
        output,
        protocol.FROZEN_PROTOCOL_SHA256,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert [row[0] for row in observed] == [
        case["case_id"] for case in protocol.primary_case_matrix()
    ]
    assert len(observed) == 12
    assert {row[1] for row in observed} == {
        "binary_chain_4",
        "mixed_2x3x2",
    }
    assert len({row[2] for row in observed}) == 1
    assert result["formal_primary_collection_complete"] is True
    assert result["case_count"] == 12
    assert result["acceptance_evaluated"] is False
    assert result["partial_matrix_classification_emitted"] is False
    assert set(result["case_manifest_files"]) == {
        case["case_id"] for case in protocol.primary_case_matrix()
    }
