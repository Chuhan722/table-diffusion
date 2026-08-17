"""Result-blind contract tests for the frozen Issue #53 P=6 evaluator."""

from __future__ import annotations

import copy
import inspect
import json

import pandas as pd
import pytest

from scripts import collect_issue53_p6_unseen as collector
from scripts import evaluate_issue53_p6_unseen as evaluator
from scripts import issue53_p6_unseen_protocol as protocol
from tests.test_collect_issue53_p6_unseen import _all_zero_trace, _fake_diagnostics


def _evidence_matrix(
    *,
    delta_l1: float = 0.005,
    stop_work: float = 12.0,
) -> list[dict]:
    rows = []
    for case in protocol.primary_case_matrix():
        terminal_l1 = 0.1
        continuation_l1 = terminal_l1 - delta_l1
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "seed": case["seed"],
                "rho": case["rho"],
                "termination_reason": "early_stopped",
                "stop_normalized_work": stop_work,
                "terminal_current_squared_loss": 10.0,
                "terminal_current_normalized_l1": terminal_l1,
                "checkpoints": [
                    {
                        "work_offset": offset,
                        "status": "observed",
                        "current_squared_loss": 9.0,
                        "current_normalized_l1": continuation_l1,
                    }
                    for offset in protocol.SHADOW_WORK_OFFSETS
                ],
            }
        )
    return rows


def _turn_into_a(row: dict) -> None:
    row.update(
        {
            "termination_reason": "fit_target_reached",
            "stop_normalized_work": 0.0,
            "terminal_current_squared_loss": 0.0,
            "terminal_current_normalized_l1": 0.0,
            "checkpoints": [],
        }
    )


def _turn_into_c(row: dict) -> None:
    row.update(
        {
            "termination_reason": "resource_cap_reached",
            "stop_normalized_work": 60.0,
            "checkpoints": [],
        }
    )


def _set_censored(row: dict, offset: int) -> None:
    checkpoint = next(
        item for item in row["checkpoints"] if item["work_offset"] == offset
    )
    checkpoint.update(
        {
            "status": "right_censored_by_resource_guard",
            "current_squared_loss": None,
            "current_normalized_l1": None,
        }
    )


def test_plan_is_result_blind_and_has_no_threshold_overrides(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not read artifacts or call generator")

    monkeypatch.setattr(evaluator, "audit_collection", forbidden)
    monkeypatch.setattr(collector, "run_evolution", forbidden)

    plan = evaluator.build_evaluation_plan()

    assert plan["mode"] == "plan_only_no_collection_read"
    assert plan["protocol_sha256"] == protocol.FROZEN_PROTOCOL_SHA256
    assert plan["required_case_count"] == 12
    assert plan["threshold_overrides_allowed"] is False
    assert plan["artifact_read_started"] is False
    assert plan["generation_started"] is False
    assert set(inspect.signature(evaluator.evaluate_collection).parameters) == {
        "collection_dir",
        "confirmed_protocol_sha256",
    }


@pytest.mark.parametrize(
    ("delta_l1", "stop_work", "classification", "fallback"),
    [
        (0.005, 12.0, "supports_p6_on_frozen_artificial_development", None),
        (0.03, 12.0, "quality_only_failure_fallback_p12", 12),
        (0.005, 100.0, "compute_only_failure_fallback_p4", 4),
        (0.03, 100.0, "reject_b_redesign", None),
    ],
)
def test_complete_matrix_has_one_deterministic_classification(
    delta_l1,
    stop_work,
    classification,
    fallback,
):
    result = evaluator.evaluate_evidence_rows(
        _evidence_matrix(delta_l1=delta_l1, stop_work=stop_work)
    )

    assert result["evidence_gates"]["all_pass"] is True
    assert result["classification"] == classification
    assert result["fallback_patience_ticks"] == fallback
    assert result["maximum_fallback_attempts"] == 1
    assert result["third_patience_candidate_allowed"] is False
    assert result["post_result_threshold_retuning_allowed"] is False


@pytest.mark.parametrize("failure", ["too_many_c", "too_few_b", "coverage"])
def test_incomplete_evidence_never_becomes_quality_or_compute_failure(failure):
    rows = _evidence_matrix()
    if failure == "too_many_c":
        for row in rows[:3]:
            _turn_into_c(row)
    elif failure == "too_few_b":
        for row in rows[:7]:
            _turn_into_a(row)
    else:
        for row in rows[:3]:
            _set_censored(row, 6)

    result = evaluator.evaluate_evidence_rows(rows)

    assert result["evidence_gates"]["all_pass"] is False
    assert result["quality_gates"]["evaluable"] is False
    assert result["quality_gates"]["pass"] is None
    assert result["compute_gate"]["evaluable"] is False
    assert result["compute_gate"]["pass"] is None
    assert result["classification"] == "insufficient_evidence_no_p_change"
    assert result["fallback_patience_ticks"] is None


def test_missing_family_checkpoint_is_insufficient_not_imputed():
    rows = _evidence_matrix()
    for row in rows:
        if row["family"] == "binary_chain_4":
            _turn_into_a(row)

    result = evaluator.evaluate_evidence_rows(rows)

    assert result["termination_counts"] == {
        "fit_target_reached": 6,
        "early_stopped": 6,
        "resource_cap_reached": 0,
    }
    assert result["evidence_gates"]["b_case_count"]["pass"] is True
    assert result["evidence_gates"]["checkpoint_coverage"]["6"]["pass"] is True
    assert result["evidence_gates"]["family_checkpoint_presence"]["pass"] is False
    assert result["classification"] == "insufficient_evidence_no_p_change"


def test_opposite_family_directions_reject_single_fallback():
    rows = _evidence_matrix()
    for row in rows:
        if row["family"] == "binary_chain_4":
            row["stop_normalized_work"] = 12.0
            for checkpoint in row["checkpoints"]:
                checkpoint["current_normalized_l1"] = 0.07
        else:
            row["stop_normalized_work"] = 100.0
            for checkpoint in row["checkpoints"]:
                checkpoint["current_normalized_l1"] = 0.095

    result = evaluator.evaluate_evidence_rows(rows)

    assert result["quality_gates"]["pass"] is False
    assert result["compute_gate"]["pass"] is True
    assert {item["direction"] for item in result["family_directions"].values()} == {
        "increase_P",
        "decrease_P",
    }
    assert result["opposite_family_direction_conflict"] is True
    assert result["classification"] == "reject_b_redesign"
    assert result["fallback_patience_ticks"] is None


def test_large_degradation_boundary_is_strictly_greater_than_point_02():
    rows = _evidence_matrix(delta_l1=0.0)
    for row in rows[:3]:
        row["terminal_current_normalized_l1"] = 0.02
        for checkpoint in row["checkpoints"]:
            checkpoint["current_normalized_l1"] = 0.0

    result = evaluator.evaluate_evidence_rows(rows)

    for checkpoint in result["quality_gates"]["checkpoints"].values():
        assert checkpoint["large_degradation_count"] == 0
        assert checkpoint["large_degradation_tail_pass"] is True
    assert result["quality_gates"]["pass"] is True


def test_right_censoring_uses_only_observed_b_cases_after_coverage_passes():
    rows = _evidence_matrix()
    for row in rows[:2]:
        _set_censored(row, 6)
        _set_censored(row, 12)

    result = evaluator.evaluate_evidence_rows(rows)

    assert result["evidence_gates"]["all_pass"] is True
    for checkpoint in result["quality_gates"]["checkpoints"].values():
        assert checkpoint["observed_count"] == 10
        assert len(checkpoint["delta_l1_values"]) == 10
    assert len(result["compute_gate"]["saving_12_values"]) == 10


def test_evidence_contract_rejects_order_unknown_fields_and_a_priority_conflicts():
    rows = _evidence_matrix()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="冻结 case 顺序"):
        evaluator.evaluate_evidence_rows(rows)

    rows = _evidence_matrix()
    rows[0]["unexpected"] = True
    with pytest.raises(ValueError, match="字段不一致"):
        evaluator.evaluate_evidence_rows(rows)

    rows = _evidence_matrix()
    rows[0]["terminal_current_squared_loss"] = 0.0
    rows[0]["terminal_current_normalized_l1"] = 0.0
    with pytest.raises(ValueError, match="A 优先级"):
        evaluator.evaluate_evidence_rows(rows)


def _synthetic_b_diagnostics(case: dict) -> dict:
    metrics = [
        {
            "state_index": state_index,
            "round": state_index,
            "phase": "initial" if state_index == 0 else "post_round",
            "current_normalized_l1": 0.1,
            "current_squared_loss": 10.0,
        }
        for state_index in range(7)
    ]
    clocks = [
        {
            "state_index": state_index,
            "round": state_index,
            "attempts": [{"participating_rows": case["n_records"]}],
            "accepted_attempt": 1,
            "candidate_evaluation_count_cumulative": state_index,
            "post_current_table_sha256": f"{state_index:064x}",
            "primary_rng_state_sha256": f"{state_index + 20:064x}",
            "factorized_gibbs_rng_state_sha256": None,
        }
        for state_index in range(1, 7)
    ]
    last_decision = {
        "phase": "post_round",
        "state_index": 6,
        "current_loss": 10.0,
        "best_loss_diagnostic_only": 10.0,
        "best_state_index_diagnostic_only": 0,
        "best_updated": False,
        "applied_participating_rows": case["n_records"],
        "cumulative_participating_rows": 6 * case["n_records"],
        "normalized_work": 6.0,
        "work_tick_completed": True,
        "completed_tick_had_progress": False,
        "completed_work_ticks": 6,
        "consecutive_no_progress_ticks": 6,
        "external_resource_cap_reached": False,
        "termination_reason": "early_stopped",
        "fit_target_reached": False,
        "inner_complete": True,
        "terminal_output_state_index": 6,
        "terminal_output_loss": 10.0,
    }
    return {
        "rounds_run": 6,
        "candidate_evaluation_count": 6,
        "current_state_metrics_history": metrics,
        "transition_clock_history": clocks,
        "accept_history": [True] * 6,
        "proposal_attempts_history": [1] * 6,
        "accepted_attempt_history": [1] * 6,
        "inner_early_stopping": {"last_decision": last_decision},
    }


def test_online_a_b_c_decision_is_replayed_from_loss_and_natural_work():
    case = protocol.primary_case_matrix()[0]
    diagnostics = _synthetic_b_diagnostics(case)

    replayed = evaluator._replay_online_stopping_decision(case, diagnostics)

    assert replayed["termination_reason"] == "early_stopped"
    assert replayed["normalized_work"] == 6.0
    assert replayed["consecutive_no_progress_ticks"] == 6

    tampered = copy.deepcopy(diagnostics)
    tampered["current_state_metrics_history"][1]["current_squared_loss"] = 0.0
    with pytest.raises(RuntimeError, match="越过了更早"):
        evaluator._replay_online_stopping_decision(case, tampered)


def test_b_shadow_artifacts_are_independently_replayed_and_audited(
    tmp_path,
    monkeypatch,
):
    case = protocol.primary_case_matrix()[0]
    workload = collector.materialize_family(case["family"])
    frame, answers, loss, trace, metrics, clocks = _all_zero_trace(
        workload,
        rounds=case["n_rounds"],
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
    synthetic_decision = _synthetic_b_diagnostics(case)["inner_early_stopping"][
        "last_decision"
    ]
    for key in (
        "current_loss",
        "best_loss_diagnostic_only",
        "terminal_output_loss",
    ):
        synthetic_decision[key] = loss
    online["inner_early_stopping"] = {
        "enabled": True,
        "patience_ticks": case["patience_ticks"],
        "last_decision": synthetic_decision,
        "resource_cap_source_diagnostic_only": None,
    }
    online["params"] = collector._generator_kwargs(case, shadow=False)
    shadow = _fake_diagnostics(
        workload,
        frame,
        answers,
        loss,
        trace,
        metrics,
        clocks,
        rounds_run=case["n_rounds"],
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
    execution_sha = "e" * 64
    manifest_path = collector._collect_case(
        cases_dir,
        case,
        workload,
        execution_sha,
    )

    evidence = evaluator._audit_case_manifest(
        manifest_path,
        case,
        execution_sha,
    )

    assert evidence["termination_reason"] == "early_stopped"
    assert evidence["stop_normalized_work"] == 6.0
    assert [item["work_offset"] for item in evidence["checkpoints"]] == [6, 12]
    assert all(item["status"] == "observed" for item in evidence["checkpoints"])


def _reference_frame(family_name: str) -> pd.DataFrame:
    family = next(
        item for item in protocol.family_manifests() if item["family"] == family_name
    )
    rows = []
    for item in family["reference_multiset"]:
        row = dict(zip(family["attribute_order"], item["state"], strict=True))
        rows.extend([row.copy() for _ in range(item["count"])])
    return pd.DataFrame(rows, columns=family["attribute_order"])


def _zero_loss_online_diagnostics(case: dict, table_sha: str) -> dict:
    params = collector._generator_kwargs(case, shadow=False)
    params["tol"] = "positive_infinity"
    decision = {
        "phase": "initial",
        "state_index": 0,
        "current_loss": 0.0,
        "best_loss_diagnostic_only": 0.0,
        "best_state_index_diagnostic_only": 0,
        "best_updated": True,
        "applied_participating_rows": 0,
        "cumulative_participating_rows": 0,
        "normalized_work": 0.0,
        "work_tick_completed": False,
        "completed_tick_had_progress": None,
        "completed_work_ticks": 0,
        "consecutive_no_progress_ticks": 0,
        "external_resource_cap_reached": False,
        "termination_reason": "fit_target_reached",
        "fit_target_reached": True,
        "inner_complete": True,
        "terminal_output_state_index": 0,
        "terminal_output_loss": 0.0,
    }
    diagnostics = dict.fromkeys(collector._ONLINE_DIAGNOSTIC_KEYS)
    diagnostics.update(
        {
            "rounds_run": 0,
            "termination_reason": "fit_target_reached",
            "stopped_early": True,
            "fit_target_reached": True,
            "inner_complete": True,
            "output_table_identity": "terminal_current",
            "output_squared_loss": 0.0,
            "final_current_squared_loss": 0.0,
            "final_current_normalized_l1": 0.0,
            "normalized_l1_error": 0.0,
            "normalized_l1_median": 0.0,
            "normalized_l1_p90": 0.0,
            "normalized_l1_max": 0.0,
            "best_loss_diagnostic_only": 0.0,
            "normalized_l1_at_best_squared_loss_diagnostic_only": 0.0,
            "state_evaluation_count": 1,
            "candidate_evaluation_count": 0,
            "candidate_budget_exhausted": False,
            "initial_table_sha256": table_sha,
            "primary_rng_post_initialization_state_sha256": "a" * 64,
            "primary_rng_state_sha256": "a" * 64,
            "current_state_metrics_history": [
                {
                    "state_index": 0,
                    "round": 0,
                    "phase": "initial",
                    "current_normalized_l1": 0.0,
                    "current_squared_loss": 0.0,
                }
            ],
            "transition_clock_history": [],
            "accept_history": [],
            "proposal_attempts_history": [],
            "accepted_attempt_history": [],
            "inner_early_stopping": {
                "enabled": True,
                "patience_ticks": case["patience_ticks"],
                "last_decision": decision,
                "resource_cap_source_diagnostic_only": None,
            },
            "elapsed_sec": 0.0,
            "sec_per_round": 0.0,
            "params": params,
        }
    )
    return diagnostics


def _build_fake_all_a_collection(tmp_path):
    destination = tmp_path / "fake_collection"
    destination.mkdir()
    execution = {
        "contract_version": collector.COLLECTION_CONTRACT_VERSION,
        "created_at_utc": "2026-08-17T00:00:00+00:00",
        "git_commit": "a" * 40,
        "git_worktree_clean_including_untracked": True,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "protocol": protocol.frozen_protocol_manifest(),
        "source_sha256": {str(path): "b" * 64 for path in collector.SOURCE_PATHS},
        "environment": {
            "python_version": "fake",
            "numpy_version": "fake",
            "pandas_version": "fake",
            "platform": "fake",
            "machine": "fake",
            "processor": "fake",
            "device": "numpy",
        },
        "execution_started": False,
        "formal_rng_instantiated": False,
        "acceptance_evaluated": False,
    }
    execution_path = destination / "execution_manifest.json"
    collector._write_json_exclusive(execution_path, execution)
    execution_sha = collector._sha256_file(execution_path)
    cases_dir = destination / "cases"
    cases_dir.mkdir()
    case_files = {}
    frames = {}
    for case in protocol.primary_case_matrix():
        workload = collector.materialize_family(case["family"])
        frame = frames.setdefault(case["family"], _reference_frame(case["family"]))
        case_dir = cases_dir / case["case_id"]
        case_dir.mkdir()
        table_path = case_dir / "terminal_current.csv"
        collector._write_frame_exclusive(table_path, frame)
        table_sha = collector._frame_sha256(frame)
        diagnostics_path = case_dir / "online_diagnostics.json"
        collector._write_json_exclusive(
            diagnostics_path,
            _zero_loss_online_diagnostics(case, table_sha),
        )
        case_manifest = {
            "contract_version": collector.COLLECTION_CONTRACT_VERSION,
            "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
            "execution_manifest_sha256": execution_sha,
            "case": case,
            "family_identity_sha256": workload.family_identity_sha256,
            "query_identity_sha256": workload.query_identity_sha256,
            "target_identity_sha256": workload.target_identity_sha256,
            "reference_multiset_passed_to_generator": False,
            "online": {
                "termination_reason": "fit_target_reached",
                "inner_complete": True,
                "stop_state_index": 0,
                "stop_normalized_work": 0.0,
                "candidate_evaluation_count": 0,
                "terminal_current_table_sha256": table_sha,
                "terminal_query_answers": workload.target.tolist(),
                "terminal_current_squared_loss": 0.0,
                "terminal_current_normalized_l1": 0.0,
                "historical_best_loss_diagnostic_only": 0.0,
                "files": {
                    "terminal_current_table": {
                        "path": table_path.name,
                        "sha256": collector._sha256_file(table_path),
                    },
                    "diagnostics": {
                        "path": diagnostics_path.name,
                        "sha256": collector._sha256_file(diagnostics_path),
                    },
                },
            },
            "shadow": {
                "collected": False,
                "role": "B_only_read_only_continuation",
                "prefix_audit": None,
                "termination_reason": None,
                "rounds_run": None,
                "candidate_evaluation_count": None,
                "checkpoints": [],
                "files": {},
            },
            "acceptance_evaluated": False,
            "partial_matrix_classification_emitted": False,
        }
        case_manifest_path = case_dir / "case_manifest.json"
        collector._write_json_exclusive(case_manifest_path, case_manifest)
        case_files[case["case_id"]] = {
            "path": str(case_manifest_path.relative_to(destination)),
            "sha256": collector._sha256_file(case_manifest_path),
        }
    collection = {
        "contract_version": collector.COLLECTION_CONTRACT_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "execution_manifest": {
            "path": execution_path.name,
            "sha256": execution_sha,
        },
        "formal_primary_collection_complete": True,
        "case_count": 12,
        "case_manifest_files": case_files,
        "collection_elapsed_sec": 0.0,
        "acceptance_evaluated": False,
        "partial_matrix_classification_emitted": False,
        "real_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    collector._write_json_exclusive(
        destination / "collection_manifest.json",
        collection,
    )
    return destination


def test_raw_artifact_auditor_recomputes_complete_fake_matrix(tmp_path):
    destination = _build_fake_all_a_collection(tmp_path)

    audit = evaluator.audit_collection(destination)
    decision = evaluator.evaluate_evidence_rows(audit["evidence_rows"])

    assert audit["all_artifacts_verified"] is True
    assert audit["case_count"] == 12
    assert len(audit["case_manifest_sha256"]) == 12
    assert all(
        row["termination_reason"] == "fit_target_reached"
        and row["terminal_current_squared_loss"] == 0.0
        and row["terminal_current_normalized_l1"] == 0.0
        for row in audit["evidence_rows"]
    )
    assert decision["classification"] == "insufficient_evidence_no_p_change"


def test_raw_artifact_auditor_rejects_tampered_file_hash(tmp_path):
    destination = _build_fake_all_a_collection(tmp_path)
    first_case = protocol.primary_case_matrix()[0]
    table_path = destination / "cases" / first_case["case_id"] / "terminal_current.csv"
    table_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 不一致"):
        evaluator.audit_collection(destination)


def test_artifact_paths_cannot_escape_collection_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="逃逸"):
        evaluator._safe_artifact_path(root, "../outside.json", "artifact")


def test_wrong_sha_fails_before_environment_artifact_or_generator(
    tmp_path,
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong SHA must fail before any read or generation")

    monkeypatch.setattr(evaluator, "_evaluation_environment", forbidden)
    monkeypatch.setattr(evaluator, "audit_collection", forbidden)
    monkeypatch.setattr(collector, "run_evolution", forbidden)

    with pytest.raises(ValueError, match="显式确认"):
        evaluator.evaluate_collection(tmp_path / "missing", "0" * 64)


def test_dirty_tree_fails_before_source_hash_artifact_or_generator(
    tmp_path,
    monkeypatch,
):
    git_calls = []

    def fake_git(_root, *arguments):
        git_calls.append(arguments)
        return "?? untracked.py"

    def forbidden(*args, **kwargs):
        raise AssertionError("dirty tree must fail before hashing/read/generation")

    monkeypatch.setattr(collector, "_git_text", fake_git)
    monkeypatch.setattr(collector, "_sha256_file", forbidden)
    monkeypatch.setattr(evaluator, "audit_collection", forbidden)
    monkeypatch.setattr(collector, "run_evolution", forbidden)

    with pytest.raises(RuntimeError, match="干净工作树"):
        evaluator.evaluate_collection(
            tmp_path,
            protocol.FROZEN_PROTOCOL_SHA256,
        )
    assert git_calls == [("status", "--porcelain", "--untracked-files=all")]


def test_formal_evaluation_writes_once_without_generator(tmp_path, monkeypatch):
    rows = _evidence_matrix()
    source_sha = {str(path): "c" * 64 for path in evaluator.SOURCE_PATHS}
    environment = {
        "evaluated_at_utc": "2026-08-17T00:00:00+00:00",
        "git_commit": "d" * 40,
        "git_worktree_clean_including_untracked": True,
        "source_sha256": source_sha,
        "runtime": {
            "python_version": "fake-python",
            "numpy_version": "fake-numpy",
            "pandas_version": "fake-pandas",
            "device": "numpy",
        },
    }
    audit = {
        "collection_manifest_path": "fake",
        "collection_manifest_sha256": "e" * 64,
        "execution_manifest_sha256": "f" * 64,
        "execution_git_commit": environment["git_commit"],
        "execution_source_sha256": {
            str(path): source_sha[str(path)] for path in collector.SOURCE_PATHS
        },
        "execution_environment": {
            **environment["runtime"],
            "platform": "fake",
            "machine": "fake",
            "processor": "fake",
        },
        "case_manifest_sha256": {},
        "case_count": 12,
        "all_artifacts_verified": True,
        "acceptance_evaluated": False,
        "evidence_rows": rows,
    }

    monkeypatch.setattr(evaluator, "_evaluation_environment", lambda _root: environment)
    monkeypatch.setattr(
        evaluator, "audit_collection", lambda _path: copy.deepcopy(audit)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluator must never call generator")

    monkeypatch.setattr(collector, "run_evolution", forbidden)

    report_path = evaluator.evaluate_collection(
        tmp_path,
        protocol.FROZEN_PROTOCOL_SHA256,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["decision"]["classification"] == (
        "supports_p6_on_frozen_artificial_development"
    )
    assert report["generator_called"] is False
    assert report["real_data_accessed"] is False
    assert report["privacy_budget_consumed"] is False
    with pytest.raises(FileExistsError, match="已存在"):
        evaluator.evaluate_collection(
            tmp_path,
            protocol.FROZEN_PROTOCOL_SHA256,
        )


def test_formal_evaluation_rejects_collection_runtime_drift(tmp_path, monkeypatch):
    source_sha = {str(path): "a" * 64 for path in evaluator.SOURCE_PATHS}
    environment = {
        "evaluated_at_utc": "2026-08-17T00:00:00+00:00",
        "git_commit": "b" * 40,
        "git_worktree_clean_including_untracked": True,
        "source_sha256": source_sha,
        "runtime": {
            "python_version": "same-python",
            "numpy_version": "new-numpy",
            "pandas_version": "same-pandas",
            "device": "numpy",
        },
    }
    audit = {
        "execution_git_commit": environment["git_commit"],
        "execution_source_sha256": {
            str(path): source_sha[str(path)] for path in collector.SOURCE_PATHS
        },
        "execution_environment": {
            "python_version": "same-python",
            "numpy_version": "old-numpy",
            "pandas_version": "same-pandas",
            "platform": "fake",
            "machine": "fake",
            "processor": "fake",
            "device": "numpy",
        },
        "evidence_rows": _evidence_matrix(),
    }
    monkeypatch.setattr(
        evaluator,
        "_evaluation_environment",
        lambda _root: environment,
    )
    monkeypatch.setattr(evaluator, "audit_collection", lambda _path: audit)

    with pytest.raises(RuntimeError, match="runtime 漂移：numpy_version"):
        evaluator.evaluate_collection(
            tmp_path,
            protocol.FROZEN_PROTOCOL_SHA256,
        )
    assert not (tmp_path / evaluator.EVALUATION_REPORT_FILENAME).exists()
