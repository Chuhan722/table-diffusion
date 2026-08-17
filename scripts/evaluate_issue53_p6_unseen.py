#!/usr/bin/env python
"""Audit and evaluate the frozen Issue #53 P=6 raw collection.

The plan command is result-blind.  Formal evaluation accepts only a complete
raw collection bound to the frozen protocol SHA, verifies every referenced
artifact, then applies the preregistered evidence, quality, compute, and
fallback rules without exposing threshold overrides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import collect_issue53_p6_unseen as collector
    from scripts import issue53_p6_unseen_protocol as protocol
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import collect_issue53_p6_unseen as collector
    import issue53_p6_unseen_protocol as protocol

from table_diffevo.inner_early_stopping import EarlyStoppingConfig, InnerEarlyStopper
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.stationarity import load_stationarity_trace

EVALUATION_CONTRACT_VERSION = "issue53-p6-unseen-evaluation-v1"
EVALUATION_REPORT_FILENAME = "p6_evaluation_report.json"
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *collector.SOURCE_PATHS,
            Path("scripts/evaluate_issue53_p6_unseen.py"),
        )
    )
)
FAMILY_NAMES = ("binary_chain_4", "mixed_2x3x2")
LARGE_DEGRADATION_DELTA_L1 = 0.02

_EVIDENCE_ROW_KEYS = {
    "case_id",
    "family",
    "seed",
    "rho",
    "termination_reason",
    "stop_normalized_work",
    "terminal_current_squared_loss",
    "terminal_current_normalized_l1",
    "checkpoints",
}
_EVIDENCE_CHECKPOINT_KEYS = {
    "work_offset",
    "status",
    "current_squared_loss",
    "current_normalized_l1",
}
_NORMAL_REASONS = {"fit_target_reached", "early_stopped"}
_ALL_REASONS = _NORMAL_REASONS | {"resource_cap_reached"}
_OBSERVED = "observed"
_CENSORED = "right_censored_by_resource_guard"
_PREFIX_AUDIT_KEYS = {
    "current_metrics_prefix_equal",
    "transition_clocks_prefix_equal",
    "accept_history_prefix_equal",
    "proposal_attempts_prefix_equal",
    "accepted_attempt_prefix_equal",
    "terminal_table_identity_equal",
    "terminal_query_vector_equal",
    "primary_rng_prefix_equal",
    "candidate_evaluations_prefix_equal",
}
_CURRENT_METRIC_KEYS = {
    "state_index",
    "round",
    "phase",
    "current_normalized_l1",
    "current_squared_loss",
}
_TRANSITION_CLOCK_KEYS = {
    "state_index",
    "round",
    "attempts",
    "accepted_attempt",
    "candidate_evaluation_count_cumulative",
    "post_current_table_sha256",
    "primary_rng_state_sha256",
    "factorized_gibbs_rng_state_sha256",
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是对象")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} 字段不一致；missing={sorted(expected - observed)}，"
            f"unknown={sorted(observed - expected)}"
        )


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} 必须是有限数值")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} 不得小于 {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} 不得大于 {maximum}")
    return normalized


def _validate_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("evidence rows 必须是列表")
    expected_cases = protocol.primary_case_matrix()
    if len(rows) != len(expected_cases):
        raise ValueError("evidence rows 必须恰好覆盖 12 条 primary cases")
    normalized_rows = json.loads(_strict_json_bytes(rows))
    for index, (row, expected_case) in enumerate(
        zip(normalized_rows, expected_cases, strict=True)
    ):
        name = f"evidence_rows[{index}]"
        _validate_exact_keys(row, _EVIDENCE_ROW_KEYS, name)
        for key in ("case_id", "family", "seed", "rho"):
            if (
                type(row[key]) is not type(expected_case[key])
                or row[key] != expected_case[key]
            ):
                raise ValueError(f"{name}.{key} 与冻结 case 顺序不一致")
        reason = row["termination_reason"]
        if reason not in _ALL_REASONS:
            raise ValueError(f"{name}.termination_reason 非法")
        _finite_float(
            row["stop_normalized_work"],
            f"{name}.stop_normalized_work",
            minimum=0.0,
        )
        terminal_loss = _finite_float(
            row["terminal_current_squared_loss"],
            f"{name}.terminal_current_squared_loss",
            minimum=0.0,
        )
        terminal_l1 = _finite_float(
            row["terminal_current_normalized_l1"],
            f"{name}.terminal_current_normalized_l1",
            minimum=0.0,
            maximum=1.0,
        )
        if reason == "fit_target_reached":
            if terminal_loss != 0.0 or terminal_l1 != 0.0:
                raise ValueError(f"{name} 的 A 必须对应精确零 terminal 指标")
        elif terminal_loss == 0.0 or terminal_l1 == 0.0:
            raise ValueError(f"{name} 的 A 优先级与 terminal 指标矛盾")
        checkpoints = row["checkpoints"]
        if reason != "early_stopped":
            if checkpoints != []:
                raise ValueError(f"{name} 只有 B 可以带 shadow checkpoints")
            continue
        if not isinstance(checkpoints, list) or len(checkpoints) != 2:
            raise ValueError(f"{name} 的 B 必须恰有 +6/+12 checkpoints")
        for checkpoint_index, (checkpoint, expected_offset) in enumerate(
            zip(checkpoints, protocol.SHADOW_WORK_OFFSETS, strict=True)
        ):
            checkpoint_name = f"{name}.checkpoints[{checkpoint_index}]"
            _validate_exact_keys(
                checkpoint,
                _EVIDENCE_CHECKPOINT_KEYS,
                checkpoint_name,
            )
            if checkpoint["work_offset"] != expected_offset:
                raise ValueError(f"{checkpoint_name} offset 顺序不一致")
            status = checkpoint["status"]
            if status not in {_OBSERVED, _CENSORED}:
                raise ValueError(f"{checkpoint_name}.status 非法")
            metric_keys = (
                "current_squared_loss",
                "current_normalized_l1",
            )
            if status == _CENSORED:
                if any(checkpoint[key] is not None for key in metric_keys):
                    raise ValueError("右删失 checkpoint 不得补入 terminal 指标")
                continue
            _finite_float(
                checkpoint["current_squared_loss"],
                f"{checkpoint_name}.current_squared_loss",
                minimum=0.0,
            )
            _finite_float(
                checkpoint["current_normalized_l1"],
                f"{checkpoint_name}.current_normalized_l1",
                minimum=0.0,
                maximum=1.0,
            )
    return normalized_rows


def _checkpoint_for(row: dict[str, Any], offset: int) -> dict[str, Any]:
    return next(
        checkpoint
        for checkpoint in row["checkpoints"]
        if checkpoint["work_offset"] == offset
    )


def _direction(quality_pass: bool, compute_pass: bool) -> str:
    if quality_pass and compute_pass:
        return "none"
    if not quality_pass and compute_pass:
        return "increase_P"
    if quality_pass and not compute_pass:
        return "decrease_P"
    return "redesign"


def evaluate_evidence_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen gates to one complete ordered primary matrix."""

    rows = _validate_evidence_rows(rows)
    acceptance = protocol.frozen_protocol_manifest()["acceptance"]
    if acceptance["large_degradation_definition"] != (
        "delta_l1_strictly_greater_than_0.02"
    ):
        raise RuntimeError("large degradation 冻结定义发生变化")
    normal_count = sum(row["termination_reason"] in _NORMAL_REASONS for row in rows)
    c_count = sum(row["termination_reason"] == "resource_cap_reached" for row in rows)
    b_rows = [row for row in rows if row["termination_reason"] == "early_stopped"]
    b_count = len(b_rows)
    coverage = {}
    family_presence = {}
    for offset in protocol.SHADOW_WORK_OFFSETS:
        observed = [
            row for row in b_rows if _checkpoint_for(row, offset)["status"] == _OBSERVED
        ]
        fraction = len(observed) / b_count if b_count else 0.0
        coverage[str(offset)] = {
            "eligible_b_case_count": b_count,
            "observed_count": len(observed),
            "right_censored_count": b_count - len(observed),
            "observed_fraction": fraction,
            "minimum_fraction": acceptance["checkpoint_coverage_minimum_fraction"],
            "pass": fraction >= acceptance["checkpoint_coverage_minimum_fraction"],
        }
        family_presence[str(offset)] = {
            family: sum(row["family"] == family for row in observed)
            for family in FAMILY_NAMES
        }
    family_evidence_present = all(
        count > 0 for counts in family_presence.values() for count in counts.values()
    )
    evidence_gates = {
        "normal_completion": {
            "observed_count": normal_count,
            "minimum_count": acceptance["normal_completion_minimum_count"],
            "pass": normal_count >= acceptance["normal_completion_minimum_count"],
        },
        "resource_cap": {
            "observed_count": c_count,
            "maximum_count": acceptance["resource_cap_maximum_count"],
            "pass": c_count <= acceptance["resource_cap_maximum_count"],
        },
        "b_case_count": {
            "observed_count": b_count,
            "minimum_count": acceptance["b_case_minimum_count"],
            "pass": b_count >= acceptance["b_case_minimum_count"],
        },
        "checkpoint_coverage": coverage,
        "family_checkpoint_presence": {
            "observed_by_checkpoint_and_family": family_presence,
            "at_least_one_each_required": True,
            "pass": family_evidence_present,
        },
    }
    evidence_pass = bool(
        evidence_gates["normal_completion"]["pass"]
        and evidence_gates["resource_cap"]["pass"]
        and evidence_gates["b_case_count"]["pass"]
        and all(item["pass"] for item in coverage.values())
        and family_evidence_present
    )
    evidence_gates["all_pass"] = evidence_pass

    quality: dict[str, Any] = {
        "evaluable": evidence_pass,
        "checkpoints": {},
        "pass": None,
    }
    compute: dict[str, Any] = {
        "evaluable": evidence_pass,
        "saving_12_values": [],
        "median_saving_12": None,
        "minimum_median_saving_12": acceptance["median_saving_12_minimum"],
        "pass": None,
    }
    family_directions: dict[str, Any] = {}
    direction_conflict = False

    if evidence_pass:
        delta_by_offset: dict[int, list[tuple[dict[str, Any], float]]] = {}
        for offset in protocol.SHADOW_WORK_OFFSETS:
            values = []
            for row in b_rows:
                checkpoint = _checkpoint_for(row, offset)
                if checkpoint["status"] != _OBSERVED:
                    continue
                delta = float(
                    row["terminal_current_normalized_l1"]
                    - checkpoint["current_normalized_l1"]
                )
                values.append((row, delta))
            delta_by_offset[offset] = values
            deltas = [delta for _, delta in values]
            aggregate_median = float(median(deltas))
            large_count = sum(delta > LARGE_DEGRADATION_DELTA_L1 for delta in deltas)
            large_fraction = large_count / len(deltas)
            family_medians = {
                family: float(
                    median(delta for row, delta in values if row["family"] == family)
                )
                for family in FAMILY_NAMES
            }
            median_pass = (
                aggregate_median
                <= acceptance["median_delta_l1_maximum_each_checkpoint"]
            )
            tail_pass = (
                large_fraction
                <= acceptance["large_degradation_maximum_fraction_each_checkpoint"]
            )
            family_pass = all(
                value
                <= acceptance["per_family_median_delta_l1_maximum_each_checkpoint"]
                for value in family_medians.values()
            )
            quality["checkpoints"][str(offset)] = {
                "observed_count": len(deltas),
                "delta_l1_values": deltas,
                "median_delta_l1": aggregate_median,
                "maximum_median_delta_l1": acceptance[
                    "median_delta_l1_maximum_each_checkpoint"
                ],
                "median_pass": median_pass,
                "large_degradation_count": large_count,
                "large_degradation_fraction": large_fraction,
                "maximum_large_degradation_fraction": acceptance[
                    "large_degradation_maximum_fraction_each_checkpoint"
                ],
                "large_degradation_tail_pass": tail_pass,
                "family_median_delta_l1": family_medians,
                "maximum_family_median_delta_l1": acceptance[
                    "per_family_median_delta_l1_maximum_each_checkpoint"
                ],
                "family_medians_pass": family_pass,
                "pass": median_pass and tail_pass and family_pass,
            }
        quality["pass"] = all(item["pass"] for item in quality["checkpoints"].values())

        observed_plus_12 = [
            row for row in b_rows if _checkpoint_for(row, 12)["status"] == _OBSERVED
        ]
        savings = [
            12.0 / (float(row["stop_normalized_work"]) + 12.0)
            for row in observed_plus_12
        ]
        compute["saving_12_values"] = savings
        compute["median_saving_12"] = float(median(savings))
        compute["pass"] = bool(
            compute["median_saving_12"] >= acceptance["median_saving_12_minimum"]
        )

        for family in FAMILY_NAMES:
            family_quality_pass = all(
                median(
                    delta
                    for row, delta in delta_by_offset[offset]
                    if row["family"] == family
                )
                <= acceptance["per_family_median_delta_l1_maximum_each_checkpoint"]
                for offset in protocol.SHADOW_WORK_OFFSETS
            )
            family_savings = [
                12.0 / (float(row["stop_normalized_work"]) + 12.0)
                for row in observed_plus_12
                if row["family"] == family
            ]
            family_compute_pass = bool(
                median(family_savings) >= acceptance["median_saving_12_minimum"]
            )
            family_directions[family] = {
                "quality_pass": family_quality_pass,
                "median_saving_12": float(median(family_savings)),
                "compute_pass": family_compute_pass,
                "direction": _direction(
                    family_quality_pass,
                    family_compute_pass,
                ),
            }
        observed_directions = {item["direction"] for item in family_directions.values()}
        direction_conflict = {
            "increase_P",
            "decrease_P",
        }.issubset(observed_directions)

    if not evidence_pass:
        classification = "insufficient_evidence_no_p_change"
        next_action = "review_c_and_observation_range_without_changing_p"
        fallback_patience = None
    elif direction_conflict:
        classification = "reject_b_redesign"
        next_action = "opposite_family_directions_reject_b"
        fallback_patience = None
    elif quality["pass"] and compute["pass"]:
        classification = "supports_p6_on_frozen_artificial_development"
        next_action = "accept_p6_for_current_development_stage"
        fallback_patience = None
    elif not quality["pass"] and compute["pass"]:
        classification = "quality_only_failure_fallback_p12"
        next_action = "one_fallback_on_independent_seeds"
        fallback_patience = 12
    elif quality["pass"] and not compute["pass"]:
        classification = "compute_only_failure_fallback_p4"
        next_action = "one_fallback_on_independent_seeds"
        fallback_patience = 4
    else:
        classification = "reject_b_redesign"
        next_action = "quality_and_compute_failure_reject_b"
        fallback_patience = None

    result = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "protocol_sha256": protocol.assert_frozen_protocol_identity(),
        "case_count": len(rows),
        "termination_counts": {
            "fit_target_reached": sum(
                row["termination_reason"] == "fit_target_reached" for row in rows
            ),
            "early_stopped": b_count,
            "resource_cap_reached": c_count,
        },
        "evidence_gates": evidence_gates,
        "quality_gates": quality,
        "compute_gate": compute,
        "family_directions": family_directions,
        "opposite_family_direction_conflict": direction_conflict,
        "classification": classification,
        "next_action": next_action,
        "fallback_patience_ticks": fallback_patience,
        "maximum_fallback_attempts": 1,
        "third_patience_candidate_allowed": False,
        "post_result_threshold_retuning_allowed": False,
        "claim_scope": (
            "two_public_artificial_families_development_only_not_convergence"
        ),
    }
    _strict_json_bytes(result)
    return result


def build_evaluation_plan() -> dict[str, Any]:
    plan = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "mode": "plan_only_no_collection_read",
        "protocol_sha256": protocol.assert_frozen_protocol_identity(),
        "required_collection_contract": collector.COLLECTION_CONTRACT_VERSION,
        "required_case_count": 12,
        "classification_values": [
            "supports_p6_on_frozen_artificial_development",
            "quality_only_failure_fallback_p12",
            "compute_only_failure_fallback_p4",
            "insufficient_evidence_no_p_change",
            "reject_b_redesign",
        ],
        "threshold_overrides_allowed": False,
        "artifact_read_started": False,
        "generation_started": False,
    }
    _strict_json_bytes(plan)
    return plan


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 {name}：{path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON 对象")
    _strict_json_bytes(value)
    return value


def _safe_artifact_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{name}.path 必须是非空相对路径")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise RuntimeError(f"{name}.path 不得是绝对路径")
    root = root.resolve()
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{name}.path 逃逸 artifact root") from exc
    if not candidate.is_file():
        raise RuntimeError(f"{name} 文件不存在：{candidate}")
    return candidate


def _verify_file_info(root: Path, info: Any, name: str) -> Path:
    _validate_exact_keys(info, {"path", "sha256"}, name)
    path = _safe_artifact_path(root, info["path"], name)
    if info["sha256"] != collector._sha256_file(path):
        raise RuntimeError(f"{name} SHA-256 不一致")
    return path


def _audit_terminal_table(
    case_dir: Path,
    online: dict[str, Any],
    workload: collector.ArtificialWorkload,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    files = online["files"]
    _validate_exact_keys(
        files,
        {"terminal_current_table", "diagnostics"},
        "online.files",
    )
    table_path = _verify_file_info(
        case_dir,
        files["terminal_current_table"],
        "online terminal table",
    )
    frame = pd.read_csv(table_path)
    if frame.columns.tolist() != workload.schema.attribute_names():
        raise RuntimeError("terminal table columns 与冻结 schema 不一致")
    if len(frame) != workload.n_records or frame.isna().any().any():
        raise RuntimeError("terminal table 行数或缺失值非法")
    for attribute in workload.schema.attributes:
        if not set(frame[attribute.name].tolist()).issubset(attribute.values):
            raise RuntimeError(f"terminal table 的 {attribute.name} 越出 domain")
    table_sha = collector._frame_sha256(frame)
    if table_sha != online["terminal_current_table_sha256"]:
        raise RuntimeError("terminal table identity SHA-256 不一致")
    answers = np.asarray(evaluate_table(frame, workload.queries), dtype=float)
    loss = float(compute_squared_loss(workload.target, answers))
    l1 = float(compute_normalized_l1(workload.target, answers, workload.n_records))
    if [float(value) for value in answers] != online["terminal_query_answers"]:
        raise RuntimeError("terminal query answers 复算不一致")
    if loss != online["terminal_current_squared_loss"]:
        raise RuntimeError("terminal squared loss 复算不一致")
    if l1 != online["terminal_current_normalized_l1"]:
        raise RuntimeError("terminal normalized L1 复算不一致")
    return frame, answers, loss, l1


def _replay_online_stopping_decision(
    case: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Recompute A/B/C from saved current losses and natural-work clocks."""

    metrics = diagnostics["current_state_metrics_history"]
    clocks = diagnostics["transition_clock_history"]
    rounds_run = diagnostics["rounds_run"]
    stopper = InnerEarlyStopper(
        EarlyStoppingConfig(
            n_records=case["n_records"],
            patience_ticks=case["patience_ticks"],
        )
    )
    previous_candidates = 0
    replayed = None
    for state_index, metric in enumerate(metrics):
        _validate_exact_keys(
            metric,
            _CURRENT_METRIC_KEYS,
            f"online current metric {state_index}",
        )
        if (
            metric["state_index"] != state_index
            or metric["round"] != state_index
            or metric["phase"] != ("initial" if state_index == 0 else "post_round")
        ):
            raise RuntimeError("online current metric state/round/phase 不一致")
        loss = _finite_float(
            metric["current_squared_loss"],
            f"online current metric {state_index} loss",
            minimum=0.0,
        )
        _finite_float(
            metric["current_normalized_l1"],
            f"online current metric {state_index} L1",
            minimum=0.0,
            maximum=1.0,
        )
        if state_index == 0:
            replayed = stopper.observe_initial(
                loss,
                resource_cap_reached=(
                    case["n_rounds"] == 0 or case["candidate_budget"] == 0
                ),
            )
        else:
            clock = clocks[state_index - 1]
            _validate_exact_keys(
                clock,
                _TRANSITION_CLOCK_KEYS,
                f"online transition clock {state_index}",
            )
            if clock["state_index"] != state_index or clock["round"] != state_index:
                raise RuntimeError("online transition clock state/round 不一致")
            attempts = clock["attempts"]
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise RuntimeError("冻结无门控核每轮必须恰有一个 proposal attempt")
            if not isinstance(attempts[0], dict):
                raise RuntimeError("online proposal attempt 必须是对象")
            participating_rows = attempts[0].get("participating_rows")
            if (
                isinstance(participating_rows, bool)
                or not isinstance(participating_rows, int)
                or participating_rows < 0
                or participating_rows > case["n_records"]
            ):
                raise RuntimeError("online proposal participating rows 非法")
            accepted_attempt = clock["accepted_attempt"]
            if accepted_attempt != 1:
                raise RuntimeError("冻结无门控核每轮 proposal 必须直接生效")
            if (
                diagnostics["accept_history"][state_index - 1] is not True
                or diagnostics["proposal_attempts_history"][state_index - 1] != 1
                or diagnostics["accepted_attempt_history"][state_index - 1] != 1
            ):
                raise RuntimeError("online 无门控 accept/attempt 历史不一致")
            cumulative_candidates = clock["candidate_evaluation_count_cumulative"]
            if (
                isinstance(cumulative_candidates, bool)
                or not isinstance(cumulative_candidates, int)
                or cumulative_candidates != previous_candidates + 1
                or cumulative_candidates > case["candidate_budget"]
            ):
                raise RuntimeError("online candidate clock 不连续或越过 C")
            previous_candidates = cumulative_candidates
            applied_rows = collector._applied_rows(clock)
            if applied_rows < 0 or applied_rows > case["n_records"]:
                raise RuntimeError("online applied participating rows 非法")
            replayed = stopper.observe_post_round(
                current_loss=loss,
                participating_rows=applied_rows,
                resource_cap_reached=(
                    cumulative_candidates >= case["candidate_budget"]
                    or state_index >= case["n_rounds"]
                ),
            )
        assert replayed is not None
        if replayed.should_stop and state_index != rounds_run:
            raise RuntimeError("保存的 online 轨迹越过了更早的 A/B/C 终点")
    assert replayed is not None
    if not replayed.should_stop:
        raise RuntimeError("保存的 online terminal 未触发 A/B/C")
    if previous_candidates != diagnostics["candidate_evaluation_count"]:
        raise RuntimeError("online terminal candidate count 与 clocks 不一致")
    serialized = asdict(replayed)
    recorded = diagnostics["inner_early_stopping"]["last_decision"]
    if _strict_json_bytes(recorded) != _strict_json_bytes(serialized):
        raise RuntimeError("online A/B/C decision 无法由保存轨迹复算")
    return serialized


def _audit_online_diagnostics(
    case_dir: Path,
    case: dict[str, Any],
    online: dict[str, Any],
    table_sha: str,
) -> dict[str, Any]:
    diagnostics_path = _verify_file_info(
        case_dir,
        online["files"]["diagnostics"],
        "online diagnostics",
    )
    diagnostics = _read_json(diagnostics_path, "online diagnostics")
    if set(diagnostics) != set(collector._ONLINE_DIAGNOSTIC_KEYS):
        raise RuntimeError("online diagnostics 字段与 collector 契约不一致")
    reason = online["termination_reason"]
    if diagnostics["termination_reason"] != reason:
        raise RuntimeError("online reason 与 diagnostics 不一致")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError("diagnostics output 不是 terminal current")
    if diagnostics["rounds_run"] != online["stop_state_index"]:
        raise RuntimeError("online stop state 与 diagnostics rounds 不一致")
    if (
        isinstance(diagnostics["rounds_run"], bool)
        or not isinstance(diagnostics["rounds_run"], int)
        or diagnostics["rounds_run"] < 0
    ):
        raise RuntimeError("online rounds_run 非法")
    clocks = diagnostics["transition_clock_history"]
    histories = {
        "transition_clock_history": clocks,
        "accept_history": diagnostics["accept_history"],
        "proposal_attempts_history": diagnostics["proposal_attempts_history"],
        "accepted_attempt_history": diagnostics["accepted_attempt_history"],
    }
    if any(
        not isinstance(history, list) or len(history) != diagnostics["rounds_run"]
        for history in histories.values()
    ):
        raise RuntimeError("online transition clocks 数量不一致")
    metrics = diagnostics["current_state_metrics_history"]
    if not isinstance(metrics, list) or len(metrics) != diagnostics["rounds_run"] + 1:
        raise RuntimeError("online current metrics 数量不一致")
    recomputed_work = collector._normalized_work_from_clocks(
        clocks,
        case["n_records"],
    )
    if recomputed_work != online["stop_normalized_work"]:
        raise RuntimeError("online stop work 复算不一致")
    if (
        diagnostics["candidate_evaluation_count"]
        != online["candidate_evaluation_count"]
    ):
        raise RuntimeError("online candidate count 不一致")
    if (
        diagnostics["final_current_squared_loss"]
        != online["terminal_current_squared_loss"]
    ):
        raise RuntimeError("online terminal loss 与 diagnostics 不一致")
    if (
        diagnostics["final_current_normalized_l1"]
        != online["terminal_current_normalized_l1"]
    ):
        raise RuntimeError("online terminal L1 与 diagnostics 不一致")
    stopping = diagnostics["inner_early_stopping"]
    _validate_exact_keys(
        stopping,
        {
            "enabled",
            "patience_ticks",
            "last_decision",
            "resource_cap_source_diagnostic_only",
        },
        "online inner_early_stopping",
    )
    if (
        stopping["enabled"] is not True
        or stopping["patience_ticks"] != case["patience_ticks"]
    ):
        raise RuntimeError("online early-stopping 配置不一致")
    decision = _replay_online_stopping_decision(case, diagnostics)
    if (
        not isinstance(decision, dict)
        or decision.get("state_index") != online["stop_state_index"]
        or decision.get("termination_reason") != reason
        or decision["terminal_output_state_index"] != online["stop_state_index"]
        or decision["normalized_work"] != online["stop_normalized_work"]
        or decision.get("terminal_output_loss")
        != online["terminal_current_squared_loss"]
        or decision.get("inner_complete") is not (reason in _NORMAL_REASONS)
        or decision.get("fit_target_reached") is not (reason == "fit_target_reached")
    ):
        raise RuntimeError("online A/B/C decision 身份不一致")
    expected_complete = reason in _NORMAL_REASONS
    if (
        diagnostics["inner_complete"] is not expected_complete
        or online["inner_complete"] is not expected_complete
    ):
        raise RuntimeError("online inner_complete 与 reason 不一致")
    if diagnostics["fit_target_reached"] is not (reason == "fit_target_reached"):
        raise RuntimeError("online fit_target_reached 标志与 reason 不一致")
    if diagnostics["stopped_early"] is not (reason in _NORMAL_REASONS):
        raise RuntimeError("online stopped_early 标志与 reason 不一致")
    if diagnostics["output_squared_loss"] != online["terminal_current_squared_loss"]:
        raise RuntimeError("online output loss 不是 terminal current loss")
    if (
        diagnostics["state_evaluation_count"] != diagnostics["rounds_run"] + 1
        or metrics[-1]["current_squared_loss"]
        != online["terminal_current_squared_loss"]
        or metrics[-1]["current_normalized_l1"]
        != online["terminal_current_normalized_l1"]
    ):
        raise RuntimeError("online current metrics terminal 身份不一致")
    minimum_observed_loss = min(metric["current_squared_loss"] for metric in metrics)
    if (
        diagnostics["best_loss_diagnostic_only"] != minimum_observed_loss
        or online["historical_best_loss_diagnostic_only"] != minimum_observed_loss
    ):
        raise RuntimeError("online historical best diagnostic 复算不一致")
    expected_budget_exhausted = (
        diagnostics["candidate_evaluation_count"] >= case["candidate_budget"]
    )
    if diagnostics["candidate_budget_exhausted"] is not expected_budget_exhausted:
        raise RuntimeError("online candidate budget exhausted 标志不一致")
    expected_resource_source = None
    if decision["external_resource_cap_reached"]:
        expected_resource_source = (
            "candidate_budget" if expected_budget_exhausted else "max_rounds"
        )
    if stopping["resource_cap_source_diagnostic_only"] != expected_resource_source:
        raise RuntimeError("online resource cap source 复算不一致")
    params = diagnostics["params"]
    expected = collector._generator_kwargs(case, shadow=False)
    comparable = {
        "n_records",
        "n_rounds",
        "seed",
        "rho",
        "eta",
        "mu",
        "device",
        "init_method",
        "distance_mode",
        "max_retries",
        "residual_directed_diffusion",
        "diffusion_direction_strength",
        "diffusion_direction_normalization",
        "factorized_gibbs_sweeps",
        "candidate_budget",
        "residual_self_cooling",
        "alpha_schedule_mode",
        "fixed_alpha",
        "diffusion_direction_reference_scale",
        "diffusion_direction_logit_clip",
        "record_transition_clocks",
        "record_stationarity_trace",
        "stop_on_exact_residual",
        "horizon_invariant",
        "inner_early_stopping_patience_ticks",
    }
    for key in comparable:
        if params[key] != expected[key]:
            raise RuntimeError(f"online diagnostics.params.{key} 漂移")
    if params["tol"] != "positive_infinity":
        raise RuntimeError("online diagnostics tol 编码不一致")
    terminal_clock_sha = (
        diagnostics["initial_table_sha256"]
        if diagnostics["rounds_run"] == 0
        else clocks[-1]["post_current_table_sha256"]
    )
    if terminal_clock_sha != table_sha:
        raise RuntimeError("online terminal table 与 transition clock 不一致")
    return diagnostics


def _audit_b_shadow(
    case_dir: Path,
    case: dict[str, Any],
    online: dict[str, Any],
    diagnostics: dict[str, Any],
    workload: collector.ArtificialWorkload,
    shadow: dict[str, Any],
) -> list[dict[str, Any]]:
    if shadow["collected"] is not True:
        raise RuntimeError("B case 缺少 shadow")
    files = shadow["files"]
    _validate_exact_keys(
        files,
        {"trace_metadata", "trace_query_array", "summary"},
        "shadow.files",
    )
    metadata_path = _verify_file_info(
        case_dir,
        files["trace_metadata"],
        "shadow trace metadata",
    )
    query_array_path = _verify_file_info(
        case_dir,
        files["trace_query_array"],
        "shadow trace query array",
    )
    if metadata_path.parent != query_array_path.parent:
        raise RuntimeError("shadow trace 两个文件不在同一目录")
    trace = load_stationarity_trace(metadata_path.parent)
    if (
        trace.query_identity_sha256 != workload.query_identity_sha256
        or trace.target_identity_sha256 != workload.target_identity_sha256
    ):
        raise RuntimeError("shadow trace query/target identity 不一致")
    if shadow["termination_reason"] != trace.termination_reason:
        raise RuntimeError("shadow termination reason 不一致")
    if shadow["termination_reason"] not in {"candidate_budget", "max_rounds"}:
        raise RuntimeError("shadow 必须只由冻结 C 终止")
    if shadow["rounds_run"] != trace.post_round_count:
        raise RuntimeError("shadow rounds 与 trace 不一致")
    if shadow["rounds_run"] != case["n_rounds"]:
        raise RuntimeError("shadow 未达到冻结 raw-round C")
    if shadow["candidate_evaluation_count"] != case["candidate_budget"]:
        raise RuntimeError("shadow 未达到冻结 candidate C")
    stop_state = online["stop_state_index"]
    if stop_state >= trace.state_count:
        raise RuntimeError("shadow trace 未覆盖 B terminal")
    stop_observation = trace.observations[stop_state]
    if (
        stop_observation["current_table_sha256"]
        != online["terminal_current_table_sha256"]
        or [float(value) for value in trace.measured_query_answers[stop_state]]
        != online["terminal_query_answers"]
    ):
        raise RuntimeError("B terminal 与 shadow 同状态身份不一致")
    metrics = diagnostics["current_state_metrics_history"]
    clocks = diagnostics["transition_clock_history"]
    if len(metrics) != stop_state + 1 or len(clocks) != stop_state:
        raise RuntimeError("B online prefix 长度不一致")
    for state_index, metric in enumerate(metrics):
        observation = trace.observations[state_index]
        expected_metric = {
            "state_index": state_index,
            "round": state_index,
            "phase": "initial" if state_index == 0 else "post_round",
            "current_normalized_l1": observation["current_normalized_l1"],
            "current_squared_loss": observation["current_squared_loss"],
        }
        if metric != expected_metric:
            raise RuntimeError("B online/shadow current metrics prefix 不一致")
    for state_index, clock in enumerate(clocks, start=1):
        observation = trace.observations[state_index]
        if (
            clock["post_current_table_sha256"] != observation["current_table_sha256"]
            or clock["primary_rng_state_sha256"]
            != observation["primary_rng_state_sha256"]
            or clock["candidate_evaluation_count_cumulative"]
            != observation["candidate_evaluation_count_cumulative"]
            or collector._applied_rows(clock)
            != observation["applied_participating_row_count"]
            or len(clock["attempts"]) != observation["proposal_attempt_count"]
            or clock["accepted_attempt"] != observation["applied_attempt_index"]
            or diagnostics["accept_history"][state_index - 1]
            is not observation["proposal_accepted"]
            or diagnostics["proposal_attempts_history"][state_index - 1]
            != observation["proposal_attempt_count"]
            or diagnostics["accepted_attempt_history"][state_index - 1]
            != observation["applied_attempt_index"]
        ):
            raise RuntimeError("B online/shadow transition prefix 不一致")
    if (
        diagnostics["primary_rng_state_sha256"]
        != stop_observation["primary_rng_state_sha256"]
        or diagnostics["candidate_evaluation_count"]
        != stop_observation["candidate_evaluation_count_cumulative"]
    ):
        raise RuntimeError("B online/shadow terminal RNG/candidate prefix 不一致")
    prefix_audit = shadow["prefix_audit"]
    _validate_exact_keys(prefix_audit, _PREFIX_AUDIT_KEYS, "shadow.prefix_audit")
    if not all(value is True for value in prefix_audit.values()):
        raise RuntimeError("B shadow prefix audit 未全部通过")
    checkpoints = collector.locate_b_shadow_checkpoints(trace, stop_state)
    if checkpoints != shadow["checkpoints"]:
        raise RuntimeError("B shadow checkpoints 复算不一致")
    summary_path = _verify_file_info(
        case_dir,
        files["summary"],
        "shadow summary",
    )
    summary = _read_json(summary_path, "shadow summary")
    expected_summary_keys = {
        "termination_reason",
        "rounds_run",
        "candidate_evaluation_count",
        "final_current_squared_loss",
        "final_current_normalized_l1",
        "final_current_table_sha256",
    }
    _validate_exact_keys(summary, expected_summary_keys, "shadow summary")
    if (
        summary["termination_reason"] != trace.termination_reason
        or summary["rounds_run"] != trace.post_round_count
        or summary["candidate_evaluation_count"]
        != trace.observations[-1]["candidate_evaluation_count_cumulative"]
        or summary["final_current_squared_loss"]
        != trace.observations[-1]["current_squared_loss"]
        or summary["final_current_normalized_l1"]
        != trace.observations[-1]["current_normalized_l1"]
        or summary["final_current_table_sha256"]
        != trace.observations[-1]["current_table_sha256"]
    ):
        raise RuntimeError("shadow summary 与 trace 末状态不一致")
    return checkpoints


def _audit_case_manifest(
    case_manifest_path: Path,
    expected_case: dict[str, Any],
    execution_manifest_sha256: str,
) -> dict[str, Any]:
    case_dir = case_manifest_path.parent
    manifest = _read_json(case_manifest_path, "case manifest")
    expected_keys = {
        "contract_version",
        "protocol_sha256",
        "execution_manifest_sha256",
        "case",
        "family_identity_sha256",
        "query_identity_sha256",
        "target_identity_sha256",
        "reference_multiset_passed_to_generator",
        "online",
        "shadow",
        "acceptance_evaluated",
        "partial_matrix_classification_emitted",
    }
    _validate_exact_keys(manifest, expected_keys, "case manifest")
    if (
        manifest["contract_version"] != collector.COLLECTION_CONTRACT_VERSION
        or manifest["protocol_sha256"] != protocol.assert_frozen_protocol_identity()
        or manifest["execution_manifest_sha256"] != execution_manifest_sha256
        or _strict_json_bytes(manifest["case"]) != _strict_json_bytes(expected_case)
    ):
        raise RuntimeError("case manifest contract/protocol/case identity 不一致")
    if (
        manifest["reference_multiset_passed_to_generator"] is not False
        or manifest["acceptance_evaluated"] is not False
        or manifest["partial_matrix_classification_emitted"] is not False
    ):
        raise RuntimeError("case collection 越过 raw-only 边界")
    workload = collector.materialize_family(expected_case["family"])
    if (
        manifest["family_identity_sha256"] != workload.family_identity_sha256
        or manifest["query_identity_sha256"] != workload.query_identity_sha256
        or manifest["target_identity_sha256"] != workload.target_identity_sha256
    ):
        raise RuntimeError("case family/query/target identity 不一致")
    online = manifest["online"]
    expected_online_keys = {
        "termination_reason",
        "inner_complete",
        "stop_state_index",
        "stop_normalized_work",
        "candidate_evaluation_count",
        "terminal_current_table_sha256",
        "terminal_query_answers",
        "terminal_current_squared_loss",
        "terminal_current_normalized_l1",
        "historical_best_loss_diagnostic_only",
        "files",
    }
    _validate_exact_keys(online, expected_online_keys, "case online")
    reason = online["termination_reason"]
    if reason not in _ALL_REASONS:
        raise RuntimeError("case online termination reason 非法")
    for key, limit in (
        ("stop_state_index", expected_case["n_rounds"]),
        ("candidate_evaluation_count", expected_case["candidate_budget"]),
    ):
        value = online[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > limit
        ):
            raise RuntimeError(f"case online {key} 越出冻结范围")
    _finite_float(
        online["stop_normalized_work"],
        "case online stop_normalized_work",
        minimum=0.0,
    )
    historical_best = _finite_float(
        online["historical_best_loss_diagnostic_only"],
        "case online historical_best_loss_diagnostic_only",
        minimum=0.0,
    )
    frame, _, loss, l1 = _audit_terminal_table(
        case_dir,
        online,
        workload,
    )
    if historical_best > loss:
        raise RuntimeError("historical best loss 不得大于 terminal current loss")
    diagnostics = _audit_online_diagnostics(
        case_dir,
        expected_case,
        online,
        collector._frame_sha256(frame),
    )
    shadow = manifest["shadow"]
    expected_shadow_keys = {
        "collected",
        "role",
        "prefix_audit",
        "termination_reason",
        "rounds_run",
        "candidate_evaluation_count",
        "checkpoints",
        "files",
    }
    _validate_exact_keys(shadow, expected_shadow_keys, "case shadow")
    if shadow["role"] != "B_only_read_only_continuation":
        raise RuntimeError("shadow role 不一致")
    if reason == "early_stopped":
        checkpoints = _audit_b_shadow(
            case_dir,
            expected_case,
            online,
            diagnostics,
            workload,
            shadow,
        )
    else:
        expected_empty_shadow = {
            "collected": False,
            "role": "B_only_read_only_continuation",
            "prefix_audit": None,
            "termination_reason": None,
            "rounds_run": None,
            "candidate_evaluation_count": None,
            "checkpoints": [],
            "files": {},
        }
        if shadow != expected_empty_shadow:
            raise RuntimeError("A/C case 不得携带 shadow")
        checkpoints = []
    return {
        "case_id": expected_case["case_id"],
        "family": expected_case["family"],
        "seed": expected_case["seed"],
        "rho": expected_case["rho"],
        "termination_reason": reason,
        "stop_normalized_work": online["stop_normalized_work"],
        "terminal_current_squared_loss": loss,
        "terminal_current_normalized_l1": l1,
        "checkpoints": [
            {
                "work_offset": checkpoint["work_offset"],
                "status": checkpoint["status"],
                "current_squared_loss": checkpoint["current_squared_loss"],
                "current_normalized_l1": checkpoint["current_normalized_l1"],
            }
            for checkpoint in checkpoints
        ],
    }


def audit_collection(collection_dir: Path) -> dict[str, Any]:
    """Read and verify a complete raw collection without classifying it."""

    collection_dir = Path(collection_dir).resolve()
    collection_path = collection_dir / "collection_manifest.json"
    collection = _read_json(collection_path, "collection manifest")
    expected_collection_keys = {
        "contract_version",
        "protocol_sha256",
        "execution_manifest",
        "formal_primary_collection_complete",
        "case_count",
        "case_manifest_files",
        "collection_elapsed_sec",
        "acceptance_evaluated",
        "partial_matrix_classification_emitted",
        "real_data_accessed",
        "privacy_budget_consumed",
    }
    _validate_exact_keys(
        collection,
        expected_collection_keys,
        "collection manifest",
    )
    if (
        collection["contract_version"] != collector.COLLECTION_CONTRACT_VERSION
        or collection["protocol_sha256"] != protocol.assert_frozen_protocol_identity()
        or collection["formal_primary_collection_complete"] is not True
        or collection["case_count"] != 12
        or collection["acceptance_evaluated"] is not False
        or collection["partial_matrix_classification_emitted"] is not False
        or collection["real_data_accessed"] is not False
        or collection["privacy_budget_consumed"] is not False
    ):
        raise RuntimeError("collection manifest contract/raw-only 状态不一致")
    _finite_float(
        collection["collection_elapsed_sec"],
        "collection_elapsed_sec",
        minimum=0.0,
    )
    execution_path = _verify_file_info(
        collection_dir,
        collection["execution_manifest"],
        "execution manifest",
    )
    execution_sha = collector._sha256_file(execution_path)
    execution = _read_json(execution_path, "execution manifest")
    expected_execution_keys = {
        "contract_version",
        "created_at_utc",
        "git_commit",
        "git_worktree_clean_including_untracked",
        "protocol_sha256",
        "protocol",
        "source_sha256",
        "environment",
        "execution_started",
        "formal_rng_instantiated",
        "acceptance_evaluated",
    }
    _validate_exact_keys(execution, expected_execution_keys, "execution manifest")
    if (
        execution["contract_version"] != collector.COLLECTION_CONTRACT_VERSION
        or execution["protocol_sha256"] != protocol.assert_frozen_protocol_identity()
        or _strict_json_bytes(execution["protocol"])
        != _strict_json_bytes(protocol.frozen_protocol_manifest())
        or execution["git_worktree_clean_including_untracked"] is not True
        or execution["execution_started"] is not False
        or execution["formal_rng_instantiated"] is not False
        or execution["acceptance_evaluated"] is not False
    ):
        raise RuntimeError("execution manifest identity 不一致")
    expected_source_paths = {str(path) for path in collector.SOURCE_PATHS}
    source_sha256 = execution["source_sha256"]
    if not isinstance(source_sha256, dict) or set(source_sha256) != (
        expected_source_paths
    ):
        raise RuntimeError("execution source SHA 矩阵不完整")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source_sha256.values()
    ):
        raise RuntimeError("execution source SHA 格式非法")
    expected_environment_keys = {
        "python_version",
        "numpy_version",
        "pandas_version",
        "platform",
        "machine",
        "processor",
        "device",
    }
    _validate_exact_keys(
        execution["environment"],
        expected_environment_keys,
        "execution environment",
    )
    if execution["environment"]["device"] != "numpy":
        raise RuntimeError("execution device 不是冻结的 NumPy CPU")
    if any(
        not isinstance(execution["environment"][key], str)
        for key in expected_environment_keys
    ):
        raise RuntimeError("execution environment 字段必须是字符串")
    expected_cases = protocol.primary_case_matrix()
    expected_case_ids = [case["case_id"] for case in expected_cases]
    case_files = collection["case_manifest_files"]
    if not isinstance(case_files, dict) or set(case_files) != set(expected_case_ids):
        raise RuntimeError("collection case manifest 矩阵不完整")
    evidence_rows = []
    case_manifest_hashes = {}
    for case in expected_cases:
        case_path = _verify_file_info(
            collection_dir,
            case_files[case["case_id"]],
            f"case manifest {case['case_id']}",
        )
        evidence_rows.append(_audit_case_manifest(case_path, case, execution_sha))
        case_manifest_hashes[case["case_id"]] = collector._sha256_file(case_path)
    _validate_evidence_rows(evidence_rows)
    result = {
        "collection_manifest_path": str(collection_path),
        "collection_manifest_sha256": collector._sha256_file(collection_path),
        "execution_manifest_sha256": execution_sha,
        "execution_git_commit": execution["git_commit"],
        "execution_source_sha256": source_sha256,
        "execution_environment": execution["environment"],
        "case_manifest_sha256": case_manifest_hashes,
        "case_count": len(evidence_rows),
        "all_artifacts_verified": True,
        "acceptance_evaluated": False,
        "evidence_rows": evidence_rows,
    }
    _strict_json_bytes(result)
    return result


def _evaluation_environment(root: Path) -> dict[str, Any]:
    status = collector._git_text(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError("正式 P6 判定要求包含 untracked 在内的干净工作树")
    missing = [str(path) for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"正式 P6 判定源文件缺失：{missing}")
    return {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": collector._git_text(root, "rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": True,
        "source_sha256": {
            str(path): collector._sha256_file(root / path) for path in SOURCE_PATHS
        },
        "runtime": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": "numpy",
        },
    }


def evaluate_collection(
    collection_dir: Path,
    confirmed_protocol_sha256: str,
) -> Path:
    """Audit once and write the only frozen aggregate decision report."""

    expected_sha = protocol.assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected_sha:
        raise ValueError("必须显式确认完整冻结 P6 protocol SHA-256")
    collection_dir = Path(collection_dir).resolve()
    report_path = collection_dir / EVALUATION_REPORT_FILENAME
    if report_path.exists():
        raise FileExistsError(f"P6 evaluation report 已存在：{report_path}")
    root = collector._repo_root()
    environment = _evaluation_environment(root)
    audit = audit_collection(collection_dir)
    if audit["execution_git_commit"] != environment["git_commit"]:
        raise RuntimeError("collection 与 evaluator Git commit 不一致")
    for path, recorded_sha256 in audit["execution_source_sha256"].items():
        if environment["source_sha256"].get(path) != recorded_sha256:
            raise RuntimeError(f"collection 记录的源文件 SHA 漂移：{path}")
    for key in ("python_version", "numpy_version", "pandas_version", "device"):
        if audit["execution_environment"][key] != environment["runtime"][key]:
            raise RuntimeError(f"collection/evaluator runtime 漂移：{key}")
    decision = evaluate_evidence_rows(audit["evidence_rows"])
    report = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "protocol_sha256": expected_sha,
        "collection_audit": {
            key: value for key, value in audit.items() if key != "evidence_rows"
        },
        "evaluation_environment": environment,
        "decision": decision,
        "real_data_accessed": False,
        "privacy_budget_consumed": False,
        "generator_called": False,
    }
    collector._write_json_exclusive(report_path, report)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--collection-dir", type=Path, required=True)
    evaluate_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(
            json.dumps(
                build_evaluation_plan(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return
    destination = evaluate_collection(
        args.collection_dir,
        args.confirm_protocol_sha,
    )
    print(f"P6 evaluation -> {destination}", flush=True)


if __name__ == "__main__":
    main()
