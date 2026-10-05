#!/usr/bin/env python3
"""结果盲收口已生成的 Issue #53 A/R 双通道两轨迹 collection。"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as source_protocol
from scripts import issue53_gap_weight_dual_ar_screen_recovery_protocol as recovery_protocol
from scripts import recover_issue53_gap_weight_sqrt_screen as recovery_engine
from scripts import run_issue53_gap_weight_dual_ar_screen as source_collector


CASE_MANIFEST = source_collector.CASE_MANIFEST
TERMINAL_TABLE = source_collector.TERMINAL_TABLE
CHECKPOINT_ARTIFACT = source_collector.CHECKPOINT_ARTIFACT
TRANSITION_AUDIT = source_collector.TRANSITION_AUDIT


def _expected_weight_identity(dataset: str) -> dict[str, Any]:
    expected = source_protocol.EXPECTED_WEIGHT_AUDIT[dataset]
    return {
        "positive_target_query_count": expected["positive_target_query_count"],
        "relative_inverse_target_normalizer": expected[
            "relative_inverse_target_normalizer"
        ],
        "relative_positive_weight_ratio": expected[
            "relative_positive_weight_ratio"
        ],
    }


def _validate_transition(path: Path, task: Any, row: dict[str, Any]) -> None:
    transition = recovery_engine._load_json(path)
    applied = row["applied_rounds"]
    clocks = transition.get("clocks")
    gap_rounds = transition.get("gap_rounds")
    factor_rounds = transition.get("factor_rounds")
    weighting = transition.get("weighting_audit")
    expected = _expected_weight_identity(task.dataset)
    if (
        transition.get("contract_version") != source_protocol.PROTOCOL_VERSION
        or transition.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or transition.get("task_id") != task.task_id
        or transition.get("round_count") != applied
        or transition.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != applied
        or not isinstance(gap_rounds, list)
        or len(gap_rounds) != applied
        or factor_rounds != []
        or not isinstance(weighting, dict)
        or weighting.get("expected_weighting") != source_protocol.DUAL_WEIGHTING
        or weighting.get("expected_channel_aggregation") != "max"
        or weighting.get("expected_zero_target_policy")
        != "absolute_channel_only"
        or weighting.get("expected_floor_applied") is not False
        or weighting.get("expected_max_weight_ratio") is not None
        or weighting.get("expected_smoothing_count") is not None
        or weighting.get("expected_positive_target_query_count")
        != expected["positive_target_query_count"]
        or weighting.get("expected_relative_inverse_target_normalizer")
        != expected["relative_inverse_target_normalizer"]
        or weighting.get("expected_relative_positive_weight_ratio")
        != expected["relative_positive_weight_ratio"]
        or weighting.get("scan_round_count")
        != row.get("gap_weighting_scan_round_count")
        or weighting.get("guard_passed") is not True
    ):
        raise RuntimeError(f"A/R 恢复转移总结构或权重身份漂移：{task.task_id}")

    clock_microsteps = 0
    for index, clock in enumerate(clocks, start=1):
        integer_names = (
            "participating_rows",
            "changed_rows",
            "changed_cells",
            "changed_queries",
            "gibbs_microsteps",
        )
        if (
            not isinstance(clock, dict)
            or clock.get("round") != index
            or clock.get("state_index") != index
            or clock.get("accepted_attempt") != 1
            or clock.get("candidate_evaluation_count_cumulative") != index
            or not all(
                recovery_engine._nonnegative_int(clock.get(name))
                for name in integer_names
            )
            or clock["changed_rows"] > clock["participating_rows"]
            or not recovery_engine._is_sha256(
                clock.get("post_current_table_sha256")
            )
            or not recovery_engine._is_sha256(clock.get("primary_rng_state_sha256"))
            or clock.get("factorized_gibbs_rng_state_sha256") is not None
        ):
            raise RuntimeError(f"A/R 恢复转移时钟漂移：{task.task_id}/{index}")
        clock_microsteps += clock["gibbs_microsteps"]

    total_microsteps = 0
    for index, item in enumerate(gap_rounds, start=1):
        active = item.get("active_switches_k") if isinstance(item, dict) else None
        expected_microsteps = (
            8 * active if recovery_engine._nonnegative_int(active) else None
        )
        if (
            not isinstance(item, dict)
            or item.get("round") != index
            or item.get("scan_applied") is not True
            or item.get("n_sweeps") != 8
            or item.get("microsteps") != expected_microsteps
            or item.get("expected_microsteps") != expected_microsteps
            or item.get("clip_hit_count") != 0
            or item.get("nonfinite_condition_count") != 0
            or item.get("expected_weighting") != source_protocol.DUAL_WEIGHTING
            or item.get("expected_channel_aggregation") != "max"
            or item.get("expected_zero_target_policy")
            != "absolute_channel_only"
            or item.get("expected_floor_applied") is not False
            or item.get("expected_max_weight_ratio") is not None
            or item.get("expected_smoothing_count") is not None
            or item.get("expected_positive_target_query_count")
            != expected["positive_target_query_count"]
            or item.get("expected_relative_inverse_target_normalizer")
            != expected["relative_inverse_target_normalizer"]
            or item.get("expected_relative_positive_weight_ratio")
            != expected["relative_positive_weight_ratio"]
            or item.get("observed_kernel")
            != "gap_l1_global_random_scan_dual_abs_relative_max"
            or item.get("observed_weighting") != source_protocol.DUAL_WEIGHTING
            or item.get("observed_channel_aggregation") != "max"
            or item.get("observed_zero_target_policy")
            != "absolute_channel_only"
            or item.get("observed_floor_applied") is not False
            or item.get("observed_positive_target_query_count")
            != expected["positive_target_query_count"]
            or item.get("observed_relative_inverse_target_normalizer")
            != expected["relative_inverse_target_normalizer"]
            or item.get("observed_relative_positive_weight_ratio")
            != expected["relative_positive_weight_ratio"]
        ):
            raise RuntimeError(f"A/R 恢复缺口 8*K 或权重漂移：{task.task_id}/{index}")
        total_microsteps += expected_microsteps
    if (
        total_microsteps != row["gap_microsteps"]
        or clock_microsteps != total_microsteps
    ):
        raise RuntimeError(f"A/R 恢复缺口微步总量漂移：{task.task_id}")


def _shard_audit() -> dict[str, bool]:
    return {
        "all_assigned_cases_present": True,
        "both_dataset_seed_tasks_complete": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_dual_ar_gap_8k_identity": True,
        "all_weighting_diagnostics_valid": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_shard_physical_index": True,
    }


def _collection_audit() -> dict[str, bool]:
    return {
        "all_2_candidate_cases_present": True,
        "both_dataset_seed_tasks_complete": True,
        "single_rtx4090_shard_assignment": True,
        "complete_shard_verified_before_merge": True,
        "all_existing_p6_autostop_enabled": True,
        "all_normal_abc_termination": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_dual_ar_gap_8k_identity": True,
        "all_weighting_diagnostics_valid": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_physical_index_1": True,
    }


@contextlib.contextmanager
def _recovery_runtime():
    """把已冻结的通用恢复引擎临时绑定到 A/R 源证据。"""

    replacements = {
        "source_protocol": source_protocol,
        "recovery_protocol": recovery_protocol,
        "source_collector": source_collector,
        "CASE_MANIFEST": CASE_MANIFEST,
        "TERMINAL_TABLE": TERMINAL_TABLE,
        "CHECKPOINT_ARTIFACT": CHECKPOINT_ARTIFACT,
        "TRANSITION_AUDIT": TRANSITION_AUDIT,
        "__file__": __file__,
        "_validate_transition": _validate_transition,
        "_shard_audit": _shard_audit,
        "_collection_audit": _collection_audit,
    }
    originals = {
        name: getattr(recovery_engine, name) for name in replacements
    }
    for name, value in replacements.items():
        setattr(recovery_engine, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(recovery_engine, name, value)


def _strict_json_text(value: Any) -> str:
    return recovery_engine._strict_json_text(value)


def _load_json(path: Path) -> dict[str, Any]:
    return recovery_engine._load_json(path)


def _git_text(root: Path, *arguments: str) -> str:
    return recovery_engine._git_text(root, *arguments)


def _legacy_validation_proxy(row: dict[str, Any]) -> dict[str, Any]:
    return recovery_engine._legacy_validation_proxy(row)


def _validate_staging_manifest(root: Path) -> dict[str, Any]:
    with _recovery_runtime():
        return recovery_engine._validate_staging_manifest(root)


def _validate_case_row(task: Any, row: dict[str, Any]) -> None:
    with _recovery_runtime():
        recovery_engine._validate_case_row(task, row)


def _validate_case_matrix(
    artifact_root: Path,
    tasks: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    with _recovery_runtime():
        return recovery_engine._validate_case_matrix(artifact_root, tasks)


def build_plan() -> dict[str, Any]:
    with _recovery_runtime():
        plan = recovery_engine.build_plan()
    plan["mode"] = "result_blind_dual_ar_postcollection_recovery_only"
    return plan


def recover(confirmed_recovery_sha256: str) -> Path:
    with _recovery_runtime():
        return recovery_engine.recover(confirmed_recovery_sha256)


def load_collection(
    root: Path,
    confirmed_collection_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    with _recovery_runtime():
        return recovery_engine.load_collection(root, confirmed_collection_sha256)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run = subparsers.add_parser("recover")
    run.add_argument("--confirm-recovery-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    path = recover(args.confirm_recovery_sha)
    print(f"A/R 恢复 collection 报告 -> {path}")
    print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()

