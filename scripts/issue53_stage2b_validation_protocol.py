#!/usr/bin/env python
"""Issue #53 Stage 2B detector 封存验证协议。

本模块冻结开发候选进入 validation 时使用的唯一 detector 配置、20 个验证 cell、
完整 8000 轮反事实尾部采集和硬通过门禁。它没有生成、读取轨迹或阈值覆盖入口；
命令行只打印计划。正式验证运行仍须另行向用户报告单卡开销并获得明确确认。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

try:
    from scripts import collect_issue53_stage2b_range_finding as collector
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import collect_issue53_stage2b_range_finding as collector

from table_diffevo.stationarity import StationarityDetectorConfig


VALIDATION_PROTOCOL_VERSION = "issue53-stage2b-detector-validation-v1"

REFERENCE_GENERATOR_GIT_COMMIT = (
    "d87503e38fc5068c60e8ebf50eb7baeda94fcb9a"
)
REFERENCE_GENERATION_PROTOCOL_SHA256 = (
    "483fd48ff88f050a7935eeb8cd4eb05e74607c1067800da39669516aa1d4b12b"
)
CALIBRATION_ANALYSIS_GIT_COMMIT = (
    "58c03863dbfb9f0e79faa981e721de64c5310a87"
)
CALIBRATION_REPORT_MANIFEST_SHA256 = (
    "faa7c821804ea8de98a50069745ef906996ca51dbb00bdab7bc862f2945c1d8e"
)
CALIBRATION_REPORT_SHA256 = (
    "562b37eb7b21c8fa3e6344c2f6c3dd1127c4337a032e2432cbf91146da73c590"
)
CALIBRATION_CHECKS_SHA256 = (
    "0d9a4d417aa3dda2adb6f5fb3a6cb81afc2b21388e4a6bd24f1c17ad4f11d008"
)
FULL_DEVELOPMENT_REPLAY_SHA256 = (
    "3f1efb0c92b2f0ccfcb08bd91efca0d7608b8c58326eb3c05522c8a994f601a1"
)

VALIDATION_ROUND_BUDGET = 8000
VALIDATION_SEEDS = (220, 221, 222, 223, 224)
REQUIRED_MOVING_STABILITY_CHECKS = 2
PERSISTENT_REDRIFT_CHECKS = 4

FROZEN_DETECTOR_CONFIG = StationarityDetectorConfig(
    window_size=400,
    query_mean_shift_tolerance=0.0022331666666666017,
    query_p95_shift_tolerance=0.005488583333333529,
    l1_mean_shift_tolerance=0.0004866000000000001,
    l1_p90_minus_p10_shift_tolerance=0.00044000000000000034,
    unique_row_rate_tolerance=0.05588666666666668,
    normalized_row_entropy_tolerance=0.019247834404109442,
    minimum_active_round_rate=0.8625,
    minimum_mean_changed_row_fraction=0.005748717631790372,
    stall_patience_checks=4,
)

_VALIDATION_SUMMARY_KEYS = {
    "dataset",
    "kernel",
    "seed",
    "status",
    "candidate_round_index",
    "persistent_redrift_detected",
}
_FULL_BUDGET_VALIDATION_STATUSES = {
    "stationary_qualified",
    "stalled",
    "horizon_reached",
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def expected_validation_cells() -> List[Dict[str, Any]]:
    return [
        {"dataset": dataset, "seed": seed, "kernel": kernel}
        for dataset in collector.DATASETS
        for seed in VALIDATION_SEEDS
        for kernel in collector.KERNELS
    ]


def frozen_validation_protocol_manifest() -> Dict[str, Any]:
    if tuple(collector.SEALED_VALIDATION_SEEDS) != VALIDATION_SEEDS:
        raise RuntimeError("validation seed 与原始封存协议不一致")
    if collector.protocol_sha256() != REFERENCE_GENERATION_PROTOCOL_SHA256:
        raise RuntimeError("原始生成协议 SHA-256 已发生变化")

    config = FROZEN_DETECTOR_CONFIG.to_dict()
    manifest = {
        "contract_version": VALIDATION_PROTOCOL_VERSION,
        "purpose": "heldout_validation_of_one_common_stationarity_detector",
        "freeze_status": "frozen_before_validation_seed_access",
        "source_evidence": {
            "reference_generator_git_commit": (
                REFERENCE_GENERATOR_GIT_COMMIT
            ),
            "reference_generation_protocol_sha256": (
                REFERENCE_GENERATION_PROTOCOL_SHA256
            ),
            "calibration_analysis_git_commit": (
                CALIBRATION_ANALYSIS_GIT_COMMIT
            ),
            "calibration_report_manifest_sha256": (
                CALIBRATION_REPORT_MANIFEST_SHA256
            ),
            "calibration_report_sha256": CALIBRATION_REPORT_SHA256,
            "calibration_checks_sha256": CALIBRATION_CHECKS_SHA256,
            "full_development_replay_sha256": (
                FULL_DEVELOPMENT_REPLAY_SHA256
            ),
            "development_trajectory_count": 12,
            "development_classification": (
                "candidate_supported_on_development"
            ),
        },
        "scope": {
            "datasets": list(collector.DATASETS),
            "kernels": list(collector.KERNELS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "expected_trajectory_count": len(expected_validation_cells()),
            "one_common_config": True,
            "per_dataset_query_kernel_exception": False,
        },
        "generator": {
            "same_as_development_protocol_sha256": (
                REFERENCE_GENERATION_PROTOCOL_SHA256
            ),
            "maximum_round_budget": VALIDATION_ROUND_BUDGET,
            "full_budget_required_for_counterfactual_tail_audit": True,
            "online_stop_during_validation_collection": False,
            "same_seed_s0_and_post_initialization_rng_paired_across_kernels": (
                True
            ),
            "device_by_dataset": {
                name: specification["device"]
                for name, specification in collector.DATASETS.items()
            },
        },
        "detector": {
            "config": config,
            "config_sha256": _sha256_json(config),
            "three_adjacent_nonoverlapping_windows": True,
            "all_three_window_pairwise_comparisons": True,
            "initial_state_excluded": True,
            "required_consecutive_moving_stability_checks": (
                REQUIRED_MOVING_STABILITY_CHECKS
            ),
            "persistent_redrift_checks": PERSISTENT_REDRIFT_CHECKS,
            "query_change_requires_history_reset": True,
            "absolute_l1_quality_is_not_a_stop_condition": True,
        },
        "acceptance": {
            "all_20_trajectories_stationary_qualified": True,
            "stalled_trajectory_count_must_equal": 0,
            "persistent_redrift_trajectory_count_must_equal": 0,
            "cell_specific_exception_allowed": False,
            "threshold_retuning_after_validation_access_allowed": False,
            "validation_failure_action": (
                "reject_frozen_config_retire_seeds_and_redesign"
            ),
        },
        "execution_safety": {
            "this_protocol_entry_reads_validation_data": False,
            "this_protocol_entry_runs_generation": False,
            "formal_run_requires_separate_user_runtime_confirmation": True,
            "single_visible_gpu_only": True,
        },
    }
    _strict_json_bytes(manifest)
    return manifest


def validation_protocol_sha256() -> str:
    return _sha256_json(frozen_validation_protocol_manifest())


def build_validation_plan() -> Dict[str, Any]:
    plan = {
        "contract_version": VALIDATION_PROTOCOL_VERSION,
        "protocol_sha256": validation_protocol_sha256(),
        "mode": "plan_only_no_generation_or_validation_read",
        "cells": expected_validation_cells(),
        "trajectory_count": len(expected_validation_cells()),
        "round_budget_per_trajectory": VALIDATION_ROUND_BUDGET,
        "total_round_budget": (
            len(expected_validation_cells()) * VALIDATION_ROUND_BUDGET
        ),
        "validation_seed_accessed": False,
        "generation_started": False,
        "execution_authorized_by_this_command": False,
    }
    _strict_json_bytes(plan)
    return plan


def _validate_exact_keys(
    value: Any,
    expected: Iterable[str],
    name: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise ValueError(
            f"{name} 字段不一致；missing="
            f"{sorted(expected_set - observed)}, "
            f"unknown={sorted(observed - expected_set)}"
        )


def evaluate_validation_summaries(
    summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply the frozen all-20/no-stall/no-redrift hard gates."""
    rows = list(summaries)
    expected_cells = {
        (row["dataset"], row["seed"], row["kernel"])
        for row in expected_validation_cells()
    }
    if len(rows) != len(expected_cells):
        raise ValueError("validation summary 必须恰好包含 20 条轨迹")

    seen = set()
    for index, row in enumerate(rows):
        _validate_exact_keys(
            row, _VALIDATION_SUMMARY_KEYS, f"summaries[{index}]"
        )
        cell = (row["dataset"], row["seed"], row["kernel"])
        if cell not in expected_cells or cell in seen:
            raise ValueError(f"未知或重复 validation cell：{cell}")
        seen.add(cell)
        if not isinstance(row["persistent_redrift_detected"], bool):
            raise ValueError("persistent_redrift_detected 必须是布尔值")
        if row["status"] not in _FULL_BUDGET_VALIDATION_STATUSES:
            raise ValueError("validation status 不是冻结的全预算状态")
        candidate_round = row["candidate_round_index"]
        if row["status"] == "stationary_qualified":
            if (
                isinstance(candidate_round, bool)
                or not isinstance(candidate_round, int)
                or candidate_round < 4 * FROZEN_DETECTOR_CONFIG.window_size
                or candidate_round > VALIDATION_ROUND_BUDGET
                or candidate_round % FROZEN_DETECTOR_CONFIG.window_size != 0
            ):
                raise ValueError("合格轨迹的候选停止轮次非法")
        else:
            if candidate_round is not None:
                raise ValueError("未合格轨迹不得带候选停止轮次")
            if row["persistent_redrift_detected"]:
                raise ValueError("未产生候选停止时不得报告停止后再漂移")
    if seen != expected_cells:
        raise ValueError("validation cell 不完整")

    qualified = sum(
        row["status"] == "stationary_qualified" for row in rows
    )
    stalled = sum(row["status"] == "stalled" for row in rows)
    redrift = sum(row["persistent_redrift_detected"] for row in rows)
    gates = {
        "expected_trajectory_count": 20,
        "stationary_qualified_count": int(qualified),
        "all_20_trajectories_stationary_qualified": qualified == 20,
        "stalled_trajectory_count": int(stalled),
        "zero_stalled_trajectories": stalled == 0,
        "persistent_redrift_trajectory_count": int(redrift),
        "zero_persistent_redrift_trajectories": redrift == 0,
    }
    supported = all((
        gates["all_20_trajectories_stationary_qualified"],
        gates["zero_stalled_trajectories"],
        gates["zero_persistent_redrift_trajectories"],
    ))
    result = {
        "contract_version": VALIDATION_PROTOCOL_VERSION,
        "protocol_sha256": validation_protocol_sha256(),
        "classification": (
            "supports_frozen_detector_on_validation"
            if supported else "does_not_support_frozen_detector_on_validation"
        ),
        "acceptance_gates": gates,
        "retuning_on_these_validation_seeds_allowed": False,
    }
    _strict_json_bytes(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan",), default="plan")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    print(json.dumps(
        build_validation_plan(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
