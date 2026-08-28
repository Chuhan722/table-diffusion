#!/usr/bin/env python3
"""离线评价 Issue #53 Stage 6E 自动停止三核正式 collection。"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_stage6d_formal as stage6d_evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as protocol
from scripts import run_issue53_stage6e_autostop as collection


EVALUATION_VERSION = "issue53-stage6e-autostop-formal-evaluation-v1"

_legacy_evaluate_cases = stage6d_evaluator._evaluate_cases
_legacy_summary = stage6d_evaluator._summary


@contextlib.contextmanager
def _stage6d_evaluation_runtime():
    names = (
        "protocol",
        "collection",
        "EVALUATION_VERSION",
        "build_plan",
        "_audit_collection",
        "_evaluate_cases",
        "_summary",
    )
    original = {name: getattr(stage6d_evaluator, name) for name in names}
    stage6d_evaluator.protocol = protocol
    stage6d_evaluator.collection = collection
    stage6d_evaluator.EVALUATION_VERSION = EVALUATION_VERSION
    stage6d_evaluator.build_plan = build_plan
    stage6d_evaluator._audit_collection = _audit_collection
    stage6d_evaluator._evaluate_cases = _evaluate_cases
    stage6d_evaluator._summary = _summary
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(stage6d_evaluator, name, value)


def _repo_root() -> Path:
    return collection._repo_root()


def build_plan() -> dict[str, Any]:
    protocol_sha = protocol.assert_frozen_protocol_identity(_repo_root())
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "collection_report": str(protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT),
        "evaluation_report": str(protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT),
        "l1_results_csv": str(protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV),
        "candidate_arm": protocol.ARM_GAP,
        "baseline_arms": list(protocol.BASELINE_ARMS),
        "stable_win_minimum": protocol.STABLE_WIN_MINIMUM,
        "paired_seed_count": len(protocol.FORMAL_SEEDS),
        "lower_risk_ratio_max": protocol.LOWER_RISK_RATIO_MAX,
        "higher_quality_ratio_min": protocol.HIGHER_QUALITY_RATIO_MIN,
        "terminal_current_only": True,
        "variable_applied_rounds_expected": True,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _artifact_path(root: Path, relative: Any) -> Path:
    with _stage6d_evaluation_runtime():
        return stage6d_evaluator._artifact_path(root, relative)


def _audit_collection(
    root: Path,
    confirmed_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    protocol.require_collection_confirmation(confirmed_sha256)
    report_path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(report_path) != confirmed_sha256:
        raise ValueError("Stage 6E collection 报告 SHA-256 与确认值不一致")
    report = collection.stage6d_runner._load_json(report_path)
    expected_audit = {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "single_rtx4090_shard_assignment": True,
        "all_triplets_single_shard": True,
        "complete_shard_verified_before_merge": True,
        "all_existing_p6_autostop_enabled": True,
        "all_normal_abc_termination": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_physical_index_1": True,
    }
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
        or report.get("generator_params_manifest_sha256")
        != protocol.generator_params_manifest_sha256()
        or report.get("shard_assignment_sha256")
        != protocol.shard_assignment_sha256()
        or report.get("case_count") != 30
        or report.get("paired_dataset_seed_count") != 10
        or report.get("collection_audit") != expected_audit
        or report.get("formal_result_valid") is not True
        or report.get("raw_reference_data_accessed") is not False
        or report.get("partial_matrix_comparison_emitted") is not False
        or report.get("method_ranking_emitted") is not False
        or report.get("l1_results_published_by_collection") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
    ):
        raise RuntimeError("Stage 6E collection 协议、完整性或信息边界漂移")

    expected_inputs = {
        dataset: {
            key: spec["input_sha256"][key]
            for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in protocol.DATASETS.items()
    }
    if report.get("generation_input_sha256") != expected_inputs:
        raise RuntimeError("Stage 6E collection 生成输入身份漂移")
    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("Stage 6E collection 执行提交缺失")
    current_commit = collection.stage6d_runner._git_text(root, "rev-parse", "HEAD")
    if (
        collection.stage6d_runner._git_text(
            root, "merge-base", execution_commit, current_commit
        )
        != execution_commit
    ):
        raise RuntimeError("Stage 6E collection 提交不是评价提交的祖先")

    shard_id = protocol.LOCAL_SHARD
    environments = report.get("environment", {}).get("shards")
    if not isinstance(environments, dict) or set(environments) != {shard_id}:
        raise RuntimeError("Stage 6E collection 单卡环境缺失")
    gpu = environments[shard_id].get("gpu")
    expected_shard = protocol.EXECUTION_SHARDS[shard_id]
    if (
        not isinstance(gpu, dict)
        or gpu.get("shard_id") != shard_id
        or gpu.get("hostname") != expected_shard["hostname"]
        or any(
            gpu.get(key) != expected
            for key, expected in expected_shard["expected_gpu"].items()
        )
        or any(
            gpu.get(key) != expected
            for key, expected in protocol.EXPECTED_SOFTWARE.items()
        )
    ):
        raise RuntimeError("Stage 6E collection GPU/软件环境漂移")

    artifact = report.get("shard_report_artifacts", {}).get(shard_id)
    expected_shard_sha = report.get("shard_report_sha256", {}).get(shard_id)
    expected_relative = str(Path("shards") / shard_id / protocol.SHARD_REPORT)
    if (
        not isinstance(artifact, dict)
        or artifact.get("path") != expected_relative
        or artifact.get("sha256") != expected_shard_sha
    ):
        raise RuntimeError("Stage 6E shard report 身份漂移")
    shard_path = _artifact_path(root, expected_relative)
    if protocol.file_sha256(shard_path) != expected_shard_sha:
        raise RuntimeError("Stage 6E shard report 文件哈希漂移")
    shard_report = collection.stage6d_runner._load_json(shard_path)
    if (
        shard_report.get("contract_version") != protocol.PROTOCOL_VERSION
        or shard_report.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or shard_report.get("shard_id") != shard_id
        or shard_report.get("execution_commit") != execution_commit
        or shard_report.get("formal_shard_complete") is not True
        or shard_report.get("partial_shard_comparison_emitted") is not False
    ):
        raise RuntimeError("Stage 6E shard report 内容漂移")
    gpu_samples = shard_report.get("gpu_samples")
    if (
        not isinstance(gpu_samples, list)
        or not gpu_samples
        or any(sample.get("physical_index") != 1 for sample in gpu_samples)
    ):
        raise RuntimeError("Stage 6E GPU 监控样本漂移")

    rows = report.get("raw_results")
    tasks = protocol.task_plan().tasks
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("Stage 6E collection 不是完整30条")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for task, row in zip(tasks, rows, strict=True):
        collection._validate_case_row(task, row)
        spec = protocol.DATASETS[task.dataset]
        if (
            row.get("query_identity_sha256") != spec["query_identity_sha256"]
            or row.get("target_vector_sha256") != spec["target_vector_sha256"]
            or row.get("trace_query_identity_sha256")
            != spec["trace_query_identity_sha256"]
            or row.get("trace_target_vector_sha256")
            != spec["trace_target_vector_sha256"]
            or row.get("gap_8k_identity") is not True
            or row.get("gap_clip_hit_count") != 0
            or row.get("factorized_gibbs_conditional_logit_clipped_count") != 0
            or row.get("direction_logit_clipped_count") != 0
            or row.get("nonfinite_count") != 0
        ):
            raise RuntimeError(f"Stage 6E case 身份或数值护栏漂移：{task.task_id}")
        for path_key, sha_key in (
            ("terminal_table_path", "terminal_table_sha256"),
            ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
            ("transition_audit_path", "transition_audit_sha256"),
        ):
            if protocol.file_sha256(_artifact_path(root, row[path_key])) != row[sha_key]:
                raise RuntimeError(f"Stage 6E artifact 哈希漂移：{task.task_id}")
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError(f"Stage 6E case 重复：{key}")
        indexed[key] = row
    if shard_report.get("raw_results") != rows:
        raise RuntimeError("Stage 6E 单分片与合并报告行不一致")
    return report, indexed


def _evaluate_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with _stage6d_evaluation_runtime():
        cases, identities = _legacy_evaluate_cases(root, indexed)
    for case in cases:
        source = indexed[(case["seed"], case["dataset"], case["arm"])]
        case["inner_early_stopping"] = source["inner_early_stopping"]
        case["cost"]["applied_rounds"] = source["applied_rounds"]
        case["cost"]["average_sec_per_applied_round"] = source[
            "average_sec_per_applied_round"
        ]
    return cases, identities


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    with _stage6d_evaluation_runtime():
        result = _legacy_summary(cases)
    for dataset in joint.DATASET_ORDER:
        for arm in joint.ARM_ORDER:
            cell = result[dataset][arm]
            cell["applied_rounds"] = stage6d_evaluator._arm_summary(
                cases, dataset, arm, "cost.applied_rounds"
            )
            cell["average_sec_per_applied_round"] = (
                stage6d_evaluator._arm_summary(
                    cases,
                    dataset,
                    arm,
                    "cost.average_sec_per_applied_round",
                )
            )
            selected = stage6d_evaluator._selected(cases, dataset, arm)
            total_elapsed = math.fsum(
                float(case["cost"]["elapsed_sec"]) for case in selected
            )
            total_rounds = sum(
                int(case["cost"]["applied_rounds"]) for case in selected
            )
            cell["timing_total_and_average_per_round"] = {
                "total_elapsed_sec": float(total_elapsed),
                "total_applied_rounds": int(total_rounds),
                "average_sec_per_applied_round": (
                    float(total_elapsed / total_rounds)
                    if total_rounds
                    else 0.0
                ),
            }
    return result


def evaluate(confirmed_collection_report_sha256: str):
    with _stage6d_evaluation_runtime(), collection._stage6d_runtime():
        return stage6d_evaluator.evaluate(confirmed_collection_report_sha256)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-collection-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(collection._strict_json_text(build_plan()), end="")
        return
    report, csv_path = evaluate(args.confirm_collection_sha)
    print(f"Stage 6E evaluation -> {report}")
    print(f"evaluation SHA-256 -> {protocol.file_sha256(report)}")
    print(f"Stage 6E L1 CSV -> {csv_path}")
    print(f"L1 CSV SHA-256 -> {protocol.file_sha256(csv_path)}")


if __name__ == "__main__":
    main()
