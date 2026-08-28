#!/usr/bin/env python3
"""独立复核经恢复收口的 Issue #53 Stage 6D（阶段 6D）第五版评价。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts import audit_issue53_stage6d_formal as legacy_auditor
from scripts import evaluate_issue53_stage6d_v5_recovered as recovery_evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6d_v5 as recovery_collection

AUDIT_VERSION = "issue53-stage6d-v5-recovered-independent-audit-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _require_sha(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name}必须是完整小写 SHA-256")


def _safe_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("独立复核产物路径必须是字符串")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("独立复核产物路径越界")
    base = (root / recovery_protocol.OUTPUT_DIR).resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("独立复核产物路径逃逸")
    return resolved


def _assert_commit_ancestry(
    root: Path, generation_commit: Any, recovery_commit: Any
) -> None:
    for value, name in ((generation_commit, "生成提交"), (recovery_commit, "恢复提交")):
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"{name}无效")
    current = recovery_collection._git_text(root, "rev-parse", "HEAD")
    if (
        recovery_collection._git_text(
            root, "merge-base", generation_commit, recovery_commit
        )
        != generation_commit
        or recovery_collection._git_text(root, "merge-base", recovery_commit, current)
        != recovery_commit
    ):
        raise RuntimeError("生成、恢复与当前提交祖先关系漂移")


def build_plan() -> dict[str, Any]:
    recovery_sha = recovery_protocol.assert_frozen_recovery_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "recovery_protocol_sha256": recovery_sha,
        "collection_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.COLLECTION_REPORT
        ),
        "evaluation_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.EVALUATION_REPORT
        ),
        "l1_results_csv": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.L1_RESULTS_CSV
        ),
        "audit_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.AUDIT_REPORT
        ),
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_l1_csv": True,
        "recompute_frozen_classification": True,
        "rerun_generation": False,
        "audit_started": False,
    }


def _audit_transition_independently(path: Path, task: Any, row: dict[str, Any]) -> None:
    payload = _load_json(path)
    rounds = generation_protocol.ROUNDS
    clocks = payload.get("clocks")
    if (
        payload.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or payload.get("protocol_sha256") != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or payload.get("task_id") != task.task_id
        or payload.get("round_count") != rounds
        or payload.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != rounds
    ):
        raise RuntimeError(f"独立复核转移总结构漂移：{task.task_id}")
    microsteps = 0
    for index, clock in enumerate(clocks, start=1):
        if (
            not isinstance(clock, dict)
            or clock.get("round") != index
            or clock.get("state_index") != index
            or clock.get("accepted_attempt") != 1
            or clock.get("candidate_evaluation_count_cumulative") != index
            or isinstance(clock.get("gibbs_microsteps"), bool)
            or not isinstance(clock.get("gibbs_microsteps"), int)
            or clock["gibbs_microsteps"] < 0
        ):
            raise RuntimeError(f"独立复核无门控时钟漂移：{task.task_id}/{index}")
        microsteps += clock["gibbs_microsteps"]
    gap_rounds = payload.get("gap_rounds")
    if not isinstance(gap_rounds, list):
        raise TypeError(f"独立复核缺口轮次缺失：{task.task_id}")
    if task.arm == generation_protocol.ARM_GAP:
        if len(gap_rounds) != rounds:
            raise RuntimeError(f"独立复核缺口轮次不完整：{task.task_id}")
        gap_total = 0
        for index, item in enumerate(gap_rounds, start=1):
            active = item.get("active_switches_k") if isinstance(item, dict) else None
            expected = (
                8 * active
                if isinstance(active, int) and not isinstance(active, bool)
                else None
            )
            if (
                not isinstance(item, dict)
                or item.get("round") != index
                or item.get("n_sweeps") != 8
                or item.get("scan_applied") is not True
                or item.get("clip_hit_count") != 0
                or item.get("nonfinite_condition_count") != 0
                or item.get("expected_microsteps") != expected
                or item.get("microsteps") != expected
            ):
                raise RuntimeError(f"独立复核缺口 8*K 漂移：{task.task_id}/{index}")
            gap_total += expected
        if gap_total != row.get("gap_microsteps") or microsteps != gap_total:
            raise RuntimeError(f"独立复核缺口 8*K 总量漂移：{task.task_id}")
    elif gap_rounds:
        raise RuntimeError(f"独立复核基线包含缺口轮次：{task.task_id}")
    elif task.arm == "factor_b_s8" and microsteps != row.get(
        "factorized_gibbs_microsteps"
    ):
        raise RuntimeError(f"独立复核 Gibbs（吉布斯）微步漂移：{task.task_id}")
    elif task.arm == "independent_b_s0" and microsteps != 0:
        raise RuntimeError(f"独立复核独立基线出现 Gibbs（吉布斯）微步：{task.task_id}")


def _audit_case_independently(
    root: Path,
    task: Any,
    row: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    rounds = generation_protocol.ROUNDS
    shard_id = generation_protocol.task_shard_id(task)
    spec = generation_protocol.DATASETS[task.dataset]
    required = {
        "task_id": task.task_id,
        "execution_shard_id": shard_id,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "requested_rounds": rounds,
        "applied_rounds": rounds,
        "termination_reason": "candidate_budget",
        "device": "cuda:0",
        "output_table_identity": "terminal_current",
        "all_applied_unconditionally": True,
        "proposal_attempt_count": rounds,
        "candidate_evaluation_count": rounds,
        "state_evaluation_count": rounds,
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_clipped_count": 0,
        "nonfinite_count": 0,
    }
    if any(row.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"独立复核 case 身份、2500 计数或护栏漂移：{task.task_id}")
    expected_case_prefix = Path("cases") / task.task_id
    terminal_relative = str(expected_case_prefix / recovery_collection.TERMINAL_TABLE)
    if row.get("terminal_table_path") != terminal_relative:
        raise RuntimeError(f"独立复核终表路径漂移：{task.task_id}")
    manifest_relative = str(expected_case_prefix / recovery_collection.CASE_MANIFEST)
    manifest_path = _safe_artifact(root, manifest_relative)
    expected_manifest_sha = inventory["shards"][shard_id]["case_manifest_sha256"][
        task.task_id
    ]
    if recovery_protocol.file_sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError(f"独立复核 case 清单哈希漂移：{task.task_id}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or manifest.get("protocol_sha256") != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or manifest.get("execution_commit")
        != recovery_protocol.SOURCE_GENERATION_COMMIT
        or manifest.get("execution_shard_id") != shard_id
        or manifest.get("generator_params")
        != generation_protocol.generator_params_manifest(
            task.dataset, task.arm, task.seed
        )
        or manifest.get("collection_row") != row
        or manifest.get("raw_reference_data_accessed") is not False
        or manifest.get("method_comparison_emitted") is not False
    ):
        raise RuntimeError(f"独立复核 case 来源清单漂移：{task.task_id}")
    attachments = (
        (
            "terminal_table_path",
            "terminal_table_sha256",
            str(expected_case_prefix / recovery_collection.TERMINAL_TABLE),
        ),
        (
            "checkpoint_artifact_path",
            "checkpoint_artifact_sha256",
            str(expected_case_prefix / recovery_collection.CHECKPOINT_ARTIFACT),
        ),
        (
            "transition_audit_path",
            "transition_audit_sha256",
            str(expected_case_prefix / recovery_collection.TRANSITION_AUDIT),
        ),
    )
    paths: dict[str, Path] = {}
    for path_key, sha_key, expected_relative in attachments:
        if row.get(path_key) != expected_relative:
            raise RuntimeError(f"独立复核附件路径漂移：{task.task_id}/{path_key}")
        path = _safe_artifact(root, expected_relative)
        _require_sha(row.get(sha_key), f"{task.task_id}/{sha_key}")
        if recovery_protocol.file_sha256(path) != row[sha_key]:
            raise RuntimeError(f"独立复核附件哈希漂移：{task.task_id}/{path_key}")
        paths[path_key] = path
    # 收口结构复核不解释检查点质量；这里只独立读取无门控转移审计。
    _audit_transition_independently(paths["transition_audit_path"], task, row)


def _audit_collection_independently(
    root: Path,
    confirmed_sha: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    _require_sha(confirmed_sha, "collection（采集）报告 SHA-256")
    path = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.COLLECTION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("恢复 collection（采集）报告 SHA-256 不一致")
    report = _load_json(path)
    expected_top = {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256),
        "source_generation_protocol_sha256": (recovery_protocol.SOURCE_PROTOCOL_SHA256),
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256),
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "formal_result_valid": True,
        "formal_collection_recovered": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in expected_top.items()):
        raise RuntimeError("独立复核恢复 collection 顶层契约漂移")
    if report.get("recovery_protocol") != recovery_protocol.recovery_protocol_manifest(
        root
    ):
        raise RuntimeError("独立复核内嵌恢复协议漂移")
    if (
        report.get("source_generation_protocol")
        != generation_protocol.frozen_protocol_manifest()
    ):
        raise RuntimeError("独立复核内嵌生成协议漂移")
    if (
        report.get("recovery_collector_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
    ):
        raise RuntimeError("独立复核恢复收口器身份漂移")
    expected_inputs = {
        dataset: {
            key: spec["input_sha256"][key] for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in generation_protocol.DATASETS.items()
    }
    if report.get("generation_input_sha256") != expected_inputs:
        raise RuntimeError("独立复核生成输入身份漂移")
    if report.get("source_gpu_monitoring_evidence") != {
        shard_id: recovery_collection._source_monitoring_limitation()
        for shard_id in generation_protocol.SHARD_ORDER
    }:
        raise RuntimeError("独立复核显卡监控限制披露漂移")
    _assert_commit_ancestry(
        root,
        report.get("generation_execution_commit"),
        report.get("recovery_commit"),
    )
    expected_execution = {
        "shard_order": list(generation_protocol.SHARD_ORDER),
        "resumed_case_counts": {
            generation_protocol.LOCAL_SHARD: 21,
            generation_protocol.A6000_SHARD: 9,
        },
        "new_generation_case_count": 0,
        "generator_invoked": False,
        "gpu_accessed": False,
        "raw_reference_data_accessed": False,
        "quality_metrics_interpreted": False,
    }
    if report.get("recovery_execution") != expected_execution:
        raise RuntimeError("独立复核恢复执行边界或 21/9 计数漂移")
    expected_audit = {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "exact_21_9_shard_assignment": True,
        "both_recovered_shard_reports_verified_before_merge": True,
        "all_case_manifests_match_frozen_inventory": True,
        "all_case_attachments_sha256_verified": True,
        "all_candidate_budget_counts_use_actual_semantics": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "zero_new_generation_cases": True,
        "source_case_files_rewritten": False,
        "gpu_monitoring_limitation_disclosed": True,
    }
    if report.get("collection_audit") != expected_audit:
        raise RuntimeError("独立复核恢复 collection 审计字段漂移")
    inventory = recovery_protocol.recovery_inventory(root)
    shard_payloads: dict[str, dict[str, Any]] = {}
    shard_hashes = report.get("shard_report_sha256")
    shard_artifacts = report.get("shard_report_artifacts")
    staging_artifacts = report.get("source_staging_manifest_artifacts")
    if not all(
        isinstance(value, dict)
        for value in (shard_hashes, shard_artifacts, staging_artifacts)
    ):
        raise TypeError("独立复核双分片证据缺失")
    for shard_id in generation_protocol.SHARD_ORDER:
        report_relative = str(
            Path("shards") / shard_id / recovery_protocol.SHARD_REPORT
        )
        staging_relative = str(
            Path("shards") / shard_id / recovery_collection.STAGING_MANIFEST
        )
        report_artifact = shard_artifacts.get(shard_id)
        staging_artifact = staging_artifacts.get(shard_id)
        if (
            not isinstance(report_artifact, dict)
            or report_artifact.get("path") != report_relative
            or report_artifact.get("sha256") != shard_hashes.get(shard_id)
            or not isinstance(staging_artifact, dict)
            or staging_artifact.get("path") != staging_relative
            or staging_artifact.get("sha256")
            != inventory["shards"][shard_id]["staging_manifest_sha256"]
        ):
            raise RuntimeError(f"独立复核分片证据地址漂移：{shard_id}")
        for relative, digest in (
            (report_relative, report_artifact["sha256"]),
            (staging_relative, staging_artifact["sha256"]),
        ):
            artifact = _safe_artifact(root, relative)
            if recovery_protocol.file_sha256(artifact) != digest:
                raise RuntimeError(f"独立复核分片证据哈希漂移：{shard_id}")
        shard_payload = _load_json(_safe_artifact(root, report_relative))
        expected_count = len(generation_protocol.tasks_for_shard(shard_id))
        expected_source = {
            "contract_version": generation_protocol.PROTOCOL_VERSION,
            "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
            "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
            "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
            "generator_params_manifest_sha256": (
                recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
            ),
            "shard_assignment_sha256": (
                recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256
            ),
        }
        expected_environment = {
            "hostname": generation_protocol.EXECUTION_SHARDS[shard_id]["hostname"],
            "gpu": dict(generation_protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]),
            "software": dict(generation_protocol.EXPECTED_SOFTWARE),
            "status": "frozen_expectation_not_reconstructed_runtime_sample",
        }
        if (
            shard_payload.get("contract_version")
            != recovery_protocol.RECOVERY_PROTOCOL_VERSION
            or shard_payload.get("recovery_protocol_sha256")
            != recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
            or shard_payload.get("recovery_commit") != report.get("recovery_commit")
            or shard_payload.get("source_generation") != expected_source
            or shard_payload.get("recovery_collector_sha256")
            != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
            or shard_payload.get("recovery_inventory_sha256")
            != recovery_protocol.RECOVERY_INVENTORY_SHA256
            or shard_payload.get("shard_id") != shard_id
            or shard_payload.get("source_hostname")
            != inventory["shards"][shard_id]["source_hostname"]
            or shard_payload.get("source_expected_execution_environment")
            != expected_environment
            or shard_payload.get("task_ids")
            != [task.task_id for task in generation_protocol.tasks_for_shard(shard_id)]
            or shard_payload.get("case_count") != expected_count
            or shard_payload.get("case_manifest_sha256")
            != inventory["shards"][shard_id]["case_manifest_sha256"]
            or shard_payload.get("recovery_execution", {}).get("resumed_case_count")
            != expected_count
            or shard_payload.get("recovery_execution", {}).get(
                "new_generation_case_count"
            )
            != 0
            or shard_payload.get("source_gpu_monitoring_evidence")
            != recovery_collection._source_monitoring_limitation()
            or shard_payload.get("formal_shard_recovered") is not True
        ):
            raise RuntimeError(f"独立复核恢复分片报告漂移：{shard_id}")
        shard_payloads[shard_id] = shard_payload
    rows = report.get("raw_results")
    tasks = generation_protocol.task_plan().tasks
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("独立复核恢复 collection 不是完整 30 条")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for task, row in zip(tasks, rows):
        if not isinstance(row, dict) or row.get("task_id") != task.task_id:
            raise RuntimeError("独立复核恢复 collection 冻结顺序漂移")
        _audit_case_independently(root, task, row, inventory)
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("独立复核恢复 case 重复")
        indexed[key] = row
    for seed in generation_protocol.FORMAL_SEEDS:
        for dataset in joint.DATASET_ORDER:
            paired = [indexed[(seed, dataset, arm)] for arm in joint.ARM_ORDER]
            for identity_key in (
                "initial_table_sha256",
                "primary_rng_post_initialization_sha256",
                "direction_reference_scale",
                "query_identity_sha256",
                "target_vector_sha256",
                "trace_query_identity_sha256",
                "trace_target_vector_sha256",
            ):
                if len({row[identity_key] for row in paired}) != 1:
                    raise RuntimeError(
                        f"独立复核三方法配对身份漂移：{seed}/{dataset}/{identity_key}"
                    )
    for shard_id in generation_protocol.SHARD_ORDER:
        expected_rows = [
            indexed[(task.seed, task.dataset, task.arm)]
            for task in generation_protocol.tasks_for_shard(shard_id)
        ]
        if shard_payloads[shard_id].get("raw_results") != expected_rows:
            raise RuntimeError(f"独立复核分片与 collection 行不一致：{shard_id}")
    return report, indexed


def _audit_evaluation(
    root: Path,
    confirmed_evaluation_sha: str,
    confirmed_collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> dict[str, Any]:
    _require_sha(confirmed_evaluation_sha, "evaluation（评价）报告 SHA-256")
    path = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.EVALUATION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_evaluation_sha:
        raise ValueError("恢复版 evaluation（评价）报告 SHA-256 不一致")
    report = _load_json(path)
    if (
        report.get("contract_version") != recovery_evaluator.EVALUATION_VERSION
        or report.get("recovery_protocol_sha256")
        != recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        or report.get("source_generation_protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or report.get("recovery_evaluator_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_evaluator"]["sha256"]
        or report.get("collection_report_sha256") != confirmed_collection_sha
        or report.get("collection_generation_commit")
        != collection_report["generation_execution_commit"]
        or report.get("collection_recovery_commit")
        != collection_report["recovery_commit"]
        or report.get("collection_recovered_after_generation") is not True
        or report.get("collection_monitoring_limitation_code")
        != recovery_protocol.MONITORING_LIMITATION_CODE
        or report.get("case_count") != 30
        or report.get("raw_reference_data_accessed") is not True
        or report.get("l1_fully_emitted") is not True
        or report.get("historical_best_used") is not False
        or report.get("checkpoint_selected_as_output") is not False
        or report.get("new_generation_performed_by_evaluator") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
    ):
        raise RuntimeError("独立复核恢复版评价顶层契约漂移")
    evaluation_commit = report.get("evaluation_commit")
    _assert_commit_ancestry(
        root,
        collection_report["generation_execution_commit"],
        collection_report["recovery_commit"],
    )
    if (
        not isinstance(evaluation_commit, str)
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
        or recovery_collection._git_text(
            root,
            "merge-base",
            collection_report["recovery_commit"],
            evaluation_commit,
        )
        != collection_report["recovery_commit"]
        or recovery_collection._git_text(
            root,
            "merge-base",
            evaluation_commit,
            recovery_collection._git_text(root, "rev-parse", "HEAD"),
        )
        != evaluation_commit
    ):
        raise RuntimeError("独立复核恢复版评价提交祖先关系漂移")
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("独立复核查询/参考身份漂移")
    if report.get("cases") != cases:
        raise RuntimeError("独立复核 30 条指标与评价报告不一致")
    expected_summary = legacy_auditor._summary_independent(cases)
    expected_classification = legacy_auditor._classification_independent(cases)
    if report.get("summary") != expected_summary:
        raise RuntimeError("独立复核评价汇总漂移")
    if report.get("frozen_classification") != expected_classification:
        raise RuntimeError("独立复核冻结结论漂移")
    csv_audit = legacy_auditor._audit_csv(
        root,
        report,
        legacy_auditor._csv_rows_independent(cases),
    )
    return {
        "evaluation_report_sha256": confirmed_evaluation_sha,
        "all_30_case_metrics_exactly_recomputed": True,
        "all_terminal_and_checkpoint_l1_exactly_recomputed": True,
        "all_pairwise_gates_exactly_recomputed": True,
        "classification_exactly_reproduced": True,
        "collection_recovery_provenance_independently_verified": True,
        "gpu_monitoring_limitation_preserved": True,
        "l1_csv": csv_audit,
        "pass": True,
    }


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("恢复版独立复核要求包含未跟踪文件在内的干净工作树")
    output = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.AUDIT_REPORT
    if output.exists():
        raise FileExistsError(f"恢复版独立复核报告已存在，不覆盖：{output}")
    collection_report, indexed = _audit_collection_independently(
        root, confirmed_collection_report_sha256
    )
    cases, identities = legacy_auditor._recompute_cases(root, indexed)
    evaluation_audit = _audit_evaluation(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report,
        cases,
        identities,
    )
    report = {
        **build_plan(),
        "mode": "independent_recompute_after_recovered_complete_collection",
        "audit_started": True,
        "audit_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovery_independent_auditor_sha256": recovery_protocol.file_sha256(
            Path(__file__)
        ),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "evaluation_report_sha256": confirmed_evaluation_report_sha256,
        "collection_generation_commit": collection_report[
            "generation_execution_commit"
        ],
        "collection_recovery_commit": collection_report["recovery_commit"],
        "collection_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "query_and_reference_identity_audit": identities,
        "recomputed_case_count": len(cases),
        "independent_classification": (
            legacy_auditor._classification_independent(cases)
        ),
        "evaluation_audit": evaluation_audit,
        "overall_pass": True,
        "new_generation_performed_by_auditor": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "parameter_retuning_performed": False,
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"恢复版独立复核临时文件已存在：{temporary}")
    try:
        temporary.write_text(
            recovery_collection._strict_json_text(report), encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--confirm-collection-sha", required=True)
    audit_parser.add_argument("--confirm-evaluation-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    path = audit(args.confirm_collection_sha, args.confirm_evaluation_sha)
    print(f"恢复版 independent audit（独立复核） -> {path}")
    print(f"复核报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
