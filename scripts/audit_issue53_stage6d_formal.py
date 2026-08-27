#!/usr/bin/env python3
"""独立复核 Issue #53 第 6D 正式评价与完整 L1 输出。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_issue53_fixed_alpha_calibration as offline_primitives
from scripts import freeze_issue53_test_query_workload_ab as blind_identity
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as protocol
from scripts import run_issue53_stage6d_formal as collection
from table_diffevo.stationarity import (
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)

AUDIT_VERSION = "issue53-stage6d-joint-formal-independent-audit-v2"
T_CRITICAL_DF4_95 = 2.7764451051977987
CSV_FIELDS = (
    "dataset",
    "arm",
    "seed",
    "record_kind",
    "round",
    "state_index",
    "is_terminal",
    "absolute_count_error_sum",
    "normalized_l1",
    "gap_e",
    "squared_loss",
)


def _repo_root() -> Path:
    return collection._repo_root()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _safe_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact（产物）路径必须是字符串")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("artifact（产物）路径越界")
    base = (root / protocol.OUTPUT_DIR).resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("artifact（产物）路径逃逸")
    return resolved


def _require_sha(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是完整小写 SHA-256")


def build_plan() -> dict[str, Any]:
    protocol_sha = protocol.assert_frozen_protocol_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "collection_report": str(protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT),
        "evaluation_report": str(protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT),
        "l1_results_csv": str(protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV),
        "audit_report": str(protocol.OUTPUT_DIR / protocol.AUDIT_REPORT),
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_l1_csv": True,
        "recompute_frozen_classification": True,
        "rerun_generation": False,
    }


def _audit_collection_independently(
    root: Path, confirmed_sha: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    _require_sha(confirmed_sha, "collection（采集）报告 SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("collection（采集）报告 SHA-256 不一致")
    report = _load_json(path)
    required_top = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "formal_result_valid": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required_top.items()):
        raise RuntimeError("collection（采集）顶层契约漂移")
    if report.get("protocol") != protocol.frozen_protocol_manifest():
        raise RuntimeError("collection（采集）内嵌协议漂移")
    if (
        report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
    ):
        raise RuntimeError("collection（采集）执行器身份漂移")
    expected_generation_inputs = {
        dataset: {
            key: spec["input_sha256"][key] for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in protocol.DATASETS.items()
    }
    if report.get("generation_input_sha256") != expected_generation_inputs:
        raise RuntimeError("collection（采集）生成输入身份漂移")
    gpu = report.get("environment", {}).get("gpu")
    if not isinstance(gpu, dict) or any(
        gpu.get(key) != expected for key, expected in protocol.EXPECTED_GPU.items()
    ):
        raise RuntimeError("collection（采集）物理1号显卡身份漂移")
    expected_collection_audit = {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
    }
    if report.get("collection_audit") != expected_collection_audit:
        raise RuntimeError("collection（采集）结构审计漂移")
    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("collection（采集）执行提交无效")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(root, "merge-base", execution_commit, current_commit)
        != execution_commit
    ):
        raise RuntimeError("collection（采集）执行提交不是当前提交祖先")

    task_by_id = {task.task_id: task for task in protocol.task_plan().tasks}
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != len(task_by_id):
        raise RuntimeError("collection（采集）行数不完整")
    indexed = {}
    for expected_task, row in zip(protocol.task_plan().tasks, rows):
        if row.get("task_id") != expected_task.task_id:
            raise RuntimeError("collection（采集）冻结顺序漂移")
        task = task_by_id[row["task_id"]]
        if (
            row.get("dataset") != task.dataset
            or row.get("arm") != task.arm
            or row.get("seed") != task.seed
            or row.get("requested_rounds") != protocol.ROUNDS
            or row.get("device") != "cuda:0"
            or row.get("output_table_identity") != "terminal_current"
            or row.get("all_applied_unconditionally") is not True
            or not isinstance(row.get("applied_rounds"), int)
            or isinstance(row.get("applied_rounds"), bool)
            or not 0 <= row["applied_rounds"] <= protocol.ROUNDS
            or row.get("termination_reason")
            not in {"candidate_budget", "exact_residual"}
            or (
                row.get("termination_reason") == "candidate_budget"
                and row["applied_rounds"] != protocol.ROUNDS
            )
            or row.get("proposal_attempt_count") != row.get("applied_rounds")
            or row.get("candidate_evaluation_count") != row.get("applied_rounds")
            or row.get("state_evaluation_count") != row.get("applied_rounds") + 1
            or row.get("gap_8k_identity") is not True
            or row.get("gap_clip_hit_count") != 0
            or row.get("factorized_gibbs_conditional_logit_clipped_count") != 0
            or row.get("direction_logit_clipped_count") != 0
            or row.get("nonfinite_count") != 0
        ):
            raise RuntimeError(f"collection（采集）case 身份/护栏漂移：{task.task_id}")
        spec = protocol.DATASETS[task.dataset]
        if (
            row.get("query_identity_sha256") != spec["query_identity_sha256"]
            or row.get("target_vector_sha256") != spec["target_vector_sha256"]
            or row.get("trace_query_identity_sha256")
            != spec["trace_query_identity_sha256"]
            or row.get("trace_target_vector_sha256")
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"collection（采集）查询身份漂移：{task.task_id}")
        for path_key, hash_key in (
            ("terminal_table_path", "terminal_table_sha256"),
            ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
            ("transition_audit_path", "transition_audit_sha256"),
        ):
            artifact = _safe_artifact(root, row[path_key])
            if protocol.file_sha256(artifact) != row[hash_key]:
                raise RuntimeError(f"collection（采集）产物哈希漂移：{task.task_id}")
        transition = _load_json(_safe_artifact(root, row["transition_audit_path"]))
        if (
            transition.get("task_id") != task.task_id
            or transition.get("round_count") != row["applied_rounds"]
            or transition.get("all_applied_unconditionally") is not True
            or len(transition.get("clocks", [])) != row["applied_rounds"]
        ):
            raise RuntimeError(f"转移审计产物漂移：{task.task_id}")
        for round_index, clock in enumerate(transition["clocks"], start=1):
            if (
                clock.get("round") != round_index
                or clock.get("state_index") != round_index
                or clock.get("accepted_attempt") != 1
                or clock.get("candidate_evaluation_count_cumulative") != round_index
            ):
                raise RuntimeError(f"无门控转移时钟漂移：{task.task_id}/{round_index}")
        if task.arm == protocol.ARM_GAP:
            gap_actual = sum(row_["microsteps"] for row_ in transition["gap_rounds"])
            gap_expected = sum(
                row_["expected_microsteps"] for row_ in transition["gap_rounds"]
            )
            if (
                len(transition["gap_rounds"]) != row["applied_rounds"]
                or gap_actual != gap_expected
                or gap_actual != row["gap_microsteps"]
            ):
                raise RuntimeError(f"缺口 8*K 独立结构复核失败：{task.task_id}")
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("collection（采集）case 重复")
        indexed[key] = row
    return report, indexed


def _load_inputs_and_references(
    root: Path, runtime: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    # 使用已经冻结并绑定哈希的公共离线算子，但不使用正式 evaluator（评价器）的任何汇总或分类函数。
    test_groups, test_identity = offline_primitives._freeze_test_groups(root)
    nltcs_groups, nltcs_identity = offline_primitives._freeze_nltcs_groups(root)
    inputs = {}
    for dataset, spec in protocol.DATASETS.items():
        for key in ("schema", "queries", "marginals"):
            if protocol.file_sha256(root / spec[key]) != spec["input_sha256"][key]:
                raise RuntimeError(f"独立复核 {dataset}.{key} 输入哈希漂移")
        schema = runtime.load_schema(str(root / spec["schema"]))
        queries = runtime.load_queries(str(root / spec["queries"]))
        payload = _load_json(root / spec["queries"])
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise TypeError(f"独立复核 {dataset} 原始查询列表缺失")
        query_set_sha = blind_identity.query_set_identity(raw_queries)
        raw_targets = [query.get("result") for query in raw_queries]
        targets = np.asarray(raw_targets, dtype=float)
        if len(raw_queries) != spec["query_count"] or len(queries) != len(raw_queries):
            raise RuntimeError(f"独立复核 {dataset} 查询数量漂移")
        loaded_targets = np.asarray([query["result"] for query in queries], dtype=float)
        if not np.array_equal(loaded_targets, targets):
            raise RuntimeError(f"独立复核 {dataset} 查询加载结果漂移")
        if query_set_sha != spec["query_identity_sha256"]:
            raise RuntimeError(f"独立复核 {dataset} 结果盲查询身份漂移")
        if protocol.canonical_sha256(raw_targets) != spec["target_vector_sha256"]:
            raise RuntimeError(f"独立复核 {dataset} 结果盲目标身份漂移")
        if (
            ordered_query_identity_sha256(queries)
            != spec["trace_query_identity_sha256"]
        ):
            raise RuntimeError(f"独立复核 {dataset} 状态轨迹查询身份漂移")
        if (
            target_answer_identity_sha256(loaded_targets)
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"独立复核 {dataset} 状态轨迹目标身份漂移")
        inputs[dataset] = {
            "schema": schema,
            "queries": queries,
            "targets": loaded_targets,
        }
    references, reference_sha = offline_primitives._load_references(root, runtime)
    expected_reference = {
        name: spec["input_sha256"]["reference"]
        for name, spec in protocol.DATASETS.items()
    }
    if reference_sha != expected_reference:
        raise RuntimeError("独立复核原始参考表哈希漂移")
    auxiliary = {
        "test_groups": test_groups,
        "nltcs_groups": nltcs_groups,
        "references": references,
        "test_group_targets": {
            name: runtime.evaluate_table(references["test_300x10"], queries)
            for name, queries in test_groups.items()
        },
        "nltcs_marginals": _load_json(root / protocol.DATASETS["nltcs"]["marginals"]),
    }
    auxiliary["nltcs_domains"] = runtime.offline._discretization_domains(
        auxiliary["nltcs_marginals"]
    )
    auxiliary["nltcs_measured_triples"] = runtime.offline._measured_cell_keys(
        inputs["nltcs"]["queries"],
        auxiliary["nltcs_marginals"],
        order=3,
    )
    identities = {
        "query_identity_frozen_before_reference_load": True,
        "test": test_identity,
        "nltcs": nltcs_identity,
        "reference_sha256": reference_sha,
    }
    return {"inputs": inputs, **auxiliary}, identities


def _arithmetic(
    target: np.ndarray, answers: np.ndarray, n_records: int
) -> dict[str, Any]:
    target_values = np.asarray(target, dtype=float)
    answer_values = np.asarray(answers, dtype=float)
    if (
        isinstance(n_records, bool)
        or not isinstance(n_records, int)
        or n_records <= 0
        or target_values.size == 0
        or target_values.shape != answer_values.shape
        or target_values.ndim != 1
        or not np.all(np.isfinite(target_values))
        or not np.all(np.isfinite(answer_values))
        or not np.all(target_values == np.rint(target_values))
        or not np.all(answer_values == np.rint(answer_values))
        or np.any(target_values < 0)
        or np.any(target_values > n_records)
        or np.any(answer_values < 0)
        or np.any(answer_values > n_records)
    ):
        raise RuntimeError("独立复核查询向量非法")
    target_int = target_values.astype(np.int64)
    answer_int = answer_values.astype(np.int64)
    error = np.abs(target_int - answer_int)
    integer_sum = sum(int(value) for value in error)
    squared_sum = sum(
        (int(target_value) - int(answer_value)) ** 2
        for target_value, answer_value in zip(target_int, answer_int, strict=True)
    )
    gap_sum = math.fsum(
        int(error_value) / max(int(target_value), 8)
        for error_value, target_value in zip(error, target_int, strict=True)
    )
    return {
        "absolute_count_error_sum": integer_sum,
        "normalized_l1": float(integer_sum / (len(error) * n_records)),
        "squared_loss": float(squared_sum / 2),
        "gap_e": float(gap_sum / len(target_int)),
    }


def _checkpoint_independent(
    root: Path,
    source: dict[str, Any],
    target: np.ndarray,
    terminal_answers: np.ndarray,
) -> dict[str, Any]:
    artifact = _load_json(_safe_artifact(root, source["checkpoint_artifact_path"]))
    if (
        artifact.get("contract_version") != protocol.PROTOCOL_VERSION
        or artifact.get("task_id") != source["task_id"]
        or artifact.get("dataset") != source["dataset"]
        or artifact.get("arm") != source["arm"]
        or artifact.get("seed") != source["seed"]
        or artifact.get("n_records")
        != protocol.DATASETS[source["dataset"]]["n_records"]
        or artifact.get("query_count") != len(target)
        or artifact.get("query_identity_sha256")
        != protocol.DATASETS[source["dataset"]]["query_identity_sha256"]
        or artifact.get("target_vector_sha256")
        != protocol.DATASETS[source["dataset"]]["target_vector_sha256"]
        or artifact.get("trace_query_identity_sha256")
        != protocol.DATASETS[source["dataset"]]["trace_query_identity_sha256"]
        or artifact.get("trace_target_vector_sha256")
        != protocol.DATASETS[source["dataset"]]["trace_target_vector_sha256"]
        or artifact.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or artifact.get("fixed_checkpoint_rounds_requested")
        != list(protocol.CHECKPOINT_ROUNDS)
        or artifact.get("historical_best_included") is not False
    ):
        raise RuntimeError(f"独立复核检查点身份漂移：{source['task_id']}")
    n_records = protocol.DATASETS[source["dataset"]]["n_records"]

    def one(row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        answers = np.asarray(row["query_answers"], dtype=float)
        metrics = _arithmetic(target, answers, n_records)
        for name, value in metrics.items():
            if row.get(name) != value:
                raise RuntimeError(
                    f"独立检查点算术漂移：{source['task_id']}/{row.get('round')}/{name}"
                )
        return (
            {
                "kind": row["kind"],
                "state_index": int(row["state_index"]),
                "round": int(row["round"]),
                "phase": row["phase"],
                **metrics,
            },
            answers,
        )

    fixed = []
    for row in artifact["fixed_checkpoints"]:
        recomputed, _ = one(row)
        fixed.append(recomputed)
    expected_rounds = [
        value
        for value in protocol.CHECKPOINT_ROUNDS
        if value <= source["applied_rounds"]
    ]
    if [row["round"] for row in fixed] != expected_rounds:
        raise RuntimeError(f"独立复核检查点覆盖漂移：{source['task_id']}")
    terminal, terminal_vector = one(artifact["terminal"])
    if not np.array_equal(terminal_vector, terminal_answers):
        raise RuntimeError(f"独立复核终表/终点查询向量不一致：{source['task_id']}")
    return {"fixed_checkpoints": fixed, "terminal": terminal}


def _recompute_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = offline_primitives._load_runtime()
    material, identities = _load_inputs_and_references(root, runtime)
    inputs = material["inputs"]
    cases = []
    for task in protocol.task_plan().tasks:
        source = indexed[(task.seed, task.dataset, task.arm)]
        table = runtime.pd.read_csv(_safe_artifact(root, source["terminal_table_path"]))
        if len(table) != protocol.DATASETS[task.dataset]["n_records"]:
            raise RuntimeError(f"独立复核终表行数漂移：{task.task_id}")
        dataset_input = inputs[task.dataset]
        terminal_answers = np.asarray(
            runtime.evaluate_table(table, dataset_input["queries"]), dtype=float
        )
        terminal = _arithmetic(
            dataset_input["targets"],
            terminal_answers,
            protocol.DATASETS[task.dataset]["n_records"],
        )
        trajectory = _checkpoint_independent(
            root, source, dataset_input["targets"], terminal_answers
        )
        if trajectory["terminal"]["normalized_l1"] != terminal["normalized_l1"]:
            raise RuntimeError(f"独立终点 L1 两路径不一致：{task.task_id}")
        if task.dataset == "test_300x10":
            metrics = offline_primitives._evaluate_test_case(
                runtime,
                table,
                dataset_input["queries"],
                dataset_input["targets"],
                material["test_groups"],
                material["test_group_targets"],
                dataset_input["schema"],
                material["references"][task.dataset],
            )
        else:
            metrics = offline_primitives._evaluate_nltcs_case(
                runtime,
                table,
                dataset_input["queries"],
                dataset_input["targets"],
                material["nltcs_groups"]["one_way_safety"],
                dataset_input["schema"],
                material["nltcs_marginals"],
                material["nltcs_domains"],
                material["nltcs_measured_triples"],
                material["references"][task.dataset],
            )
        metrics["measured"]["overall"]["absolute_count_error_sum"] = terminal[
            "absolute_count_error_sum"
        ]
        metrics["measured"]["overall"]["gap_e"] = terminal["gap_e"]
        if not math.isclose(
            metrics["measured"]["overall"]["normalized_l1_mean"],
            terminal["normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or (
            metrics["measured"]["overall"]["squared_loss_diagnostic_only"]
            != terminal["squared_loss"]
        ):
            raise RuntimeError(f"独立 measured（测量）算术漂移：{task.task_id}")
        metrics["measured"]["overall"]["normalized_l1_mean"] = terminal["normalized_l1"]
        cases.append(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "arm": task.arm,
                "seed": task.seed,
                "termination_reason": source["termination_reason"],
                "applied_rounds": source["applied_rounds"],
                "terminal_table_sha256": source["terminal_table_sha256"],
                "metrics": metrics,
                "trajectory_l1": trajectory,
                "cost": {
                    name: source[name]
                    for name in (
                        "elapsed_sec",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "state_evaluation_count",
                        "candidate_evaluation_count",
                        "distance_evaluation_count",
                        "direction_evaluation_count",
                        "gap_microsteps",
                        "factorized_gibbs_microsteps",
                        "factorized_gibbs_conditional_logit_evaluated_count",
                    )
                },
            }
        )
    return cases, identities


def _value(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for component in path.split("."):
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("独立汇总指标不是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("独立汇总指标不是有限值")
    return result


def _cell(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> list[dict[str, Any]]:
    rows = sorted(
        [case for case in cases if case["dataset"] == dataset and case["arm"] == arm],
        key=lambda case: case["seed"],
    )
    if [row["seed"] for row in rows] != list(protocol.FORMAL_SEEDS):
        raise RuntimeError("独立汇总缺少冻结五随机种子")
    return rows


def _cell_summary(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str, path: str
) -> dict[str, Any]:
    rows = _cell(cases, dataset, arm)
    values = [_value(row, path) for row in rows]
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
        "values_by_seed": {str(row["seed"]): value for row, value in zip(rows, values)},
    }


def _pair(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    baseline_arm: str,
    path: str,
    lower_is_better: bool,
) -> dict[str, Any]:
    gap = {
        row["seed"]: _value(row, path)
        for row in _cell(cases, dataset, protocol.ARM_GAP)
    }
    base = {
        row["seed"]: _value(row, path) for row in _cell(cases, dataset, baseline_arm)
    }
    differences = [gap[seed] - base[seed] for seed in protocol.FORMAL_SEEDS]
    mean_difference = float(statistics.fmean(differences))
    half = (
        T_CRITICAL_DF4_95 * statistics.stdev(differences) / math.sqrt(len(differences))
    )
    gap_mean = float(statistics.fmean(gap.values()))
    base_mean = float(statistics.fmean(base.values()))
    return {
        "metric": path,
        "candidate_arm": protocol.ARM_GAP,
        "baseline_arm": baseline_arm,
        "lower_is_better": lower_is_better,
        "candidate_mean": gap_mean,
        "baseline_mean": base_mean,
        "candidate_over_baseline": gap_mean / base_mean if base_mean != 0.0 else None,
        "mean_paired_difference": mean_difference,
        "paired_wins": int(
            sum(value < 0 if lower_is_better else value > 0 for value in differences)
        ),
        "paired_ties": int(sum(value == 0.0 for value in differences)),
        "paired_losses": int(
            sum(value > 0 if lower_is_better else value < 0 for value in differences)
        ),
        "candidate_values_by_seed": {
            str(seed): gap[seed] for seed in protocol.FORMAL_SEEDS
        },
        "baseline_values_by_seed": {
            str(seed): base[seed] for seed in protocol.FORMAL_SEEDS
        },
        "candidate_minus_baseline_by_seed": {
            str(seed): value for seed, value in zip(protocol.FORMAL_SEEDS, differences)
        },
        "paired_difference_95pct_t_interval_diagnostic_only": [
            mean_difference - half,
            mean_difference + half,
        ],
    }


def _stability(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str
) -> dict[str, Any]:
    count = _pair(
        cases,
        dataset,
        baseline,
        "metrics.measured.overall.absolute_count_error_sum",
        True,
    )
    l1 = _pair(
        cases,
        dataset,
        baseline,
        "metrics.measured.overall.normalized_l1_mean",
        True,
    )
    gap_sum = int(sum(count["candidate_values_by_seed"].values()))
    base_sum = int(sum(count["baseline_values_by_seed"].values()))
    stable = gap_sum < base_sum and count["paired_wins"] >= protocol.STABLE_WIN_MINIMUM
    reverse = (
        base_sum < gap_sum and count["paired_losses"] >= protocol.STABLE_WIN_MINIMUM
    )
    return {
        "count_error": count,
        "normalized_l1": l1,
        "candidate_five_seed_aggregate_count_error_sum": gap_sum,
        "baseline_five_seed_aggregate_count_error_sum": base_sum,
        "paired_strict_wins_required": protocol.STABLE_WIN_MINIMUM,
        "stable_gap_gain": bool(stable),
        "stable_baseline_advantage": bool(reverse),
        "mixed_no_stable_winner": bool(not stable and not reverse),
    }


def _risk_lower(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str, path: str
) -> dict[str, Any]:
    result = _pair(cases, dataset, baseline, path, True)
    candidate = result["candidate_mean"]
    base = result["baseline_mean"]
    result["maximum_ratio"] = protocol.LOWER_RISK_RATIO_MAX
    result["pass"] = bool(
        candidate == 0.0
        if base == 0.0
        else candidate / base <= protocol.LOWER_RISK_RATIO_MAX
    )
    return result


def _risk_higher(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str, path: str
) -> dict[str, Any]:
    result = _pair(cases, dataset, baseline, path, False)
    result["minimum_ratio"] = protocol.HIGHER_QUALITY_RATIO_MIN
    result["pass"] = bool(
        result["candidate_mean"]
        >= protocol.HIGHER_QUALITY_RATIO_MIN * result["baseline_mean"]
    )
    return result


def _safety_independent(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str
) -> dict[str, Any]:
    if dataset == "test_300x10":
        lower_paths = {
            name: f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            for name in protocol.TEST_GROUP_ORDER
        }
    else:
        lower_paths = {
            "one_way_safety": "metrics.offline_query_groups.one_way_safety.normalized_l1_mean",
            "unmeasured_3way": "metrics.offline_query_groups.unmeasured_3way.normalized_l1_mean",
            "all_4way": "metrics.offline_query_groups.all_4way.normalized_l1_mean",
            "binned_joint_tvd": "metrics.binned_joint.tvd",
        }
    higher_paths = {
        "synthetic_mass_in_reference_support": "metrics.reference_support.synthetic_mass_in_reference_support",
        "reference_mass_covered": "metrics.reference_support.reference_mass_covered",
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": "metrics.diversity.effective_unique_row_ratio",
        "attribute_effective_support_ratio_mean": "metrics.diversity.attribute_effective_support_ratio_mean",
        "attribute_effective_support_ratio_min": "metrics.diversity.attribute_effective_support_ratio_min",
    }
    lower = {
        name: _risk_lower(cases, dataset, baseline, path)
        for name, path in lower_paths.items()
    }
    higher = {
        name: _risk_higher(cases, dataset, baseline, path)
        for name, path in higher_paths.items()
    }
    validity = {
        str(row["seed"]): _value(row, "metrics.validity.valid_row_rate")
        for row in _cell(cases, dataset, protocol.ARM_GAP)
    }
    validity_pass = all(value == 1.0 for value in validity.values())
    return {
        "baseline_arm": baseline,
        "offline_lower_is_better": lower,
        "support_and_diversity_higher_is_better": higher,
        "gap_valid_row_rate_by_seed": validity,
        "validity_pass": validity_pass,
        "pass": bool(
            validity_pass
            and all(item["pass"] for item in lower.values())
            and all(item["pass"] for item in higher.values())
        ),
    }


def _decision_independent(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    measured = {
        baseline: _stability(cases, dataset, baseline)
        for baseline in protocol.BASELINE_ARMS
    }
    safety = {
        baseline: _safety_independent(cases, dataset, baseline)
        for baseline in protocol.BASELINE_ARMS
    }
    measured_pass = all(value["stable_gap_gain"] for value in measured.values())
    safety_pass = all(value["pass"] for value in safety.values())
    classification = (
        "no_stable_gap_gain"
        if not measured_pass
        else (
            "gap_measured_gain_with_quality_risk"
            if not safety_pass
            else "gap_quality_supported"
        )
    )
    return {
        "classification": classification,
        "measured_pass_against_both_baselines": measured_pass,
        "measured_by_baseline": measured,
        "safety_pass_against_both_baselines": safety_pass,
        "safety_by_baseline": safety,
    }


def _classification_independent(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    decisions = {
        dataset: _decision_independent(cases, dataset)
        for dataset in joint.DATASET_ORDER
    }
    values = [value["classification"] for value in decisions.values()]
    if all(value == "gap_quality_supported" for value in values):
        cross = "shared_gap_kernel_support"
    elif all(
        value in {"gap_quality_supported", "gap_measured_gain_with_quality_risk"}
        for value in values
    ):
        cross = "shared_measured_gain_with_quality_risk"
    elif any(
        value in {"gap_quality_supported", "gap_measured_gain_with_quality_risk"}
        for value in values
    ):
        cross = "dataset_dependent_gap_response"
    else:
        cross = "no_shared_gap_support"
    return {"by_dataset": decisions, "cross_dataset_response": cross}


def _summary_independent(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "terminal_absolute_count_error_sum": "metrics.measured.overall.absolute_count_error_sum",
        "terminal_normalized_l1": "metrics.measured.overall.normalized_l1_mean",
        "terminal_gap_e": "metrics.measured.overall.gap_e",
        "terminal_squared_loss": "metrics.measured.overall.squared_loss_diagnostic_only",
        "elapsed_sec": "cost.elapsed_sec",
        "gap_microsteps": "cost.gap_microsteps",
        "factorized_gibbs_microsteps": "cost.factorized_gibbs_microsteps",
    }
    result = {}
    for dataset in joint.DATASET_ORDER:
        result[dataset] = {}
        for arm in joint.ARM_ORDER:
            cell = {
                name: _cell_summary(cases, dataset, arm, path)
                for name, path in paths.items()
            }
            checkpoints = {}
            for round_index in protocol.CHECKPOINT_ROUNDS:
                values = [
                    float(checkpoint["normalized_l1"])
                    for case in _cell(cases, dataset, arm)
                    for checkpoint in case["trajectory_l1"]["fixed_checkpoints"]
                    if checkpoint["round"] == round_index
                ]
                checkpoints[str(round_index)] = {
                    "observed_seed_count": len(values),
                    "mean_normalized_l1": (
                        float(statistics.fmean(values)) if values else None
                    ),
                    "values": values,
                    "selection_role": "diagnostic_only",
                }
            cell["checkpoint_normalized_l1"] = checkpoints
            result[dataset][arm] = cell
    return result


def _csv_rows_independent(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        terminal = case["trajectory_l1"]["terminal"]
        fixed_rounds = set()
        for checkpoint in case["trajectory_l1"]["fixed_checkpoints"]:
            fixed_rounds.add(checkpoint["round"])
            rows.append(
                {
                    "dataset": case["dataset"],
                    "arm": case["arm"],
                    "seed": case["seed"],
                    "record_kind": "fixed_checkpoint",
                    "round": checkpoint["round"],
                    "state_index": checkpoint["state_index"],
                    "is_terminal": checkpoint["round"] == terminal["round"],
                    "absolute_count_error_sum": checkpoint["absolute_count_error_sum"],
                    "normalized_l1": checkpoint["normalized_l1"],
                    "gap_e": checkpoint["gap_e"],
                    "squared_loss": checkpoint["squared_loss"],
                }
            )
        if terminal["round"] not in fixed_rounds:
            rows.append(
                {
                    "dataset": case["dataset"],
                    "arm": case["arm"],
                    "seed": case["seed"],
                    "record_kind": "terminal",
                    "round": terminal["round"],
                    "state_index": terminal["state_index"],
                    "is_terminal": True,
                    "absolute_count_error_sum": terminal["absolute_count_error_sum"],
                    "normalized_l1": terminal["normalized_l1"],
                    "gap_e": terminal["gap_e"],
                    "squared_loss": terminal["squared_loss"],
                }
            )
    return rows


def _audit_csv(
    root: Path, evaluation: dict[str, Any], expected: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    path = root / protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV
    metadata = evaluation.get("l1_results_csv")
    if not isinstance(metadata, dict):
        raise TypeError("evaluation（评价）缺少 L1 CSV 元数据")
    observed_sha = protocol.file_sha256(path)
    if (
        metadata.get("path") != protocol.L1_RESULTS_CSV
        or metadata.get("sha256") != observed_sha
        or metadata.get("row_count") != len(expected)
        or metadata.get("columns") != list(CSV_FIELDS)
    ):
        raise RuntimeError("L1 CSV 身份或行列元数据漂移")
    observed = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError("L1 CSV 列顺序漂移")
        for row in reader:
            observed.append(
                {
                    "dataset": row["dataset"],
                    "arm": row["arm"],
                    "seed": int(row["seed"]),
                    "record_kind": row["record_kind"],
                    "round": int(row["round"]),
                    "state_index": int(row["state_index"]),
                    "is_terminal": row["is_terminal"] == "True",
                    "absolute_count_error_sum": int(row["absolute_count_error_sum"]),
                    "normalized_l1": float(row["normalized_l1"]),
                    "gap_e": float(row["gap_e"]),
                    "squared_loss": float(row["squared_loss"]),
                }
            )
    if observed != list(expected):
        raise RuntimeError("L1 CSV 与独立终表/检查点复算不一致")
    return {
        "sha256": observed_sha,
        "row_count": len(observed),
        "all_rows_exactly_recomputed": True,
    }


def _audit_evaluation(
    root: Path,
    confirmed_sha: str,
    collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha(confirmed_sha, "evaluation（评价）报告 SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("evaluation（评价）报告 SHA-256 不一致")
    report = _load_json(path)
    if (
        report.get("contract_version") != "issue53-stage6d-joint-formal-evaluation-v2"
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("collection_report_sha256") != collection_sha
        or report.get("collection_execution_commit")
        != collection_report["execution_commit"]
        or report.get("case_count") != 30
        or report.get("raw_reference_data_accessed") is not True
        or report.get("l1_fully_emitted") is not True
        or report.get("historical_best_used") is not False
        or report.get("checkpoint_selected_as_output") is not False
        or report.get("new_generation_performed_by_evaluator") is not False
        or report.get("parameter_retuning_performed") is not False
    ):
        raise RuntimeError("evaluation（评价）顶层契约漂移")
    evaluation_commit = report.get("evaluation_commit")
    if (
        not isinstance(evaluation_commit, str)
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
    ):
        raise RuntimeError("evaluation（评价）执行提交无效")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(root, "merge-base", evaluation_commit, current_commit)
        != evaluation_commit
    ):
        raise RuntimeError("evaluation（评价）执行提交不是当前提交祖先")
    if (
        collection._git_text(
            root,
            "merge-base",
            collection_report["execution_commit"],
            evaluation_commit,
        )
        != collection_report["execution_commit"]
    ):
        raise RuntimeError("collection（采集）执行提交不是评价提交祖先")
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("evaluation（评价）查询/参考身份漂移")
    if report.get("cases") != cases:
        raise RuntimeError("evaluation（评价）30条指标与独立复算不一致")
    expected_summary = _summary_independent(cases)
    if report.get("summary") != expected_summary:
        raise RuntimeError("evaluation（评价）汇总与独立复算不一致")
    expected_classification = _classification_independent(cases)
    if report.get("frozen_classification") != expected_classification:
        raise RuntimeError("evaluation（评价）冻结分类与独立复算不一致")
    csv_audit = _audit_csv(root, report, _csv_rows_independent(cases))
    return report, {
        "evaluation_report_sha256": confirmed_sha,
        "all_30_case_metrics_exactly_recomputed": True,
        "all_terminal_and_checkpoint_l1_exactly_recomputed": True,
        "all_pairwise_gates_exactly_recomputed": True,
        "classification_exactly_reproduced": True,
        "l1_csv": csv_audit,
        "pass": True,
    }


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("独立复核要求包含未跟踪文件在内的干净工作树")
    output = root / protocol.OUTPUT_DIR / protocol.AUDIT_REPORT
    if output.exists():
        raise FileExistsError(f"独立复核报告已存在，不覆盖：{output}")
    collection_report, indexed = _audit_collection_independently(
        root, confirmed_collection_report_sha256
    )
    cases, identities = _recompute_cases(root, indexed)
    _, evaluation_audit = _audit_evaluation(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report,
        cases,
        identities,
    )
    report = {
        **build_plan(),
        "mode": "independent_recompute_from_terminal_tables_and_checkpoint_vectors",
        "audit_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "evaluation_report_sha256": confirmed_evaluation_report_sha256,
        "query_and_reference_identity_audit": identities,
        "recomputed_case_count": len(cases),
        "independent_classification": _classification_independent(cases),
        "evaluation_audit": evaluation_audit,
        "overall_pass": True,
        "new_generation_performed_by_auditor": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "parameter_retuning_performed": False,
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"独立复核临时文件已存在：{temporary}")
    try:
        temporary.write_text(collection._strict_json_text(report), encoding="utf-8")
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
        print(collection._strict_json_text(build_plan()), end="")
        return
    path = audit(args.confirm_collection_sha, args.confirm_evaluation_sha)
    print(f"第 6D independent audit（独立复核） -> {path}")
    print(f"audit SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
