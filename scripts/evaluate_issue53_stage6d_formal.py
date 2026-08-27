#!/usr/bin/env python3
"""离线评价 Issue #53 第 6D 阶段完整正式 collection（采集）。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_issue53_fixed_alpha_calibration as offline_helpers
from scripts import freeze_issue53_test_query_workload_ab as blind_identity
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as protocol
from scripts import run_issue53_stage6d_formal as collection
from table_diffevo.stationarity import (
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)

EVALUATION_VERSION = "issue53-stage6d-joint-formal-evaluation-v2"
T_CRITICAL_DF4_95 = 2.7764451051977987
L1_CSV_FIELDS = (
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
    return collection._load_json(path)


def _strict_json_text(value: Any) -> str:
    return collection._strict_json_text(value)


def _nested(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric（指标）{path} 不是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric（指标）{path} 不是有限数")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("均值输入不得为空")
    return float(statistics.fmean(values))


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
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact（产物）相对路径必须是字符串")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact（产物）路径越界")
    resolved = (root / protocol.OUTPUT_DIR / path).resolve()
    expected_root = (root / protocol.OUTPUT_DIR).resolve()
    if resolved != expected_root and expected_root not in resolved.parents:
        raise ValueError("artifact（产物）路径逃逸正式输出目录")
    return resolved


def _audit_collection(
    root: Path,
    confirmed_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    protocol.require_collection_confirmation(confirmed_sha256)
    report_path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    observed_sha = protocol.file_sha256(report_path)
    if observed_sha != confirmed_sha256:
        raise ValueError("collection（采集）报告 SHA-256 与确认值不一致")
    report = _load_json(report_path)
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
        or report.get("case_count") != 30
        or report.get("paired_dataset_seed_count") != 10
        or report.get("formal_result_valid") is not True
        or report.get("raw_reference_data_accessed") is not False
        or report.get("partial_matrix_comparison_emitted") is not False
        or report.get("method_ranking_emitted") is not False
        or report.get("l1_results_published_by_collection") is not False
        or report.get("parameter_retuning_performed") is not False
    ):
        raise RuntimeError("collection（采集）协议、完整性或信息边界漂移")
    expected_audit = {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
    }
    if report.get("collection_audit") != expected_audit:
        raise RuntimeError("collection（采集）结构审计未完整通过")
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

    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("collection（采集）执行提交缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(root, "merge-base", execution_commit, current_commit)
        != execution_commit
    ):
        raise RuntimeError("collection（采集）提交不是评价提交的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError("collection（采集）不是完整30条")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    expected_tasks = {task.task_id: task for task in protocol.task_plan().tasks}
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in expected_tasks:
            raise RuntimeError(f"collection（采集）任务地址非法：{task_id!r}")
        task = expected_tasks[task_id]
        collection._validate_case_row(task, row)
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError(f"collection（采集）任务重复：{key}")
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
            raise RuntimeError(f"collection（采集）身份或数值护栏漂移：{key}")
        for path_key, sha_key in (
            ("terminal_table_path", "terminal_table_sha256"),
            ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
            ("transition_audit_path", "transition_audit_sha256"),
        ):
            artifact = _artifact_path(root, row[path_key])
            if protocol.file_sha256(artifact) != row[sha_key]:
                raise RuntimeError(
                    f"collection artifact（采集产物）哈希漂移：{key}/{path_key}"
                )
        indexed[key] = row
    expected = {
        (task.seed, task.dataset, task.arm) for task in protocol.task_plan().tasks
    }
    if set(indexed) != expected:
        raise RuntimeError("collection（采集）30条任务身份不完整")
    return report, indexed


def _load_dataset_inputs(root: Path, runtime: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for dataset, spec in protocol.DATASETS.items():
        for key in ("schema", "queries", "marginals"):
            if protocol.file_sha256(root / spec[key]) != spec["input_sha256"][key]:
                raise RuntimeError(f"{dataset}.{key} 当前输入 SHA-256 漂移")
        schema = runtime.load_schema(str(root / spec["schema"]))
        queries = runtime.load_queries(str(root / spec["queries"]))
        payload = _load_json(root / spec["queries"])
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise TypeError(f"{dataset} 原始测量查询列表缺失")
        query_set_sha = blind_identity.query_set_identity(raw_queries)
        raw_targets = [query.get("result") for query in raw_queries]
        targets = np.asarray(raw_targets, dtype=float)
        if len(raw_queries) != spec["query_count"] or len(queries) != len(raw_queries):
            raise RuntimeError(f"{dataset} 测量查询输入数量漂移")
        loaded_targets = np.asarray([query["result"] for query in queries], dtype=float)
        if not np.array_equal(loaded_targets, targets):
            raise RuntimeError(f"{dataset} 查询加载结果与原始载荷漂移")
        if query_set_sha != spec["query_identity_sha256"]:
            raise RuntimeError(f"{dataset} 结果盲查询集合身份漂移")
        if protocol.canonical_sha256(raw_targets) != spec["target_vector_sha256"]:
            raise RuntimeError(f"{dataset} 结果盲目标向量身份漂移")
        if (
            ordered_query_identity_sha256(queries)
            != spec["trace_query_identity_sha256"]
        ):
            raise RuntimeError(f"{dataset} 状态轨迹查询身份漂移")
        if (
            target_answer_identity_sha256(loaded_targets)
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"{dataset} 状态轨迹目标身份漂移")
        result[dataset] = {
            "schema": schema,
            "queries": queries,
            "targets": loaded_targets,
        }
    return result


def _measured_exact(
    runtime: Any,
    table: Any,
    queries: Sequence[dict[str, Any]],
    targets: np.ndarray,
    n_records: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    answers = np.asarray(runtime.evaluate_table(table, list(queries)), dtype=float)
    computed = collection._metrics_from_answers(targets, answers, n_records)
    return answers, computed


def _audit_checkpoint_artifact(
    root: Path,
    source: dict[str, Any],
    targets: np.ndarray,
    terminal_answers: np.ndarray,
) -> dict[str, Any]:
    artifact = _load_json(_artifact_path(root, source["checkpoint_artifact_path"]))
    if (
        artifact.get("contract_version") != protocol.PROTOCOL_VERSION
        or artifact.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or artifact.get("task_id") != source["task_id"]
        or artifact.get("dataset") != source["dataset"]
        or artifact.get("arm") != source["arm"]
        or artifact.get("seed") != source["seed"]
        or artifact.get("n_records")
        != protocol.DATASETS[source["dataset"]]["n_records"]
        or artifact.get("query_count") != len(targets)
        or artifact.get("query_identity_sha256")
        != protocol.DATASETS[source["dataset"]]["query_identity_sha256"]
        or artifact.get("target_vector_sha256")
        != protocol.DATASETS[source["dataset"]]["target_vector_sha256"]
        or artifact.get("trace_query_identity_sha256")
        != protocol.DATASETS[source["dataset"]]["trace_query_identity_sha256"]
        or artifact.get("trace_target_vector_sha256")
        != protocol.DATASETS[source["dataset"]]["trace_target_vector_sha256"]
        or artifact.get("fixed_checkpoint_rounds_requested")
        != list(protocol.CHECKPOINT_ROUNDS)
        or artifact.get("historical_best_included") is not False
    ):
        raise RuntimeError(f"检查点 artifact（产物）身份漂移：{source['task_id']}")
    n_records = protocol.DATASETS[source["dataset"]]["n_records"]

    def recompute(row: dict[str, Any]) -> dict[str, Any]:
        answers = np.asarray(row["query_answers"], dtype=float)
        metrics = collection._metrics_from_answers(targets, answers, n_records)
        for key, value in metrics.items():
            if row.get(key) != value:
                raise RuntimeError(
                    f"检查点 {source['task_id']}/{row.get('round')} {key} 复算漂移"
                )
        return {
            "kind": row["kind"],
            "state_index": int(row["state_index"]),
            "round": int(row["round"]),
            "phase": row["phase"],
            **metrics,
        }

    fixed_rows = artifact.get("fixed_checkpoints")
    if not isinstance(fixed_rows, list):
        raise TypeError("固定检查点列表缺失")
    observed_rounds = [int(row["round"]) for row in fixed_rows]
    expected_rounds = [
        value
        for value in protocol.CHECKPOINT_ROUNDS
        if value <= source["applied_rounds"]
    ]
    if observed_rounds != expected_rounds:
        raise RuntimeError(f"固定检查点覆盖漂移：{source['task_id']}")
    fixed = [recompute(row) for row in fixed_rows]
    terminal_source = artifact.get("terminal")
    if not isinstance(terminal_source, dict):
        raise TypeError("terminal（终点）检查点缺失")
    terminal_vector = np.asarray(terminal_source["query_answers"], dtype=float)
    if not np.array_equal(terminal_vector, terminal_answers):
        raise RuntimeError(f"终表与终点查询向量不一致：{source['task_id']}")
    terminal = recompute(terminal_source)
    if terminal["round"] != source["applied_rounds"]:
        raise RuntimeError(f"终点轮次与 collection（采集）不一致：{source['task_id']}")
    return {"fixed_checkpoints": fixed, "terminal": terminal}


def _evaluate_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = offline_helpers._load_runtime()
    # 查询身份先冻结，之后才加载原始参考表。
    test_groups, test_identity = offline_helpers._freeze_test_groups(root)
    nltcs_groups, nltcs_identity = offline_helpers._freeze_nltcs_groups(root)
    inputs = _load_dataset_inputs(root, runtime)
    references, reference_sha = offline_helpers._load_references(root, runtime)
    if reference_sha != {
        name: spec["input_sha256"]["reference"]
        for name, spec in protocol.DATASETS.items()
    }:
        raise RuntimeError("原始参考表 SHA-256 漂移")

    test_group_targets = {
        name: runtime.evaluate_table(references["test_300x10"], queries)
        for name, queries in test_groups.items()
    }
    nltcs_marginals = _load_json(root / protocol.DATASETS["nltcs"]["marginals"])
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    nltcs_measured_triples = runtime.offline._measured_cell_keys(
        inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )

    cases = []
    for task in protocol.task_plan().tasks:
        source = indexed[(task.seed, task.dataset, task.arm)]
        table_path = _artifact_path(root, source["terminal_table_path"])
        table = runtime.pd.read_csv(table_path)
        spec = protocol.DATASETS[task.dataset]
        if len(table) != spec["n_records"]:
            raise RuntimeError(f"终表行数漂移：{task.task_id}")
        dataset_input = inputs[task.dataset]
        terminal_answers, terminal_exact = _measured_exact(
            runtime,
            table,
            dataset_input["queries"],
            dataset_input["targets"],
            spec["n_records"],
        )
        checkpoints = _audit_checkpoint_artifact(
            root, source, dataset_input["targets"], terminal_answers
        )
        if checkpoints["terminal"]["normalized_l1"] != terminal_exact["normalized_l1"]:
            raise RuntimeError(f"终表与检查点 L1 不一致：{task.task_id}")

        if task.dataset == "test_300x10":
            metrics = offline_helpers._evaluate_test_case(
                runtime,
                table,
                dataset_input["queries"],
                dataset_input["targets"],
                test_groups,
                test_group_targets,
                dataset_input["schema"],
                references[task.dataset],
            )
        else:
            metrics = offline_helpers._evaluate_nltcs_case(
                runtime,
                table,
                dataset_input["queries"],
                dataset_input["targets"],
                nltcs_groups["one_way_safety"],
                dataset_input["schema"],
                nltcs_marginals,
                nltcs_domains,
                nltcs_measured_triples,
                references[task.dataset],
            )
        overall = metrics["measured"]["overall"]
        overall["absolute_count_error_sum"] = terminal_exact["absolute_count_error_sum"]
        overall["gap_e"] = terminal_exact["gap_e"]
        if not math.isclose(
            overall["normalized_l1_mean"],
            terminal_exact["normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or (
            overall["squared_loss_diagnostic_only"] != terminal_exact["squared_loss"]
        ):
            raise RuntimeError(f"离线 measured（测量）指标复算漂移：{task.task_id}")
        # 统一使用整数误差总和除以 m*N，避免浮点求均值顺序造成末位漂移。
        overall["normalized_l1_mean"] = terminal_exact["normalized_l1"]
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
                "trajectory_l1": checkpoints,
                "cost": {
                    key: source[key]
                    for key in (
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
    identities = {
        "query_identity_frozen_before_reference_load": True,
        "test": test_identity,
        "nltcs": nltcs_identity,
        "reference_sha256": reference_sha,
    }
    return cases, identities


def _selected(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> list[dict[str, Any]]:
    selected = sorted(
        [case for case in cases if case["dataset"] == dataset and case["arm"] == arm],
        key=lambda case: case["seed"],
    )
    if [case["seed"] for case in selected] != list(protocol.FORMAL_SEEDS):
        raise RuntimeError(f"{dataset}/{arm} 缺少五个冻结随机种子")
    return selected


def _arm_summary(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str, path: str
) -> dict[str, Any]:
    selected = _selected(cases, dataset, arm)
    values = [_nested(case, path) for case in selected]
    return {
        "mean": _mean(values),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
        "values_by_seed": {
            str(case["seed"]): value for case, value in zip(selected, values)
        },
    }


def _paired(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    baseline: str,
    path: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    candidate_rows = _selected(cases, dataset, protocol.ARM_GAP)
    baseline_rows = _selected(cases, dataset, baseline)
    candidate = {case["seed"]: _nested(case, path) for case in candidate_rows}
    baseline_values = {case["seed"]: _nested(case, path) for case in baseline_rows}
    differences = [
        candidate[seed] - baseline_values[seed] for seed in protocol.FORMAL_SEEDS
    ]
    wins = sum(value < 0 if lower_is_better else value > 0 for value in differences)
    losses = sum(value > 0 if lower_is_better else value < 0 for value in differences)
    mean_difference = _mean(differences)
    half_width = (
        T_CRITICAL_DF4_95 * statistics.stdev(differences) / math.sqrt(len(differences))
    )
    candidate_mean = _mean(list(candidate.values()))
    baseline_mean = _mean(list(baseline_values.values()))
    return {
        "metric": path,
        "candidate_arm": protocol.ARM_GAP,
        "baseline_arm": baseline,
        "lower_is_better": lower_is_better,
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "candidate_over_baseline": (
            candidate_mean / baseline_mean if baseline_mean != 0.0 else None
        ),
        "mean_paired_difference": mean_difference,
        "paired_wins": int(wins),
        "paired_ties": int(sum(value == 0.0 for value in differences)),
        "paired_losses": int(losses),
        "candidate_values_by_seed": {
            str(seed): candidate[seed] for seed in protocol.FORMAL_SEEDS
        },
        "baseline_values_by_seed": {
            str(seed): baseline_values[seed] for seed in protocol.FORMAL_SEEDS
        },
        "candidate_minus_baseline_by_seed": {
            str(seed): value for seed, value in zip(protocol.FORMAL_SEEDS, differences)
        },
        "paired_difference_95pct_t_interval_diagnostic_only": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
    }


def _measured_stability(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str
) -> dict[str, Any]:
    count_path = "metrics.measured.overall.absolute_count_error_sum"
    l1_path = "metrics.measured.overall.normalized_l1_mean"
    count = _paired(cases, dataset, baseline, count_path, lower_is_better=True)
    l1 = _paired(cases, dataset, baseline, l1_path, lower_is_better=True)
    candidate_sum = int(sum(count["candidate_values_by_seed"].values()))
    baseline_sum = int(sum(count["baseline_values_by_seed"].values()))
    stable = (
        candidate_sum < baseline_sum
        and count["paired_wins"] >= protocol.STABLE_WIN_MINIMUM
    )
    reverse = (
        baseline_sum < candidate_sum
        and count["paired_losses"] >= protocol.STABLE_WIN_MINIMUM
    )
    return {
        "count_error": count,
        "normalized_l1": l1,
        "candidate_five_seed_aggregate_count_error_sum": candidate_sum,
        "baseline_five_seed_aggregate_count_error_sum": baseline_sum,
        "paired_strict_wins_required": protocol.STABLE_WIN_MINIMUM,
        "stable_gap_gain": bool(stable),
        "stable_baseline_advantage": bool(reverse),
        "mixed_no_stable_winner": bool(not stable and not reverse),
    }


def _lower_gate(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str, path: str
) -> dict[str, Any]:
    comparison = _paired(cases, dataset, baseline, path, lower_is_better=True)
    candidate = comparison["candidate_mean"]
    base = comparison["baseline_mean"]
    passed = (
        candidate == 0.0
        if base == 0.0
        else (candidate / base <= protocol.LOWER_RISK_RATIO_MAX)
    )
    comparison.update(
        {
            "maximum_ratio": protocol.LOWER_RISK_RATIO_MAX,
            "pass": bool(passed),
        }
    )
    return comparison


def _higher_gate(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str, path: str
) -> dict[str, Any]:
    comparison = _paired(cases, dataset, baseline, path, lower_is_better=False)
    candidate = comparison["candidate_mean"]
    base = comparison["baseline_mean"]
    passed = candidate >= protocol.HIGHER_QUALITY_RATIO_MIN * base
    comparison.update(
        {
            "minimum_ratio": protocol.HIGHER_QUALITY_RATIO_MIN,
            "pass": bool(passed),
        }
    )
    return comparison


def _safety(
    cases: Sequence[dict[str, Any]], dataset: str, baseline: str
) -> dict[str, Any]:
    if dataset == "test_300x10":
        lower_paths = {
            name: f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            for name in protocol.TEST_GROUP_ORDER
        }
    else:
        lower_paths = {
            "one_way_safety": (
                "metrics.offline_query_groups.one_way_safety.normalized_l1_mean"
            ),
            "unmeasured_3way": (
                "metrics.offline_query_groups.unmeasured_3way.normalized_l1_mean"
            ),
            "all_4way": ("metrics.offline_query_groups.all_4way.normalized_l1_mean"),
            "binned_joint_tvd": "metrics.binned_joint.tvd",
        }
    higher_paths = {
        "synthetic_mass_in_reference_support": (
            "metrics.reference_support.synthetic_mass_in_reference_support"
        ),
        "reference_mass_covered": "metrics.reference_support.reference_mass_covered",
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": "metrics.diversity.effective_unique_row_ratio",
        "attribute_effective_support_ratio_mean": (
            "metrics.diversity.attribute_effective_support_ratio_mean"
        ),
        "attribute_effective_support_ratio_min": (
            "metrics.diversity.attribute_effective_support_ratio_min"
        ),
    }
    lower = {
        name: _lower_gate(cases, dataset, baseline, path)
        for name, path in lower_paths.items()
    }
    higher = {
        name: _higher_gate(cases, dataset, baseline, path)
        for name, path in higher_paths.items()
    }
    validity = {
        str(case["seed"]): _nested(case, "metrics.validity.valid_row_rate")
        for case in _selected(cases, dataset, protocol.ARM_GAP)
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


def _checkpoint_summary(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> dict[str, Any]:
    selected = _selected(cases, dataset, arm)
    result = {}
    for round_index in protocol.CHECKPOINT_ROUNDS:
        rows = [
            checkpoint
            for case in selected
            for checkpoint in case["trajectory_l1"]["fixed_checkpoints"]
            if checkpoint["round"] == round_index
        ]
        values = [float(row["normalized_l1"]) for row in rows]
        result[str(round_index)] = {
            "observed_seed_count": len(values),
            "mean_normalized_l1": _mean(values) if values else None,
            "values": values,
            "selection_role": "diagnostic_only",
        }
    return result


def _dataset_decision(cases: Sequence[dict[str, Any]], dataset: str) -> dict[str, Any]:
    measured = {
        baseline: _measured_stability(cases, dataset, baseline)
        for baseline in protocol.BASELINE_ARMS
    }
    safety = {
        baseline: _safety(cases, dataset, baseline)
        for baseline in protocol.BASELINE_ARMS
    }
    measured_pass = all(item["stable_gap_gain"] for item in measured.values())
    safety_pass = all(item["pass"] for item in safety.values())
    if not measured_pass:
        classification = "no_stable_gap_gain"
    elif not safety_pass:
        classification = "gap_measured_gain_with_quality_risk"
    else:
        classification = "gap_quality_supported"
    return {
        "classification": classification,
        "measured_pass_against_both_baselines": measured_pass,
        "measured_by_baseline": measured,
        "safety_pass_against_both_baselines": safety_pass,
        "safety_by_baseline": safety,
    }


def _cross_dataset(decisions: dict[str, dict[str, Any]]) -> str:
    values = [decision["classification"] for decision in decisions.values()]
    if all(value == "gap_quality_supported" for value in values):
        return "shared_gap_kernel_support"
    if all(
        value in {"gap_quality_supported", "gap_measured_gain_with_quality_risk"}
        for value in values
    ):
        return "shared_measured_gain_with_quality_risk"
    if any(
        value in {"gap_quality_supported", "gap_measured_gain_with_quality_risk"}
        for value in values
    ):
        return "dataset_dependent_gap_response"
    return "no_shared_gap_support"


def _summary(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "terminal_absolute_count_error_sum": (
            "metrics.measured.overall.absolute_count_error_sum"
        ),
        "terminal_normalized_l1": "metrics.measured.overall.normalized_l1_mean",
        "terminal_gap_e": "metrics.measured.overall.gap_e",
        "terminal_squared_loss": (
            "metrics.measured.overall.squared_loss_diagnostic_only"
        ),
        "elapsed_sec": "cost.elapsed_sec",
        "gap_microsteps": "cost.gap_microsteps",
        "factorized_gibbs_microsteps": "cost.factorized_gibbs_microsteps",
    }
    return {
        dataset: {
            arm: {
                **{
                    name: _arm_summary(cases, dataset, arm, path)
                    for name, path in paths.items()
                },
                "checkpoint_normalized_l1": _checkpoint_summary(cases, dataset, arm),
            }
            for arm in joint.ARM_ORDER
        }
        for dataset in joint.DATASET_ORDER
    }


def _l1_csv_rows(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=L1_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _publish_evaluation(
    destination: Path,
    report: dict[str, Any],
    csv_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    report_path = destination / protocol.EVALUATION_REPORT
    csv_path = destination / protocol.L1_RESULTS_CSV
    if report_path.exists() or csv_path.exists():
        raise FileExistsError("第 6D 正式评价或 L1 输出已存在，不覆盖")
    temporary = Path(
        tempfile.mkdtemp(prefix=".stage6d-evaluation.tmp-", dir=destination)
    )
    try:
        temporary_csv = temporary / protocol.L1_RESULTS_CSV
        _write_csv(temporary_csv, csv_rows)
        report["l1_results_csv"] = {
            "path": protocol.L1_RESULTS_CSV,
            "sha256": protocol.file_sha256(temporary_csv),
            "row_count": len(csv_rows),
            "columns": list(L1_CSV_FIELDS),
        }
        temporary_report = temporary / protocol.EVALUATION_REPORT
        temporary_report.write_text(_strict_json_text(report), encoding="utf-8")
        try:
            os.replace(temporary_csv, csv_path)
            os.replace(temporary_report, report_path)
        except OSError:
            if csv_path.exists() and not report_path.exists():
                csv_path.unlink()
            raise
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return report_path, csv_path


def evaluate(confirmed_collection_report_sha256: str) -> tuple[Path, Path]:
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("第 6D 正式评价要求包含未跟踪文件在内的干净工作树")
    collection_report, indexed = _audit_collection(
        root, confirmed_collection_report_sha256
    )
    cases, identities = _evaluate_cases(root, indexed)
    decisions = {
        dataset: _dataset_decision(cases, dataset) for dataset in joint.DATASET_ORDER
    }
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_complete_collection",
        "evaluation_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_commit": collection_report["execution_commit"],
        "raw_reference_data_accessed": True,
        "query_and_reference_identity_audit": identities,
        "case_count": len(cases),
        "cases": cases,
        "summary": _summary(cases),
        "frozen_classification": {
            "by_dataset": decisions,
            "cross_dataset_response": _cross_dataset(decisions),
        },
        "l1_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "cross_dataset_or_query_group_weighted_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    return _publish_evaluation(root / protocol.OUTPUT_DIR, report, _l1_csv_rows(cases))


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
        print(_strict_json_text(build_plan()), end="")
        return
    report, l1_csv = evaluate(args.confirm_collection_sha)
    print(f"第 6D 正式 evaluation（评价） -> {report}")
    print(f"evaluation SHA-256 -> {protocol.file_sha256(report)}")
    print(f"L1 结果 -> {l1_csv}")
    print(f"L1 CSV SHA-256 -> {protocol.file_sha256(l1_csv)}")


if __name__ == "__main__":
    main()
