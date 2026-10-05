#!/usr/bin/env python3
"""独立复核 Issue #53 R8 单种子筛查评价与完整指标 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_issue53_fixed_alpha_calibration as offline_primitives
from scripts import freeze_issue53_test_query_workload_ab as blind_identity
from scripts import issue53_gap_weight_r8_screen_execution_protocol as protocol
from scripts import run_issue53_gap_weight_r8_screen as collection
from table_diffevo.stationarity import (
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)


AUDIT_VERSION = "issue53-gap-weight-r8-paired-screen-independent-audit-v1"
EXPECTED_EVALUATION_VERSION = (
    "issue53-gap-weight-r8-paired-screen-evaluation-v1"
)
CSV_FIELDS = (
    "dataset",
    "arm",
    "seed",
    "record_kind",
    "round",
    "state_index",
    "is_terminal",
    "metric_group",
    "metric_name",
    "query_count",
    "value",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"独立审计 JSON 根必须是对象：{path}")
    return value


def _safe_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("独立审计产物路径必须是字符串")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("独立审计产物路径越界")
    base = (root / protocol.OUTPUT_DIR).resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("独立审计产物路径逃逸")
    return resolved


def _require_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name}必须是完整小写 SHA-256")
    return value


def build_plan() -> dict[str, Any]:
    protocol_sha = protocol.assert_frozen_protocol_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "scientific_protocol_sha256": protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "collection_report": str(
            protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
        ),
        "evaluation_report": str(
            protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
        ),
        "screen_metrics_csv": str(
            protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV
        ),
        "audit_report": str(protocol.OUTPUT_DIR / protocol.AUDIT_REPORT),
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_target_bins_and_both_proxies": True,
        "recompute_screen_metrics_csv": True,
        "recompute_frozen_classification": True,
        "imports_r8_evaluator_arithmetic": False,
        "rerun_generation": False,
    }


def _audit_collection_independently(
    root: Path, confirmed_sha: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    _require_sha(confirmed_sha, "collection SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("独立审计 collection SHA-256 不一致")
    report = _load_json(path)
    required = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "scientific_protocol_sha256": protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "case_count": 4,
        "paired_dataset_seed_count": 2,
        "formal_result_valid": True,
        "screen_collection_valid": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "quality_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("独立审计 collection 顶层契约漂移")
    if (
        report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
        or report.get("generator_params_manifest_sha256")
        != protocol.generator_params_manifest_sha256()
        or report.get("shard_assignment_sha256")
        != protocol.shard_assignment_sha256()
    ):
        raise RuntimeError("独立审计 collection 协议绑定漂移")
    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("独立审计 collection 执行提交无效")
    current = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(root, "merge-base", execution_commit, current)
        != execution_commit
    ):
        raise RuntimeError("独立审计 collection 提交祖先关系漂移")

    shard_id = protocol.LOCAL_SHARD
    shard_relative = str(Path("shards") / shard_id / protocol.SHARD_REPORT)
    artifacts = report.get("shard_report_artifacts")
    hashes = report.get("shard_report_sha256")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(hashes, dict)
        or set(artifacts) != {shard_id}
        or set(hashes) != {shard_id}
        or artifacts[shard_id]
        != {"path": shard_relative, "sha256": hashes[shard_id]}
    ):
        raise RuntimeError("独立审计分片报告绑定漂移")
    shard_path = _safe_artifact(root, shard_relative)
    if protocol.file_sha256(shard_path) != hashes[shard_id]:
        raise RuntimeError("独立审计分片报告哈希漂移")
    shard = _load_json(shard_path)
    if (
        shard.get("contract_version") != protocol.PROTOCOL_VERSION
        or shard.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or shard.get("shard_id") != shard_id
        or shard.get("execution_commit") != execution_commit
        or shard.get("task_ids")
        != [task.task_id for task in protocol.tasks_for_shard(shard_id)]
        or shard.get("formal_shard_complete") is not True
        or shard.get("partial_shard_comparison_emitted") is not False
    ):
        raise RuntimeError("独立审计分片内容漂移")
    samples = shard.get("gpu_samples")
    if (
        not isinstance(samples, list)
        or not samples
        or any(
            sample.get("physical_index")
            != protocol.EXPECTED_GPU["physical_index"]
            for sample in samples
        )
    ):
        raise RuntimeError("独立审计 GPU 样本漂移")

    tasks = protocol.task_plan().tasks
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("独立审计 collection 四条 case 不完整")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    with collection._execution_runtime():
        for task, row in zip(tasks, rows, strict=True):
            if not isinstance(row, dict) or row.get("task_id") != task.task_id:
                raise RuntimeError("独立审计 collection 行顺序漂移")
            collection._validate_case_row(task, row)
            if (
                row.get("gap_8k_identity") is not True
                or row.get("gap_weighting_guard_passed") is not True
                or row.get("gap_clip_hit_count") != 0
                or row.get("factorized_gibbs_conditional_logit_clipped_count")
                != 0
                or row.get("direction_logit_clipped_count") != 0
                or row.get("nonfinite_count") != 0
            ):
                raise RuntimeError(f"独立审计数值护栏漂移：{task.task_id}")
            terminal = row.get("terminal_table_path")
            if not isinstance(terminal, str):
                raise TypeError(f"独立审计终表路径非法：{task.task_id}")
            manifest = _load_json(
                _safe_artifact(
                    root,
                    str(Path(terminal).parent / collection.CASE_MANIFEST),
                )
            )
            if (
                manifest.get("contract_version") != protocol.PROTOCOL_VERSION
                or manifest.get("protocol_sha256")
                != protocol.FROZEN_PROTOCOL_SHA256
                or manifest.get("execution_commit") != execution_commit
                or manifest.get("generator_params")
                != protocol.generator_params_manifest(
                    task.dataset, task.arm, task.seed
                )
                or manifest.get("collection_row") != row
                or manifest.get("raw_reference_data_accessed") is not False
                or manifest.get("method_comparison_emitted") is not False
            ):
                raise RuntimeError(f"独立审计 case 清单漂移：{task.task_id}")
            for path_key, hash_key in (
                ("terminal_table_path", "terminal_table_sha256"),
                ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
                ("transition_audit_path", "transition_audit_sha256"),
            ):
                if (
                    protocol.file_sha256(_safe_artifact(root, row[path_key]))
                    != row[hash_key]
                ):
                    raise RuntimeError(
                        f"独立审计产物哈希漂移：{task.task_id}/{path_key}"
                    )
            transition = _load_json(
                _safe_artifact(root, row["transition_audit_path"])
            )
            gap_rows = transition.get("gap_rounds")
            if (
                transition.get("task_id") != task.task_id
                or transition.get("round_count") != row["applied_rounds"]
                or transition.get("all_applied_unconditionally") is not True
                or not isinstance(gap_rows, list)
                or len(gap_rows) != row["applied_rounds"]
                or sum(item["microsteps"] for item in gap_rows)
                != sum(item["expected_microsteps"] for item in gap_rows)
                or sum(item["microsteps"] for item in gap_rows)
                != row["gap_microsteps"]
            ):
                raise RuntimeError(f"独立审计 8*K 失败：{task.task_id}")
            expected_weighting = (
                protocol.LEGACY_WEIGHTING
                if task.arm == "gap_legacy_s8"
                else protocol.BOUNDED_WEIGHTING
            )
            applied_scans = [item for item in gap_rows if item["scan_applied"]]
            if (
                not applied_scans
                or any(
                    item.get("expected_weighting") != expected_weighting
                    or item.get("observed_weighting") != expected_weighting
                    for item in applied_scans
                )
            ):
                raise RuntimeError(f"独立审计权重模式失败：{task.task_id}")
            if expected_weighting == protocol.BOUNDED_WEIGHTING and any(
                item.get("observed_max_weight_ratio")
                != protocol.MAX_WEIGHT_RATIO
                or item.get("observed_smoothing_count")
                != protocol.smoothing_count(task.dataset)
                or not 1.0
                <= float(item.get("observed_actual_weight_ratio"))
                <= protocol.MAX_WEIGHT_RATIO * (1.0 + 1e-12)
                for item in applied_scans
            ):
                raise RuntimeError(
                    f"独立审计 R8 权重证据失败：{task.task_id}"
                )
            key = (task.seed, task.dataset, task.arm)
            if key in indexed:
                raise RuntimeError("独立审计 collection case 重复")
            indexed[key] = row
    expected = {(task.seed, task.dataset, task.arm) for task in tasks}
    if set(indexed) != expected:
        raise RuntimeError("独立审计 collection 地址不完整")
    if shard.get("raw_results") != [
        indexed[(task.seed, task.dataset, task.arm)] for task in tasks
    ]:
        raise RuntimeError("独立审计分片与合并行不一致")
    return report, indexed


def _load_inputs_and_references(
    root: Path, runtime: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    # 仅复用结果前冻结的公共离线查询生成/表评价原语；
    # 不导入 R8 evaluator。
    test_groups, test_identity = offline_primitives._freeze_test_groups(root)
    nltcs_groups, nltcs_identity = offline_primitives._freeze_nltcs_groups(root)
    inputs: dict[str, Any] = {}
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        for key in ("schema", "queries", "marginals"):
            if protocol.file_sha256(root / spec[key]) != spec["input_sha256"][key]:
                raise RuntimeError(f"独立审计 {dataset}.{key} 输入哈希漂移")
        raw = _load_json(root / spec["queries"]).get("queries")
        if not isinstance(raw, list):
            raise TypeError(f"独立审计 {dataset} 原始查询缺失")
        queries = runtime.load_queries(str(root / spec["queries"]))
        raw_targets = [row.get("result") for row in raw]
        targets = np.asarray(raw_targets, dtype=float)
        loaded_targets = np.asarray(
            [query["result"] for query in queries], dtype=float
        )
        if (
            len(raw) != spec["query_count"]
            or len(queries) != len(raw)
            or not np.array_equal(targets, loaded_targets)
            or blind_identity.query_set_identity(raw)
            != spec["query_identity_sha256"]
            or protocol.canonical_sha256(raw_targets)
            != spec["target_vector_sha256"]
            or ordered_query_identity_sha256(queries)
            != spec["trace_query_identity_sha256"]
            or target_answer_identity_sha256(loaded_targets)
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"独立审计 {dataset} 查询身份漂移")
        member_indices: list[int] = []
        for item in protocol.TARGET_COUNT_BINS[dataset]:
            members = [
                {
                    "index": index,
                    "id": row.get("id"),
                    "target": int(row["result"]),
                }
                for index, row in enumerate(raw)
                if item["lower_inclusive"]
                <= int(row["result"])
                <= item["upper_inclusive"]
            ]
            if (
                len(members) != item["query_count"]
                or protocol.canonical_sha256(members)
                != item["membership_sha256"]
            ):
                raise RuntimeError(
                    f"独立审计 {dataset}/{item['name']} 分箱身份漂移"
                )
            member_indices.extend(member["index"] for member in members)
        if sorted(member_indices) != list(range(len(raw))) or len(
            member_indices
        ) != len(set(member_indices)):
            raise RuntimeError(f"独立审计 {dataset} 分箱不完整或重叠")
        inputs[dataset] = {
            "schema": runtime.load_schema(str(root / spec["schema"])),
            "queries": queries,
            "raw_queries": raw,
            "targets": loaded_targets,
        }
    references, reference_sha = offline_primitives._load_references(root, runtime)
    if reference_sha != {
        name: spec["input_sha256"]["reference"]
        for name, spec in protocol.DATASETS.items()
    }:
        raise RuntimeError("独立审计参考表哈希漂移")
    nltcs_marginals = _load_json(
        root / protocol.DATASETS["nltcs"]["marginals"]
    )
    material = {
        "inputs": inputs,
        "references": references,
        "test_groups": test_groups,
        "test_group_targets": {
            name: runtime.evaluate_table(references["test_300x10"], queries)
            for name, queries in test_groups.items()
        },
        "nltcs_groups": nltcs_groups,
        "nltcs_marginals": nltcs_marginals,
        "nltcs_domains": runtime.offline._discretization_domains(
            nltcs_marginals
        ),
        "nltcs_measured_triples": runtime.offline._measured_cell_keys(
            inputs["nltcs"]["queries"], nltcs_marginals, order=3
        ),
    }
    identities = {
        "query_identity_frozen_before_reference_load": True,
        "test": test_identity,
        "nltcs": nltcs_identity,
        "reference_sha256": reference_sha,
    }
    return material, identities


def _independent_vector_metrics(
    dataset: str,
    raw_queries: Sequence[Mapping[str, Any]],
    targets: Sequence[Any],
    answers: Sequence[Any],
    n_records: int,
) -> dict[str, Any]:
    target_values = np.asarray(targets, dtype=float)
    answer_values = np.asarray(answers, dtype=float)
    if (
        target_values.ndim != 1
        or answer_values.shape != target_values.shape
        or target_values.size == 0
        or not np.all(np.isfinite(target_values))
        or not np.all(np.isfinite(answer_values))
        or not np.all(target_values == np.rint(target_values))
        or not np.all(answer_values == np.rint(answer_values))
        or np.any(target_values < 0)
        or np.any(target_values > n_records)
        or np.any(answer_values < 0)
        or np.any(answer_values > n_records)
    ):
        raise RuntimeError("独立审计查询向量非法")
    targets_int = [int(value) for value in target_values]
    answers_int = [int(value) for value in answer_values]
    errors = [
        abs(target - answer)
        for target, answer in zip(targets_int, answers_int, strict=True)
    ]
    count_sum = sum(errors)
    smoothing = float(
        max(8.0, np.float64(n_records) / np.float64(7.0))
    )
    legacy_proxy = math.fsum(
        error / max(target, 8)
        for error, target in zip(errors, targets_int, strict=True)
    ) / len(errors)
    bounded_proxy = math.fsum(
        error / (min(max(target, 0), n_records) + smoothing)
        for error, target in zip(errors, targets_int, strict=True)
    ) / len(errors)
    bins: dict[str, Any] = {}
    for item in protocol.TARGET_COUNT_BINS[dataset]:
        indices = [
            index
            for index, target in enumerate(targets_int)
            if item["lower_inclusive"] <= target <= item["upper_inclusive"]
        ]
        subtotal = sum(errors[index] for index in indices)
        bins[item["name"]] = {
            "query_count": len(indices),
            "absolute_count_error_sum": subtotal,
            "absolute_count_error_mean": float(subtotal / len(indices)),
            "normalized_l1": float(subtotal / (len(indices) * n_records)),
            "membership_sha256": item["membership_sha256"],
        }
    orders: dict[str, Any] = {}
    for order in sorted({len(query["conditions"]) for query in raw_queries}):
        indices = [
            index
            for index, query in enumerate(raw_queries)
            if len(query["conditions"]) == order
        ]
        subtotal = sum(errors[index] for index in indices)
        orders[str(order)] = {
            "query_count": len(indices),
            "absolute_count_error_sum": subtotal,
            "absolute_count_error_mean": float(subtotal / len(indices)),
            "normalized_l1": float(subtotal / (len(indices) * n_records)),
        }
    return {
        "query_count": len(errors),
        "absolute_count_error_sum": count_sum,
        "normalized_l1": float(count_sum / (len(errors) * n_records)),
        "legacy_relative_gap_proxy": float(legacy_proxy),
        "bounded_r8_gap_proxy": float(bounded_proxy),
        "bounded_r8_smoothing_count": smoothing,
        "target_count_bins": bins,
        "query_orders": orders,
    }


def _checkpoint_independent(
    root: Path,
    source: dict[str, Any],
    dataset_input: dict[str, Any],
    terminal_answers: np.ndarray,
) -> dict[str, Any]:
    artifact = _load_json(
        _safe_artifact(root, source["checkpoint_artifact_path"])
    )
    dataset = source["dataset"]
    spec = protocol.DATASETS[dataset]
    required = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "task_id": source["task_id"],
        "dataset": dataset,
        "arm": source["arm"],
        "seed": source["seed"],
        "n_records": spec["n_records"],
        "query_count": spec["query_count"],
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "fixed_checkpoint_rounds_requested": list(protocol.CHECKPOINT_ROUNDS),
        "historical_best_included": False,
    }
    if any(artifact.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"独立审计检查点身份漂移：{source['task_id']}")

    def recompute(row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        answers = np.asarray(row.get("query_answers"), dtype=float)
        vector = _independent_vector_metrics(
            dataset,
            dataset_input["raw_queries"],
            dataset_input["targets"],
            answers,
            spec["n_records"],
        )
        squared = float(
            sum(
                (int(target) - int(answer)) ** 2
                for target, answer in zip(
                    dataset_input["targets"], answers, strict=True
                )
            )
            / 2
        )
        if (
            row.get("absolute_count_error_sum")
            != vector["absolute_count_error_sum"]
            or row.get("normalized_l1") != vector["normalized_l1"]
            or row.get("gap_e") != vector["legacy_relative_gap_proxy"]
            or row.get("squared_loss") != squared
        ):
            identity = f"{source['task_id']}/{row.get('round')}"
            raise RuntimeError(f"独立审计检查点算术漂移：{identity}")
        return (
            {
                "kind": row["kind"],
                "state_index": int(row["state_index"]),
                "round": int(row["round"]),
                "phase": row["phase"],
                "squared_loss": squared,
                **vector,
            },
            answers,
        )

    fixed_source = artifact.get("fixed_checkpoints")
    if not isinstance(fixed_source, list):
        raise TypeError("独立审计固定检查点缺失")
    fixed = [recompute(row)[0] for row in fixed_source]
    expected_rounds = [
        value
        for value in protocol.CHECKPOINT_ROUNDS
        if value <= source["applied_rounds"]
    ]
    if [row["round"] for row in fixed] != expected_rounds:
        raise RuntimeError(f"独立审计检查点覆盖漂移：{source['task_id']}")
    terminal_source = artifact.get("terminal")
    if not isinstance(terminal_source, dict):
        raise TypeError("独立审计终点检查点缺失")
    terminal, terminal_vector = recompute(terminal_source)
    if (
        terminal["round"] != source["applied_rounds"]
        or not np.array_equal(terminal_vector, terminal_answers)
    ):
        raise RuntimeError(f"独立审计终点身份漂移：{source['task_id']}")
    return {"fixed_checkpoints": fixed, "terminal": terminal}


def _recompute_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = offline_primitives._load_runtime()
    material, identities = _load_inputs_and_references(root, runtime)
    inputs = material["inputs"]
    cases: list[dict[str, Any]] = []
    for task in protocol.task_plan().tasks:
        source = indexed[(task.seed, task.dataset, task.arm)]
        table = runtime.pd.read_csv(
            _safe_artifact(root, source["terminal_table_path"])
        )
        spec = protocol.DATASETS[task.dataset]
        if len(table) != spec["n_records"]:
            raise RuntimeError(f"独立审计终表行数漂移：{task.task_id}")
        dataset_input = inputs[task.dataset]
        terminal_answers = np.asarray(
            runtime.evaluate_table(table, dataset_input["queries"]), dtype=float
        )
        vector = _independent_vector_metrics(
            task.dataset,
            dataset_input["raw_queries"],
            dataset_input["targets"],
            terminal_answers,
            spec["n_records"],
        )
        trajectory = _checkpoint_independent(
            root, source, dataset_input, terminal_answers
        )
        if any(
            trajectory["terminal"][key] != vector[key] for key in vector
        ):
            raise RuntimeError(
                f"独立审计终表/终点指标漂移：{task.task_id}"
            )
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
        overall = metrics["measured"]["overall"]
        if not math.isclose(
            float(overall["normalized_l1_mean"]),
            vector["normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"独立审计 measured L1 漂移：{task.task_id}")
        overall.update(
            {
                "absolute_count_error_sum": vector[
                    "absolute_count_error_sum"
                ],
                "normalized_l1_mean": vector["normalized_l1"],
            }
        )
        for order, exact in vector["query_orders"].items():
            observed = metrics["measured"]["by_order"][order]
            if not math.isclose(
                float(observed["normalized_l1_mean"]),
                exact["normalized_l1"],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    f"独立审计查询阶数指标漂移：{task.task_id}/{order}"
                )
            observed.update(
                {
                    "absolute_count_error_sum": exact[
                        "absolute_count_error_sum"
                    ],
                    "absolute_count_error_mean": exact[
                        "absolute_count_error_mean"
                    ],
                    "normalized_l1_mean": exact["normalized_l1"],
                }
            )
        metrics["measured"]["proxy_objectives"] = {
            "legacy_relative_gap": vector["legacy_relative_gap_proxy"],
            "bounded_r8_gap": vector["bounded_r8_gap_proxy"],
            "bounded_r8_smoothing_count": vector[
                "bounded_r8_smoothing_count"
            ],
        }
        metrics["measured"]["target_count_bins"] = vector[
            "target_count_bins"
        ]
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
                "trajectory": trajectory,
                "kernel_audit": {
                    key: source[key]
                    for key in (
                        "gap_weighting",
                        "gap_max_weight_ratio",
                        "gap_smoothing_count",
                        "gap_weighting_scan_round_count",
                        "gap_actual_weight_ratio_min",
                        "gap_actual_weight_ratio_max",
                        "gap_weighting_guard_passed",
                    )
                },
                "cost": {
                    key: source[key]
                    for key in (
                        "elapsed_sec",
                        "average_sec_per_applied_round",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "state_evaluation_count",
                        "candidate_evaluation_count",
                        "distance_evaluation_count",
                        "direction_evaluation_count",
                        "gap_microsteps",
                    )
                },
                "inner_early_stopping": source["inner_early_stopping"],
            }
        )
    return cases, identities


def _nested(record: Mapping[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"独立审计指标 {path} 非数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"独立审计指标 {path} 非有限")
    return result


def _one_case(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case["dataset"] == dataset and case["arm"] == arm
    ]
    if len(selected) != 1 or selected[0]["seed"] != protocol.DEVELOPMENT_SEED:
        raise RuntimeError(f"独立审计 {dataset}/{arm} case 不唯一")
    return selected[0]


def _independent_ratio(candidate: float, legacy: float) -> dict[str, Any]:
    if legacy == 0.0:
        if candidate == 0.0:
            ratio: float | None = 1.0
            state = "both_zero_ratio_one"
        else:
            ratio = None
            state = "positive_infinity"
    else:
        ratio = float(candidate / legacy)
        state = "finite"
    return {
        "bounded_r8": float(candidate),
        "legacy": float(legacy),
        "bounded_over_legacy": ratio,
        "ratio_state": state,
        "lower_is_better": True,
    }


def _ratio_value(record: Mapping[str, Any]) -> float:
    return (
        float("inf")
        if record["ratio_state"] == "positive_infinity"
        else float(record["bounded_over_legacy"])
    )


def _independent_comparison(
    cases: Sequence[dict[str, Any]], dataset: str, path: str
) -> dict[str, Any]:
    bounded = _nested(_one_case(cases, dataset, "gap_bounded_r8_s8"), path)
    legacy = _nested(_one_case(cases, dataset, "gap_legacy_s8"), path)
    result = _independent_ratio(bounded, legacy)
    result["metric"] = path
    return result


def _summary_independent(
    cases: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], bool, str]:
    summary: dict[str, Any] = {}
    for dataset in protocol.DATASET_ORDER:
        measured = _independent_comparison(
            cases, dataset, "metrics.measured.overall.normalized_l1_mean"
        )
        proxies = {
            name: _independent_comparison(
                cases,
                dataset,
                f"metrics.measured.proxy_objectives.{name}",
            )
            for name in ("legacy_relative_gap", "bounded_r8_gap")
        }
        bins = {
            item["name"]: _independent_comparison(
                cases,
                dataset,
                "metrics.measured.target_count_bins."
                f"{item['name']}.absolute_count_error_mean",
            )
            for item in protocol.TARGET_COUNT_BINS[dataset]
        }
        orders = {
            str(order): _independent_comparison(
                cases,
                dataset,
                "metrics.measured.by_order."
                f"{order}.absolute_count_error_mean",
            )
            for order in protocol.DATASETS[dataset]["order_counts"]
        }
        group_names = (
            protocol.TEST_GROUP_ORDER
            if dataset == "test_300x10"
            else tuple(protocol.NLTCS_GROUP_COUNTS)
        )
        groups = {
            name: _independent_comparison(
                cases,
                dataset,
                f"metrics.offline_query_groups.{name}.normalized_l1_mean",
            )
            for name in group_names
        }
        curves: dict[str, Any] = {}
        for arm in protocol.ARM_ORDER:
            case = _one_case(cases, dataset, arm)
            by_round = {
                int(row["round"]): row
                for row in case["trajectory"]["fixed_checkpoints"]
            }
            curves[arm] = {
                str(round_index): (
                    {
                        "normalized_l1": by_round[round_index][
                            "normalized_l1"
                        ],
                        "legacy_relative_gap_proxy": by_round[round_index][
                            "legacy_relative_gap_proxy"
                        ],
                        "bounded_r8_gap_proxy": by_round[round_index][
                            "bounded_r8_gap_proxy"
                        ],
                        "selection_role": "diagnostic_only",
                    }
                    if round_index in by_round
                    else None
                )
                for round_index in protocol.CHECKPOINT_ROUNDS
            }
        summary[dataset] = {
            "measured_normalized_l1": measured,
            "proxy_objectives": proxies,
            "target_count_bins": bins,
            "query_orders": orders,
            "offline_query_groups": groups,
            "checkpoint_curves": curves,
            "terminal_cost_by_arm": {
                arm: _one_case(cases, dataset, arm)["cost"]
                for arm in protocol.ARM_ORDER
            },
        }
    valid = all(
        _nested(case, "metrics.validity.valid_row_rate") == 1.0
        and case["kernel_audit"]["gap_weighting_guard_passed"] is True
        for case in cases
    )
    nltcs = summary["nltcs"]
    test = summary["test_300x10"]
    # 独立展开冻结优先级，不调用 scientific.classify_screen。
    if not valid:
        classification = "execution_invalid"
    elif (
        _ratio_value(nltcs["measured_normalized_l1"]) >= 1.0
        or _ratio_value(
            nltcs["target_count_bins"][protocol.NLTCS_COMMON_BIN]
        )
        >= 1.0
    ):
        classification = "bounded_weighting_mechanism_not_supported"
    elif (
        _ratio_value(nltcs["target_count_bins"][protocol.NLTCS_RARE_BIN])
        > protocol.NLTCS_RARE_RATIO_MAX
    ):
        classification = "common_gain_with_rare_query_regression"
    elif (
        _ratio_value(test["measured_normalized_l1"])
        > protocol.TEST_OVERALL_RATIO_MAX
        or any(
            _ratio_value(summary[dataset]["offline_query_groups"]["one_way_safety"])
            > protocol.ONE_WAY_RATIO_MAX
            for dataset in protocol.DATASET_ORDER
        )
    ):
        classification = "measured_gain_with_safety_risk"
    else:
        classification = "advance_to_five_seed_confirmation"
    return summary, bool(valid), classification


def _csv_rows_independent(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        terminal_round = case["trajectory"]["terminal"]["round"]
        states = list(case["trajectory"]["fixed_checkpoints"])
        if all(state["round"] != terminal_round for state in states):
            states.append(case["trajectory"]["terminal"])
        for state in states:
            identity = {
                "dataset": case["dataset"],
                "arm": case["arm"],
                "seed": case["seed"],
                "record_kind": state["kind"],
                "round": state["round"],
                "state_index": state["state_index"],
                "is_terminal": state["round"] == terminal_round,
            }

            def add(group: str, name: str, count: int, value: float) -> None:
                rows.append(
                    {
                        **identity,
                        "metric_group": group,
                        "metric_name": name,
                        "query_count": count,
                        "value": value,
                    }
                )

            add(
                "measured",
                "normalized_l1",
                state["query_count"],
                state["normalized_l1"],
            )
            add(
                "proxy",
                "legacy_relative_gap",
                state["query_count"],
                state["legacy_relative_gap_proxy"],
            )
            add(
                "proxy",
                "bounded_r8_gap",
                state["query_count"],
                state["bounded_r8_gap_proxy"],
            )
            for name, metric in state["target_count_bins"].items():
                add(
                    "target_count_bin",
                    name,
                    metric["query_count"],
                    metric["absolute_count_error_mean"],
                )
            for name, metric in state["query_orders"].items():
                add(
                    "query_order",
                    name,
                    metric["query_count"],
                    metric["absolute_count_error_mean"],
                )
    return rows


def _assert_equal(left: Any, right: Any, path: str = "root") -> None:
    """结构严格、有限浮点近机器精度比较。"""

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"独立审计字段集合漂移：{path}")
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"独立审计列表长度漂移：{path}")
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_equal(left_item, right_item, f"{path}[{index}]")
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"独立审计数值漂移：{path}")
        return
    if left != right:
        raise RuntimeError(f"独立审计值漂移：{path}")


def _audit_csv(
    root: Path, report: dict[str, Any], expected_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    binding = report.get("screen_metrics_csv")
    if (
        not isinstance(binding, dict)
        or binding.get("path") != protocol.L1_RESULTS_CSV
        or binding.get("row_count") != len(expected_rows)
        or binding.get("columns") != list(CSV_FIELDS)
    ):
        raise RuntimeError("独立审计 CSV 绑定漂移")
    path = _safe_artifact(root, protocol.L1_RESULTS_CSV)
    if protocol.file_sha256(path) != binding.get("sha256"):
        raise RuntimeError("独立审计 CSV 哈希漂移")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        observed = list(reader)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError("独立审计 CSV 列漂移")
    expected_strings = [
        {field: str(row[field]) for field in CSV_FIELDS} for row in expected_rows
    ]
    if observed != expected_strings:
        raise RuntimeError("独立审计 CSV 行或算术漂移")
    return {
        "path": protocol.L1_RESULTS_CSV,
        "sha256": binding["sha256"],
        "row_count": len(observed),
        "columns_exact": True,
        "rows_exact": True,
    }


def _audit_evaluation(
    root: Path,
    confirmed_sha: str,
    collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha(confirmed_sha, "evaluation SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("独立审计 evaluation SHA-256 不一致")
    report = _load_json(path)
    if (
        report.get("contract_version") != EXPECTED_EVALUATION_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("scientific_protocol_sha256")
        != protocol.SCIENTIFIC_PROTOCOL_SHA256
        or report.get("collection_report_sha256") != collection_sha
        or report.get("collection_execution_commit")
        != collection_report["execution_commit"]
        or report.get("case_count") != 4
        or report.get("raw_reference_data_accessed") is not True
        or report.get("all_four_cases_evaluated") is not True
        or report.get("terminal_and_reached_checkpoints_fully_emitted")
        is not True
        or report.get("historical_best_used") is not False
        or report.get("checkpoint_selected_as_output") is not False
        or report.get("new_generation_performed_by_evaluator") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
        or report.get("automatic_followup_authorized") is not False
    ):
        raise RuntimeError("独立审计 evaluation 顶层契约漂移")
    evaluation_commit = report.get("evaluation_commit")
    if (
        not isinstance(evaluation_commit, str)
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
    ):
        raise RuntimeError("独立审计 evaluation 提交无效")
    current = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(root, "merge-base", evaluation_commit, current)
        != evaluation_commit
        or collection._git_text(
            root,
            "merge-base",
            collection_report["execution_commit"],
            evaluation_commit,
        )
        != collection_report["execution_commit"]
    ):
        raise RuntimeError("独立审计 collection/evaluation 祖先关系漂移")
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("独立审计查询/参考身份漂移")
    _assert_equal(report.get("cases"), cases, "cases")
    summary, execution_valid, classification = _summary_independent(cases)
    _assert_equal(report.get("summary"), summary, "summary")
    if (
        report.get("execution_valid") is not execution_valid
        or report.get("quality_interpretation_allowed") is not execution_valid
        or report.get("frozen_classification") != classification
        or report.get("classification_precedence")
        != protocol.scientific.frozen_protocol_manifest()["screen_decision"][
            "precedence"
        ]
    ):
        raise RuntimeError("独立审计冻结分类漂移")
    csv_audit = _audit_csv(root, report, _csv_rows_independent(cases))
    return report, {
        "evaluation_report_sha256": confirmed_sha,
        "all_4_case_metrics_recomputed": True,
        "all_terminal_and_checkpoint_metrics_recomputed": True,
        "both_proxy_objectives_recomputed": True,
        "all_target_count_bins_recomputed": True,
        "all_offline_groups_recomputed": True,
        "classification_independently_reproduced": True,
        "screen_metrics_csv": csv_audit,
        "pass": True,
    }


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    if collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "R8 独立审计要求包含未跟踪文件在内的干净工作树"
        )
    collection_report, indexed = _audit_collection_independently(
        root, confirmed_collection_report_sha256
    )
    cases, identities = _recompute_cases(root, indexed)
    _evaluation_report, evidence = _audit_evaluation(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report,
        cases,
        identities,
    )
    destination = root / protocol.OUTPUT_DIR / protocol.AUDIT_REPORT
    if destination.exists():
        raise FileExistsError("R8 独立审计报告已存在，不覆盖")
    payload = {
        **build_plan(),
        "mode": "independent_recomputation_after_complete_evaluation",
        "audit_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        **evidence,
        "evaluator_arithmetic_imported": False,
        "generation_rerun": False,
        "evaluation_artifacts_modified": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".r8-screen-audit.tmp-", dir=root / protocol.OUTPUT_DIR
        )
    )
    try:
        temporary_path = temporary / protocol.AUDIT_REPORT
        temporary_path.write_text(
            collection._strict_json_text(payload), encoding="utf-8"
        )
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return destination


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
    path = audit(
        args.confirm_collection_sha, args.confirm_evaluation_sha
    )
    print(f"R8 screen independent audit -> {path}")
    print(f"audit SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
