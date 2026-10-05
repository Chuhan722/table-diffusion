#!/usr/bin/env python3
"""离线评价 Issue #53 R8 单种子两臂完整筛查 collection。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_issue53_fixed_alpha_calibration as offline_helpers
from scripts import freeze_issue53_test_query_workload_ab as blind_identity
from scripts import issue53_gap_weight_r8_screen_execution_protocol as protocol
from scripts import run_issue53_gap_weight_r8_screen as collection
from table_diffevo.stationarity import (
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)


EVALUATION_VERSION = "issue53-gap-weight-r8-paired-screen-evaluation-v1"
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
    return collection._load_json(path)


def _strict_json_text(value: Any) -> str:
    return collection._strict_json_text(value)


def _artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("R8 筛查产物相对路径必须是字符串")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("R8 筛查产物路径越界")
    output_root = (root / protocol.OUTPUT_DIR).resolve()
    resolved = (output_root / path).resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError("R8 筛查产物路径逃逸输出目录")
    return resolved


def _expected_collection_audit() -> dict[str, bool]:
    return {
        "all_4_cases_present": True,
        "both_dataset_seed_pairs_complete": True,
        "single_rtx4090_shard_assignment": True,
        "both_pairs_single_shard": True,
        "complete_shard_verified_before_merge": True,
        "all_existing_p6_autostop_enabled": True,
        "all_normal_abc_termination": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "both_arms_gap_8k_identity": True,
        "all_weighting_diagnostics_valid": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_physical_index_1": True,
    }


def _audit_collection(
    root: Path, confirmed_sha256: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    protocol.require_collection_confirmation(confirmed_sha256)
    report_path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(report_path) != confirmed_sha256:
        raise ValueError("R8 collection 报告 SHA-256 与确认值不一致")
    report = _load_json(report_path)
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("scientific_protocol_sha256")
        != protocol.SCIENTIFIC_PROTOCOL_SHA256
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
        or report.get("generator_params_manifest_sha256")
        != protocol.generator_params_manifest_sha256()
        or report.get("shard_assignment_sha256")
        != protocol.shard_assignment_sha256()
        or report.get("case_count") != 4
        or report.get("paired_dataset_seed_count") != 2
        or report.get("collection_audit") != _expected_collection_audit()
        or report.get("formal_result_valid") is not True
        or report.get("screen_collection_valid") is not True
        or report.get("raw_reference_data_accessed") is not False
        or report.get("partial_matrix_comparison_emitted") is not False
        or report.get("method_ranking_emitted") is not False
        or report.get("quality_results_published_by_collection") is not False
        or report.get("parameter_retuning_performed") is not False
    ):
        raise RuntimeError("R8 collection 协议、完整性或信息边界漂移")

    expected_inputs = {
        dataset: {
            key: spec["input_sha256"][key]
            for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in protocol.DATASETS.items()
    }
    if report.get("generation_input_sha256") != expected_inputs:
        raise RuntimeError("R8 collection 生成输入身份漂移")
    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("R8 collection 执行提交缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        collection._git_text(
            root, "merge-base", execution_commit, current_commit
        )
        != execution_commit
    ):
        raise RuntimeError("R8 collection 提交不是评价提交的祖先")

    shard_id = protocol.LOCAL_SHARD
    environments = report.get("environment", {}).get("shards")
    if not isinstance(environments, dict) or set(environments) != {shard_id}:
        raise RuntimeError("R8 collection 单分片环境缺失")
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
        raise RuntimeError("R8 collection GPU/软件环境漂移")

    shard_artifacts = report.get("shard_report_artifacts")
    shard_hashes = report.get("shard_report_sha256")
    expected_relative = str(Path("shards") / shard_id / protocol.SHARD_REPORT)
    if (
        not isinstance(shard_artifacts, dict)
        or not isinstance(shard_hashes, dict)
        or set(shard_artifacts) != {shard_id}
        or set(shard_hashes) != {shard_id}
    ):
        raise RuntimeError("R8 collection 分片证据缺失")
    shard_binding = shard_artifacts[shard_id]
    if (
        not isinstance(shard_binding, dict)
        or shard_binding.get("path") != expected_relative
        or shard_binding.get("sha256") != shard_hashes[shard_id]
    ):
        raise RuntimeError("R8 collection 分片报告绑定漂移")
    shard_path = _artifact_path(root, expected_relative)
    if protocol.file_sha256(shard_path) != shard_hashes[shard_id]:
        raise RuntimeError("R8 collection 分片报告哈希漂移")
    shard_report = _load_json(shard_path)
    if (
        shard_report.get("contract_version") != protocol.PROTOCOL_VERSION
        or shard_report.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or shard_report.get("shard_id") != shard_id
        or shard_report.get("execution_commit") != execution_commit
        or shard_report.get("task_ids")
        != [task.task_id for task in protocol.tasks_for_shard(shard_id)]
        or shard_report.get("formal_shard_complete") is not True
        or shard_report.get("partial_shard_comparison_emitted") is not False
    ):
        raise RuntimeError("R8 collection 分片报告内容漂移")
    samples = shard_report.get("gpu_samples")
    if (
        not isinstance(samples, list)
        or not samples
        or any(
            sample.get("physical_index")
            != protocol.EXPECTED_GPU["physical_index"]
            for sample in samples
        )
    ):
        raise RuntimeError("R8 collection GPU 监控样本漂移")

    rows = report.get("raw_results")
    tasks = protocol.task_plan().tasks
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("R8 collection 不是完整四条")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    with collection._execution_runtime():
        for task, row in zip(tasks, rows, strict=True):
            if not isinstance(row, dict) or row.get("task_id") != task.task_id:
                raise RuntimeError("R8 collection 冻结任务顺序漂移")
            collection._validate_case_row(task, row)
            spec = protocol.DATASETS[task.dataset]
            if (
                row.get("query_identity_sha256")
                != spec["query_identity_sha256"]
                or row.get("target_vector_sha256")
                != spec["target_vector_sha256"]
                or row.get("trace_query_identity_sha256")
                != spec["trace_query_identity_sha256"]
                or row.get("trace_target_vector_sha256")
                != spec["trace_target_vector_sha256"]
                or row.get("gap_8k_identity") is not True
                or row.get("gap_weighting_guard_passed") is not True
                or row.get("gap_clip_hit_count") != 0
                or row.get("factorized_gibbs_conditional_logit_clipped_count")
                != 0
                or row.get("direction_logit_clipped_count") != 0
                or row.get("nonfinite_count") != 0
            ):
                raise RuntimeError(f"R8 collection 数值身份漂移：{task.task_id}")
            terminal_relative = row.get("terminal_table_path")
            if not isinstance(terminal_relative, str):
                raise TypeError(f"R8 collection 终表路径非法：{task.task_id}")
            manifest_path = (
                Path(terminal_relative).parent / collection.CASE_MANIFEST
            )
            manifest = _load_json(_artifact_path(root, str(manifest_path)))
            if (
                manifest.get("contract_version") != protocol.PROTOCOL_VERSION
                or manifest.get("protocol_sha256")
                != protocol.FROZEN_PROTOCOL_SHA256
                or manifest.get("execution_commit") != execution_commit
                or manifest.get("execution_shard_id") != shard_id
                or manifest.get("generator_params")
                != protocol.generator_params_manifest(
                    task.dataset, task.arm, task.seed
                )
                or manifest.get("collection_row") != row
                or manifest.get("raw_reference_data_accessed") is not False
                or manifest.get("method_comparison_emitted") is not False
            ):
                raise RuntimeError(f"R8 case 清单漂移：{task.task_id}")
            for path_key, sha_key in (
                ("terminal_table_path", "terminal_table_sha256"),
                ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
                ("transition_audit_path", "transition_audit_sha256"),
            ):
                artifact = _artifact_path(root, row[path_key])
                if protocol.file_sha256(artifact) != row[sha_key]:
                    raise RuntimeError(
                        f"R8 case 产物哈希漂移：{task.task_id}/{path_key}"
                    )
            transition = _load_json(
                _artifact_path(root, row["transition_audit_path"])
            )
            weighting = transition.get("weighting_audit")
            if (
                transition.get("task_id") != task.task_id
                or transition.get("round_count") != row["applied_rounds"]
                or transition.get("all_applied_unconditionally") is not True
                or not isinstance(weighting, dict)
                or weighting.get("guard_passed") is not True
                or weighting.get("expected_weighting")
                != row["gap_weighting"]
                or weighting.get("scan_round_count")
                != row["gap_weighting_scan_round_count"]
            ):
                raise RuntimeError(f"R8 转移审计漂移：{task.task_id}")
            key = (task.seed, task.dataset, task.arm)
            if key in indexed:
                raise RuntimeError(f"R8 collection case 重复：{key}")
            indexed[key] = row
    expected_keys = {
        (task.seed, task.dataset, task.arm) for task in tasks
    }
    if set(indexed) != expected_keys:
        raise RuntimeError("R8 collection 四条任务身份不完整")
    expected_shard_rows = [
        indexed[(task.seed, task.dataset, task.arm)]
        for task in protocol.tasks_for_shard(shard_id)
    ]
    if shard_report.get("raw_results") != expected_shard_rows:
        raise RuntimeError("R8 collection 分片与合并行不一致")
    return report, indexed


def _load_dataset_inputs(
    root: Path, runtime: Any
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        for key in ("schema", "queries", "marginals"):
            if protocol.file_sha256(root / spec[key]) != spec["input_sha256"][key]:
                raise RuntimeError(f"{dataset}.{key} 当前输入 SHA-256 漂移")
        payload = _load_json(root / spec["queries"])
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise TypeError(f"{dataset} 原始测量查询缺失")
        queries = runtime.load_queries(str(root / spec["queries"]))
        targets = np.asarray([row.get("result") for row in raw_queries], dtype=float)
        loaded_targets = np.asarray(
            [query["result"] for query in queries], dtype=float
        )
        if (
            len(raw_queries) != spec["query_count"]
            or len(queries) != len(raw_queries)
            or not np.array_equal(targets, loaded_targets)
            or blind_identity.query_set_identity(raw_queries)
            != spec["query_identity_sha256"]
            or protocol.canonical_sha256(
                [row.get("result") for row in raw_queries]
            )
            != spec["target_vector_sha256"]
            or ordered_query_identity_sha256(queries)
            != spec["trace_query_identity_sha256"]
            or target_answer_identity_sha256(loaded_targets)
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"{dataset} 查询/目标身份漂移")
        # 在读取结果前重新证明每个结果前分箱的成员身份。
        all_indices: list[int] = []
        for bin_spec in protocol.TARGET_COUNT_BINS[dataset]:
            members = [
                {
                    "index": index,
                    "id": row.get("id"),
                    "target": int(row["result"]),
                }
                for index, row in enumerate(raw_queries)
                if bin_spec["lower_inclusive"]
                <= int(row["result"])
                <= bin_spec["upper_inclusive"]
            ]
            if (
                len(members) != bin_spec["query_count"]
                or protocol.canonical_sha256(members)
                != bin_spec["membership_sha256"]
            ):
                raise RuntimeError(f"{dataset}/{bin_spec['name']} 分箱身份漂移")
            all_indices.extend(item["index"] for item in members)
        if sorted(all_indices) != list(range(len(raw_queries))) or len(
            all_indices
        ) != len(set(all_indices)):
            raise RuntimeError(f"{dataset} 目标分箱不是完整互斥划分")
        loaded[dataset] = {
            "schema": runtime.load_schema(str(root / spec["schema"])),
            "queries": queries,
            "raw_queries": raw_queries,
            "targets": loaded_targets,
        }
    return loaded


def _integer_vectors(
    targets: Sequence[Any], answers: Sequence[Any], n_records: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(targets, dtype=float)
    answer = np.asarray(answers, dtype=float)
    if (
        target.ndim != 1
        or answer.shape != target.shape
        or target.size == 0
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(answer))
        or not np.all(target == np.rint(target))
        or not np.all(answer == np.rint(answer))
        or np.any(target < 0)
        or np.any(target > n_records)
        or np.any(answer < 0)
        or np.any(answer > n_records)
    ):
        raise ValueError("R8 指标目标/答案不是合法同形整数计数向量")
    target_int = target.astype(np.int64)
    answer_int = answer.astype(np.int64)
    return target_int, answer_int, np.abs(target_int - answer_int)


def _vector_metrics(
    dataset: str,
    raw_queries: Sequence[Mapping[str, Any]],
    targets: Sequence[Any],
    answers: Sequence[Any],
    n_records: int,
) -> dict[str, Any]:
    target, _answer, absolute = _integer_vectors(targets, answers, n_records)
    count_sum = int(sum(int(value) for value in absolute))
    legacy_proxy = math.fsum(
        int(error) / max(int(target_value), 8)
        for error, target_value in zip(absolute, target, strict=True)
    ) / len(absolute)
    smoothing = protocol.smoothing_count(dataset)
    bounded_proxy = math.fsum(
        int(error) / (min(max(int(target_value), 0), n_records) + smoothing)
        for error, target_value in zip(absolute, target, strict=True)
    ) / len(absolute)

    target_bins: dict[str, dict[str, Any]] = {}
    for bin_spec in protocol.TARGET_COUNT_BINS[dataset]:
        indices = [
            index
            for index, value in enumerate(target)
            if bin_spec["lower_inclusive"]
            <= int(value)
            <= bin_spec["upper_inclusive"]
        ]
        bin_sum = int(sum(int(absolute[index]) for index in indices))
        target_bins[bin_spec["name"]] = {
            "query_count": len(indices),
            "absolute_count_error_sum": bin_sum,
            "absolute_count_error_mean": float(bin_sum / len(indices)),
            "normalized_l1": float(bin_sum / (len(indices) * n_records)),
            "membership_sha256": bin_spec["membership_sha256"],
        }

    query_orders: dict[str, dict[str, Any]] = {}
    for order in sorted({len(query["conditions"]) for query in raw_queries}):
        indices = [
            index
            for index, query in enumerate(raw_queries)
            if len(query["conditions"]) == order
        ]
        order_sum = int(sum(int(absolute[index]) for index in indices))
        query_orders[str(order)] = {
            "query_count": len(indices),
            "absolute_count_error_sum": order_sum,
            "absolute_count_error_mean": float(order_sum / len(indices)),
            "normalized_l1": float(order_sum / (len(indices) * n_records)),
        }
    return {
        "query_count": len(absolute),
        "absolute_count_error_sum": count_sum,
        "normalized_l1": float(count_sum / (len(absolute) * n_records)),
        "legacy_relative_gap_proxy": float(legacy_proxy),
        "bounded_r8_gap_proxy": float(bounded_proxy),
        "bounded_r8_smoothing_count": smoothing,
        "target_count_bins": target_bins,
        "query_orders": query_orders,
    }


def _audit_checkpoint_artifact(
    root: Path,
    source: dict[str, Any],
    dataset_input: dict[str, Any],
    terminal_answers: np.ndarray,
) -> dict[str, Any]:
    artifact = _load_json(
        _artifact_path(root, source["checkpoint_artifact_path"])
    )
    dataset = source["dataset"]
    spec = protocol.DATASETS[dataset]
    if (
        artifact.get("contract_version") != protocol.PROTOCOL_VERSION
        or artifact.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or artifact.get("task_id") != source["task_id"]
        or artifact.get("dataset") != dataset
        or artifact.get("arm") != source["arm"]
        or artifact.get("seed") != source["seed"]
        or artifact.get("n_records") != spec["n_records"]
        or artifact.get("query_count") != spec["query_count"]
        or artifact.get("query_identity_sha256")
        != spec["query_identity_sha256"]
        or artifact.get("target_vector_sha256")
        != spec["target_vector_sha256"]
        or artifact.get("trace_query_identity_sha256")
        != spec["trace_query_identity_sha256"]
        or artifact.get("trace_target_vector_sha256")
        != spec["trace_target_vector_sha256"]
        or artifact.get("fixed_checkpoint_rounds_requested")
        != list(protocol.CHECKPOINT_ROUNDS)
        or artifact.get("historical_best_included") is not False
    ):
        raise RuntimeError(f"R8 检查点身份漂移：{source['task_id']}")

    def recompute(row: dict[str, Any]) -> dict[str, Any]:
        answers = np.asarray(row.get("query_answers"), dtype=float)
        vector = _vector_metrics(
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
            raise RuntimeError(f"R8 检查点指标复算漂移：{identity}")
        return {
            "kind": row["kind"],
            "state_index": int(row["state_index"]),
            "round": int(row["round"]),
            "phase": row["phase"],
            "squared_loss": squared,
            **vector,
        }

    fixed_source = artifact.get("fixed_checkpoints")
    if not isinstance(fixed_source, list):
        raise TypeError("R8 固定检查点列表缺失")
    expected_rounds = [
        value
        for value in protocol.CHECKPOINT_ROUNDS
        if value <= source["applied_rounds"]
    ]
    if [row.get("round") for row in fixed_source] != expected_rounds:
        raise RuntimeError(f"R8 固定检查点覆盖漂移：{source['task_id']}")
    fixed = [recompute(row) for row in fixed_source]
    terminal_source = artifact.get("terminal")
    if not isinstance(terminal_source, dict):
        raise TypeError("R8 terminal 检查点缺失")
    if not np.array_equal(
        np.asarray(terminal_source.get("query_answers"), dtype=float),
        terminal_answers,
    ):
        raise RuntimeError(f"R8 终表与终点向量不一致：{source['task_id']}")
    terminal = recompute(terminal_source)
    if terminal["round"] != source["applied_rounds"]:
        raise RuntimeError(f"R8 终点轮次漂移：{source['task_id']}")
    return {"fixed_checkpoints": fixed, "terminal": terminal}


def _evaluate_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = offline_helpers._load_runtime()
    # 所有查询身份在原始参考表加载前冻结。
    test_groups, test_identity = offline_helpers._freeze_test_groups(root)
    nltcs_groups, nltcs_identity = offline_helpers._freeze_nltcs_groups(root)
    inputs = _load_dataset_inputs(root, runtime)
    references, reference_sha = offline_helpers._load_references(root, runtime)
    expected_reference_sha = {
        name: spec["input_sha256"]["reference"]
        for name, spec in protocol.DATASETS.items()
    }
    if reference_sha != expected_reference_sha:
        raise RuntimeError("R8 离线参考表 SHA-256 漂移")
    test_group_targets = {
        name: runtime.evaluate_table(references["test_300x10"], queries)
        for name, queries in test_groups.items()
    }
    nltcs_marginals = _load_json(
        root / protocol.DATASETS["nltcs"]["marginals"]
    )
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    nltcs_measured_triples = runtime.offline._measured_cell_keys(
        inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )

    cases: list[dict[str, Any]] = []
    for task in protocol.task_plan().tasks:
        source = indexed[(task.seed, task.dataset, task.arm)]
        table = runtime.pd.read_csv(
            _artifact_path(root, source["terminal_table_path"])
        )
        spec = protocol.DATASETS[task.dataset]
        if len(table) != spec["n_records"]:
            raise RuntimeError(f"R8 终表行数漂移：{task.task_id}")
        dataset_input = inputs[task.dataset]
        terminal_answers = np.asarray(
            runtime.evaluate_table(table, dataset_input["queries"]),
            dtype=float,
        )
        terminal_vector = _vector_metrics(
            task.dataset,
            dataset_input["raw_queries"],
            dataset_input["targets"],
            terminal_answers,
            spec["n_records"],
        )
        trajectory = _audit_checkpoint_artifact(
            root, source, dataset_input, terminal_answers
        )
        if any(
            trajectory["terminal"][key] != terminal_vector[key]
            for key in terminal_vector
        ):
            raise RuntimeError(f"R8 终表与终点指标不一致：{task.task_id}")

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
        if not math.isclose(
            float(overall["normalized_l1_mean"]),
            terminal_vector["normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"R8 measured L1 复算漂移：{task.task_id}")
        overall.update(
            {
                "absolute_count_error_sum": terminal_vector[
                    "absolute_count_error_sum"
                ],
                "normalized_l1_mean": terminal_vector["normalized_l1"],
            }
        )
        for order, exact in terminal_vector["query_orders"].items():
            observed = metrics["measured"]["by_order"][order]
            if not math.isclose(
                float(observed["normalized_l1_mean"]),
                exact["normalized_l1"],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    f"R8 查询阶数指标漂移：{task.task_id}/{order}"
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
            "legacy_relative_gap": terminal_vector[
                "legacy_relative_gap_proxy"
            ],
            "bounded_r8_gap": terminal_vector["bounded_r8_gap_proxy"],
            "bounded_r8_smoothing_count": terminal_vector[
                "bounded_r8_smoothing_count"
            ],
        }
        metrics["measured"]["target_count_bins"] = terminal_vector[
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
    identities = {
        "query_identity_frozen_before_reference_load": True,
        "test": test_identity,
        "nltcs": nltcs_identity,
        "reference_sha256": reference_sha,
    }
    return cases, identities


def _nested(record: Mapping[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"R8 指标 {path} 不是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"R8 指标 {path} 不是有限数")
    return result


def _case(cases: Sequence[dict[str, Any]], dataset: str, arm: str) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case["dataset"] == dataset and case["arm"] == arm
    ]
    if len(selected) != 1 or selected[0]["seed"] != protocol.DEVELOPMENT_SEED:
        raise RuntimeError(f"{dataset}/{arm} 缺少唯一冻结开发种子")
    return selected[0]


def _ratio_record(candidate: float, baseline: float) -> dict[str, Any]:
    if candidate < 0.0 or baseline < 0.0:
        raise ValueError("R8/legacy 比值只接受非负指标")
    if baseline == 0.0:
        if candidate == 0.0:
            ratio = 1.0
            state = "both_zero_ratio_one"
        else:
            ratio = None
            state = "positive_infinity"
    else:
        ratio = float(candidate / baseline)
        state = "finite"
    return {
        "bounded_r8": float(candidate),
        "legacy": float(baseline),
        "bounded_over_legacy": ratio,
        "ratio_state": state,
        "lower_is_better": True,
    }


def _decision_ratio(record: Mapping[str, Any]) -> float:
    if record["ratio_state"] == "positive_infinity":
        return float("inf")
    return float(record["bounded_over_legacy"])


def _comparison(
    cases: Sequence[dict[str, Any]], dataset: str, path: str
) -> dict[str, Any]:
    bounded = _nested(_case(cases, dataset, "gap_bounded_r8_s8"), path)
    legacy = _nested(_case(cases, dataset, "gap_legacy_s8"), path)
    result = _ratio_record(bounded, legacy)
    result["metric"] = path
    return result


def _summary_and_decision(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in protocol.DATASET_ORDER:
        measured = _comparison(
            cases, dataset, "metrics.measured.overall.normalized_l1_mean"
        )
        proxies = {
            name: _comparison(
                cases,
                dataset,
                f"metrics.measured.proxy_objectives.{name}",
            )
            for name in ("legacy_relative_gap", "bounded_r8_gap")
        }
        target_bins = {
            item["name"]: _comparison(
                cases,
                dataset,
                "metrics.measured.target_count_bins."
                f"{item['name']}.absolute_count_error_mean",
            )
            for item in protocol.TARGET_COUNT_BINS[dataset]
        }
        orders = {
            str(order): _comparison(
                cases,
                dataset,
                f"metrics.measured.by_order.{order}.absolute_count_error_mean",
            )
            for order in protocol.DATASETS[dataset]["order_counts"]
        }
        group_names = (
            protocol.TEST_GROUP_ORDER
            if dataset == "test_300x10"
            else tuple(protocol.NLTCS_GROUP_COUNTS)
        )
        offline_groups = {
            name: _comparison(
                cases,
                dataset,
                f"metrics.offline_query_groups.{name}.normalized_l1_mean",
            )
            for name in group_names
        }
        checkpoint_curves: dict[str, Any] = {}
        for arm in protocol.ARM_ORDER:
            case = _case(cases, dataset, arm)
            by_round = {
                int(row["round"]): row
                for row in case["trajectory"]["fixed_checkpoints"]
            }
            checkpoint_curves[arm] = {
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
        datasets[dataset] = {
            "measured_normalized_l1": measured,
            "proxy_objectives": proxies,
            "target_count_bins": target_bins,
            "query_orders": orders,
            "offline_query_groups": offline_groups,
            "checkpoint_curves": checkpoint_curves,
            "terminal_cost_by_arm": {
                arm: _case(cases, dataset, arm)["cost"]
                for arm in protocol.ARM_ORDER
            },
        }

    execution_valid = all(
        _nested(case, "metrics.validity.valid_row_rate") == 1.0
        and case["kernel_audit"]["gap_weighting_guard_passed"] is True
        for case in cases
    )
    nltcs = datasets["nltcs"]
    test = datasets["test_300x10"]
    classification = protocol.classify_screen(
        execution_valid=execution_valid,
        nltcs_measured_l1_ratio=_decision_ratio(
            nltcs["measured_normalized_l1"]
        ),
        nltcs_common_bin_ratio=_decision_ratio(
            nltcs["target_count_bins"][protocol.NLTCS_COMMON_BIN]
        ),
        nltcs_rare_bin_ratio=_decision_ratio(
            nltcs["target_count_bins"][protocol.NLTCS_RARE_BIN]
        ),
        test_measured_l1_ratio=_decision_ratio(
            test["measured_normalized_l1"]
        ),
        one_way_ratio_by_dataset={
            dataset: _decision_ratio(
                datasets[dataset]["offline_query_groups"]["one_way_safety"]
            )
            for dataset in protocol.DATASET_ORDER
        },
    )
    return {
        "datasets": datasets,
        "execution_valid": bool(execution_valid),
        "quality_interpretation_allowed": bool(execution_valid),
        "frozen_classification": classification,
        "classification_precedence": protocol.scientific.frozen_protocol_manifest()[
            "screen_decision"
        ]["precedence"],
        "automatic_followup_authorized": False,
    }


def _csv_rows(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        terminal_round = case["trajectory"]["terminal"]["round"]
        states = list(case["trajectory"]["fixed_checkpoints"])
        if all(row["round"] != terminal_round for row in states):
            states.append(case["trajectory"]["terminal"])
        for state in states:
            base = {
                "dataset": case["dataset"],
                "arm": case["arm"],
                "seed": case["seed"],
                "record_kind": state["kind"],
                "round": state["round"],
                "state_index": state["state_index"],
                "is_terminal": state["round"] == terminal_round,
            }

            def append(group: str, name: str, count: int, value: float) -> None:
                rows.append(
                    {
                        **base,
                        "metric_group": group,
                        "metric_name": name,
                        "query_count": count,
                        "value": value,
                    }
                )

            append(
                "measured",
                "normalized_l1",
                state["query_count"],
                state["normalized_l1"],
            )
            append(
                "proxy",
                "legacy_relative_gap",
                state["query_count"],
                state["legacy_relative_gap_proxy"],
            )
            append(
                "proxy",
                "bounded_r8_gap",
                state["query_count"],
                state["bounded_r8_gap_proxy"],
            )
            for name, metric in state["target_count_bins"].items():
                append(
                    "target_count_bin",
                    name,
                    metric["query_count"],
                    metric["absolute_count_error_mean"],
                )
            for name, metric in state["query_orders"].items():
                append(
                    "query_order",
                    name,
                    metric["query_count"],
                    metric["absolute_count_error_mean"],
                )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _publish(
    destination: Path,
    report: dict[str, Any],
    csv_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    report_path = destination / protocol.EVALUATION_REPORT
    csv_path = destination / protocol.L1_RESULTS_CSV
    if report_path.exists() or csv_path.exists():
        raise FileExistsError("R8 筛查评价产物已存在，不覆盖")
    temporary = Path(
        tempfile.mkdtemp(prefix=".r8-screen-evaluation.tmp-", dir=destination)
    )
    try:
        temporary_csv = temporary / protocol.L1_RESULTS_CSV
        _write_csv(temporary_csv, csv_rows)
        report["screen_metrics_csv"] = {
            "path": protocol.L1_RESULTS_CSV,
            "sha256": protocol.file_sha256(temporary_csv),
            "row_count": len(csv_rows),
            "columns": list(CSV_FIELDS),
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


def build_plan() -> dict[str, Any]:
    protocol_sha = protocol.assert_frozen_protocol_identity(_repo_root())
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_reference_or_generation_access",
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
        "new_generation_allowed": False,
        "generation_started": False,
        "requires_complete_collection_sha_confirmation": True,
    }


def evaluate(confirmed_collection_report_sha256: str) -> tuple[Path, Path]:
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    if collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "R8 筛查评价要求包含未跟踪文件在内的干净工作树"
        )
    collection_report, indexed = _audit_collection(
        root, confirmed_collection_report_sha256
    )
    cases, identities = _evaluate_cases(root, indexed)
    summary = _summary_and_decision(cases)
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_complete_r8_collection",
        "evaluation_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_commit": collection_report["execution_commit"],
        "query_and_reference_identity_audit": identities,
        "case_count": len(cases),
        "cases": cases,
        "summary": summary["datasets"],
        "execution_valid": summary["execution_valid"],
        "quality_interpretation_allowed": summary[
            "quality_interpretation_allowed"
        ],
        "frozen_classification": summary["frozen_classification"],
        "classification_precedence": summary["classification_precedence"],
        "automatic_followup_authorized": False,
        "raw_reference_data_accessed": True,
        "all_four_cases_evaluated": len(cases) == 4,
        "terminal_and_reached_checkpoints_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    return _publish(
        root / protocol.OUTPUT_DIR, report, _csv_rows(cases)
    )


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
    report, csv_path = evaluate(args.confirm_collection_sha)
    print(f"R8 screen evaluation -> {report}")
    print(f"R8 screen metrics -> {csv_path}")
    print(f"evaluation SHA-256 -> {protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
