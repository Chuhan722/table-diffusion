#!/usr/bin/env python3
"""Evaluate the frozen Issue #53 test query-workload A/B collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import freeze_issue53_test_query_workload_ab as freeze
    from scripts import materialize_issue53_test_query_workload_b as materialize
    from scripts import run_issue53_test_query_workload_ab as collection
else:
    import freeze_issue53_test_query_workload_ab as freeze
    import materialize_issue53_test_query_workload_b as materialize
    import run_issue53_test_query_workload_ab as collection


EVALUATION_VERSION = "issue53-test-query-workload-ab-evaluation-v1"
EVALUATION_MODE = "evaluate_frozen_collection_after_query_identity_audit"
EVALUATION_REPORT = "evaluation_report.json"
ERROR_ARTIFACT = "query_seed_errors.csv"

GROUP_ORDER = (
    "one_way_safety",
    "common_unseen_2way",
    "fixed_heldout_3way",
    "fixed_heldout_4way",
)
PRIMARY_GROUPS = (
    "common_unseen_2way",
    "fixed_heldout_3way",
    "fixed_heldout_4way",
)
CANDIDATE_GEOMETRIES = ("sqrt_relative", "relative")
EXPECTED_GROUP_COUNTS = freeze.EXPECTED_EVALUATION_COUNTS
EXPECTED_GROUP_IDENTITIES = {
    "one_way_safety": (
        "b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c"
    ),
    "common_unseen_2way": (
        "fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2"
    ),
    "fixed_heldout_3way": (
        "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a"
    ),
    "fixed_heldout_4way": (
        "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171"
    ),
}

ERROR_FIELDS = (
    "dataset",
    "seed",
    "workload",
    "geometry",
    "query_group",
    "query_group_index",
    "query_id",
    "query_order",
    "query_fingerprint_sha256",
    "target_count",
    "current_count",
    "abs_error",
)


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_or_reference_read",
        "collection_protocol_sha256": collection.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            collection.OUTPUT_DIR / collection.COLLECTION_REPORT
        ),
        "evaluation_report": str(collection.OUTPUT_DIR / EVALUATION_REPORT),
        "query_groups_in_report_order": list(GROUP_ORDER),
        "expected_group_counts": EXPECTED_GROUP_COUNTS,
        "expected_group_identity_sha256": EXPECTED_GROUP_IDENTITIES,
        "primary_groups": list(PRIMARY_GROUPS),
        "comparison_order": [
            "within_each_geometry_workload_B_minus_A",
            "within_workload_B_candidate_geometry_minus_absolute",
        ],
        "workload_comparisons": [
            {"geometry": geometry, "candidate": "B", "baseline": "A"}
            for geometry in collection.GEOMETRIES
        ],
        "workload_b_geometry_comparisons": [
            {"candidate": geometry, "baseline": "absolute"}
            for geometry in CANDIDATE_GEOMETRIES
        ],
        "pareto_rule": {
            "all_primary_group_mean_delta_lte_zero": True,
            "at_least_one_primary_group_mean_delta_lt_zero": True,
            "same_improved_group_paired_seed_better_minimum": 4,
            "paired_seed_count": 5,
        },
        "one_way_safety_rule": "mean_delta_lte_zero",
        "cross_group_aggregate_allowed": False,
        "scientific_overrides_allowed": False,
        "raw_reference_access_only_after_query_identity_audit": True,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def build_evaluation_preamble() -> dict[str, Any]:
    return {**build_plan(), "mode": EVALUATION_MODE}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _audit_collection(
    root: Path,
    confirmed_report_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    report_path = root / collection.OUTPUT_DIR / collection.COLLECTION_REPORT
    observed_sha = collection._sha256_file(report_path)
    if observed_sha != confirmed_report_sha256:
        raise ValueError(
            "collection report SHA 与显式确认值不一致："
            f"confirmed={confirmed_report_sha256}, observed={observed_sha}"
        )
    report = _load_json(report_path)
    if report.get("contract_version") != collection.PROTOCOL_VERSION:
        raise RuntimeError("collection contract version 漂移")
    if report.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("collection protocol SHA 漂移")
    if report.get("protocol") != collection.frozen_protocol_manifest():
        raise RuntimeError("collection protocol manifest 漂移")
    if report.get("case_count") != 30:
        raise RuntimeError("collection case count 不是 30")
    if report.get("raw_reference_data_accessed"):
        raise RuntimeError("collection 不得访问 raw reference")
    if report.get("privacy_budget_consumed"):
        raise RuntimeError("collection 不得消耗隐私预算")
    if report.get("parameter_retuning_performed"):
        raise RuntimeError("collection 不得结果后调参")

    execution_commit = report.get("execution_git_commit")
    if not isinstance(execution_commit, str):
        raise TypeError("collection execution commit 缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    merge_base = collection._git_text(
        root,
        "merge-base",
        execution_commit,
        current_commit,
    )
    if merge_base != execution_commit:
        raise RuntimeError("collection execution commit 不是评价提交的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError("collection raw results 不完整")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["workload"], row["geometry"])
        if key in indexed:
            raise RuntimeError(f"collection case 重复：{key}")
        if (
            key[0] not in collection.SEEDS
            or key[1] not in collection.WORKLOADS
            or key[2] not in collection.GEOMETRIES
        ):
            raise RuntimeError(f"collection case 不在冻结矩阵：{key}")
        if row.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256:
            raise RuntimeError(f"collection case protocol 漂移：{key}")
        if row.get("git_commit") != execution_commit:
            raise RuntimeError(f"collection case commit 漂移：{key}")
        if row.get("query_identity_sha256") != collection.WORKLOADS[key[1]][
            "query_identity_sha256"
        ]:
            raise RuntimeError(f"collection case query identity 漂移：{key}")
        if row.get("target_vector_sha256") != collection.WORKLOADS[key[1]][
            "target_vector_sha256"
        ]:
            raise RuntimeError(f"collection case target identity 漂移：{key}")
        if key[1] == "B" and (
            row.get("measured_four_way_query_count") != 5
            or not row.get("four_way_queries_in_full_objective_and_early_stop")
            or not row.get("factorized_gibbs_inactive")
        ):
            raise RuntimeError(f"collection B 4-way 路径审计失败：{key}")
        table_path = (
            root
            / collection.OUTPUT_DIR
            / f"seed_{key[0]}"
            / key[1]
            / key[2]
            / "terminal_current.csv"
        )
        if collection._sha256_file(table_path) != row["terminal_table_sha256"]:
            raise RuntimeError(f"collection terminal table SHA 漂移：{key}")
        indexed[key] = row
    expected = {
        (seed, workload, geometry)
        for seed in collection.SEEDS
        for workload, geometry in collection.CASE_ORDER
    }
    if set(indexed) != expected:
        raise RuntimeError("collection 30-case 身份不完整")
    return report, indexed


def _freeze_query_groups(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Freeze every common evaluation identity before opening the CSV."""

    marginals = _load_json(root / freeze.MARGINALS_PATH)
    measured = _load_json(root / freeze.WORKLOAD_A_PATH).get("queries")
    if not isinstance(measured, list):
        raise TypeError("workload A queries 缺失")
    frozen = freeze.freeze_query_identities(marginals, measured)
    groups = frozen["evaluation_groups"]
    observed_counts = {name: len(groups[name]) for name in GROUP_ORDER}
    observed_identities = {
        name: freeze.query_set_identity(groups[name]) for name in GROUP_ORDER
    }
    if observed_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError("公共评价查询数量漂移")
    if observed_identities != EXPECTED_GROUP_IDENTITIES:
        raise RuntimeError("公共评价查询身份漂移")
    freeze._assert_result_free(groups)
    return groups, {
        "identity_frozen_before_reference_load": True,
        "group_counts": observed_counts,
        "group_query_identity_sha256": observed_identities,
        "cross_group_aggregate_allowed": False,
    }


def _attach_reference_answers(
    root: Path,
    groups: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    reference_path = root / materialize.REFERENCE_PATH
    observed_sha = collection._sha256_file(reference_path)
    if observed_sha != materialize.REFERENCE_SHA256:
        raise RuntimeError("raw reference SHA 漂移")
    rows = materialize._load_reference(reference_path)
    answers = {
        name: materialize.evaluate_queries(rows, groups[name]) for name in GROUP_ORDER
    }

    stored_heldout = _load_json(
        root / "configs/test_300x10/heldout_issue53_v1.json"
    )["queries"]
    for order, name in ((3, "fixed_heldout_3way"), (4, "fixed_heldout_4way")):
        stored_answers = [
            query["result"]
            for query in stored_heldout
            if freeze.query_order(query) == order
        ]
        if answers[name] != stored_answers:
            raise RuntimeError(f"{name} reference answers 与既有存档不一致")
    return answers, {
        "reference_sha256": observed_sha,
        "group_target_vector_sha256": {
            name: materialize._target_vector_sha256(values)
            for name, values in answers.items()
        },
        "fixed_heldout_answers_match_existing_archive": True,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
    }


def _build_error_rows(
    root: Path,
    collection_rows: dict[tuple[int, str, str], dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    targets: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = []
    table_audit = {}
    for seed in collection.SEEDS:
        for workload, geometry in collection.CASE_ORDER:
            key = (seed, workload, geometry)
            row = collection_rows[key]
            table_path = (
                root
                / collection.OUTPUT_DIR
                / f"seed_{seed}"
                / workload
                / geometry
                / "terminal_current.csv"
            )
            table = materialize._load_reference(table_path)
            case_name = f"seed_{seed}/{workload}/{geometry}"
            table_audit[case_name] = {
                "path": str(table_path.relative_to(root)),
                "sha256": collection._sha256_file(table_path),
                "matches_collection": (
                    collection._sha256_file(table_path)
                    == row["terminal_table_sha256"]
                ),
            }
            for group_name in GROUP_ORDER:
                queries = groups[group_name]
                observed = materialize.evaluate_queries(table, queries)
                expected = targets[group_name]
                if len(observed) != len(expected):
                    raise RuntimeError(f"{case_name}/{group_name} answer 数量漂移")
                for index, (query, target, current) in enumerate(
                    zip(queries, expected, observed, strict=True)
                ):
                    errors.append({
                        "dataset": collection.DATASET,
                        "seed": seed,
                        "workload": workload,
                        "geometry": geometry,
                        "query_group": group_name,
                        "query_group_index": index,
                        "query_id": query.get("id", f"{group_name}_{index}"),
                        "query_order": freeze.query_order(query),
                        "query_fingerprint_sha256": freeze.query_fingerprint(query),
                        "target_count": target,
                        "current_count": current,
                        "abs_error": abs(target - current),
                    })
    expected_rows = sum(EXPECTED_GROUP_COUNTS.values()) * 30
    if len(errors) != expected_rows:
        raise RuntimeError(
            f"query-seed error 行数漂移：expected={expected_rows}, observed={len(errors)}"
        )
    if not all(row["matches_collection"] for row in table_audit.values()):
        raise RuntimeError("terminal table audit 失败")
    return errors, table_audit


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile 输入不能为空")
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _cell_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["abs_error"]) for row in rows]
    if len(rows) % len(collection.SEEDS) != 0:
        raise RuntimeError("cell rows 不能按 seed 均分")
    expected_per_seed = len(rows) // len(collection.SEEDS)
    per_seed = {
        seed: _mean(
            [float(row["abs_error"]) for row in rows if row["seed"] == seed]
        )
        for seed in collection.SEEDS
    }
    if any(
        len([row for row in rows if row["seed"] == seed]) != expected_per_seed
        for seed in collection.SEEDS
    ):
        raise RuntimeError("cell seed 配对数量不均衡")
    return {
        "mean_abs_error_count": _mean(values),
        "mean_abs_error_normalized": _mean(values) / collection.N_RECORDS,
        "median_abs_error_count": float(statistics.median(values)),
        "p90_abs_error_count": _percentile(values, 90),
        "max_abs_error_count": max(values),
        "exact_match_rate": _mean([value == 0 for value in values]),
        "paired_seed_mean_abs_error_count": {
            str(seed): value for seed, value in per_seed.items()
        },
    }


def _paired_comparison(
    rows: Sequence[dict[str, Any]],
    candidate_filter: Callable[[dict[str, Any]], bool],
    baseline_filter: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    candidate = {
        (row["seed"], row["query_group_index"]): float(row["abs_error"])
        for row in rows
        if candidate_filter(row)
    }
    baseline = {
        (row["seed"], row["query_group_index"]): float(row["abs_error"])
        for row in rows
        if baseline_filter(row)
    }
    if not candidate or set(candidate) != set(baseline):
        raise RuntimeError("paired comparison 身份不完整")
    deltas = {key: candidate[key] - baseline[key] for key in candidate}
    per_seed = {
        seed: _mean([value for (row_seed, _), value in deltas.items() if row_seed == seed])
        for seed in collection.SEEDS
    }
    return {
        "mean_abs_error_delta_count": _mean(list(deltas.values())),
        "query_seed_candidate_better_count": sum(value < 0 for value in deltas.values()),
        "query_seed_tie_count": sum(value == 0 for value in deltas.values()),
        "query_seed_candidate_worse_count": sum(value > 0 for value in deltas.values()),
        "paired_seed_mean_delta_count": {
            str(seed): value for seed, value in per_seed.items()
        },
        "paired_seed_candidate_better_count": sum(value < 0 for value in per_seed.values()),
        "paired_seed_tie_count": sum(value == 0 for value in per_seed.values()),
        "paired_seed_candidate_worse_count": sum(value > 0 for value in per_seed.values()),
    }


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = {row["query_group"] for row in rows}
    if len(names) != 1:
        raise ValueError("summarize_group 只能接收一个查询组")
    query_count = len({row["query_group_index"] for row in rows})
    expected = (
        query_count
        * len(collection.SEEDS)
        * len(collection.WORKLOADS)
        * len(collection.GEOMETRIES)
    )
    if query_count <= 0 or len(rows) != expected:
        raise RuntimeError("evaluation group rows 不完整")

    cells = {workload: {} for workload in collection.WORKLOADS}
    for workload in collection.WORKLOADS:
        for geometry in collection.GEOMETRIES:
            selected = [
                row
                for row in rows
                if row["workload"] == workload and row["geometry"] == geometry
            ]
            if len(selected) != query_count * len(collection.SEEDS):
                raise RuntimeError(f"{workload}/{geometry} group cell 不完整")
            cells[workload][geometry] = _cell_metrics(selected)

    workload_pairs = {
        geometry: _paired_comparison(
            rows,
            lambda row, geometry=geometry: (
                row["workload"] == "B" and row["geometry"] == geometry
            ),
            lambda row, geometry=geometry: (
                row["workload"] == "A" and row["geometry"] == geometry
            ),
        )
        for geometry in collection.GEOMETRIES
    }
    geometry_pairs = {
        geometry: _paired_comparison(
            rows,
            lambda row, geometry=geometry: (
                row["workload"] == "B" and row["geometry"] == geometry
            ),
            lambda row: (
                row["workload"] == "B" and row["geometry"] == "absolute"
            ),
        )
        for geometry in CANDIDATE_GEOMETRIES
    }
    return {
        "query_count": query_count,
        "query_seed_count_per_cell": query_count * len(collection.SEEDS),
        "cells": cells,
        "workload_b_minus_a_by_geometry": workload_pairs,
        "workload_b_geometry_minus_absolute": geometry_pairs,
    }


def _gate_one_comparison(
    primary: dict[str, dict[str, Any]],
    safety: dict[str, Any],
    *,
    normal_completion: bool,
    support_classification: str,
    mixed_classification: str,
) -> dict[str, Any]:
    all_nonpositive = all(
        row["mean_abs_error_delta_count"] <= 0 for row in primary.values()
    )
    stable_improved_groups = [
        group
        for group, row in primary.items()
        if row["mean_abs_error_delta_count"] < 0
        and row["paired_seed_candidate_better_count"] >= 4
    ]
    unseen_pass = all_nonpositive and bool(stable_improved_groups)
    safety_pass = safety["mean_abs_error_delta_count"] <= 0
    if not normal_completion:
        classification = "inconclusive_resource_cap"
    elif unseen_pass and safety_pass:
        classification = support_classification
    elif unseen_pass:
        classification = "higher_order_gain_with_1way_tradeoff"
    else:
        classification = mixed_classification
    return {
        "classification": classification,
        "normal_completion": normal_completion,
        "unseen_pareto_pass": unseen_pass,
        "all_primary_group_mean_delta_lte_zero": all_nonpositive,
        "stable_improved_primary_groups": stable_improved_groups,
        "one_way_safety_pass": safety_pass,
        "one_way_mean_delta_count": safety["mean_abs_error_delta_count"],
        "primary_groups": {
            group: {
                "mean_abs_error_delta_count": row["mean_abs_error_delta_count"],
                "paired_seed_candidate_better_count": row[
                    "paired_seed_candidate_better_count"
                ],
                "paired_seed_tie_count": row["paired_seed_tie_count"],
                "paired_seed_candidate_worse_count": row[
                    "paired_seed_candidate_worse_count"
                ],
            }
            for group, row in primary.items()
        },
    }


def evaluate_frozen_gates(
    group_reports: dict[str, Any],
    normal_completion_by_cell: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    workload_effects = {}
    for geometry in collection.GEOMETRIES:
        primary = {
            group: group_reports[group]["workload_b_minus_a_by_geometry"][geometry]
            for group in PRIMARY_GROUPS
        }
        safety = group_reports["one_way_safety"][
            "workload_b_minus_a_by_geometry"
        ][geometry]
        normal = (
            normal_completion_by_cell[("A", geometry)]
            and normal_completion_by_cell[("B", geometry)]
        )
        workload_effects[geometry] = _gate_one_comparison(
            primary,
            safety,
            normal_completion=normal,
            support_classification="supports_workload_B_under_geometry",
            mixed_classification="mixed_no_workload_replacement",
        )

    classifications = [
        workload_effects[geometry]["classification"]
        for geometry in collection.GEOMETRIES
    ]
    workload_consistency = (
        "geometry_independent_workload_effect"
        if len(set(classifications)) == 1
        else "geometry_dependent_workload_effect"
    )

    geometry_effects = {}
    for geometry in CANDIDATE_GEOMETRIES:
        primary = {
            group: group_reports[group][
                "workload_b_geometry_minus_absolute"
            ][geometry]
            for group in PRIMARY_GROUPS
        }
        safety = group_reports["one_way_safety"][
            "workload_b_geometry_minus_absolute"
        ][geometry]
        normal = (
            normal_completion_by_cell[("B", "absolute")]
            and normal_completion_by_cell[("B", geometry)]
        )
        geometry_effects[geometry] = _gate_one_comparison(
            primary,
            safety,
            normal_completion=normal,
            support_classification="supports_geometry_under_workload_B",
            mixed_classification="mixed_no_unified_geometry_candidate",
        )
    return {
        "normal_completion_by_cell": {
            f"{workload}/{geometry}": value
            for (workload, geometry), value in normal_completion_by_cell.items()
        },
        "workload_b_vs_a_by_geometry": workload_effects,
        "workload_effect_consistency": workload_consistency,
        "workload_b_geometry_vs_absolute": geometry_effects,
    }


def _normal_completion_by_cell(
    rows: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[tuple[str, str], bool]:
    normal_reasons = {"fit_target_reached", "early_stopped"}
    return {
        (workload, geometry): all(
            rows[(seed, workload, geometry)]["termination_reason"]
            in normal_reasons
            for seed in collection.SEEDS
        )
        for workload in collection.WORKLOADS
        for geometry in collection.GEOMETRIES
    }


def _write_error_csv(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return collection._sha256_file(path)


def evaluate(confirmed_collection_report_sha256: str) -> Path:
    if len(confirmed_collection_report_sha256) != 64:
        raise ValueError("必须显式确认完整 collection report SHA-256")
    root = collection._repo_root()
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式评价要求包含 untracked 在内的干净工作树")
    destination = root / collection.OUTPUT_DIR
    report_path = destination / EVALUATION_REPORT
    error_path = destination / ERROR_ARTIFACT
    if report_path.exists() or error_path.exists():
        raise FileExistsError("正式评价产物已存在，不覆盖")

    collection_report, indexed_rows = _audit_collection(
        root,
        confirmed_collection_report_sha256,
    )
    groups, query_audit = _freeze_query_groups(root)
    targets, answer_audit = _attach_reference_answers(root, groups)
    errors, table_audit = _build_error_rows(
        root,
        indexed_rows,
        groups,
        targets,
    )
    group_reports = {
        group: summarize_group(
            [row for row in errors if row["query_group"] == group]
        )
        for group in GROUP_ORDER
    }
    completion = _normal_completion_by_cell(indexed_rows)
    gates = evaluate_frozen_gates(group_reports, completion)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".query-seed-errors.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_error_path = Path(handle.name)
    temporary_error_path.unlink()
    artifact_sha = _write_error_csv(temporary_error_path, errors)

    report = {
        **build_evaluation_preamble(),
        "evaluation_git_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_git_commit": collection_report[
            "execution_git_commit"
        ],
        "collection_summary": collection_report["summary"],
        "query_identity_frozen_before_reference_load": True,
        "query_identity_audit": query_audit,
        "reference_answer_audit": answer_audit,
        "terminal_table_audit": table_audit,
        "groups": group_reports,
        "frozen_gate_evaluation": gates,
        "cross_group_aggregate_present": False,
        "new_generation_performed_by_evaluator": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "canonical_selection_performed": False,
        "claim_scope": "test_query_workload_ab_not_cross_dataset_canonical",
        "artifacts": {
            "query_seed_errors": {
                "path": ERROR_ARTIFACT,
                "row_count": len(errors),
                "sha256": artifact_sha,
            }
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".evaluation-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_report_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_error_path, error_path)
    os.replace(temporary_report_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-collection-report-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = evaluate(args.confirm_collection_report_sha)
    print(f"query-workload A/B evaluation -> {path}")
    print(f"evaluation SHA-256 -> {collection._sha256_file(path)}")


if __name__ == "__main__":
    main()
