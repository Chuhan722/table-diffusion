#!/usr/bin/env python3
"""离线评价 Issue #53 固定 α 响应曲线的冻结采集。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__:
    from scripts import freeze_issue53_test_query_workload_ab as freeze_test
    from scripts import run_issue53_fixed_alpha_calibration as collection
else:
    import freeze_issue53_test_query_workload_ab as freeze_test
    import run_issue53_fixed_alpha_calibration as collection


EVALUATION_VERSION = "issue53-fixed-alpha-calibration-evaluation-v1"
EVALUATION_REPORT = "evaluation_report.json"
RISK_RATIO_MAX = 1.05
DIVERSITY_RATIO_MIN = 0.95
WORK_RATIO_MAX = 1.05
STABLE_WIN_MINIMUM = 4
BASELINE_ALPHA = 16.0
PROBE_ALPHAS = (12.0, 24.0)
NORMAL_REASONS = {"fit_target_reached", "early_stopped"}

REFERENCE_PATHS = {
    "test_300x10": Path("data/test_300x10/test_300x10.csv"),
    "nltcs": Path("data/nltcs/nltcs.train.data"),
}
REFERENCE_SHA256 = {
    "test_300x10": (
        "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
    ),
    "nltcs": (
        "e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30"
    ),
}
TEST_IDENTITY_ARTIFACT = Path(
    "configs/test_300x10/issue53_query_workload_ab_v1.json"
)
TEST_IDENTITY_ARTIFACT_SHA256 = (
    "a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36"
)
TEST_GROUP_ORDER = (
    "one_way_safety",
    "common_unseen_2way",
    "fixed_heldout_3way",
    "fixed_heldout_4way",
)
TEST_GROUP_COUNTS = {
    "one_way_safety": 25,
    "common_unseen_2way": 521,
    "fixed_heldout_3way": 512,
    "fixed_heldout_4way": 512,
}
TEST_GROUP_IDENTITIES = {
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
NLTCS_GROUP_COUNTS = {
    "one_way_safety": 32,
    "unmeasured_3way": 3958,
    "all_4way": 29120,
}
NLTCS_GROUP_IDENTITIES = {
    "one_way_safety": (
        "bbc8fc5d1b1ed0e5cd318a2168fe3887297b1c6aa33634736d0c693e96785c13"
    ),
    "unmeasured_3way": (
        "9c43437d6366e3cce0438fdf79e104d70ebabc112db9236b3feef5220b5eb588"
    ),
    "all_4way": (
        "1b92f8d80e775cffd637450d3d5015c78d43f7d9a870faf1603c99c88ec5d408"
    ),
}


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_or_reference_read",
        "collection_protocol_sha256": collection.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            collection.OUTPUT_DIR / collection.COLLECTION_REPORT
        ),
        "evaluation_report": str(collection.OUTPUT_DIR / EVALUATION_REPORT),
        "reference_sha256": REFERENCE_SHA256,
        "test_groups": {
            "counts": TEST_GROUP_COUNTS,
            "identity_sha256": TEST_GROUP_IDENTITIES,
        },
        "nltcs_groups": {
            "counts": NLTCS_GROUP_COUNTS,
            "identity_sha256": NLTCS_GROUP_IDENTITIES,
        },
        "baseline_alpha": BASELINE_ALPHA,
        "probe_alphas": list(PROBE_ALPHAS),
        "stable_measured_gain": {
            "probe_mean_lt_alpha16_mean": True,
            "paired_seed_wins_minimum": STABLE_WIN_MINIMUM,
            "paired_seed_count": len(collection.SEEDS),
        },
        "offline_risk_ratio_max": RISK_RATIO_MAX,
        "diversity_ratio_min": DIVERSITY_RATIO_MIN,
        "normalized_work_ratio_max": WORK_RATIO_MAX,
        "concentration_monotonic_metric": "tail_mean_last_at_most_100_rounds",
        "fixed_alpha_selection_allowed": False,
        "adaptive_alpha_design_allowed": False,
        "cross_dataset_or_cross_group_score_allowed": False,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _load_runtime() -> SimpleNamespace:
    import numpy as np
    import pandas as pd

    from scripts import compare_factorized_gibbs_closed_loop as offline
    from table_diffevo.marginals import load_marginals
    from table_diffevo.quality import (
        diversity_metrics,
        query_error_metrics,
        reference_support_metrics,
        schema_validity_metrics,
    )
    from table_diffevo.queries import evaluate_table, load_queries
    from table_diffevo.schema import load_schema

    return SimpleNamespace(
        np=np,
        pd=pd,
        offline=offline,
        load_marginals=load_marginals,
        diversity_metrics=diversity_metrics,
        query_error_metrics=query_error_metrics,
        reference_support_metrics=reference_support_metrics,
        schema_validity_metrics=schema_validity_metrics,
        evaluate_table=evaluate_table,
        load_queries=load_queries,
        load_schema=load_schema,
    )


def _table_path(
    root: Path, *, seed: int, dataset: str, alpha: float
) -> Path:
    return (
        root
        / collection.OUTPUT_DIR
        / f"seed_{seed}"
        / dataset
        / collection._alpha_label(alpha)
        / "terminal_current.csv"
    )


def _audit_collection(
    root: Path,
    confirmed_report_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, float], dict[str, Any]]]:
    report_path = root / collection.OUTPUT_DIR / collection.COLLECTION_REPORT
    observed_sha = collection._sha256_file(report_path)
    if observed_sha != confirmed_report_sha256:
        raise ValueError(
            "collection report SHA 与显式确认值不一致："
            f"confirmed={confirmed_report_sha256}, observed={observed_sha}"
        )
    report = _load_json(report_path)
    if (
        report.get("contract_version") != collection.PROTOCOL_VERSION
        or report.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != collection.frozen_protocol_manifest()
        or report.get("case_count") != 30
    ):
        raise RuntimeError("collection 协议或 30-case 身份漂移")
    if (
        report.get("raw_reference_data_accessed")
        or report.get("privacy_budget_consumed")
        or report.get("parameter_retuning_performed")
        or report.get("fixed_alpha_selection_allowed")
        or report.get("adaptive_alpha_design_in_scope")
    ):
        raise RuntimeError("collection 信息流或研究边界漂移")

    execution_commit = report.get("execution_git_commit")
    if not isinstance(execution_commit, str):
        raise TypeError("collection execution commit 缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if collection._git_text(
        root, "merge-base", execution_commit, current_commit
    ) != execution_commit:
        raise RuntimeError("collection execution commit 不是评价提交的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError("collection raw results 不完整")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["dataset"], float(row["alpha"]))
        if key in indexed:
            raise RuntimeError(f"collection case 重复：{key}")
        if (
            key[0] not in collection.SEEDS
            or key[1] not in collection.DATASETS
            or key[2] not in collection.ALPHAS
            or row.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256
            or row.get("git_commit") != execution_commit
        ):
            raise RuntimeError(f"collection case 身份漂移：{key}")
        spec = collection.DATASETS[key[1]]
        if (
            row.get("query_identity_sha256")
            != spec["query_identity_sha256"]
            or row.get("target_vector_sha256")
            != spec["target_vector_sha256"]
        ):
            raise RuntimeError(f"collection query/target 身份漂移：{key}")
        path = _table_path(
            root, seed=key[0], dataset=key[1], alpha=key[2]
        )
        if collection._sha256_file(path) != row["terminal_table_sha256"]:
            raise RuntimeError(f"collection terminal table SHA 漂移：{key}")
        indexed[key] = row
    expected = {
        (seed, dataset, alpha)
        for seed in collection.SEEDS
        for dataset, alpha in collection.CASE_ORDER
    }
    if set(indexed) != expected:
        raise RuntimeError("collection 30-case 身份不完整")
    return report, indexed


def _freeze_test_groups(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if (
        collection._sha256_file(root / TEST_IDENTITY_ARTIFACT)
        != TEST_IDENTITY_ARTIFACT_SHA256
    ):
        raise RuntimeError("test 查询身份 artifact SHA 漂移")
    marginals = _load_json(root / freeze_test.MARGINALS_PATH)
    old_measured = _load_json(root / freeze_test.WORKLOAD_A_PATH).get("queries")
    if not isinstance(old_measured, list):
        raise TypeError("test 旧查询身份输入缺失")
    frozen = freeze_test.freeze_query_identities(marginals, old_measured)
    groups = frozen["evaluation_groups"]
    counts = {name: len(groups[name]) for name in TEST_GROUP_ORDER}
    identities = {
        name: freeze_test.query_set_identity(groups[name])
        for name in TEST_GROUP_ORDER
    }
    if counts != TEST_GROUP_COUNTS or identities != TEST_GROUP_IDENTITIES:
        raise RuntimeError("test 离线查询组身份漂移")
    freeze_test._assert_result_free(groups)
    return groups, {
        "identity_frozen_before_reference_load": True,
        "counts": counts,
        "identity_sha256": identities,
    }


def _nltcs_one_way_queries(
    marginals: dict[str, Any], *, include_results: bool
) -> list[dict[str, Any]]:
    queries = []
    for attribute, specification in marginals["attributes"].items():
        values = specification["values"]
        counts = specification["counts"]
        for index, value in enumerate(values):
            query = {
                "conditions": [
                    {
                        "attribute": attribute,
                        "operator": "==",
                        "value": value,
                    }
                ]
            }
            if include_results:
                query["result"] = counts[index]
            queries.append(query)
    return queries


def _nltcs_enumerated_queries(
    marginals: dict[str, Any],
    measured_queries: Sequence[dict[str, Any]],
    order: int,
) -> list[dict[str, Any]]:
    attributes = list(marginals["attributes"])
    measured = {
        freeze_test.query_fingerprint(query)
        for query in measured_queries
        if len(query["conditions"]) == order
    }
    queries = []
    for names in itertools.combinations(attributes, order):
        domains = [marginals["attributes"][name]["values"] for name in names]
        for values in itertools.product(*domains):
            query = {
                "conditions": [
                    {
                        "attribute": name,
                        "operator": "==",
                        "value": value,
                    }
                    for name, value in zip(names, values)
                ]
            }
            if order == 3 and freeze_test.query_fingerprint(query) in measured:
                continue
            queries.append(query)
    return queries


def _freeze_nltcs_groups(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    marginals = _load_json(root / collection.DATASETS["nltcs"]["marginals"])
    measured = _load_json(root / collection.DATASETS["nltcs"]["queries"])[
        "queries"
    ]
    groups = {
        "one_way_safety": _nltcs_one_way_queries(
            marginals, include_results=True
        ),
        "unmeasured_3way": _nltcs_enumerated_queries(
            marginals, measured, order=3
        ),
        "all_4way": _nltcs_enumerated_queries(
            marginals, measured, order=4
        ),
    }
    counts = {name: len(value) for name, value in groups.items()}
    identities = {
        name: freeze_test.query_set_identity(value)
        for name, value in groups.items()
    }
    if counts != NLTCS_GROUP_COUNTS or identities != NLTCS_GROUP_IDENTITIES:
        raise RuntimeError("nltcs 离线查询组身份漂移")
    return groups, {
        "identity_frozen_before_reference_load": True,
        "counts": counts,
        "identity_sha256": identities,
    }


def _load_references(
    root: Path, runtime: SimpleNamespace
) -> tuple[dict[str, Any], dict[str, str]]:
    observed = {
        name: collection._sha256_file(root / path)
        for name, path in REFERENCE_PATHS.items()
    }
    if observed != REFERENCE_SHA256:
        raise RuntimeError("离线参考表 SHA 漂移")
    references = {}
    for name in collection.DATASET_ORDER:
        schema = runtime.load_schema(
            str(root / collection.DATASETS[name]["schema"])
        )
        columns = schema.attribute_names()
        path = root / REFERENCE_PATHS[name]
        if name == "test_300x10":
            frame = runtime.pd.read_csv(path)
            if list(frame.columns) != columns:
                frame.columns = columns
        else:
            frame = runtime.pd.read_csv(path, header=None, names=columns)
        frame = frame[columns]
        if len(frame) != collection.DATASETS[name]["n_records"]:
            raise RuntimeError(f"{name} reference 行数漂移")
        references[name] = frame
    return references, observed


def _with_count_errors(metrics: dict[str, Any], n_records: int) -> dict[str, Any]:
    result = dict(metrics)
    for suffix in ("mean", "median", "p90", "max"):
        result[f"absolute_count_error_{suffix}"] = float(
            metrics[f"normalized_l1_{suffix}"] * n_records
        )
    return result


def _offline_cell_metrics(
    metrics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    return {
        "query_count": int(metrics["n_queries"]),
        "normalized_l1_mean": float(metrics["mean"]),
        "normalized_l1_median": float(metrics["median"]),
        "normalized_l1_p90": float(metrics["p90"]),
        "normalized_l1_max": float(metrics["max"]),
        "absolute_count_error_mean": float(metrics["mean"] * n_records),
        "absolute_count_error_median": float(metrics["median"] * n_records),
        "absolute_count_error_p90": float(metrics["p90"] * n_records),
        "absolute_count_error_max": float(metrics["max"] * n_records),
    }


def _measured_metrics(
    runtime: SimpleNamespace,
    table: Any,
    queries: Sequence[dict[str, Any]],
    targets: Sequence[float],
    n_records: int,
) -> dict[str, Any]:
    answers = runtime.np.asarray(runtime.evaluate_table(table, list(queries)))
    target_array = runtime.np.asarray(targets, dtype=float)
    result = {
        "overall": _with_count_errors(
            runtime.query_error_metrics(target_array, answers, n_records),
            n_records,
        ),
        "by_order": {},
    }
    orders = sorted({len(query["conditions"]) for query in queries})
    for order in orders:
        indices = [
            index
            for index, query in enumerate(queries)
            if len(query["conditions"]) == order
        ]
        result["by_order"][str(order)] = _with_count_errors(
            runtime.query_error_metrics(
                target_array[indices], answers[indices], n_records
            ),
            n_records,
        )
    return result


def _evaluate_test_case(
    runtime: SimpleNamespace,
    table: Any,
    measured_queries: Sequence[dict[str, Any]],
    measured_targets: Sequence[float],
    groups: dict[str, list[dict[str, Any]]],
    group_targets: dict[str, Sequence[float]],
    schema: Any,
    reference: Any,
) -> dict[str, Any]:
    n_records = len(table)
    offline_groups = {}
    for name in TEST_GROUP_ORDER:
        answers = runtime.evaluate_table(table, groups[name])
        offline_groups[name] = _with_count_errors(
            runtime.query_error_metrics(
                group_targets[name], answers, n_records
            ),
            n_records,
        )
    return {
        "measured": _measured_metrics(
            runtime,
            table,
            measured_queries,
            measured_targets,
            n_records,
        ),
        "offline_query_groups": offline_groups,
        "validity": runtime.schema_validity_metrics(table, schema),
        "diversity": runtime.diversity_metrics(table, schema),
        "reference_support": runtime.reference_support_metrics(
            reference, table, schema
        ),
    }


def _evaluate_nltcs_case(
    runtime: SimpleNamespace,
    table: Any,
    measured_queries: Sequence[dict[str, Any]],
    measured_targets: Sequence[float],
    one_way_queries: Sequence[dict[str, Any]],
    schema: Any,
    marginals: dict[str, Any],
    domains: dict[str, Any],
    measured_triples: set[Any],
    reference: Any,
) -> dict[str, Any]:
    n_records = len(table)
    one_way_targets = [query["result"] for query in one_way_queries]
    one_way_answers = runtime.evaluate_table(table, list(one_way_queries))
    offline = runtime.offline._offline_metrics(
        reference,
        table,
        marginals,
        domains,
        measured_triples,
    )
    return {
        "measured": _measured_metrics(
            runtime,
            table,
            measured_queries,
            measured_targets,
            n_records,
        ),
        "offline_query_groups": {
            "one_way_safety": _with_count_errors(
                runtime.query_error_metrics(
                    one_way_targets, one_way_answers, n_records
                ),
                n_records,
            ),
            "unmeasured_3way": _offline_cell_metrics(
                offline["unmeasured_3way"], n_records
            ),
            "all_4way": _offline_cell_metrics(
                offline["unmeasured_4way"], n_records
            ),
        },
        "raw_joint": offline["raw_joint"],
        "binned_joint": offline["binned_joint"],
        "validity": runtime.schema_validity_metrics(table, schema),
        "diversity": runtime.diversity_metrics(table, schema),
        "reference_support": runtime.reference_support_metrics(
            reference, table, schema
        ),
    }


def _nested(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {path} 不是数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {path} 不是有限数")
    return value


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _summary(
    cases: Sequence[dict[str, Any]], dataset: str, alpha: float, path: str
) -> dict[str, Any]:
    selected = sorted(
        (
            case
            for case in cases
            if case["dataset"] == dataset and case["alpha"] == alpha
        ),
        key=lambda case: case["seed"],
    )
    if len(selected) != len(collection.SEEDS):
        raise RuntimeError(f"{dataset}/alpha{alpha}/{path} 缺少五种子")
    values = [_nested(case, path) for case in selected]
    return {
        "mean": _mean(values),
        "median": float(statistics.median(values)),
        "values_by_seed": {
            str(case["seed"]): value for case, value in zip(selected, values)
        },
    }


def _paired(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    candidate_alpha: float,
    baseline_alpha: float,
    path: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    indexed = {
        (case["seed"], case["alpha"]): _nested(case, path)
        for case in cases
        if case["dataset"] == dataset
    }
    differences = []
    candidate_values = []
    baseline_values = []
    for seed in collection.SEEDS:
        candidate = indexed[(seed, candidate_alpha)]
        baseline = indexed[(seed, baseline_alpha)]
        candidate_values.append(candidate)
        baseline_values.append(baseline)
        differences.append(candidate - baseline)
    better = [value < 0 if lower_is_better else value > 0 for value in differences]
    worse = [value > 0 if lower_is_better else value < 0 for value in differences]
    mean_difference = _mean(differences)
    difference_std = statistics.stdev(differences)
    half_width = 2.7764451051977987 * difference_std / math.sqrt(len(differences))
    candidate_mean = _mean(candidate_values)
    baseline_mean = _mean(baseline_values)
    return {
        "metric": path,
        "candidate_alpha": candidate_alpha,
        "baseline_alpha": baseline_alpha,
        "lower_is_better": lower_is_better,
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "mean_difference": mean_difference,
        "candidate_over_baseline": (
            candidate_mean / baseline_mean if baseline_mean != 0 else None
        ),
        "paired_wins": int(sum(better)),
        "paired_ties": int(sum(value == 0 for value in differences)),
        "paired_losses": int(sum(worse)),
        "differences_by_seed": {
            str(seed): value
            for seed, value in zip(collection.SEEDS, differences)
        },
        "difference_95pct_t_interval": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
    }


def _noninferior_ratio(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    alpha: float,
    path: str,
    maximum: float,
) -> tuple[bool, dict[str, Any]]:
    comparison = _paired(
        cases,
        dataset,
        alpha,
        BASELINE_ALPHA,
        path,
        lower_is_better=True,
    )
    baseline = comparison["baseline_mean"]
    candidate = comparison["candidate_mean"]
    passed = candidate == 0 if baseline == 0 else candidate / baseline <= maximum
    comparison["maximum_ratio"] = maximum
    comparison["pass"] = bool(passed)
    return bool(passed), comparison


def _classify_probe(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    alpha: float,
    all_normal: bool,
) -> dict[str, Any]:
    measured = _paired(
        cases,
        dataset,
        alpha,
        BASELINE_ALPHA,
        "metrics.measured.overall.normalized_l1_mean",
        lower_is_better=True,
    )
    stable_measured_gain = (
        measured["candidate_mean"] < measured["baseline_mean"]
        and measured["paired_wins"] >= STABLE_WIN_MINIMUM
    )
    safety_paths = {
        "test_300x10": {
            name: (
                f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            )
            for name in TEST_GROUP_ORDER
        },
        "nltcs": {
            "one_way_safety": (
                "metrics.offline_query_groups.one_way_safety.normalized_l1_mean"
            ),
            "unmeasured_3way": (
                "metrics.offline_query_groups.unmeasured_3way.normalized_l1_mean"
            ),
            "all_4way": (
                "metrics.offline_query_groups.all_4way.normalized_l1_mean"
            ),
            "binned_joint_tvd": "metrics.binned_joint.tvd",
        },
    }[dataset]
    safety = {}
    safety_pass = True
    for name, path in safety_paths.items():
        passed, comparison = _noninferior_ratio(
            cases, dataset, alpha, path, RISK_RATIO_MAX
        )
        safety[name] = comparison
        safety_pass = safety_pass and passed

    diversity = {}
    diversity_pass = True
    for name, path in {
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
    }.items():
        comparison = _paired(
            cases,
            dataset,
            alpha,
            BASELINE_ALPHA,
            path,
            lower_is_better=False,
        )
        baseline = comparison["baseline_mean"]
        candidate = comparison["candidate_mean"]
        passed = candidate >= DIVERSITY_RATIO_MIN * baseline
        comparison["minimum_ratio"] = DIVERSITY_RATIO_MIN
        comparison["pass"] = bool(passed)
        diversity[name] = comparison
        diversity_pass = diversity_pass and passed

    validity_values = [
        _nested(case, "metrics.validity.valid_row_rate")
        for case in cases
        if case["dataset"] == dataset and case["alpha"] == alpha
    ]
    validity_pass = len(validity_values) == len(collection.SEEDS) and all(
        value == 1.0 for value in validity_values
    )
    compute_pass, compute = _noninferior_ratio(
        cases,
        dataset,
        alpha,
        "normalized_work_at_stop",
        WORK_RATIO_MAX,
    )

    if not all_normal:
        classification = "inconclusive_resource_cap"
    elif not stable_measured_gain:
        classification = "no_stable_measured_gain"
    elif not (safety_pass and diversity_pass and validity_pass):
        classification = "measured_gain_with_quality_or_diversity_risk"
    elif not compute_pass:
        classification = "quality_supported_with_compute_tradeoff"
    else:
        classification = "supported_fixed_response_point"
    return {
        "classification": classification,
        "stable_measured_gain": stable_measured_gain,
        "measured": measured,
        "offline_safety_pass": safety_pass,
        "offline_safety": safety,
        "diversity_pass": diversity_pass,
        "diversity": diversity,
        "validity_pass": validity_pass,
        "validity_values": validity_values,
        "compute_pass": compute_pass,
        "compute": compute,
    }


def _direction_check(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    candidate_alpha: float,
    baseline_alpha: float,
    path: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    comparison = _paired(
        cases,
        dataset,
        candidate_alpha,
        baseline_alpha,
        path,
        lower_is_better=lower_is_better,
    )
    if lower_is_better:
        mean_direction = (
            comparison["candidate_mean"] <= comparison["baseline_mean"]
        )
    else:
        mean_direction = (
            comparison["candidate_mean"] >= comparison["baseline_mean"]
        )
    comparison["mean_direction_pass"] = bool(mean_direction)
    comparison["paired_direction_pass"] = (
        comparison["paired_wins"] >= STABLE_WIN_MINIMUM
    )
    comparison["pass"] = bool(
        comparison["mean_direction_pass"]
        and comparison["paired_direction_pass"]
    )
    return comparison


def _concentration_response(
    cases: Sequence[dict[str, Any]], dataset: str, all_normal: bool
) -> dict[str, Any]:
    effective_path = (
        "donor_concentration.effective_donor_fraction.tail_mean"
    )
    row_max_path = "donor_concentration.row_max_prob_mean.tail_mean"
    checks = {
        "effective_alpha12_ge_alpha16": _direction_check(
            cases,
            dataset,
            12.0,
            16.0,
            effective_path,
            lower_is_better=False,
        ),
        "effective_alpha16_ge_alpha24": _direction_check(
            cases,
            dataset,
            16.0,
            24.0,
            effective_path,
            lower_is_better=False,
        ),
        "row_max_alpha12_le_alpha16": _direction_check(
            cases,
            dataset,
            12.0,
            16.0,
            row_max_path,
            lower_is_better=True,
        ),
        "row_max_alpha16_le_alpha24": _direction_check(
            cases,
            dataset,
            16.0,
            24.0,
            row_max_path,
            lower_is_better=True,
        ),
    }
    if not all_normal:
        classification = "inconclusive_resource_cap"
    elif all(check["pass"] for check in checks.values()):
        classification = "concentration_response_monotonic"
    else:
        classification = "concentration_response_mixed"
    return {"classification": classification, "checks": checks}


def _build_summaries(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "measured_l1": "metrics.measured.overall.normalized_l1_mean",
        "measured_count_error": (
            "metrics.measured.overall.absolute_count_error_mean"
        ),
        "normalized_work": "normalized_work_at_stop",
        "rounds": "rounds_run",
        "row_max_prob_tail_mean": (
            "donor_concentration.row_max_prob_mean.tail_mean"
        ),
        "effective_donor_fraction_tail_mean": (
            "donor_concentration.effective_donor_fraction.tail_mean"
        ),
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
    }
    return {
        dataset: {
            collection._alpha_label(alpha): {
                name: _summary(cases, dataset, alpha, path)
                for name, path in paths.items()
            }
            for alpha in collection.ALPHAS
        }
        for dataset in collection.DATASET_ORDER
    }


def _frozen_classification(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    all_normal = all(
        case["termination_reason"] in NORMAL_REASONS for case in cases
    )
    probe_results = {
        dataset: {
            collection._alpha_label(alpha): _classify_probe(
                cases, dataset, alpha, all_normal
            )
            for alpha in PROBE_ALPHAS
        }
        for dataset in collection.DATASET_ORDER
    }
    concentration = {
        dataset: _concentration_response(cases, dataset, all_normal)
        for dataset in collection.DATASET_ORDER
    }
    supported = {
        dataset: {
            alpha
            for alpha in PROBE_ALPHAS
            if probe_results[dataset][collection._alpha_label(alpha)][
                "classification"
            ]
            == "supported_fixed_response_point"
        }
        for dataset in collection.DATASET_ORDER
    }
    shared = supported["test_300x10"].intersection(supported["nltcs"])
    opposite = (
        12.0 in supported["test_300x10"]
        and 24.0 in supported["nltcs"]
    ) or (
        24.0 in supported["test_300x10"]
        and 12.0 in supported["nltcs"]
    )
    if not all_normal:
        cross_dataset = "inconclusive_resource_cap"
    elif shared:
        cross_dataset = "shared_fixed_response_direction"
    elif opposite:
        cross_dataset = "dataset_dependent_fixed_response"
    else:
        cross_dataset = "mixed_fixed_response"
    return {
        "all_30_cases_normal": all_normal,
        "normal_case_count": sum(
            case["termination_reason"] in NORMAL_REASONS for case in cases
        ),
        "resource_cap_case_count": sum(
            case["termination_reason"] == "resource_cap_reached"
            for case in cases
        ),
        "fixed_probe_vs_alpha16": probe_results,
        "concentration_response": concentration,
        "supported_probe_alphas_by_dataset": {
            dataset: sorted(values) for dataset, values in supported.items()
        },
        "cross_dataset_response": cross_dataset,
        "fixed_alpha_selected": None,
        "adaptive_alpha_design": None,
    }


def evaluate(confirmed_collection_report_sha256: str) -> Path:
    if len(confirmed_collection_report_sha256) != 64:
        raise ValueError("必须显式确认完整 collection report SHA-256")
    root = collection._repo_root()
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式评价要求包含 untracked 在内的干净工作树")
    destination = root / collection.OUTPUT_DIR
    report_path = destination / EVALUATION_REPORT
    if report_path.exists():
        raise FileExistsError(f"评价报告已存在，不覆盖：{report_path}")

    collection_report, indexed = _audit_collection(
        root, confirmed_collection_report_sha256
    )
    test_groups, test_identity_audit = _freeze_test_groups(root)
    nltcs_groups, nltcs_identity_audit = _freeze_nltcs_groups(root)
    runtime = _load_runtime()
    references, reference_sha = _load_references(root, runtime)

    dataset_inputs = {}
    for dataset in collection.DATASET_ORDER:
        spec = collection.DATASETS[dataset]
        schema = runtime.load_schema(str(root / spec["schema"]))
        queries = runtime.load_queries(str(root / spec["queries"]))
        payload = _load_json(root / spec["queries"])
        targets = [query["result"] for query in payload["queries"]]
        dataset_inputs[dataset] = {
            "schema": schema,
            "queries": queries,
            "targets": targets,
        }

    test_group_targets = {
        name: runtime.evaluate_table(references["test_300x10"], queries)
        for name, queries in test_groups.items()
    }
    nltcs_marginals = _load_json(
        root / collection.DATASETS["nltcs"]["marginals"]
    )
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    nltcs_measured_triples = runtime.offline._measured_cell_keys(
        dataset_inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )

    cases = []
    for seed in collection.SEEDS:
        for dataset, alpha in collection.CASE_ORDER:
            source = indexed[(seed, dataset, alpha)]
            path = _table_path(root, seed=seed, dataset=dataset, alpha=alpha)
            table = runtime.pd.read_csv(path)
            inputs = dataset_inputs[dataset]
            if dataset == "test_300x10":
                metrics = _evaluate_test_case(
                    runtime,
                    table,
                    inputs["queries"],
                    inputs["targets"],
                    test_groups,
                    test_group_targets,
                    inputs["schema"],
                    references[dataset],
                )
            else:
                metrics = _evaluate_nltcs_case(
                    runtime,
                    table,
                    inputs["queries"],
                    inputs["targets"],
                    nltcs_groups["one_way_safety"],
                    inputs["schema"],
                    nltcs_marginals,
                    nltcs_domains,
                    nltcs_measured_triples,
                    references[dataset],
                )
            measured_l1 = metrics["measured"]["overall"]["normalized_l1_mean"]
            if measured_l1 != source["terminal_current_normalized_l1"]:
                raise RuntimeError(
                    f"{dataset}/alpha{alpha}/seed{seed} measured L1 复算漂移"
                )
            cases.append(
                {
                    "dataset": dataset,
                    "alpha": alpha,
                    "seed": seed,
                    "termination_reason": source["termination_reason"],
                    "rounds_run": source["rounds_run"],
                    "normalized_work_at_stop": source[
                        "normalized_work_at_stop"
                    ],
                    "terminal_table_sha256": source["terminal_table_sha256"],
                    "donor_concentration": source["donor_concentration"],
                    "metrics": metrics,
                }
            )
            print(
                f"[evaluate {dataset}/alpha={alpha:g}/seed={seed}] "
                f"L1={measured_l1:.10f}",
                flush=True,
            )

    report = {
        **build_plan(),
        "mode": "evaluate_frozen_collection_after_identity_audit",
        "evaluation_git_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_git_commit": collection_report[
            "execution_git_commit"
        ],
        "query_identity_frozen_before_reference_load": True,
        "query_identity_audit": {
            "test_300x10": test_identity_audit,
            "nltcs": nltcs_identity_audit,
        },
        "reference_sha256": reference_sha,
        "case_count": len(cases),
        "cases": cases,
        "summary": _build_summaries(cases),
        "frozen_classification": _frozen_classification(cases),
        "cross_dataset_or_cross_group_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "fixed_alpha_selection_performed": False,
        "adaptive_alpha_design_performed": False,
        "claim_scope": "fixed_alpha_response_calibration_only",
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".evaluation-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, report_path)
    return report_path


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
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = evaluate(args.confirm_collection_sha)
    print(f"固定 alpha evaluation -> {path}")
    print(f"evaluation SHA-256 -> {collection._sha256_file(path)}")


if __name__ == "__main__":
    main()
