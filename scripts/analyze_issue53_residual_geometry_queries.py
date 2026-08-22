#!/usr/bin/env python
"""Post-result, read-only query diagnostic for the Issue #53 residual matrix."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import compare_issue53_residual_geometry_earlystop as source
from scripts import run_issue53_p6_dataset_smoke as base
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.queries import evaluate_table, load_queries

CONTRACT_VERSION = "issue53-residual-geometry-query-diagnostic-v1"
SOURCE_REPORT_SHA256 = (
    "241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143"
)
SOURCE_EXECUTION_COMMIT = "fe8fb797a718bf0e9a89668d46fbd5726c1c3082"
SOURCE_REPORT = source.OUTPUT_DIR / "report.json"
OUTPUT_DIR = Path("outputs/issue53_residual_geometry_query_diagnostic_v1")

FLOOR_COUNT = 8
RARE_FREQUENCY = 0.05
COMMON_FREQUENCY = 0.20
FREQUENCY_BANDS = ("zero", "below_floor", "rare", "medium", "common")
OVERLAP_BANDS = ("low", "middle", "high")
PAIRWISE_COMPARISONS = (
    ("sqrt_relative", "absolute"),
    ("relative", "sqrt_relative"),
    ("relative", "absolute"),
)


def frequency_band(target_count: int, n_records: int) -> str:
    """Assign a fixed, result-independent target-frequency band."""

    if n_records <= 0:
        raise ValueError("n_records 必须为正")
    if target_count < 0:
        raise ValueError("target_count 不能为负")
    if target_count == 0:
        return "zero"
    if target_count < FLOOR_COUNT:
        return "below_floor"
    frequency = target_count / n_records
    if frequency < RARE_FREQUENCY:
        return "rare"
    if frequency < COMMON_FREQUENCY:
        return "medium"
    return "common"


def _query_features(
    queries: list[dict[str, Any]], n_records: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Build target/order/attribute-overlap features without reading outcomes."""

    if not queries:
        raise ValueError("queries 不能为空")
    attribute_sets: list[frozenset[str]] = []
    for query in queries:
        attributes = frozenset(
            str(condition["attribute"]) for condition in query["conditions"]
        )
        if not attributes:
            raise ValueError(f"查询 {query.get('id')} 没有属性")
        attribute_sets.append(attributes)

    overlap_scores = []
    for index, attributes in enumerate(attribute_sets):
        similarities = [
            len(attributes & other) / len(attributes | other)
            for other_index, other in enumerate(attribute_sets)
            if other_index != index
        ]
        overlap_scores.append(float(np.mean(similarities)) if similarities else 0.0)

    q25, q75 = np.quantile(np.asarray(overlap_scores), [0.25, 0.75])
    thresholds = {"q25": float(q25), "q75": float(q75)}
    if not len(queries) == len(attribute_sets) == len(overlap_scores):
        raise RuntimeError("query feature 数量漂移")
    features = []
    for index, (query, attributes, overlap) in enumerate(
        zip(queries, attribute_sets, overlap_scores)
    ):
        target = int(query["result"])
        if float(query["result"]) != target:
            raise ValueError(f"查询 {query.get('id')} target 不是整数计数")
        if overlap <= q25:
            overlap_band = "low"
        elif overlap >= q75:
            overlap_band = "high"
        else:
            overlap_band = "middle"
        features.append(
            {
                "query_index": index,
                "query_id": str(query["id"]),
                "query_type": str(query.get("type", "unknown")),
                "query_order": len(attributes),
                "attributes": "|".join(sorted(attributes)),
                "target_count": target,
                "target_frequency": target / n_records,
                "frequency_band": frequency_band(target, n_records),
                "structural_overlap_mean_jaccard": overlap,
                "structural_overlap_band": overlap_band,
            }
        )
    return features, thresholds


def _attach_fractional_win_credit(frame: pd.DataFrame) -> pd.DataFrame:
    """Give tied minimum-error arms equal credit for each query/seed."""

    result = frame.copy()
    group = result.groupby(["dataset", "query_index", "seed"], sort=False)["abs_error"]
    minimum = group.transform("min")
    is_winner = result["abs_error"].eq(minimum)
    winner_count = is_winner.groupby(
        [result["dataset"], result["query_index"], result["seed"]]
    ).transform("sum")
    result["fractional_win_credit"] = is_winner.astype(float) / winner_count
    return result


def _pairwise_summary(pivot: pd.DataFrame, n_records: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for candidate, baseline in PAIRWISE_COMPARISONS:
        delta = pivot[candidate] - pivot[baseline]
        summaries[f"{candidate}_minus_{baseline}"] = {
            "mean_abs_error_delta_count": float(delta.mean()),
            "mean_abs_error_delta_normalized": float(delta.mean() / n_records),
            "candidate_better_count": int((delta < 0).sum()),
            "tie_count": int((delta == 0).sum()),
            "candidate_worse_count": int((delta > 0).sum()),
        }
    return summaries


def _summarize_subset(
    frame: pd.DataFrame,
    *,
    n_records: int,
    total_query_count: int,
    dataset_arm_abs_error_sums: dict[str, float],
) -> dict[str, Any]:
    """Summarize one result-independent query stratum."""

    query_count = int(frame["query_index"].nunique())
    seed_count = int(frame["seed"].nunique())
    if query_count == 0 or seed_count == 0:
        raise ValueError("不能汇总空分层")
    expected_rows = query_count * seed_count * len(source.ARMS)
    if len(frame) != expected_rows:
        raise RuntimeError("分层中的 query/seed/arm 不完整")

    features = frame.drop_duplicates(["dataset", "query_index"])
    arms: dict[str, Any] = {}
    for arm in source.ARMS:
        arm_rows = frame[frame["arm"] == arm]
        abs_error_sum = float(arm_rows["abs_error"].sum())
        arms[arm] = {
            "mean_abs_error_count": float(arm_rows["abs_error"].mean()),
            "mean_abs_error_normalized": float(
                arm_rows["abs_error"].mean() / n_records
            ),
            "mean_dataset_l1_contribution": float(
                abs_error_sum / (seed_count * n_records * total_query_count)
            ),
            "share_of_arm_total_abs_error": float(
                abs_error_sum / dataset_arm_abs_error_sums[arm]
            ),
            "exact_match_rate": float((arm_rows["abs_error"] == 0).mean()),
            "fractional_query_seed_win_rate": float(
                arm_rows["fractional_win_credit"].sum() / (query_count * seed_count)
            ),
        }

    pivot = frame.pivot(
        index=["query_index", "seed"], columns="arm", values="abs_error"
    )
    if set(pivot.columns) != set(source.ARMS) or pivot.isna().any().any():
        raise RuntimeError("pairwise pivot 不完整")
    return {
        "query_count": query_count,
        "query_seed_count": query_count * seed_count,
        "target_count_min": int(features["target_count"].min()),
        "target_count_max": int(features["target_count"].max()),
        "target_frequency_mean": float(features["target_frequency"].mean()),
        "arms": arms,
        "pairwise": _pairwise_summary(pivot, n_records),
    }


def _group_summaries(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    n_records: int,
    total_query_count: int,
    dataset_arm_abs_error_sums: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for values, subset in frame.groupby(grouper, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        if len(columns) != len(values):
            raise RuntimeError("group columns 与 values 数量漂移")
        group_values = dict(zip(columns, values))
        key = "|".join(f"{name}={value}" for name, value in group_values.items())
        result[key] = {
            "group_values": group_values,
            **_summarize_subset(
                subset,
                n_records=n_records,
                total_query_count=total_query_count,
                dataset_arm_abs_error_sums=dataset_arm_abs_error_sums,
            ),
        }
    return result


def _query_summary(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "dataset",
        "query_index",
        "query_id",
        "query_type",
        "query_order",
        "attributes",
        "target_count",
        "target_frequency",
        "frequency_band",
        "structural_overlap_mean_jaccard",
        "structural_overlap_band",
    ]
    features = frame[feature_columns].drop_duplicates(["dataset", "query_index"])
    means = (
        frame.groupby(["dataset", "query_index", "arm"], sort=True)["abs_error"]
        .mean()
        .unstack("arm")
        .reset_index()
    )
    means = means.rename(columns={arm: f"{arm}_mean_abs_error" for arm in source.ARMS})
    result = features.merge(means, on=["dataset", "query_index"], validate="one_to_one")
    result["sqrt_relative_minus_absolute_mean_abs_error"] = (
        result["sqrt_relative_mean_abs_error"] - result["absolute_mean_abs_error"]
    )
    result["relative_minus_sqrt_relative_mean_abs_error"] = (
        result["relative_mean_abs_error"] - result["sqrt_relative_mean_abs_error"]
    )
    result["relative_minus_absolute_mean_abs_error"] = (
        result["relative_mean_abs_error"] - result["absolute_mean_abs_error"]
    )
    mean_columns = [f"{arm}_mean_abs_error" for arm in source.ARMS]

    def winners(row: pd.Series) -> str:
        minimum = min(float(row[column]) for column in mean_columns)
        return "|".join(
            arm
            for arm in source.ARMS
            if np.isclose(
                float(row[f"{arm}_mean_abs_error"]),
                minimum,
                rtol=0.0,
                atol=1e-12,
            )
        )

    result["mean_abs_error_winner_arms"] = result.apply(winners, axis=1)
    return result.sort_values(["dataset", "query_index"]).reset_index(drop=True)


def _audit_source_report(root: Path) -> dict[str, Any]:
    report_path = root / SOURCE_REPORT
    observed_sha = base._sha256_file(report_path)
    if observed_sha != SOURCE_REPORT_SHA256:
        raise RuntimeError(
            f"source report SHA 漂移：expected={SOURCE_REPORT_SHA256}, "
            f"observed={observed_sha}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["contract_version"] != source.PROTOCOL_VERSION:
        raise RuntimeError("source contract version 漂移")
    if report["protocol_sha256"] != source.FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("source protocol SHA 漂移")
    if report["execution_git_commit"] != SOURCE_EXECUTION_COMMIT:
        raise RuntimeError("source execution commit 漂移")
    if report["case_count"] != 18 or len(report["raw_results"]) != 18:
        raise RuntimeError("source 18-case 矩阵不完整")
    if report["raw_reference_data_accessed"] or report["privacy_budget_consumed"]:
        raise RuntimeError("source 信息流边界漂移")
    return report


def _build_error_frame(
    root: Path, report: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_results = {
        (row["dataset"], int(row["seed"]), row["arm"]): row
        for row in report["raw_results"]
    }
    expected = {
        (dataset, seed, arm)
        for dataset in source.DATASETS
        for seed in source.SEEDS
        for arm in source.ARMS
    }
    if set(raw_results) != expected:
        raise RuntimeError("source case identity 不完整或重复")

    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for dataset, spec in source.DATASETS.items():
        query_path = root / spec["queries"]
        query_sha = base._sha256_file(query_path)
        if query_sha != spec["sha256"]["queries"]:
            raise RuntimeError(f"{dataset} query SHA 漂移")
        queries = load_queries(str(query_path))
        if len(queries) != spec["query_count"]:
            raise RuntimeError(f"{dataset} query count 漂移")
        features, overlap_thresholds = _query_features(queries, spec["n_records"])
        target = np.asarray(
            [feature["target_count"] for feature in features], dtype=float
        )
        metadata[dataset] = {
            "query_sha256": query_sha,
            "overlap_quantile_thresholds": overlap_thresholds,
        }

        for seed in source.SEEDS:
            for arm in source.ARMS:
                source_row = raw_results[(dataset, seed, arm)]
                table_path = (
                    root
                    / source.OUTPUT_DIR
                    / f"seed_{seed}"
                    / dataset
                    / arm
                    / "terminal_current.csv"
                )
                if base._sha256_file(table_path) != source_row["terminal_table_sha256"]:
                    raise RuntimeError(f"{dataset}/{seed}/{arm} terminal SHA 漂移")
                table = pd.read_csv(table_path)
                if len(table) != spec["n_records"]:
                    raise RuntimeError(f"{dataset}/{seed}/{arm} 行数漂移")
                answers = np.asarray(evaluate_table(table, queries), dtype=float)
                observed_l1 = compute_normalized_l1(target, answers, spec["n_records"])
                if not np.isclose(
                    observed_l1,
                    source_row["terminal_current_normalized_l1"],
                    rtol=0.0,
                    atol=1e-15,
                ):
                    raise RuntimeError(f"{dataset}/{seed}/{arm} L1 复算漂移")
                if len(features) != len(answers):
                    raise RuntimeError(f"{dataset}/{seed}/{arm} answer 数量漂移")
                for feature, answer in zip(features, answers):
                    signed_error = int(answer) - feature["target_count"]
                    records.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "arm": arm,
                            **feature,
                            "terminal_answer": int(answer),
                            "signed_error": signed_error,
                            "abs_error": abs(signed_error),
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    frame = _attach_fractional_win_credit(frame)
    return frame.sort_values(["dataset", "query_index", "seed", "arm"]).reset_index(
        drop=True
    ), metadata


def _dataset_report(
    dataset: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    spec = source.DATASETS[dataset]
    subset = frame[frame["dataset"] == dataset]
    total_query_count = spec["query_count"]
    arm_sums = {
        arm: float(subset[subset["arm"] == arm]["abs_error"].sum())
        for arm in source.ARMS
    }
    overall = _summarize_subset(
        subset,
        n_records=spec["n_records"],
        total_query_count=total_query_count,
        dataset_arm_abs_error_sums=arm_sums,
    )
    crosscheck = {}
    for arm in source.ARMS:
        recomputed = overall["arms"][arm]["mean_dataset_l1_contribution"]
        source_mean = source_report["datasets"][dataset]["arms"][arm][
            "terminal_normalized_l1_mean"
        ]
        if not np.isclose(recomputed, source_mean, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"{dataset}/{arm} aggregate L1 复算漂移")
        crosscheck[arm] = {
            "source_mean_terminal_l1": source_mean,
            "recomputed_mean_terminal_l1": recomputed,
        }

    query_features = subset.drop_duplicates(["dataset", "query_index"])
    frequency_counts = {
        band: int((query_features["frequency_band"] == band).sum())
        for band in FREQUENCY_BANDS
    }
    order_counts = {
        str(int(order)): int(count)
        for order, count in query_features["query_order"].value_counts().items()
    }
    overlap_counts = {
        band: int((query_features["structural_overlap_band"] == band).sum())
        for band in OVERLAP_BANDS
    }
    common_kwargs = {
        "n_records": spec["n_records"],
        "total_query_count": total_query_count,
        "dataset_arm_abs_error_sums": arm_sums,
    }
    return {
        "n_records": spec["n_records"],
        "query_count": total_query_count,
        "query_sha256": metadata["query_sha256"],
        "target_count_min": int(query_features["target_count"].min()),
        "target_count_max": int(query_features["target_count"].max()),
        "frequency_band_counts": frequency_counts,
        "query_order_counts": dict(sorted(order_counts.items())),
        "structural_overlap_band_counts": overlap_counts,
        "structural_overlap_quantile_thresholds": metadata[
            "overlap_quantile_thresholds"
        ],
        "overall": overall,
        "by_frequency_band": _group_summaries(
            subset, ["frequency_band"], **common_kwargs
        ),
        "by_query_order": _group_summaries(subset, ["query_order"], **common_kwargs),
        "by_structural_overlap_band": _group_summaries(
            subset, ["structural_overlap_band"], **common_kwargs
        ),
        "by_frequency_and_order": _group_summaries(
            subset, ["frequency_band", "query_order"], **common_kwargs
        ),
        "by_frequency_and_overlap": _group_summaries(
            subset,
            ["frequency_band", "structural_overlap_band"],
            **common_kwargs,
        ),
        "source_l1_crosscheck": crosscheck,
    }


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "mode": "post_result_read_only_diagnostic",
        "source_report": str(SOURCE_REPORT),
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "output_dir": str(OUTPUT_DIR),
        "frequency_bands": {
            "zero": "target == 0",
            "below_floor": f"0 < target < {FLOOR_COUNT}",
            "rare": f"target >= {FLOOR_COUNT} and target/n_records < {RARE_FREQUENCY}",
            "medium": (f"{RARE_FREQUENCY} <= target/n_records < {COMMON_FREQUENCY}"),
            "common": f"target/n_records >= {COMMON_FREQUENCY}",
        },
        "query_order": "number_of_distinct_condition_attributes",
        "structural_overlap": (
            "mean_attribute_set_jaccard_against_all_other_workload_queries; "
            "low<=input_q25, high>=input_q75, otherwise middle"
        ),
        "dimensions": [
            "frequency_band",
            "query_order",
            "structural_overlap_band",
            "frequency_band_x_query_order",
            "frequency_band_x_structural_overlap_band",
        ],
        "arms": list(source.ARMS),
        "seeds": list(source.SEEDS),
        "generation_started": False,
        "raw_reference_data_accessed": False,
        "canonical_selection_allowed": False,
        "scientific_overrides_allowed": False,
    }


def run_analysis(confirmed_source_report_sha256: str) -> Path:
    if confirmed_source_report_sha256 != SOURCE_REPORT_SHA256:
        raise ValueError("必须显式确认完整 source report SHA-256")
    root = base._repo_root()
    if base._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("诊断要求包含 untracked 在内的干净工作树")
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"诊断输出已存在，不覆盖：{destination}")

    source_report = _audit_source_report(root)
    errors, metadata = _build_error_frame(root, source_report)
    query_summary = _query_summary(errors)
    report = {
        **build_plan(),
        "analysis_git_commit": base._git_text(root, "rev-parse", "HEAD"),
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "source_protocol_sha256": source.FROZEN_PROTOCOL_SHA256,
        "post_result_diagnostic": True,
        "thresholds_selected_without_query_level_outcomes": True,
        "new_generation_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "datasets": {
            dataset: _dataset_report(
                dataset,
                errors,
                metadata[dataset],
                source_report,
            )
            for dataset in source.DATASETS
        },
        "claim_scope": (
            "result_aware_development_mechanism_diagnostic_not_selection_evidence"
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".issue53-query-diagnostic.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        error_path = temporary / "query_seed_errors.csv"
        summary_path = temporary / "query_summary.csv"
        errors.to_csv(error_path, index=False)
        query_summary.to_csv(summary_path, index=False)
        report["artifacts"] = {
            "query_seed_errors": {
                "path": "query_seed_errors.csv",
                "row_count": len(errors),
                "sha256": base._sha256_file(error_path),
            },
            "query_summary": {
                "path": "query_summary.csv",
                "row_count": len(query_summary),
                "sha256": base._sha256_file(summary_path),
            },
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-source-report-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = run_analysis(args.confirm_source_report_sha)
    print(f"query diagnostic -> {path}")
    print(f"report SHA-256 -> {base._sha256_file(path)}")


if __name__ == "__main__":
    main()
