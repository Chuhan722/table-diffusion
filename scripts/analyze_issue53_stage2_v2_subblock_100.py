#!/usr/bin/env python
"""Issue #53 V2 的单一 100 轮小块假设审查。

本入口只读取既有的 12 条 development 轨迹，并且只检查事先声明的
``100`` 轮小块。它不比较或选择其他小块长度，不含阈值、收敛分类、候选停止
轮次、B4 确认或在线停止。

每条 8000 轮轨迹先形成 80 个完整小块，再按四个小块向前移动，形成终点为
1200、1600、...、8000 的 18 组十二小块证据。报告描述：零尺度/无穷证据、
去趋势残差的相邻相关、R/D/S/T/O 的自然量程，以及实际运动与表结构水平。
这些量只能发现 ``100`` 的明显反例，不能证明它对任意未来数据都普适。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from scripts import analyze_issue53_stage2b_range_finding as range_analyzer
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import analyze_issue53_stage2b_range_finding as range_analyzer

from table_diffevo.stationarity import load_stationarity_trace
from table_diffevo.stationarity_v2 import (
    STATIONARITY_V2_CANDIDATE_EVIDENCE_CONTRACT_VERSION,
    STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION,
    V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE,
    V2_SCALAR_EVIDENCE_POINT_COUNT,
    V2_SUBBLOCKS_PER_BLOCK,
    V2CandidateEvidence,
    V2QueryDistributionSummary,
    V2ScalarEvidence,
    collect_v2_subblock_summaries,
    compute_v2_candidate_evidence,
)


V2_SUBBLOCK_100_AUDIT_CONTRACT_VERSION = (
    "issue53-stage2-v2-subblock-100-development-audit-v1"
)
SUBBLOCK_ROUND_COUNT = V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE
EXPECTED_POST_ROUND_COUNT = 8000
EXPECTED_COMPLETE_SUBBLOCK_COUNT = 80
EXPECTED_CANDIDATE_COUNT_PER_TRACE = 18
EXPECTED_TOTAL_CANDIDATE_COUNT = 216

SCALAR_EVIDENCE_FIELDS: Dict[str, str] = {
    "l1_level": "当前归一化 L1 的小块均值",
    "l1_spread": "小块内 L1 的 P90-P10 宽度",
    "unique_row_rate": "唯一行比例",
    "normalized_row_entropy": "归一化行熵",
    "active_round_rate": "实际发生记录变化的轮次比例",
    "changed_row_fraction": "平均改变记录比例",
    "changed_query_fraction": "平均改变查询比例",
    "normalized_query_movement": "平均归一化查询 L1 运动量",
}

SUMMARY_METRICS = (
    "query_absolute_direction_change_finite_mean",
    "query_absolute_direction_change_finite_p95",
    "query_absolute_direction_change_finite_maximum",
    "query_trend_strength_finite_mean",
    "query_trend_strength_finite_p95",
    "query_trend_strength_positive_infinity_fraction",
    "query_outlier_strength_finite_mean",
    "query_outlier_strength_finite_p95",
    "query_outlier_strength_positive_infinity_fraction",
    "query_zero_scale_fraction",
    "query_residual_lag1_absolute_p95",
    "query_residual_lag1_undefined_fraction",
    "l1_level_trend_strength_finite",
    "l1_level_outlier_strength_finite",
    "l1_level_residual_lag1_correlation",
    "l1_spread_trend_strength_finite",
    "l1_spread_outlier_strength_finite",
    "l1_spread_residual_lag1_correlation",
    "unique_row_rate_reference_level",
    "normalized_row_entropy_reference_level",
    "active_round_rate_reference_level",
    "changed_row_fraction_reference_level",
    "changed_query_fraction_reference_level",
    "normalized_query_movement_reference_level",
)

FORBIDDEN_DECISION_FIELDS = {
    "stable",
    "converged",
    "stalled",
    "status",
    "classification",
    "candidate_stop_round",
    "stop_round",
    "threshold",
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def candidate_first_subblock_numbers(
    complete_subblock_count: int,
) -> tuple[int, ...]:
    """Return all twelve-subblock candidates on four-subblock boundaries."""

    if isinstance(complete_subblock_count, bool) or not isinstance(
        complete_subblock_count, (int, np.integer)
    ):
        raise ValueError("complete_subblock_count must be an integer")
    count = int(complete_subblock_count)
    if count < V2_SCALAR_EVIDENCE_POINT_COUNT:
        return ()
    last_start = count - V2_SCALAR_EVIDENCE_POINT_COUNT + 1
    return tuple(range(1, last_start + 1, V2_SUBBLOCKS_PER_BLOCK))


def residual_lag_one_correlation(
    evidence: V2ScalarEvidence,
) -> float | None:
    """Pearson correlation of adjacent detrended residuals.

    The robust line used by the V2 primitive is removed first because a long
    trend is already represented by D/T.  Correlation is undefined when either
    side has exactly zero variation; this case is returned as ``None`` rather
    than filled with zero or an epsilon.
    """

    if not isinstance(evidence, V2ScalarEvidence):
        raise ValueError("evidence must be V2ScalarEvidence")
    residuals = np.asarray(evidence.residuals, dtype=np.float64)
    left = residuals[:-1]
    right = residuals[1:]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    ))
    if denominator == 0.0:
        return None
    correlation = float(np.dot(left_centered, right_centered) / denominator)
    if not np.isfinite(correlation):
        raise RuntimeError("finite residuals produced non-finite correlation")
    return float(np.clip(correlation, -1.0, 1.0))


def _summarize_optional_correlations(
    values: Sequence[float | None],
) -> Dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    if finite.size and not np.all(np.isfinite(finite)):
        raise RuntimeError("correlation summary received a non-finite value")
    total_count = len(values)
    result: Dict[str, float | int | None] = {
        "value_count": total_count,
        "finite_count": int(finite.size),
        "undefined_count": total_count - int(finite.size),
        "undefined_fraction": (
            (total_count - int(finite.size)) / total_count
            if total_count else None
        ),
    }
    names_and_values: Dict[str, float | None]
    if finite.size:
        absolute = np.abs(finite)
        names_and_values = {
            "signed_minimum": float(np.min(finite)),
            "signed_p05": float(np.percentile(finite, 5, method="linear")),
            "signed_median": float(np.median(finite)),
            "signed_mean": float(np.mean(finite)),
            "signed_p95": float(np.percentile(finite, 95, method="linear")),
            "signed_maximum": float(np.max(finite)),
            "absolute_median": float(np.median(absolute)),
            "absolute_p95": float(
                np.percentile(absolute, 95, method="linear")
            ),
            "absolute_maximum": float(np.max(absolute)),
        }
    else:
        names_and_values = {
            "signed_minimum": None,
            "signed_p05": None,
            "signed_median": None,
            "signed_mean": None,
            "signed_p95": None,
            "signed_maximum": None,
            "absolute_median": None,
            "absolute_p95": None,
            "absolute_maximum": None,
        }
    result.update(names_and_values)
    return result


def _add_query_distribution(
    row: Dict[str, Any],
    prefix: str,
    summary: V2QueryDistributionSummary,
) -> None:
    for field_name in (
        "value_count",
        "finite_count",
        "positive_infinity_count",
        "finite_mean",
        "finite_p95",
        "finite_maximum",
        "finite_maximum_query_index",
        "first_positive_infinity_query_index",
    ):
        row[f"{prefix}_{field_name}"] = getattr(summary, field_name)
    row[f"{prefix}_positive_infinity_fraction"] = (
        summary.positive_infinity_count / summary.value_count
    )


def _add_scalar_evidence(
    row: Dict[str, Any],
    prefix: str,
    evidence: V2ScalarEvidence,
) -> None:
    row[f"{prefix}_reference_level"] = evidence.R
    row[f"{prefix}_direction_change"] = evidence.D
    row[f"{prefix}_residual_scale"] = evidence.S
    row[f"{prefix}_zero_scale"] = evidence.zero_scale
    row[f"{prefix}_trend_strength_finite"] = (
        evidence.T if np.isfinite(evidence.T) else None
    )
    row[f"{prefix}_trend_strength_positive_infinity"] = bool(
        np.isposinf(evidence.T)
    )
    row[f"{prefix}_outlier_strength_finite"] = (
        evidence.O if np.isfinite(evidence.O) else None
    )
    row[f"{prefix}_outlier_strength_positive_infinity"] = bool(
        np.isposinf(evidence.O)
    )
    correlation = residual_lag_one_correlation(evidence)
    row[f"{prefix}_residual_lag1_correlation"] = correlation
    row[f"{prefix}_residual_lag1_correlation_defined"] = (
        correlation is not None
    )


def flatten_candidate_evidence(
    item: range_analyzer.AuditedTraceInput,
    evidence: V2CandidateEvidence,
) -> Dict[str, Any]:
    """Flatten one threshold-free candidate without inventing a verdict."""

    if evidence.contract_version != (
        STATIONARITY_V2_CANDIDATE_EVIDENCE_CONTRACT_VERSION
    ):
        raise RuntimeError("V2 candidate evidence contract 不一致")
    row: Dict[str, Any] = {
        "dataset": item.dataset,
        "kernel": item.kernel,
        "seed": int(item.seed),
        "query_count": int(evidence.query_count),
        "n_records": int(evidence.n_records),
        "subblock_round_count": int(evidence.subblock_round_count),
        "first_subblock_number": int(evidence.first_subblock_number),
        "last_subblock_number": int(evidence.last_subblock_number),
        "start_round_index": int(evidence.start_round_index),
        "end_round_index": int(evidence.end_round_index),
        "query_zero_scale_count": int(evidence.query_zero_scale_count),
        "query_zero_scale_fraction": (
            evidence.query_zero_scale_count / evidence.query_count
        ),
    }
    _add_query_distribution(
        row,
        "query_absolute_direction_change",
        evidence.query_absolute_direction_change,
    )
    _add_query_distribution(
        row, "query_trend_strength", evidence.query_trend_strength
    )
    _add_query_distribution(
        row, "query_outlier_strength", evidence.query_outlier_strength
    )

    query_correlations = [
        residual_lag_one_correlation(query_evidence)
        for query_evidence in evidence.per_query_evidence
    ]
    for name, value in _summarize_optional_correlations(
        query_correlations
    ).items():
        row[f"query_residual_lag1_{name}"] = value

    scalar_objects = {
        "l1_level": evidence.l1_level_evidence,
        "l1_spread": evidence.l1_spread_evidence,
        "unique_row_rate": evidence.unique_row_rate_evidence,
        "normalized_row_entropy": (
            evidence.normalized_row_entropy_evidence
        ),
        "active_round_rate": evidence.active_round_rate_evidence,
        "changed_row_fraction": evidence.changed_row_fraction_evidence,
        "changed_query_fraction": evidence.changed_query_fraction_evidence,
        "normalized_query_movement": (
            evidence.normalized_query_movement_evidence
        ),
    }
    for prefix, scalar_evidence in scalar_objects.items():
        _add_scalar_evidence(row, prefix, scalar_evidence)

    if FORBIDDEN_DECISION_FIELDS.intersection(row):
        raise RuntimeError("V2 100 轮证据意外包含判定字段")
    _strict_json_bytes(row)
    return row


def build_candidate_rows(
    inputs: Sequence[range_analyzer.AuditedTraceInput],
) -> List[Dict[str, Any]]:
    """Build the fixed 100-round, 18-candidate evidence table."""

    if SUBBLOCK_ROUND_COUNT != 100:
        raise RuntimeError("本审查协议必须且只能使用 100 轮小块")
    rows: List[Dict[str, Any]] = []
    for item in inputs:
        trace = load_stationarity_trace(item.run_dir / "trace")
        collection = collect_v2_subblock_summaries(
            trace, subblock_round_count=SUBBLOCK_ROUND_COUNT
        )
        if collection.contract_version != (
            STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION
        ):
            raise RuntimeError("V2 subblock collection contract 不一致")
        if (
            collection.post_round_count != EXPECTED_POST_ROUND_COUNT
            or collection.complete_subblock_count
            != EXPECTED_COMPLETE_SUBBLOCK_COUNT
            or collection.trailing_post_round_count != 0
        ):
            raise RuntimeError("正式轨迹没有形成恰好 80 个完整 100 轮小块")
        starts = candidate_first_subblock_numbers(
            collection.complete_subblock_count
        )
        if len(starts) != EXPECTED_CANDIDATE_COUNT_PER_TRACE:
            raise RuntimeError("每条轨迹必须恰好形成 18 个候选证据")
        rows.extend(
            flatten_candidate_evidence(
                item,
                compute_v2_candidate_evidence(
                    collection, first_subblock_number=start
                ),
            )
            for start in starts
        )

    if len(rows) != EXPECTED_TOTAL_CANDIDATE_COUNT:
        raise RuntimeError("正式开发审查必须恰好包含 216 行候选证据")
    grouped: Dict[tuple[str, str, int], List[int]] = {}
    for row in rows:
        key = (row["dataset"], row["kernel"], row["seed"])
        grouped.setdefault(key, []).append(int(row["end_round_index"]))
    expected_endpoints = list(range(1200, 8001, 400))
    if len(grouped) != len(inputs) or any(
        endpoints != expected_endpoints for endpoints in grouped.values()
    ):
        raise RuntimeError("候选终点必须逐轨迹严格为 1200..8000、步长 400")
    return rows


def _finite_column_summary(
    rows: Sequence[Dict[str, Any]], field_name: str
) -> Dict[str, float | int | None]:
    raw = [row[field_name] for row in rows]
    finite = np.asarray(
        [value for value in raw if value is not None], dtype=np.float64
    )
    if finite.size and not np.all(np.isfinite(finite)):
        raise RuntimeError(f"{field_name} 包含非有限普通数值")
    result: Dict[str, float | int | None] = {
        "row_count": len(raw),
        "finite_count": int(finite.size),
        "explicitly_absent_count": len(raw) - int(finite.size),
    }
    if finite.size:
        result.update({
            "minimum": float(np.min(finite)),
            "p05": float(np.percentile(finite, 5, method="linear")),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95, method="linear")),
            "maximum": float(np.max(finite)),
        })
    else:
        result.update({
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        })
    return result


def build_descriptive_summary(
    rows: Sequence[Dict[str, Any]], source_audit: Dict[str, Any]
) -> Dict[str, Any]:
    if len(rows) != EXPECTED_TOTAL_CANDIDATE_COUNT:
        raise RuntimeError("描述性汇总要求完整的 216 行证据")
    cells: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["dataset"], row["kernel"]), []).append(row)
    if len(cells) != 4 or any(len(cell_rows) != 54 for cell_rows in cells.values()):
        raise RuntimeError("四个 dataset×kernel cell 必须各有 54 行证据")

    cell_summaries = []
    for (dataset, kernel), cell_rows in sorted(cells.items()):
        cell_summaries.append({
            "dataset": dataset,
            "kernel": kernel,
            "query_count": sorted({row["query_count"] for row in cell_rows}),
            "candidate_row_count": len(cell_rows),
            "metrics": {
                metric: _finite_column_summary(cell_rows, metric)
                for metric in SUMMARY_METRICS
            },
        })

    result = {
        "contract_version": V2_SUBBLOCK_100_AUDIT_CONTRACT_VERSION,
        "role": {
            "purpose": "falsification_audit_of_one_predeclared_subblock_length",
            "tested_subblock_round_counts": [100],
            "alternative_length_selection": False,
            "dataset_specific_window_rule": False,
            "threshold_parameters": "absent",
            "convergence_or_stall_classification": "absent",
            "candidate_stop_round": "absent",
            "b4_confirmation": "absent",
            "online_stop": False,
            "generator_rerun": False,
            "validation_seed_access": False,
        },
        "source_audit": source_audit,
        "candidate_schedule": {
            "subblock_round_count": 100,
            "subblocks_per_candidate": 12,
            "candidate_stride_subblocks": 4,
            "candidate_end_rounds": list(range(1200, 8001, 400)),
            "candidate_count_per_trace": 18,
            "trajectory_count": 12,
            "candidate_row_count": 216,
        },
        "dependence_audit": {
            "quantity": "lag_one_pearson_correlation_of_detrended_residuals",
            "trend_removed_by": "theil_sen_median_pairwise_slope_and_median_intercept",
            "undefined_rule": (
                "return_null_and_count_explicitly_when_either_adjacent_"
                "residual_vector_has_exactly_zero_variation"
            ),
            "acceptance_threshold": "absent",
        },
        "zero_scale_audit": {
            "epsilon_added": False,
            "finite_and_positive_infinity_values_separated": True,
        },
        "query_count_boundary": {
            "raw_query_maximum_retained_for_diagnosis": True,
            "query_count_maximum_correction": "not_yet_defined",
            "raw_maximum_must_not_set_a_cross_dataset_threshold": True,
        },
        "metric_descriptions_zh": SCALAR_EVIDENCE_FIELDS,
        "cell_summaries": cell_summaries,
        "interpretation_limit": (
            "本报告只能发现 100 轮假设在现有 development 轨迹上的明显反例；"
            "没有反例不等于证明对任意数据普适。"
        ),
    }
    _strict_json_bytes(result)
    return result


def _write_csv_exclusive(
    path: Path, rows: Sequence[Dict[str, Any]]
) -> None:
    if not rows:
        raise RuntimeError("candidate evidence CSV 不得为空")
    fieldnames = list(rows[0])
    expected = set(fieldnames)
    if any(set(row) != expected for row in rows):
        raise RuntimeError("candidate evidence 行字段不一致")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    """Publish one immutable, threshold-free development audit."""

    if output_dir.exists():
        raise FileExistsError(f"V2 100 轮审查目录已存在，拒绝覆盖：{output_dir}")
    environment = range_analyzer.analysis_environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 V2 100 轮审查要求干净工作树")
    inputs, source_audit = range_analyzer.audit_formal_inputs(input_dir)
    rows = build_candidate_rows(inputs)
    summary = build_descriptive_summary(rows, source_audit)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.partial-",
    ))
    try:
        evidence_path = temporary / "candidate_evidence.csv"
        summary_path = temporary / "audit_summary.json"
        _write_csv_exclusive(evidence_path, rows)
        _write_json_exclusive(summary_path, summary)
        artifacts = {
            path.name: {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (evidence_path, summary_path)
        }
        manifest = {
            "contract_version": V2_SUBBLOCK_100_AUDIT_CONTRACT_VERSION,
            "formal_threshold_free_development_audit": True,
            "tested_subblock_round_counts": [100],
            "source_audit": source_audit,
            "analysis_environment": environment,
            "artifacts": artifacts,
        }
        _write_json_exclusive(temporary / "report_manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_plan(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    result = {
        "contract_version": V2_SUBBLOCK_100_AUDIT_CONTRACT_VERSION,
        "mode": "plan_only_no_trace_read",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tested_subblock_round_counts": [100],
        "alternative_length_selection": False,
        "dataset_specific_window_rule": False,
        "expected_trajectory_count": 12,
        "expected_candidate_count_per_trajectory": 18,
        "threshold_parameters_present": False,
        "classification_output_present": False,
        "validation_seeds_may_be_read": False,
    }
    _strict_json_bytes(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "report"), default="plan"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/issue53_stage2b_range_finding"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/issue53_stage2_v2_subblock_100_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "plan":
        print(json.dumps(
            build_plan(args.input_dir, args.output_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    destination = generate_report(args.input_dir, args.output_dir)
    print(f"V2 fixed-100 audit -> {destination}", flush=True)


if __name__ == "__main__":
    main()
