#!/usr/bin/env python
"""Issue #53 Stage 2B 最坏查询漂移的无阈值开发量程审查。

本入口只读取既有的 12 条 development 轨迹，并只计算 ``W=400`` 下完整落在
``6001..8000`` 的三个既定检查点。它为每组三窗口保留现有 query mean/P95
证据，同时描述所有查询坐标中的最大窗口均值漂移。

脚本没有阈值、稳定性分类、候选停止或 detector 配置入口，不读取 validation
轨迹，也不重跑生成器。默认 ``plan`` 不读取轨迹；``report`` 要求干净工作树并
以不可覆盖、目录级原子方式发布 36 行原始证据和描述性量程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from scripts import analyze_issue53_stage2b_range_finding as range_analyzer
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import analyze_issue53_stage2b_range_finding as range_analyzer

from table_diffevo.stationarity import (
    StationarityTrace,
    collect_stationarity_range_evidence,
    load_stationarity_trace,
)


QUERY_MAX_RANGE_REPORT_CONTRACT_VERSION = (
    "issue53-stage2b-query-max-range-report-v1"
)
WINDOW_SIZE = 400
CALIBRATION_ROUND_START = 6001
CALIBRATION_ROUND_END = 8000
TERMINAL_ROUNDS = (7200, 7600, 8000)
EXPECTED_CHECKS_PER_TRAJECTORY = 3
WINDOW_PAIRS = ((0, 1), (0, 2), (1, 2))


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


def compute_query_max_evidence(
    trace: StationarityTrace,
    window_round_ranges: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Describe per-query window-mean drift for exactly three windows."""
    trace.validate()
    if len(window_round_ranges) != 3:
        raise ValueError("最坏查询漂移证据必须恰好比较三个窗口")

    round_to_position = {
        int(trace.observations[position]["round_index"]): position
        for position in trace.post_round_positions()
    }
    window_means: List[np.ndarray] = []
    normalized_ranges: List[List[int]] = []
    for raw_range in window_round_ranges:
        if len(raw_range) != 2:
            raise ValueError("每个窗口必须提供起止轮次")
        start, end = (int(raw_range[0]), int(raw_range[1]))
        if start <= 0 or end < start:
            raise ValueError("窗口轮次范围非法")
        rounds = list(range(start, end + 1))
        if len(rounds) != WINDOW_SIZE:
            raise ValueError(f"每个窗口必须恰好包含 {WINDOW_SIZE} 个 post_round")
        try:
            positions = [round_to_position[round_index] for round_index in rounds]
        except KeyError as exc:
            raise ValueError("轨迹缺少窗口要求的 post_round") from exc
        answers = trace.measured_query_answers[np.asarray(positions)]
        window_means.append(np.mean(answers / trace.n_records, axis=0))
        normalized_ranges.append([start, end])

    pairwise_rows: List[Dict[str, Any]] = []
    for left, right in WINDOW_PAIRS:
        shift = np.abs(window_means[left] - window_means[right])
        maximum_index = int(np.argmax(shift))
        pairwise_rows.append({
            "left_window": left + 1,
            "right_window": right + 1,
            "query_mean_shift": float(np.mean(shift)),
            "query_p95_shift": float(
                np.percentile(shift, 95, method="linear")
            ),
            "query_max_shift": float(shift[maximum_index]),
            "max_query_index": maximum_index,
        })

    maximum_pair = max(
        pairwise_rows,
        key=lambda row: (
            row["query_max_shift"],
            -row["left_window"],
            -row["right_window"],
            -row["max_query_index"],
        ),
    )
    result = {
        "window_round_ranges": normalized_ranges,
        "query_mean_shift": max(
            row["query_mean_shift"] for row in pairwise_rows
        ),
        "query_p95_shift": max(
            row["query_p95_shift"] for row in pairwise_rows
        ),
        "query_max_shift": maximum_pair["query_max_shift"],
        "max_query_index": maximum_pair["max_query_index"],
        "max_shift_window_pair": [
            maximum_pair["left_window"],
            maximum_pair["right_window"],
        ],
        "pairwise": pairwise_rows,
    }
    _strict_json_bytes(result)
    return result


def _selected_checks(trace: StationarityTrace) -> List[Dict[str, Any]]:
    checks = collect_stationarity_range_evidence(trace, [WINDOW_SIZE])
    selected = [
        check
        for check in checks
        if int(check["round_index"]) in TERMINAL_ROUNDS
        and int(check["window_round_ranges"][0][0])
        >= CALIBRATION_ROUND_START
        and int(check["window_round_ranges"][-1][-1])
        <= CALIBRATION_ROUND_END
    ]
    if len(selected) != EXPECTED_CHECKS_PER_TRAJECTORY:
        raise RuntimeError("每条轨迹必须恰好贡献三个完整晚期检查")
    if tuple(int(row["round_index"]) for row in selected) != TERMINAL_ROUNDS:
        raise RuntimeError("晚期检查终止轮次不符合固定审查范围")
    return selected


def _flatten_check(
    item: range_analyzer.AuditedTraceInput,
    trace: StationarityTrace,
    check: Dict[str, Any],
) -> Dict[str, Any]:
    evidence = compute_query_max_evidence(
        trace, check["window_round_ranges"]
    )
    if not np.isclose(
        evidence["query_mean_shift"],
        check["query_mean_shift"],
        rtol=0.0,
        atol=1e-15,
    ) or not np.isclose(
        evidence["query_p95_shift"],
        check["query_p95_shift"],
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("新增量程与现有 query mean/P95 公式不一致")

    row: Dict[str, Any] = {
        "dataset": item.dataset,
        "kernel": item.kernel,
        "seed": int(item.seed),
        "query_count": int(trace.query_count),
        "window_size": WINDOW_SIZE,
        "round_index": int(check["round_index"]),
        "query_mean_shift": float(evidence["query_mean_shift"]),
        "query_p95_shift": float(evidence["query_p95_shift"]),
        "query_max_shift": float(evidence["query_max_shift"]),
        "max_query_index": int(evidence["max_query_index"]),
        "max_shift_window_left": int(evidence["max_shift_window_pair"][0]),
        "max_shift_window_right": int(evidence["max_shift_window_pair"][1]),
    }
    for index, (start, end) in enumerate(
        evidence["window_round_ranges"], start=1
    ):
        row[f"window_{index}_start_round"] = int(start)
        row[f"window_{index}_end_round"] = int(end)
    for pair in evidence["pairwise"]:
        prefix = f"window_{pair['left_window']}_{pair['right_window']}"
        row[f"{prefix}_query_max_shift"] = float(pair["query_max_shift"])
        row[f"{prefix}_max_query_index"] = int(pair["max_query_index"])
    return row


def collect_query_max_frame(
    inputs: Sequence[range_analyzer.AuditedTraceInput],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in inputs:
        trace = load_stationarity_trace(item.run_dir / "trace")
        rows.extend(
            _flatten_check(item, trace, check)
            for check in _selected_checks(trace)
        )
    frame = pd.DataFrame(rows)
    expected_rows = len(inputs) * EXPECTED_CHECKS_PER_TRAJECTORY
    if len(frame) != expected_rows or expected_rows != 36:
        raise RuntimeError("最坏查询漂移量程必须恰好包含 36 行")
    counts = frame.groupby(["dataset", "kernel"], sort=True).size()
    if len(counts) != 4 or not all(int(value) == 9 for value in counts):
        raise RuntimeError("四个 dataset×kernel cell 必须各有九行证据")
    return frame.sort_values(
        ["dataset", "kernel", "seed", "round_index"],
        kind="stable",
    ).reset_index(drop=True)


def _quantile_summary(values: pd.Series) -> Dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("描述性量程必须由有限非空数值构成")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p90": float(np.percentile(array, 90, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "max": float(np.max(array)),
    }


def build_descriptive_summary(
    frame: pd.DataFrame,
    source_audit: Dict[str, Any],
) -> Dict[str, Any]:
    cell_ranges = []
    for (dataset, kernel), group in frame.groupby(
        ["dataset", "kernel"], sort=True
    ):
        cell_ranges.append({
            "dataset": dataset,
            "kernel": kernel,
            "query_max_shift": _quantile_summary(group["query_max_shift"]),
        })
    summary = {
        "contract_version": QUERY_MAX_RANGE_REPORT_CONTRACT_VERSION,
        "role": {
            "purpose": "threshold_free_query_max_range_audit_only",
            "detector_config_changed": False,
            "threshold_selected": False,
            "stationarity_classification": False,
            "candidate_stop_round": False,
            "generator_rerun": False,
            "validation_seed_access": False,
        },
        "protocol": {
            "window_size": WINDOW_SIZE,
            "calibration_round_range": [
                CALIBRATION_ROUND_START,
                CALIBRATION_ROUND_END,
            ],
            "terminal_rounds": list(TERMINAL_ROUNDS),
            "checks_per_trajectory": EXPECTED_CHECKS_PER_TRAJECTORY,
            "query_max_definition": (
                "maximum absolute normalized query-answer window-mean "
                "shift over all queries and all three window pairs"
            ),
            "linear_percentile_method": True,
        },
        "source_audit": source_audit,
        "evidence_row_count": int(len(frame)),
        "global_query_max_shift": _quantile_summary(
            frame["query_max_shift"]
        ),
        "cell_ranges": cell_ranges,
    }
    _strict_json_bytes(summary)
    return summary


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"query max range report 已存在，拒绝覆盖：{output_dir}"
        )
    environment = range_analyzer.analysis_environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 query max range report 要求干净工作树")
    inputs, source_audit = range_analyzer.audit_formal_inputs(input_dir)
    frame = collect_query_max_frame(inputs)
    summary = build_descriptive_summary(frame, source_audit)
    summary["analysis_environment"] = environment
    _strict_json_bytes(summary)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.partial-",
    ))
    try:
        checks_path = temporary / "query_max_checks.csv"
        summary_path = temporary / "query_max_summary.json"
        frame.to_csv(checks_path, index=False, float_format="%.17g")
        _write_json_exclusive(summary_path, summary)
        artifacts = {
            path.name: {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "contract_version": QUERY_MAX_RANGE_REPORT_CONTRACT_VERSION,
            "formal_threshold_free_query_max_range_report": True,
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
    plan = {
        "contract_version": QUERY_MAX_RANGE_REPORT_CONTRACT_VERSION,
        "mode": "plan_only_no_trace_read",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "expected_trajectory_count": 12,
        "window_size": WINDOW_SIZE,
        "calibration_round_range": [
            CALIBRATION_ROUND_START,
            CALIBRATION_ROUND_END,
        ],
        "terminal_rounds": list(TERMINAL_ROUNDS),
        "threshold_parameters_present": False,
        "classification_output_present": False,
        "detector_config_changed": False,
        "generator_rerun": False,
        "validation_seeds_may_be_read": False,
    }
    _strict_json_bytes(plan)
    return plan


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
        default=Path("outputs/issue53_stage2b_query_max_range"),
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
    print(f"threshold-free query max range report -> {destination}")


if __name__ == "__main__":
    main()
