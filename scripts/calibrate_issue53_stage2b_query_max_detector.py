#!/usr/bin/env python
"""Issue #53 Stage 2B 最坏查询护栏的开发候选与完整离线回放。

本入口保留已进入 validation 的原版 detector 配置，只从既有 12 条 development
轨迹的三个晚期检查导出一个新增的 ``query_max_shift_tolerance``：每个
dataset×kernel cell 取线性 P95，再取四格最大值。随后使用版本化 query-max
detector 回放全部 8000 轮，并与原版停止轮次配对比较。

脚本没有阈值覆盖参数，不读取 validation 轨迹、不重跑生成器、也不接在线停止。
默认 ``plan`` 不读取轨迹；``report`` 要求干净工作树并原子发布开发候选报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from scripts import analyze_issue53_stage2b_range_finding as range_analyzer
    from scripts import calibrate_issue53_stage2b_detector as base_calibration
    from scripts import issue53_stage2b_validation_protocol as old_protocol
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import analyze_issue53_stage2b_range_finding as range_analyzer
    import calibrate_issue53_stage2b_detector as base_calibration
    import issue53_stage2b_validation_protocol as old_protocol

from table_diffevo.stationarity import (
    STATIONARITY_QUERY_MAX_REPLAY_CONTRACT_VERSION,
    QueryMaxStationarityDetectorConfig,
    StationarityDetectorConfig,
    collect_query_max_stationarity_range_evidence,
    load_stationarity_trace,
    replay_query_max_stationarity,
    replay_stationarity,
)


QUERY_MAX_CALIBRATION_REPORT_CONTRACT_VERSION = (
    "issue53-stage2b-query-max-detector-calibration-report-v1"
)
REFERENCE_QUERY_MAX_RANGE_ANALYSIS_COMMIT = (
    "1505fd5fb3dc8bd8931bd76c0199517d4eef5576"
)
REFERENCE_QUERY_MAX_CHECKS_SHA256 = (
    "0556945c2a09d08e45f747bdc53bf11ccbb0ebfe21adeeabd04e88bee26f9092"
)
REFERENCE_QUERY_MAX_SUMMARY_SHA256 = (
    "70390c99f6cdac24568db35f502356601cf70d1a0721bec68d99b235b1439d9a"
)
WINDOW_SIZE = 400
CALIBRATION_ROUND_START = 6001
CALIBRATION_ROUND_END = 8000
CALIBRATION_TERMINAL_ROUNDS = (7200, 7600, 8000)
STABILITY_QUANTILE = 0.95
EXPECTED_CALIBRATION_CHECKS_PER_TRAJECTORY = 3
PERSISTENT_REDRIFT_CHECKS = 4
QUERY_MAX_METRIC = "query_max_shift"
QUERY_MAX_CONFIG_FIELD = "query_max_shift_tolerance"


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


def _old_frozen_config() -> StationarityDetectorConfig:
    config = old_protocol.FROZEN_DETECTOR_CONFIG
    if type(config) is not StationarityDetectorConfig:
        raise RuntimeError("原版冻结 detector 配置类型发生变化")
    manifest = old_protocol.frozen_validation_protocol_manifest()
    if manifest["detector"]["config"] != config.to_dict():
        raise RuntimeError("原版冻结 detector 配置与协议不一致")
    return config


def _flatten_raw_check(
    item: range_analyzer.AuditedTraceInput,
    check: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": item.dataset,
        "kernel": item.kernel,
        "seed": int(item.seed),
        "window_size": int(check["window_size"]),
        "completed_block_count": int(check["completed_block_count"]),
        "state_index": int(check["state_index"]),
        "round_index": int(check["round_index"]),
    }
    for index, (start, end) in enumerate(
        check["window_round_ranges"], start=1
    ):
        row[f"window_{index}_start_round"] = int(start)
        row[f"window_{index}_end_round"] = int(end)
    for metric in (*base_calibration.EVIDENCE_METRICS, QUERY_MAX_METRIC):
        row[metric] = float(check[metric])
    return row


def collect_calibration_frame(
    inputs: Sequence[range_analyzer.AuditedTraceInput],
) -> pd.DataFrame:
    """Collect the same 36 late checks with one additional raw metric."""
    rows: List[Dict[str, Any]] = []
    for item in inputs:
        trace = load_stationarity_trace(item.run_dir / "trace")
        checks = collect_query_max_stationarity_range_evidence(
            trace, [WINDOW_SIZE]
        )
        selected = [
            check
            for check in checks
            if int(check["round_index"]) in CALIBRATION_TERMINAL_ROUNDS
            and int(check["window_round_ranges"][0][0])
            >= CALIBRATION_ROUND_START
            and int(check["window_round_ranges"][-1][-1])
            <= CALIBRATION_ROUND_END
        ]
        if len(selected) != EXPECTED_CALIBRATION_CHECKS_PER_TRAJECTORY:
            raise RuntimeError("每条轨迹必须恰好贡献三个完整晚期检查")
        if tuple(int(row["round_index"]) for row in selected) != (
            CALIBRATION_TERMINAL_ROUNDS
        ):
            raise RuntimeError("晚期检查终止轮次不符合固定协议")
        rows.extend(_flatten_raw_check(item, check) for check in selected)

    frame = pd.DataFrame(rows)
    if len(frame) != 36:
        raise RuntimeError("query-max 校准证据必须恰好为 36 行")
    counts = frame.groupby(["dataset", "kernel"], sort=True).size()
    if len(counts) != 4 or not all(int(value) == 9 for value in counts):
        raise RuntimeError("四个 dataset×kernel cell 必须各有九行证据")
    return frame


def _linear_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("阈值分位数输入必须是有限非空数值")
    return float(np.percentile(array, 100.0 * quantile, method="linear"))


def derive_candidate_config(
    calibration_frame: pd.DataFrame,
) -> tuple[QueryMaxStationarityDetectorConfig, Dict[str, Any]]:
    required = {"dataset", "kernel", QUERY_MAX_METRIC}
    missing = sorted(required - set(calibration_frame.columns))
    if missing:
        raise ValueError(f"calibration_frame 缺少字段：{missing}")
    groups = list(calibration_frame.groupby(
        ["dataset", "kernel"], sort=True
    ))
    if len(groups) != 4 or any(len(group) != 9 for _, group in groups):
        raise ValueError("校准必须恰好包含四格、每格九行")

    per_cell = {
        f"{dataset}|{kernel}": _linear_quantile(
            group[QUERY_MAX_METRIC], STABILITY_QUANTILE
        )
        for (dataset, kernel), group in groups
    }
    common = max(per_cell.values())
    config = QueryMaxStationarityDetectorConfig(
        base_config=_old_frozen_config(),
        query_max_shift_tolerance=common,
    )
    derivation = {
        "metric": QUERY_MAX_METRIC,
        "config_field": QUERY_MAX_CONFIG_FIELD,
        "quantile": STABILITY_QUANTILE,
        "per_cell": per_cell,
        "common_rule": "maximum_of_four_cell_p95",
        "common_value": common,
        "linear_percentile_method": True,
        "manual_margin_or_rounding": False,
    }
    _strict_json_bytes(derivation)
    return config, derivation


def _classify_raw_check(
    check: Dict[str, Any],
    config: QueryMaxStationarityDetectorConfig,
) -> tuple[bool, bool, List[str], List[str]]:
    base_values = config.base_config.to_dict()
    failed_stability = [
        metric
        for metric, config_field in (
            *base_calibration.STABILITY_CONFIG_FIELDS.items(),
            (QUERY_MAX_METRIC, QUERY_MAX_CONFIG_FIELD),
        )
        if float(check[metric])
        > float(
            config.query_max_shift_tolerance
            if config_field == QUERY_MAX_CONFIG_FIELD
            else base_values[config_field]
        )
    ]
    failed_movement = [
        metric
        for metric, config_field in (
            base_calibration.MOVEMENT_CONFIG_FIELDS.items()
        )
        if float(check[metric]) < float(base_values[config_field])
    ]
    return (
        not failed_stability,
        not failed_movement,
        failed_stability,
        failed_movement,
    )


def annotate_full_checks(
    checks: Sequence[Dict[str, Any]],
    config: QueryMaxStationarityDetectorConfig,
    candidate_round_index: int | None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    post_candidate_unstable_streak = 0
    maximum_post_candidate_unstable_streak = 0
    for check in checks:
        stable, movement, failed_stability, failed_movement = (
            _classify_raw_check(check, config)
        )
        round_index = int(check["round_index"])
        after_candidate = (
            candidate_round_index is not None
            and round_index > candidate_round_index
        )
        if after_candidate:
            if stable:
                post_candidate_unstable_streak = 0
            else:
                post_candidate_unstable_streak += 1
                maximum_post_candidate_unstable_streak = max(
                    maximum_post_candidate_unstable_streak,
                    post_candidate_unstable_streak,
                )
        row = dict(check)
        row.update({
            "stable": stable,
            "movement_sufficient": movement,
            "failed_stability_metrics": failed_stability,
            "failed_movement_metrics": failed_movement,
            "after_candidate_stop": bool(after_candidate),
            "post_candidate_unstable_streak": int(
                post_candidate_unstable_streak
            ),
        })
        rows.append(row)
    audit = {
        "maximum_post_candidate_unstable_streak": int(
            maximum_post_candidate_unstable_streak
        ),
        "persistent_redrift_check_count": PERSISTENT_REDRIFT_CHECKS,
        "persistent_redrift_detected": bool(
            maximum_post_candidate_unstable_streak
            >= PERSISTENT_REDRIFT_CHECKS
        ),
    }
    return rows, audit


def _flatten_annotated_check(
    item: range_analyzer.AuditedTraceInput,
    check: Dict[str, Any],
) -> Dict[str, Any]:
    row = _flatten_raw_check(item, check)
    row.update({
        "stable": bool(check["stable"]),
        "movement_sufficient": bool(check["movement_sufficient"]),
        "failed_stability_metrics": "|".join(
            check["failed_stability_metrics"]
        ),
        "failed_movement_metrics": "|".join(
            check["failed_movement_metrics"]
        ),
        "after_candidate_stop": bool(check["after_candidate_stop"]),
        "post_candidate_unstable_streak": int(
            check["post_candidate_unstable_streak"]
        ),
    })
    return row


def replay_full_development(
    inputs: Sequence[range_analyzer.AuditedTraceInput],
    config: QueryMaxStationarityDetectorConfig,
) -> tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    trajectory_rows: List[Dict[str, Any]] = []
    check_rows: List[Dict[str, Any]] = []
    for item in inputs:
        trace = load_stationarity_trace(item.run_dir / "trace")
        baseline = replay_stationarity(trace, config.base_config)
        replay = replay_query_max_stationarity(trace, config)
        raw_checks = collect_query_max_stationarity_range_evidence(
            trace, [WINDOW_SIZE]
        )
        annotated, redrift = annotate_full_checks(
            raw_checks, config, replay.candidate_round_index
        )

        replay_by_round = {
            int(check["round_index"]): check for check in replay.checks
        }
        for check in annotated:
            round_index = int(check["round_index"])
            if round_index in replay_by_round:
                official = replay_by_round[round_index]
                if (
                    bool(check["stable"]) != bool(official["stable"])
                    or bool(check["movement_sufficient"])
                    != bool(official["movement_sufficient"])
                    or not np.isclose(
                        float(check[QUERY_MAX_METRIC]),
                        float(official[QUERY_MAX_METRIC]),
                        rtol=0.0,
                        atol=1e-15,
                    )
                ):
                    raise RuntimeError(
                        "完整审计分类与 query-max replay 公式不一致"
                    )
            check_rows.append(_flatten_annotated_check(item, check))

        baseline_round = baseline.candidate_round_index
        candidate_round = replay.candidate_round_index
        trajectory_rows.append({
            "dataset": item.dataset,
            "kernel": item.kernel,
            "seed": int(item.seed),
            "baseline_status": baseline.status,
            "baseline_candidate_round_index": baseline_round,
            "status": replay.status,
            "candidate_state_index": replay.candidate_state_index,
            "candidate_round_index": candidate_round,
            "candidate_round_delta_from_baseline": (
                None
                if baseline_round is None or candidate_round is None
                else int(candidate_round - baseline_round)
            ),
            "official_replay_check_count": len(replay.checks),
            "full_audit_check_count": len(annotated),
            **redrift,
        })

    check_frame = pd.DataFrame(check_rows)
    expected_checks = (
        int(range_analyzer.collector.DEVELOPMENT_ROUND_BUDGET)
        // WINDOW_SIZE - 2
    )
    observed = check_frame.groupby(
        ["dataset", "kernel", "seed"], sort=True
    ).size()
    if len(observed) != len(inputs) or not all(
        int(value) == expected_checks for value in observed
    ):
        raise RuntimeError("完整审计没有覆盖每条轨迹的全部检查")

    qualified = [
        row for row in trajectory_rows
        if row["status"] == "stationary_qualified"
    ]
    redrift = [
        row for row in trajectory_rows
        if row["persistent_redrift_detected"]
    ]
    acceptance = {
        "expected_trajectory_count": 12,
        "qualified_trajectory_count": len(qualified),
        "all_development_trajectories_qualified": len(qualified) == 12,
        "stalled_trajectory_count": sum(
            row["status"] == "stalled" for row in trajectory_rows
        ),
        "persistent_redrift_trajectory_count": len(redrift),
        "no_persistent_post_candidate_redrift": len(redrift) == 0,
    }
    acceptance["candidate_supported_on_development"] = bool(
        acceptance["all_development_trajectories_qualified"]
        and acceptance["stalled_trajectory_count"] == 0
        and acceptance["no_persistent_post_candidate_redrift"]
    )
    return trajectory_rows, check_frame, acceptance


def build_report(
    input_dir: Path,
    environment: Dict[str, Any],
) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    inputs, source_audit = range_analyzer.audit_formal_inputs(input_dir)
    calibration_frame = collect_calibration_frame(inputs)
    config, derivation = derive_candidate_config(calibration_frame)
    trajectories, full_checks, acceptance = replay_full_development(
        inputs, config
    )
    report = {
        "contract_version": QUERY_MAX_CALIBRATION_REPORT_CONTRACT_VERSION,
        "role": {
            "purpose": "query_max_development_candidate_only",
            "old_frozen_detector_modified": False,
            "online_stopping_enabled": False,
            "generator_rerun": False,
            "validation_seed_access": False,
            "candidate_config_frozen_for_validation": False,
        },
        "source_audit": source_audit,
        "analysis_environment": environment,
        "pre_threshold_range_evidence": {
            "analysis_git_commit": (
                REFERENCE_QUERY_MAX_RANGE_ANALYSIS_COMMIT
            ),
            "checks_sha256": REFERENCE_QUERY_MAX_CHECKS_SHA256,
            "summary_sha256": REFERENCE_QUERY_MAX_SUMMARY_SHA256,
        },
        "calibration_protocol": {
            "window_size": WINDOW_SIZE,
            "calibration_round_range": [
                CALIBRATION_ROUND_START,
                CALIBRATION_ROUND_END,
            ],
            "calibration_terminal_rounds": list(
                CALIBRATION_TERMINAL_ROUNDS
            ),
            "calibration_check_count": int(len(calibration_frame)),
            "checks_per_trajectory": (
                EXPECTED_CALIBRATION_CHECKS_PER_TRAJECTORY
            ),
            "stability_quantile": STABILITY_QUANTILE,
            "common_rule": "maximum_of_four_cell_p95",
            "persistent_redrift_checks": PERSISTENT_REDRIFT_CHECKS,
            "linear_percentile_method": True,
            "manual_margin_or_rounding": False,
        },
        "query_max_replay_contract_version": (
            STATIONARITY_QUERY_MAX_REPLAY_CONTRACT_VERSION
        ),
        "old_frozen_detector_config": config.base_config.to_dict(),
        "candidate_detector_config": config.to_dict(),
        "threshold_derivation": derivation,
        "development_replay": {
            "trajectories": trajectories,
            "acceptance_gates": acceptance,
        },
    }
    _strict_json_bytes(report)
    return report, calibration_frame, full_checks


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"query-max calibration report 已存在，拒绝覆盖：{output_dir}"
        )
    environment = range_analyzer.analysis_environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 query-max calibration report 要求干净工作树")
    report, calibration_frame, full_checks = build_report(
        input_dir, environment
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.partial-",
    ))
    try:
        report_path = temporary / "calibration_report.json"
        calibration_path = temporary / "calibration_checks.csv"
        full_checks_path = temporary / "full_replay_checks.csv"
        _write_json_exclusive(report_path, report)
        calibration_frame.to_csv(
            calibration_path, index=False, float_format="%.17g"
        )
        full_checks.to_csv(
            full_checks_path, index=False, float_format="%.17g"
        )
        artifact_paths = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        artifacts = {
            str(path.relative_to(temporary)): {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        }
        manifest = {
            "contract_version": (
                QUERY_MAX_CALIBRATION_REPORT_CONTRACT_VERSION
            ),
            "formal_query_max_development_candidate_report": True,
            "source_audit": report["source_audit"],
            "analysis_environment": environment,
            "artifacts": artifacts,
        }
        _write_json_exclusive(
            temporary / "report_manifest.json", manifest
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_plan(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    plan = {
        "contract_version": QUERY_MAX_CALIBRATION_REPORT_CONTRACT_VERSION,
        "mode": "plan_only_no_trace_read",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "expected_trajectory_count": 12,
        "window_size": WINDOW_SIZE,
        "calibration_round_range": [
            CALIBRATION_ROUND_START,
            CALIBRATION_ROUND_END,
        ],
        "calibration_terminal_rounds": list(
            CALIBRATION_TERMINAL_ROUNDS
        ),
        "stability_quantile": STABILITY_QUANTILE,
        "common_rule": "maximum_of_four_cell_p95",
        "threshold_override_parameters_present": False,
        "old_frozen_detector_modified": False,
        "generator_rerun": False,
        "online_stopping_enabled": False,
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
        default=Path("outputs/issue53_stage2b_query_max_calibration"),
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
    print(f"query-max detector calibration report -> {destination}")


if __name__ == "__main__":
    main()
