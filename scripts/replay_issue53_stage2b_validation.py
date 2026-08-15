#!/usr/bin/env python
"""Issue #53 Stage 2B 冻结 V1 detector 的正式 validation 回放。

本入口只读取已完整发布的20条 validation 轨迹。它重新审计 collection
manifest、每条 run manifest、trace 文件哈希、8000轮预算和配对
``s0/S0/初始化后 RNG``，随后使用验证前冻结的唯一 V1 detector 配置执行离线
回放，并在完整反事实尾部上审计候选停止后的连续四次不稳定再漂移。

脚本没有阈值覆盖、生成重跑、在线停止或 query-max 参数。默认 ``plan`` 不读取
validation 输入；``report`` 必须显式确认冻结协议 SHA-256 和本次 collection
manifest SHA-256，并从干净工作树原子发布不可覆盖的正式报告。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from scripts import calibrate_issue53_stage2b_detector as calibration
    from scripts import collect_issue53_stage2b_validation as collection
    from scripts import issue53_stage2b_validation_protocol as protocol
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import calibrate_issue53_stage2b_detector as calibration
    import collect_issue53_stage2b_validation as collection
    import issue53_stage2b_validation_protocol as protocol

from table_diffevo.stationarity import (
    STATIONARITY_REPLAY_CONTRACT_VERSION,
    StationarityDetectorConfig,
    collect_stationarity_range_evidence,
    load_stationarity_trace,
    replay_stationarity,
)


VALIDATION_REPLAY_REPORT_CONTRACT_VERSION = (
    "issue53-stage2b-detector-validation-report-v1"
)
REFERENCE_VALIDATION_COLLECTION_GIT_COMMIT = (
    "0388997755e409a76b97424de4388854df282700"
)

_COLLECTION_MANIFEST_KEYS = {
    "contract_version",
    "validation_protocol_sha256",
    "formal_validation_collection_complete",
    "trajectory_count",
    "rounds_per_trajectory",
    "total_round_count",
    "detector_replay_performed",
    "partial_validation_classification_read",
    "run_manifest_sha256",
    "collection_elapsed_sec",
}


@dataclass(frozen=True)
class AuditedValidationInput:
    dataset: str
    kernel: str
    seed: int
    run_dir: Path
    manifest: Dict[str, Any]
    manifest_sha256: str


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


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} 必须是 SHA-256 十六进制字符串")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{name} 必须是 SHA-256 十六进制字符串"
        ) from exc
    return value


def _validate_exact_keys(
    value: Any,
    expected: Iterable[str],
    name: str,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是对象")
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise RuntimeError(
            f"{name} 字段不一致；missing="
            f"{sorted(expected_set - observed)}, "
            f"unknown={sorted(observed - expected_set)}"
        )


def _load_json_object(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取{name}：{path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name}必须是 JSON 对象：{path}")
    return value


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


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def analysis_environment_manifest() -> Dict[str, Any]:
    return {
        "analysis_git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": (
            _git_output("status", "--porcelain") == ""
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "started_at": datetime.now().astimezone().isoformat(),
    }


def _expected_cells() -> set[tuple[str, int, str]]:
    return {
        (row["dataset"], int(row["seed"]), row["kernel"])
        for row in protocol.expected_validation_cells()
    }


def _cell_name(cell: tuple[str, int, str]) -> str:
    return f"{cell[0]}/seed_{cell[1]}/{cell[2]}"


def _frozen_config() -> StationarityDetectorConfig:
    config = protocol.FROZEN_DETECTOR_CONFIG
    if type(config) is not StationarityDetectorConfig:
        raise RuntimeError("冻结 V1 detector 配置类型发生变化")
    manifest = protocol.frozen_validation_protocol_manifest()
    if manifest["detector"]["config"] != config.to_dict():
        raise RuntimeError("冻结 V1 detector 配置与验证协议不一致")
    if config.window_size != calibration.WINDOW_SIZE:
        raise RuntimeError("冻结窗口与原校准回放不一致")
    if (
        protocol.PERSISTENT_REDRIFT_CHECKS
        != calibration.PERSISTENT_REDRIFT_CHECKS
    ):
        raise RuntimeError("持续再漂移门禁与原校准回放不一致")
    return config


def _execution_environment_groups(
    inputs: Sequence[AuditedValidationInput],
) -> List[Dict[str, Any]]:
    groups: Dict[bytes, Dict[str, Any]] = {}
    for item in inputs:
        environment = item.manifest["environment"]
        identity = _strict_json_bytes(environment)
        if identity not in groups:
            groups[identity] = {
                "environment": environment,
                "cells": [],
            }
        groups[identity]["cells"].append(
            _cell_name((item.dataset, item.seed, item.kernel))
        )
    result = []
    for group in groups.values():
        cells = sorted(group["cells"])
        result.append({
            "environment": group["environment"],
            "trajectory_count": len(cells),
            "cells": cells,
        })
    result.sort(key=lambda row: row["cells"][0])
    return result


def audit_validation_collection(
    input_dir: Path,
    confirmed_collection_manifest_sha256: str,
) -> tuple[List[AuditedValidationInput], Dict[str, Any]]:
    """Re-audit the complete collection before any detector replay."""
    confirmed_sha = _validate_sha256(
        confirmed_collection_manifest_sha256,
        "confirmed_collection_manifest_sha256",
    )
    if not input_dir.is_dir():
        raise RuntimeError(f"validation collection 目录不存在：{input_dir}")
    collection_path = input_dir / "collection_manifest.json"
    if not collection_path.is_file():
        raise RuntimeError("validation collection 缺少 collection manifest")
    observed_collection_sha = _sha256_file(collection_path)
    if observed_collection_sha != confirmed_sha:
        raise RuntimeError("collection manifest SHA-256 与显式确认值不一致")

    collection_manifest = _load_json_object(
        collection_path, "collection manifest"
    )
    _validate_exact_keys(
        collection_manifest,
        _COLLECTION_MANIFEST_KEYS,
        "collection manifest",
    )
    expected_protocol_sha = protocol.validation_protocol_sha256()
    expected_cells = _expected_cells()
    expected_count = len(expected_cells)
    expected_rounds = protocol.VALIDATION_ROUND_BUDGET
    if (
        collection_manifest["contract_version"]
        != collection.VALIDATION_COLLECTION_CONTRACT_VERSION
        or collection_manifest["validation_protocol_sha256"]
        != expected_protocol_sha
        or collection_manifest["formal_validation_collection_complete"]
        is not True
        or collection_manifest["trajectory_count"] != expected_count
        or collection_manifest["rounds_per_trajectory"]
        != expected_rounds
        or collection_manifest["total_round_count"]
        != expected_count * expected_rounds
        or collection_manifest["detector_replay_performed"] is not False
        or collection_manifest["partial_validation_classification_read"]
        is not False
    ):
        raise RuntimeError("collection manifest 身份或封存状态不一致")
    elapsed = collection_manifest["collection_elapsed_sec"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise RuntimeError("collection elapsed_sec 必须是正有限数值")

    run_hashes = collection_manifest["run_manifest_sha256"]
    expected_cell_names = {_cell_name(cell) for cell in expected_cells}
    _validate_exact_keys(
        run_hashes,
        expected_cell_names,
        "collection run_manifest_sha256",
    )
    for cell_name, value in run_hashes.items():
        _validate_sha256(value, f"run_manifest_sha256[{cell_name}]")

    expected_manifest_paths = {
        input_dir / cell[0] / f"seed_{cell[1]}" / cell[2]
        / "run_manifest.json"
        for cell in expected_cells
    }
    observed_manifest_paths = set(input_dir.rglob("run_manifest.json"))
    if observed_manifest_paths != expected_manifest_paths:
        missing_paths = sorted(
            str(path)
            for path in expected_manifest_paths - observed_manifest_paths
        )
        unknown_paths = sorted(
            str(path)
            for path in observed_manifest_paths - expected_manifest_paths
        )
        raise RuntimeError(
            "validation run manifest 集合不完整或包含未知路径；"
            f"missing={missing_paths}, unknown={unknown_paths}"
        )

    audited: List[AuditedValidationInput] = []
    pair_bindings: Dict[tuple[str, int], tuple[Any, ...]] = {}
    clip_totals = {
        "direction_evaluated_count": 0,
        "direction_clipped_count": 0,
        "gibbs_conditional_evaluated_count": 0,
        "gibbs_conditional_clipped_count": 0,
    }
    for cell in sorted(expected_cells):
        run_dir = input_dir / cell[0] / f"seed_{cell[1]}" / cell[2]
        manifest_path = run_dir / "run_manifest.json"
        manifest_sha = _sha256_file(manifest_path)
        if manifest_sha != run_hashes[_cell_name(cell)]:
            raise RuntimeError(f"run manifest 哈希与 collection 不一致：{cell}")
        manifest = collection.audit_validation_run(
            run_dir,
            expected_git_commit=(
                REFERENCE_VALIDATION_COLLECTION_GIT_COMMIT
            ),
        )
        observed_cell = (
            manifest["dataset"],
            int(manifest["seed"]),
            manifest["kernel"],
        )
        if observed_cell != cell:
            raise RuntimeError("run manifest 返回身份与路径 cell 不一致")
        if manifest["validation_protocol_sha256"] != expected_protocol_sha:
            raise RuntimeError("run manifest 验证协议 SHA-256 不一致")

        pair_key = (cell[0], cell[1])
        binding = (
            manifest["s0_preflight"]["direction_reference_scale"],
            manifest["run_summary"]["initial_table_sha256"],
            manifest["s0_preflight"][
                "primary_rng_post_initialization_state_sha256"
            ],
        )
        if pair_key in pair_bindings and pair_bindings[pair_key] != binding:
            raise RuntimeError("配对 kernel 没有共享 s0/S0/初始化后 RNG")
        pair_bindings[pair_key] = binding

        clip_audit = manifest["run_summary"]["clip_audit"]
        direction = clip_audit["direction"]
        gibbs = clip_audit["gibbs_conditional"]
        clip_totals["direction_evaluated_count"] += int(
            direction["evaluated_count"]
        )
        clip_totals["direction_clipped_count"] += int(
            direction["clipped_count"]
        )
        clip_totals["gibbs_conditional_evaluated_count"] += int(
            gibbs["evaluated_count"]
        )
        clip_totals["gibbs_conditional_clipped_count"] += int(
            gibbs["clipped_count"]
        )
        audited.append(AuditedValidationInput(
            dataset=cell[0],
            kernel=cell[2],
            seed=cell[1],
            run_dir=run_dir,
            manifest=manifest,
            manifest_sha256=manifest_sha,
        ))

    expected_pair_count = 2 * len(protocol.VALIDATION_SEEDS)
    if len(pair_bindings) != expected_pair_count:
        raise RuntimeError("validation dataset×seed 配对绑定数不完整")

    descriptor = {
        "input_root": str(input_dir.resolve()),
        "collection_manifest_sha256": observed_collection_sha,
        "collection_contract_version": (
            collection.VALIDATION_COLLECTION_CONTRACT_VERSION
        ),
        "validation_protocol_sha256": expected_protocol_sha,
        "source_validation_collection_git_commit": (
            REFERENCE_VALIDATION_COLLECTION_GIT_COMMIT
        ),
        "formal_validation_collection_complete": True,
        "detector_replay_performed_during_collection": False,
        "partial_validation_classification_read_during_collection": False,
        "trajectory_count": len(audited),
        "rounds_per_trajectory": expected_rounds,
        "total_round_count": expected_count * expected_rounds,
        "validation_seeds": list(protocol.VALIDATION_SEEDS),
        "paired_s0_s0_rng_binding_count": len(pair_bindings),
        "clip_audit_totals": clip_totals,
        "run_manifest_sha256": dict(sorted(run_hashes.items())),
        "execution_environment_groups": _execution_environment_groups(
            audited
        ),
    }
    _strict_json_bytes(descriptor)
    return audited, descriptor


def _flatten_full_check(
    item: AuditedValidationInput,
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
    for metric in calibration.EVIDENCE_METRICS:
        row[metric] = float(check[metric])
    for index in range(3):
        number = index + 1
        row[f"window_{number}_l1_mean"] = float(
            check["window_l1_means"][index]
        )
        row[f"window_{number}_l1_p90_minus_p10"] = float(
            check["window_l1_p90_minus_p10"][index]
        )
        row[f"window_{number}_l1_p95"] = float(
            check["window_l1_p95"][index]
        )
        row[f"window_{number}_active_round_rate"] = float(
            check["window_active_round_rates"][index]
        )
        row[f"window_{number}_mean_changed_row_fraction"] = float(
            check["window_mean_changed_row_fractions"][index]
        )
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


def _audit_official_replay_prefix(
    official_checks: Sequence[Dict[str, Any]],
    annotated_checks: Sequence[Dict[str, Any]],
) -> None:
    by_round = {
        int(check["round_index"]): check for check in annotated_checks
    }
    for official in official_checks:
        round_index = int(official["round_index"])
        if round_index not in by_round:
            raise RuntimeError("正式 replay 检查不在完整预算检查网格中")
        full = by_round[round_index]
        if (
            bool(full["stable"]) != bool(official["stable"])
            or bool(full["movement_sufficient"])
            != bool(official["movement_sufficient"])
        ):
            raise RuntimeError("完整审计分类与正式 V1 replay 公式不一致")
        for metric in calibration.EVIDENCE_METRICS:
            if not np.isclose(
                float(full[metric]),
                float(official[metric]),
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError(
                    "完整审计证据与正式 V1 replay 公式不一致"
                )


def replay_full_validation(
    inputs: Sequence[AuditedValidationInput],
) -> tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """Replay the frozen V1 detector and apply the preregistered gates."""
    expected_cells = _expected_cells()
    observed_cells = {
        (item.dataset, item.seed, item.kernel) for item in inputs
    }
    if len(inputs) != len(expected_cells) or observed_cells != expected_cells:
        raise RuntimeError("正式 V1 replay 必须恰好输入20个验证 cell")

    config = _frozen_config()
    expected_rounds = protocol.VALIDATION_ROUND_BUDGET
    expected_check_rounds = list(range(
        3 * config.window_size,
        expected_rounds + 1,
        config.window_size,
    ))
    trajectory_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    full_check_rows: List[Dict[str, Any]] = []

    for item in sorted(
        inputs, key=lambda row: (row.dataset, row.seed, row.kernel)
    ):
        trace = load_stationarity_trace(item.run_dir / "trace")
        replay = replay_stationarity(trace, config)
        if (
            replay.contract_version != STATIONARITY_REPLAY_CONTRACT_VERSION
            or replay.detector_config != config.to_dict()
            or replay.trace["query_identity_sha256"]
            != item.manifest["query_identity_sha256"]
            or replay.trace["target_identity_sha256"]
            != item.manifest["target_identity_sha256"]
            or replay.trace["post_round_count"] != expected_rounds
            or replay.trace["termination_reason"] != "max_rounds"
        ):
            raise RuntimeError("正式 V1 replay 身份或完整预算绑定不一致")

        raw_checks = collect_stationarity_range_evidence(
            trace, [config.window_size]
        )
        observed_rounds = [int(row["round_index"]) for row in raw_checks]
        if observed_rounds != expected_check_rounds:
            raise RuntimeError("完整 V1 检查没有覆盖冻结的全部窗口网格")
        stationary_candidate_state = (
            replay.candidate_state_index
            if replay.status == "stationary_qualified"
            else None
        )
        stationary_candidate_round = (
            replay.candidate_round_index
            if replay.status == "stationary_qualified"
            else None
        )
        stall_state = (
            replay.candidate_state_index
            if replay.status == "stalled"
            else None
        )
        stall_round = (
            replay.candidate_round_index
            if replay.status == "stalled"
            else None
        )
        annotated, redrift = calibration.annotate_full_checks(
            raw_checks,
            config,
            stationary_candidate_round,
        )
        _audit_official_replay_prefix(replay.checks, annotated)
        full_check_rows.extend(
            _flatten_full_check(item, check) for check in annotated
        )

        if (
            replay.candidate_state_index is not None
            and replay.candidate_state_index
            != replay.candidate_round_index
        ):
            raise RuntimeError("候选状态序号与真实轮次没有对齐")
        trajectory_row = {
            "dataset": item.dataset,
            "kernel": item.kernel,
            "seed": int(item.seed),
            "status": replay.status,
            "candidate_state_index": stationary_candidate_state,
            "candidate_round_index": stationary_candidate_round,
            "stall_state_index": stall_state,
            "stall_round_index": stall_round,
            "official_replay_check_count": len(replay.checks),
            "full_audit_check_count": len(annotated),
            "trace_identity_sha256": replay.trace[
                "trace_identity_sha256"
            ],
            "run_manifest_sha256": item.manifest_sha256,
            "source_gpu_device_name": item.manifest["environment"]["gpu"][
                "device_name"
            ],
            **redrift,
        }
        trajectory_rows.append(trajectory_row)
        summary_rows.append({
            "dataset": item.dataset,
            "kernel": item.kernel,
            "seed": int(item.seed),
            "status": replay.status,
            "candidate_round_index": stationary_candidate_round,
            "persistent_redrift_detected": redrift[
                "persistent_redrift_detected"
            ],
        })

    expected_checks_per_trajectory = len(expected_check_rounds)
    check_frame = pd.DataFrame(full_check_rows)
    observed_check_counts = check_frame.groupby(
        ["dataset", "kernel", "seed"], sort=True
    ).size()
    if (
        len(observed_check_counts) != len(inputs)
        or not all(
            int(value) == expected_checks_per_trajectory
            for value in observed_check_counts
        )
    ):
        raise RuntimeError("完整 V1 检查行数不符合20条全预算轨迹")

    validation_result = protocol.evaluate_validation_summaries(summary_rows)
    if validation_result["protocol_sha256"] != (
        protocol.validation_protocol_sha256()
    ):
        raise RuntimeError("验证门禁返回的协议身份不一致")
    return trajectory_rows, check_frame, validation_result


def build_report(
    input_dir: Path,
    environment: Dict[str, Any],
    confirmed_collection_manifest_sha256: str,
) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    inputs, source_audit = audit_validation_collection(
        input_dir,
        confirmed_collection_manifest_sha256,
    )
    config = _frozen_config()
    trajectories, full_checks, validation_result = (
        replay_full_validation(inputs)
    )
    trajectory_frame = pd.DataFrame(trajectories)
    report = {
        "contract_version": VALIDATION_REPLAY_REPORT_CONTRACT_VERSION,
        "role": {
            "purpose": "formal_heldout_validation_of_frozen_v1_detector",
            "formal_heldout_validation_replay": True,
            "validation_seed_access": True,
            "generator_rerun": False,
            "online_stopping_enabled": False,
            "query_max_detector_used": False,
            "threshold_override_parameters_present": False,
            "absolute_l1_quality_used_as_stop_condition": False,
            "retuning_on_these_validation_seeds_allowed": False,
        },
        "source_audit": source_audit,
        "analysis_environment": environment,
        "validation_protocol": {
            "contract_version": protocol.VALIDATION_PROTOCOL_VERSION,
            "sha256": protocol.validation_protocol_sha256(),
            "frozen_detector_config": config.to_dict(),
            "window_size": config.window_size,
            "required_consecutive_moving_stability_checks": (
                protocol.REQUIRED_MOVING_STABILITY_CHECKS
            ),
            "persistent_redrift_checks": (
                protocol.PERSISTENT_REDRIFT_CHECKS
            ),
            "full_budget_counterfactual_tail_audited": True,
            "threshold_retuning_after_validation_access_allowed": False,
        },
        "stationarity_replay_contract_version": (
            STATIONARITY_REPLAY_CONTRACT_VERSION
        ),
        "validation_replay": {
            "trajectories": trajectories,
            "result": validation_result,
        },
    }
    _strict_json_bytes(report)
    return report, trajectory_frame, full_checks


def generate_report(
    input_dir: Path,
    output_dir: Path,
    confirmed_protocol_sha256: str,
    confirmed_collection_manifest_sha256: str,
) -> Path:
    expected_protocol_sha = protocol.validation_protocol_sha256()
    if confirmed_protocol_sha256 != expected_protocol_sha:
        raise ValueError("必须显式确认完整冻结 validation protocol SHA-256")
    confirmed_collection_sha = _validate_sha256(
        confirmed_collection_manifest_sha256,
        "confirmed_collection_manifest_sha256",
    )
    if output_dir.exists():
        raise FileExistsError(
            f"正式 V1 validation report 已存在，拒绝覆盖：{output_dir}"
        )
    environment = analysis_environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 V1 validation replay 要求干净工作树")

    report, trajectories, full_checks = build_report(
        input_dir,
        environment,
        confirmed_collection_sha,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.partial-",
    ))
    try:
        report_path = temporary / "validation_report.json"
        trajectory_path = temporary / "trajectory_results.csv"
        full_checks_path = temporary / "full_replay_checks.csv"
        _write_json_exclusive(report_path, report)
        trajectories.to_csv(
            trajectory_path, index=False, float_format="%.17g"
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
        report_manifest = {
            "contract_version": VALIDATION_REPLAY_REPORT_CONTRACT_VERSION,
            "formal_frozen_v1_validation_report": True,
            "validation_protocol_sha256": expected_protocol_sha,
            "collection_manifest_sha256": confirmed_collection_sha,
            "classification": report["validation_replay"]["result"][
                "classification"
            ],
            "retuning_on_these_validation_seeds_allowed": False,
            "analysis_environment": environment,
            "artifacts": artifacts,
        }
        _write_json_exclusive(
            temporary / "report_manifest.json", report_manifest
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_plan(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    config = _frozen_config()
    plan = {
        "contract_version": VALIDATION_REPLAY_REPORT_CONTRACT_VERSION,
        "mode": "plan_only_no_validation_trace_read",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "validation_protocol_sha256": (
            protocol.validation_protocol_sha256()
        ),
        "expected_trajectory_count": len(_expected_cells()),
        "rounds_per_trajectory": protocol.VALIDATION_ROUND_BUDGET,
        "window_size": config.window_size,
        "expected_full_check_count_per_trajectory": (
            protocol.VALIDATION_ROUND_BUDGET // config.window_size - 2
        ),
        "persistent_redrift_checks": protocol.PERSISTENT_REDRIFT_CHECKS,
        "requires_confirmed_protocol_sha256": True,
        "requires_confirmed_collection_manifest_sha256": True,
        "requires_clean_worktree": True,
        "threshold_override_parameters_present": False,
        "generator_rerun": False,
        "online_stopping_enabled": False,
        "query_max_detector_used": False,
        "validation_traces_read": False,
        "classification_output_present": False,
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
        default=Path("outputs/issue53_stage2b_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/issue53_stage2b_v1_validation_replay"),
    )
    parser.add_argument("--confirm-protocol-sha")
    parser.add_argument("--confirm-collection-manifest-sha")
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
    if args.confirm_protocol_sha is None:
        raise RuntimeError("report 模式必须显式确认 validation protocol SHA")
    if args.confirm_collection_manifest_sha is None:
        raise RuntimeError("report 模式必须显式确认 collection manifest SHA")
    destination = generate_report(
        args.input_dir,
        args.output_dir,
        args.confirm_protocol_sha,
        args.confirm_collection_manifest_sha,
    )
    print(f"formal V1 validation replay -> {destination}", flush=True)


if __name__ == "__main__":
    main()
