#!/usr/bin/env python
"""Issue #53 Stage 2B 无阈值收敛量程报告。

本脚本只读取冻结的 12 条 development current-state 轨迹，按预先固定的候选窗口
计算与 Stage 2A detector 完全同公式的原始证据。它没有阈值参数，不调用 detector
回放，不输出收敛/停滞分类或候选停止轮次，也拒绝读取封存 validation seed。

默认 ``plan`` 只打印计划；``report`` 要求干净工作树并以不可覆盖、目录级原子方式
发布 CSV、JSON 和描述性图片。
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

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from scripts import collect_issue53_stage2b_range_finding as collector
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import collect_issue53_stage2b_range_finding as collector
from table_diffevo.stationarity import (  # noqa: E402
    STATIONARITY_RANGE_EVIDENCE_CONTRACT_VERSION,
    StationarityTrace,
    collect_stationarity_range_evidence,
    load_stationarity_trace,
)


RANGE_REPORT_CONTRACT_VERSION = "issue53-stage2b-range-report-v1"
CANDIDATE_WINDOW_SIZES = (100, 200, 400, 800, 1000)
ROUND_BAND_LABELS = (
    "round_0001_2000",
    "round_2001_4000",
    "round_4001_6000",
    "round_6001_8000",
)

STABILITY_METRICS: Dict[str, str] = {
    "query_mean_shift": "查询均值在三个窗口间的最大两两平均绝对漂移",
    "query_p95_shift": "查询漂移在三个窗口两两比较中的最大 P95",
    "l1_mean_shift": "窗口平均归一化 L1 的最大两两差",
    "l1_p90_minus_p10_shift": "窗口 L1 的 P90-P10 宽度最大两两差",
    "unique_row_rate_shift": "窗口平均唯一行比例的最大两两差",
    "normalized_row_entropy_shift": "窗口平均归一化行熵的最大两两差",
}
MOVEMENT_METRICS: Dict[str, str] = {
    "minimum_observed_active_round_rate": (
        "三个窗口中最低的活跃轮次比例"
    ),
    "minimum_observed_mean_changed_row_fraction": (
        "三个窗口中最低的平均改变行比例"
    ),
}
SCALAR_METRICS = tuple(STABILITY_METRICS) + tuple(MOVEMENT_METRICS)
WINDOW_VECTOR_METRICS = (
    "window_l1_means",
    "window_l1_p90_minus_p10",
    "window_l1_p95",
    "window_active_round_rates",
    "window_mean_changed_row_fractions",
)
FORBIDDEN_CLASSIFICATION_FIELDS = {
    "stable",
    "movement_sufficient",
    "check_status",
    "moving_stability_streak",
    "insufficient_movement_streak",
    "status",
    "candidate_state_index",
    "candidate_round_index",
}

_MANIFEST_KEYS = {
    "contract_version",
    "protocol_sha256",
    "mode",
    "formal_development_calibration",
    "dataset",
    "kernel",
    "seed",
    "maximum_round_budget",
    "input_sha256",
    "query_identity_sha256",
    "target_identity_sha256",
    "s0_preflight",
    "generator_params",
    "reference_process_contract",
    "run_summary",
    "trace_files",
    "environment",
}


@dataclass(frozen=True)
class AuditedTraceInput:
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


def _validate_exact_keys(
    value: Any,
    expected: Iterable[str],
    name: str,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是对象")
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        unknown = sorted(observed - expected_set)
        raise RuntimeError(
            f"{name} 字段不符合冻结契约；missing={missing}, unknown={unknown}"
        )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def analysis_environment_manifest() -> Dict[str, Any]:
    status = _git_output("status", "--porcelain")
    return {
        "analysis_git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": status == "",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "started_at": datetime.now().astimezone().isoformat(),
    }


def _expected_cells() -> set[tuple[str, int, str]]:
    return {
        (dataset, seed, kernel)
        for dataset in collector.DATASETS
        for seed in collector.DEVELOPMENT_SEEDS
        for kernel in collector.KERNELS
    }


def _load_json_object(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取{name}：{path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name}必须是 JSON 对象：{path}")
    return value


def _audit_trace_and_manifest(
    run_dir: Path,
    manifest: Dict[str, Any],
) -> StationarityTrace:
    dataset = manifest["dataset"]
    kernel = manifest["kernel"]
    seed = manifest["seed"]
    summary = manifest["run_summary"]
    specification = collector.DATASETS[dataset]

    if run_dir != (
        run_dir.parents[2] / dataset / f"seed_{seed}" / kernel
    ):
        raise RuntimeError("run manifest 路径与 cell 身份不一致")
    if manifest["contract_version"] != (
        collector.STAGE2B_RANGE_FINDING_PROTOCOL_VERSION
    ):
        raise RuntimeError("range-finding contract version 不一致")
    if manifest["protocol_sha256"] != collector.protocol_sha256():
        raise RuntimeError("range-finding protocol SHA-256 不一致")
    if (
        manifest["mode"] != "development"
        or manifest["formal_development_calibration"] is not True
    ):
        raise RuntimeError("输入不是正式 development calibration 轨迹")
    if manifest["maximum_round_budget"] != (
        collector.DEVELOPMENT_ROUND_BUDGET
    ):
        raise RuntimeError("输入轮数与冻结 development 预算不一致")
    if seed not in collector.DEVELOPMENT_SEEDS:
        raise RuntimeError("输入包含非 development seed")
    if seed in collector.SEALED_VALIDATION_SEEDS:
        raise RuntimeError("拒绝读取封存 validation seed")
    if manifest["input_sha256"] != specification["sha256"]:
        raise RuntimeError("公开 workload 输入哈希不一致")
    if manifest["environment"][
        "git_worktree_clean_including_untracked"
    ] is not True:
        raise RuntimeError("正式输入不是从干净工作树生成")
    if manifest["generator_params"]["device"] != specification["device"]:
        raise RuntimeError("输入执行后端与冻结 workload 协议不一致")
    if int(manifest["generator_params"]["seed"]) != seed:
        raise RuntimeError("generator seed 与 cell 身份不一致")

    _validate_exact_keys(
        manifest["trace_files"],
        {"metadata", "query_array"},
        "trace_files",
    )
    expected_paths = {
        "metadata": "trace/stationarity_trace.json",
        "query_array": "trace/measured_query_answers.npz",
    }
    for role, expected_path in expected_paths.items():
        info = manifest["trace_files"][role]
        _validate_exact_keys(info, {"path", "sha256"}, f"trace_files.{role}")
        if info["path"] != expected_path:
            raise RuntimeError("trace 文件相对路径与冻结契约不一致")
        if _sha256_file(run_dir / expected_path) != info["sha256"]:
            raise RuntimeError("trace 文件哈希不一致")

    trace = load_stationarity_trace(run_dir / "trace")
    expected_rounds = int(collector.DEVELOPMENT_ROUND_BUDGET)
    if (
        trace.n_records != specification["n_records"]
        or trace.query_count != specification["query_count"]
        or trace.post_round_count != expected_rounds
        or trace.state_count != expected_rounds + 1
        or trace.termination_reason != "max_rounds"
    ):
        raise RuntimeError("strict trace 的规模或终止原因不一致")
    if (
        trace.query_identity_sha256 != manifest["query_identity_sha256"]
        or trace.target_identity_sha256 != manifest["target_identity_sha256"]
    ):
        raise RuntimeError("strict trace 身份与 manifest 不一致")
    if (
        summary["rounds_run"] != expected_rounds
        or summary["termination_reason"] != "max_rounds"
        or summary["candidate_evaluation_count"] != expected_rounds
    ):
        raise RuntimeError("run summary 没有跑满冻结预算")
    if not all(
        observation["proposal_accepted"]
        for observation in trace.observations[1:]
    ):
        raise RuntimeError("no-gate 输入中出现未应用 proposal")
    if trace.observations[-1]["current_table_sha256"] != summary[
        "final_table_sha256"
    ]:
        raise RuntimeError("trace 终态与 run summary 不一致")
    if trace.observations[-1][
        "candidate_evaluation_count_cumulative"
    ] != expected_rounds:
        raise RuntimeError("trace 候选评价计数不一致")
    return trace


def audit_formal_inputs(
    input_dir: Path,
) -> tuple[List[AuditedTraceInput], Dict[str, Any]]:
    if not input_dir.is_dir():
        raise RuntimeError(f"正式轨迹目录不存在：{input_dir}")
    expected = _expected_cells()
    manifest_paths = sorted(input_dir.rglob("run_manifest.json"))
    if len(manifest_paths) != len(expected):
        raise RuntimeError(
            f"正式轨迹必须恰好有 {len(expected)} 个 manifest，"
            f"实际 {len(manifest_paths)}"
        )

    seen: set[tuple[str, int, str]] = set()
    source_commits: set[str] = set()
    protocol_hashes: set[str] = set()
    audited: List[AuditedTraceInput] = []
    pair_bindings: Dict[tuple[str, int], tuple[Any, ...]] = {}
    manifest_hashes: Dict[str, str] = {}
    clip_totals = {
        "direction_evaluated_count": 0,
        "direction_clipped_count": 0,
        "gibbs_conditional_evaluated_count": 0,
        "gibbs_conditional_clipped_count": 0,
    }
    for manifest_path in manifest_paths:
        manifest = _load_json_object(manifest_path, "run manifest")
        _validate_exact_keys(manifest, _MANIFEST_KEYS, "run manifest")
        cell = (
            manifest.get("dataset"),
            manifest.get("seed"),
            manifest.get("kernel"),
        )
        if cell not in expected or cell in seen:
            raise RuntimeError(f"未知或重复正式 cell：{cell}")
        seen.add(cell)
        run_dir = manifest_path.parent
        trace = _audit_trace_and_manifest(run_dir, manifest)
        del trace

        source_commits.add(manifest["environment"]["git_commit"])
        protocol_hashes.add(manifest["protocol_sha256"])
        pair_key = (manifest["dataset"], manifest["seed"])
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
        manifest_hash = _sha256_file(manifest_path)
        cell_name = f"{cell[0]}/seed_{cell[1]}/{cell[2]}"
        manifest_hashes[cell_name] = manifest_hash
        clip_audit = manifest["run_summary"]["clip_audit"]
        direction = clip_audit["direction"]
        gibbs = clip_audit["gibbs_conditional"]
        if (
            direction["clipped_count"] > direction["evaluated_count"]
            or gibbs["clipped_count"] > gibbs["evaluated_count"]
        ):
            raise RuntimeError("正式输入的 clip 计数非法")
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
        audited.append(AuditedTraceInput(
            dataset=str(cell[0]),
            kernel=str(cell[2]),
            seed=int(cell[1]),
            run_dir=run_dir,
            manifest=manifest,
            manifest_sha256=manifest_hash,
        ))

    if seen != expected:
        raise RuntimeError(f"正式 cell 不完整：missing={sorted(expected - seen)}")
    if len(source_commits) != 1 or len(protocol_hashes) != 1:
        raise RuntimeError("正式轨迹没有绑定同一生成提交与协议")
    if len(pair_bindings) != (
        len(collector.DATASETS) * len(collector.DEVELOPMENT_SEEDS)
    ):
        raise RuntimeError("dataset×seed 配对绑定数不完整")

    audited.sort(key=lambda row: (row.dataset, row.seed, row.kernel))
    descriptor = {
        "input_root": str(input_dir.resolve()),
        "trajectory_count": len(audited),
        "source_generator_git_commit": next(iter(source_commits)),
        "source_protocol_sha256": next(iter(protocol_hashes)),
        "maximum_round_budget": collector.DEVELOPMENT_ROUND_BUDGET,
        "development_seeds": list(collector.DEVELOPMENT_SEEDS),
        "sealed_validation_seeds_read": False,
        "paired_s0_s0_rng_binding_count": len(pair_bindings),
        "clip_audit_totals": clip_totals,
        "run_manifest_sha256": dict(sorted(manifest_hashes.items())),
    }
    _strict_json_bytes(descriptor)
    return audited, descriptor


def _round_band(round_index: int) -> str:
    budget = int(collector.DEVELOPMENT_ROUND_BUDGET)
    band_index = min(((round_index - 1) * 4) // budget, 3)
    return ROUND_BAND_LABELS[band_index]


def _flatten_check(
    item: AuditedTraceInput,
    check: Dict[str, Any],
) -> Dict[str, Any]:
    if FORBIDDEN_CLASSIFICATION_FIELDS.intersection(check):
        raise RuntimeError("量程证据意外包含阈值分类字段")
    row: Dict[str, Any] = {
        "dataset": item.dataset,
        "kernel": item.kernel,
        "seed": item.seed,
        "window_size": check["window_size"],
        "completed_block_count": check["completed_block_count"],
        "state_index": check["state_index"],
        "round_index": check["round_index"],
        "round_band": _round_band(check["round_index"]),
    }
    for window_index, (start, end) in enumerate(
        check["window_round_ranges"], start=1
    ):
        row[f"window_{window_index}_start_round"] = start
        row[f"window_{window_index}_end_round"] = end
    for metric in SCALAR_METRICS:
        row[metric] = check[metric]
    for metric in WINDOW_VECTOR_METRICS:
        values = check[metric]
        if len(values) != 3:
            raise RuntimeError(f"{metric} 必须恰好包含三个窗口值")
        singular = metric.removeprefix("window_")
        for window_index, value in enumerate(values, start=1):
            row[f"window_{window_index}_{singular}"] = value
    if FORBIDDEN_CLASSIFICATION_FIELDS.intersection(row):
        raise RuntimeError("扁平量程行意外包含阈值分类字段")
    return row


def build_range_frames(
    inputs: Sequence[AuditedTraceInput],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    range_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    for item in inputs:
        trace = load_stationarity_trace(item.run_dir / "trace")
        checks = collect_stationarity_range_evidence(
            trace, CANDIDATE_WINDOW_SIZES
        )
        range_rows.extend(_flatten_check(item, check) for check in checks)
        for observation in trace.observations:
            state_rows.append({
                "dataset": item.dataset,
                "kernel": item.kernel,
                "seed": item.seed,
                "round_index": observation["round_index"],
                "current_normalized_l1": observation[
                    "current_normalized_l1"
                ],
                "unique_row_rate": observation["unique_row_rate"],
                "normalized_row_entropy": observation[
                    "normalized_row_entropy"
                ],
                "active_round": int(
                    observation["actual_changed_row_count"] > 0
                ),
                "changed_row_fraction": (
                    observation["actual_changed_row_count"]
                    / trace.n_records
                ),
            })
    range_frame = pd.DataFrame(range_rows)
    state_frame = pd.DataFrame(state_rows)
    if range_frame.empty or state_frame.empty:
        raise RuntimeError("量程报告不得为空")
    if FORBIDDEN_CLASSIFICATION_FIELDS.intersection(range_frame.columns):
        raise RuntimeError("量程 CSV 不得包含分类字段")
    expected_checks = {
        window: int(collector.DEVELOPMENT_ROUND_BUDGET) // window - 2
        for window in CANDIDATE_WINDOW_SIZES
    }
    observed_checks = range_frame.groupby(
        ["dataset", "kernel", "seed", "window_size"]
    ).size()
    for (*_cell, window), count in observed_checks.items():
        if int(count) != expected_checks[int(window)]:
            raise RuntimeError("每条轨迹的量程检查数与冻结窗口不一致")
    expected_group_count = len(inputs) * len(CANDIDATE_WINDOW_SIZES)
    if len(observed_checks) != expected_group_count:
        raise RuntimeError("量程检查没有完整覆盖 trajectory×window")
    observed_states = state_frame.groupby(
        ["dataset", "kernel", "seed"]
    ).size()
    expected_states = int(collector.DEVELOPMENT_ROUND_BUDGET) + 1
    if len(observed_states) != len(inputs) or not all(
        int(count) == expected_states for count in observed_states
    ):
        raise RuntimeError("当前态序列没有完整覆盖全部轨迹")
    return range_frame, state_frame


def _quantile_summary(values: pd.Series) -> Dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("描述性量程必须由有限非空数值构成")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10, method="linear")),
        "p25": float(np.percentile(array, 25, method="linear")),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p75": float(np.percentile(array, 75, method="linear")),
        "p90": float(np.percentile(array, 90, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "max": float(np.max(array)),
    }


def build_descriptive_summary(
    range_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    source_audit: Dict[str, Any],
) -> Dict[str, Any]:
    range_groups = []
    group_columns = ["dataset", "kernel", "window_size", "round_band"]
    for keys, group in range_frame.groupby(group_columns, sort=True):
        dataset, kernel, window_size, round_band = keys
        range_groups.append({
            "dataset": dataset,
            "kernel": kernel,
            "window_size": int(window_size),
            "round_band": round_band,
            "terminal_check_count": int(len(group)),
            "terminal_round_min": int(group["round_index"].min()),
            "terminal_round_max": int(group["round_index"].max()),
            "metrics": {
                metric: _quantile_summary(group[metric])
                for metric in SCALAR_METRICS
            },
        })

    trajectories = []
    for keys, group in state_frame.groupby(
        ["dataset", "kernel", "seed"], sort=True
    ):
        dataset, kernel, seed = keys
        ordered = group.sort_values("round_index")
        l1 = ordered["current_normalized_l1"].to_numpy(dtype=float)
        trajectories.append({
            "dataset": dataset,
            "kernel": kernel,
            "seed": int(seed),
            "initial_normalized_l1": float(l1[0]),
            "final_normalized_l1": float(l1[-1]),
            "minimum_observed_normalized_l1": float(np.min(l1)),
            "maximum_observed_normalized_l1": float(np.max(l1)),
        })

    expected_counts = {
        str(window): int(collector.DEVELOPMENT_ROUND_BUDGET) // window - 2
        for window in CANDIDATE_WINDOW_SIZES
    }
    result = {
        "contract_version": RANGE_REPORT_CONTRACT_VERSION,
        "range_evidence_contract_version": (
            STATIONARITY_RANGE_EVIDENCE_CONTRACT_VERSION
        ),
        "role": {
            "purpose": "descriptive_range_finding_only",
            "threshold_parameters": "absent",
            "stationarity_or_stall_classification": "absent",
            "candidate_stop_round": "absent",
            "generator_rerun": False,
            "validation_seed_access": False,
        },
        "source_audit": source_audit,
        "candidate_window_sizes": list(CANDIDATE_WINDOW_SIZES),
        "window_semantics": {
            "three_adjacent_nonoverlapping_blocks": True,
            "checks_only_at_completed_block_boundaries": True,
            "initial_state_excluded": True,
            "expected_checks_per_trace": expected_counts,
        },
        "metric_descriptions_zh": {
            **STABILITY_METRICS,
            **MOVEMENT_METRICS,
        },
        "range_check_row_count": int(len(range_frame)),
        "state_row_count": int(len(state_frame)),
        "trajectory_descriptions": trajectories,
        "range_groups": range_groups,
    }
    _strict_json_bytes(result)
    return result


def _cell_axes() -> tuple[Any, np.ndarray]:
    plt = _load_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    return figure, axes


def _load_pyplot() -> Any:
    """Load the optional plotting backend only for formal report output."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _cell_position(dataset: str, kernel: str) -> tuple[int, int]:
    return list(collector.DATASETS).index(dataset), list(
        collector.KERNELS
    ).index(kernel)


def _save_raw_l1_plot(state_frame: pd.DataFrame, path: Path) -> None:
    plt = _load_pyplot()
    figure, axes = _cell_axes()
    colors = {200: "#1f77b4", 201: "#ff7f0e", 202: "#2ca02c"}
    for (dataset, kernel, seed), group in state_frame.groupby(
        ["dataset", "kernel", "seed"], sort=True
    ):
        row, column = _cell_position(dataset, kernel)
        axes[row, column].plot(
            group["round_index"],
            group["current_normalized_l1"],
            color=colors[int(seed)],
            linewidth=0.8,
            alpha=0.8,
            label=f"seed {seed}",
        )
    for dataset in collector.DATASETS:
        for kernel in collector.KERNELS:
            row, column = _cell_position(dataset, kernel)
            axis = axes[row, column]
            axis.set_title(f"{dataset} | {kernel}")
            axis.set_xlabel("Round (轮次)")
            axis.set_ylabel("Current normalized L1 (当前态归一化 L1)")
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("Raw current-state L1 trajectories (原始当前态 L1 轨迹)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_metric_plot(
    range_frame: pd.DataFrame,
    metric: str,
    description: str,
    path: Path,
) -> None:
    plt = _load_pyplot()
    figure, axes = _cell_axes()
    color_map = plt.get_cmap("viridis")
    colors = {
        window: color_map(index / (len(CANDIDATE_WINDOW_SIZES) - 1))
        for index, window in enumerate(CANDIDATE_WINDOW_SIZES)
    }
    for (dataset, kernel, window), group in range_frame.groupby(
        ["dataset", "kernel", "window_size"], sort=True
    ):
        row, column = _cell_position(dataset, kernel)
        axis = axes[row, column]
        by_round = group.groupby("round_index", sort=True)[metric]
        rounds = np.asarray(sorted(group["round_index"].unique()), dtype=int)
        lower = by_round.min().reindex(rounds).to_numpy(dtype=float)
        median = by_round.median().reindex(rounds).to_numpy(dtype=float)
        upper = by_round.max().reindex(rounds).to_numpy(dtype=float)
        color = colors[int(window)]
        axis.plot(
            rounds,
            median,
            color=color,
            linewidth=1.2,
            label=f"W={window}",
        )
        axis.fill_between(rounds, lower, upper, color=color, alpha=0.10)
    for dataset in collector.DATASETS:
        for kernel in collector.KERNELS:
            row, column = _cell_position(dataset, kernel)
            axis = axes[row, column]
            axis.set_title(f"{dataset} | {kernel}")
            axis.set_xlabel("Terminal round (检查终止轮次)")
            axis.set_ylabel(metric)
            if metric in STABILITY_METRICS:
                positive = range_frame.loc[
                    (range_frame["dataset"] == dataset)
                    & (range_frame["kernel"] == kernel)
                    & (range_frame[metric] > 0.0),
                    metric,
                ]
                linear_threshold = (
                    max(float(positive.min()) * 0.5, 1e-15)
                    if not positive.empty else 1e-15
                )
                axis.set_yscale("symlog", linthresh=linear_threshold)
            axis.grid(alpha=0.25)
            axis.legend(ncol=2, fontsize=8)
    figure.suptitle(f"{metric}\n{description}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"range report 已存在，拒绝覆盖：{output_dir}")
    environment = analysis_environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 range report 要求干净工作树")
    inputs, source_audit = audit_formal_inputs(input_dir)
    range_frame, state_frame = build_range_frames(inputs)
    summary = build_descriptive_summary(
        range_frame, state_frame, source_audit
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.partial-",
    ))
    try:
        range_path = temporary / "range_checks.csv"
        states_path = temporary / "current_state_series.csv"
        summary_path = temporary / "range_summary.json"
        range_frame.to_csv(range_path, index=False, float_format="%.17g")
        state_frame.to_csv(states_path, index=False, float_format="%.17g")
        _write_json_exclusive(summary_path, summary)

        plots_dir = temporary / "plots"
        plots_dir.mkdir()
        _save_raw_l1_plot(state_frame, plots_dir / "raw_current_l1.png")
        for metric, description in {
            **STABILITY_METRICS,
            **MOVEMENT_METRICS,
        }.items():
            _save_metric_plot(
                range_frame,
                metric,
                description,
                plots_dir / f"evidence_{metric}.png",
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
            "contract_version": RANGE_REPORT_CONTRACT_VERSION,
            "formal_threshold_free_range_report": True,
            "candidate_window_sizes": list(CANDIDATE_WINDOW_SIZES),
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
        "contract_version": RANGE_REPORT_CONTRACT_VERSION,
        "mode": "plan_only_no_trace_read",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "candidate_window_sizes": list(CANDIDATE_WINDOW_SIZES),
        "expected_trajectory_count": len(_expected_cells()),
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
        default=Path("outputs/issue53_stage2b_range_report"),
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
    print(f"threshold-free range report -> {destination}", flush=True)


if __name__ == "__main__":
    main()
