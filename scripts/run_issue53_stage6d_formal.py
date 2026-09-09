#!/usr/bin/env python3
"""采集 Issue #53 第 6D 阶段冻结的两数据三方法正式长轨迹。"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from scripts import freeze_issue53_test_query_workload_ab as blind_identity
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as protocol
from scripts import run_issue53_stage6c_joint_smoke as gpu_helpers
from table_diffevo.evolution import run_evolution
from table_diffevo.experiment_parallel import run_ordered_process_tasks
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema
from table_diffevo.stationarity import (
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)

CASE_MANIFEST = "case_manifest.json"
TERMINAL_TABLE = "terminal_current.csv"
CHECKPOINT_ARTIFACT = "checkpoint_query_answers.json"
TRANSITION_AUDIT = "transition_audit.json"
STAGING_MANIFEST = "staging_manifest.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_strict_json_text(value), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_clean_worktree(root: Path) -> str:
    status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("第 6D 正式采集要求包含未跟踪文件在内的干净工作树")
    return _git_text(root, "rev-parse", "HEAD")


def _generation_input_audit(root: Path) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for dataset, spec in protocol.DATASETS.items():
        observed[dataset] = {}
        for key in ("schema", "queries", "marginals"):
            digest = protocol.file_sha256(root / spec[key])
            if digest != spec["input_sha256"][key]:
                raise RuntimeError(f"{dataset}.{key} SHA-256 漂移")
            observed[dataset][key] = digest
    return observed


def _query_target_identity_audit(
    root: Path,
    dataset: str,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str]]:
    spec = protocol.DATASETS[dataset]
    payload = _load_json(root / spec["queries"])
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        raise TypeError(f"{dataset} 原始查询列表缺失")
    if len(raw_queries) != spec["query_count"]:
        raise RuntimeError(f"{dataset} 原始查询数量漂移")
    query_set_sha = blind_identity.query_set_identity(raw_queries)
    if query_set_sha != spec["query_identity_sha256"]:
        raise RuntimeError(f"{dataset} 结果盲查询集合身份漂移")
    if blind_identity._order_counts(raw_queries) != spec["order_counts"]:
        raise RuntimeError(f"{dataset} 查询阶数构成漂移")
    raw_targets = []
    for index, query in enumerate(raw_queries):
        value = query.get("result")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= spec["n_records"]
        ):
            raise RuntimeError(f"{dataset} 第{index}个目标不是合法整数计数")
        raw_targets.append(value)
    target_set_sha = protocol.canonical_sha256(raw_targets)
    if target_set_sha != spec["target_vector_sha256"]:
        raise RuntimeError(f"{dataset} 结果盲目标向量身份漂移")

    queries = load_queries(str(root / spec["queries"]))
    targets = np.asarray([query["result"] for query in queries], dtype=float)
    if len(queries) != len(raw_queries) or not np.array_equal(
        targets, np.asarray(raw_targets, dtype=float)
    ):
        raise RuntimeError(f"{dataset} 查询加载结果与原始载荷漂移")
    trace_query_sha = ordered_query_identity_sha256(queries)
    trace_target_sha = target_answer_identity_sha256(targets)
    if trace_query_sha != spec["trace_query_identity_sha256"]:
        raise RuntimeError(f"{dataset} 状态轨迹查询身份漂移")
    if trace_target_sha != spec["trace_target_vector_sha256"]:
        raise RuntimeError(f"{dataset} 状态轨迹目标身份漂移")
    return (
        queries,
        targets,
        {
            "query_identity_sha256": query_set_sha,
            "target_vector_sha256": target_set_sha,
            "trace_query_identity_sha256": trace_query_sha,
            "trace_target_vector_sha256": trace_target_sha,
        },
    )


def _gpu_idle_audit(shard_id: str) -> dict[str, Any]:
    if shard_id not in protocol.SHARD_ORDER:
        raise ValueError(f"未知第 6D 执行分片：{shard_id!r}")
    shard = protocol.EXECUTION_SHARDS[shard_id]
    expected_gpu = shard["expected_gpu"]
    hostname = platform.node()
    if hostname != shard["hostname"]:
        raise RuntimeError(
            f"{shard_id} 主机身份不一致：{hostname!r}，expected={shard['hostname']!r}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_gpu["cuda_visible_devices"]:
        raise RuntimeError(
            f"{shard_id} 的 CUDA_VISIBLE_DEVICES 必须精确等于 "
            f"{expected_gpu['cuda_visible_devices']}"
        )
    physical_index = int(expected_gpu["physical_index"])
    identity = gpu_helpers._parse_csv_row(
        gpu_helpers._nvidia_smi(
            f"--id={physical_index}",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        4,
    )

    import pandas as pd
    import torch

    torch.use_deterministic_algorithms(True)
    snapshot = {
        "shard_id": shard_id,
        "hostname": hostname,
        "physical_index": int(identity[0]),
        "cuda_visible_devices": visible,
        "uuid": identity[1],
        "name": identity[2],
        "memory_total_mib": int(identity[3]),
        "process_device": "cuda:0",
        "visible_device_count": int(torch.cuda.device_count()),
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for key, expected in expected_gpu.items():
        if snapshot.get(key) != expected:
            raise RuntimeError(
                f"{shard_id} 显卡身份不一致：{key}="
                f"{snapshot.get(key)!r}，expected={expected!r}"
            )
    for key, expected in protocol.EXPECTED_SOFTWARE.items():
        if snapshot.get(key) != expected:
            raise RuntimeError(
                f"{shard_id} 软件环境不一致：{key}="
                f"{snapshot.get(key)!r}，expected={expected!r}"
            )
    if snapshot["cuda_available"] is not True:
        raise RuntimeError("CUDA（显卡计算）不可用")
    if snapshot["deterministic_algorithms"] is not True:
        raise RuntimeError("显卡确定性算法没有开启")

    row = gpu_helpers._parse_csv_row(
        gpu_helpers._nvidia_smi(
            f"--id={physical_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ),
        3,
    )
    utilization = int(row[0])
    memory_used = int(row[1])
    processes = gpu_helpers._nvidia_smi(
        f"--id={physical_index}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ).strip()
    if utilization != 0 or processes:
        raise RuntimeError(f"{shard_id} 冻结显卡当前不空闲，禁止启动或恢复正式采集")
    if memory_used > 256:
        raise RuntimeError(
            f"{shard_id} 显卡虽无计算进程但基础显存为 "
            f"{memory_used} MiB，超过256 MiB护栏"
        )
    snapshot["preflight_utilization_percent"] = utilization
    snapshot["preflight_memory_used_mib"] = memory_used
    snapshot["preflight_compute_processes"] = []
    return snapshot


def _sample_gpu(physical_index: int) -> dict[str, Any]:
    row = gpu_helpers._parse_csv_row(
        gpu_helpers._nvidia_smi(
            f"--id={physical_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ),
        3,
    )
    return {
        "elapsed_sec": 0.0,
        "physical_index": int(physical_index),
        "utilization_percent": int(row[0]),
        "memory_used_mib": int(row[1]),
        "memory_total_mib": int(row[2]),
    }


class _GpuMonitor:
    """只采样当前分片冻结的物理显卡，不复用写死物理1号的旧监控器。"""

    def __init__(self, physical_index: int, interval_sec: float | None = None):
        self.physical_index = int(physical_index)
        self.interval_sec = float(
            gpu_helpers.GPU_SAMPLE_INTERVAL_SEC
            if interval_sec is None
            else interval_sec
        )
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self) -> None:
        try:
            sample = _sample_gpu(self.physical_index)
            sample["elapsed_sec"] = float(time.perf_counter() - self._started)
            self.samples.append(sample)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            self.error = f"{type(exc).__name__}: {exc}"
            self._stop.set()

    def _loop(self) -> None:
        self._record()
        while not self._stop.wait(self.interval_sec):
            self._record()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("显卡监控不得重复启动")
        self._started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop,
            name="issue53-stage6d-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def finish(self) -> list[dict[str, Any]]:
        if self._thread is None:
            raise RuntimeError("显卡监控尚未启动")
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("显卡监控线程无法停止")
        if self.error is not None:
            raise RuntimeError(f"显卡利用率监控失败：{self.error}")
        if not self.samples:
            raise RuntimeError("显卡利用率监控没有采到样本")
        if any(
            sample.get("physical_index") != self.physical_index
            for sample in self.samples
        ):
            raise RuntimeError("显卡利用率样本物理编号漂移")
        return list(self.samples)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("正式 artifact 不允许非有限浮点数")
        return float(value)
    return value


def _preflight_generator_param_manifests(
    tasks: Sequence[joint.JointTrajectoryTask],
) -> tuple[dict[str, dict[str, Any]], str]:
    """在占用显卡前验证全部正式任务的参数清单都能严格写入 JSON。"""

    manifests: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if task.task_id in manifests:
            raise RuntimeError(f"正式任务地址重复：{task.task_id}")
        manifest = protocol.generator_params_manifest(
            task.dataset,
            task.arm,
            task.seed,
        )
        if _json_safe(manifest) != manifest:
            raise RuntimeError(f"正式生成参数清单不可严格序列化：{task.task_id}")
        manifests[task.task_id] = manifest
    if len(manifests) != len(tasks):
        raise RuntimeError("正式生成参数清单预检数量不完整")
    digest = protocol.canonical_sha256(manifests)
    formal_task_ids = tuple(task.task_id for task in protocol.task_plan().tasks)
    if tuple(task.task_id for task in tasks) == formal_task_ids and (
        manifests != protocol.generator_params_manifest_matrix()
        or digest != protocol.generator_params_manifest_sha256()
    ):
        raise RuntimeError("正式30条生成参数清单统一身份漂移")
    return manifests, digest


def _frame_sha256(frame: Any) -> str:
    return gpu_helpers._frame_sha256(frame)


def _metrics_from_answers(
    target: np.ndarray,
    answers: np.ndarray,
    n_records: int,
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=float)
    answer_array = np.asarray(answers, dtype=float)
    if isinstance(n_records, bool) or not isinstance(n_records, int) or n_records <= 0:
        raise ValueError("正式表行数必须是正整数")
    if (
        target_array.shape != answer_array.shape
        or target_array.ndim != 1
        or target_array.size == 0
    ):
        raise ValueError("目标与检查点查询向量形状不一致")
    if not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(answer_array)):
        raise ValueError("目标或检查点查询答案含非有限值")
    if not np.all(target_array == np.rint(target_array)):
        raise ValueError("正式目标答案不是整数计数")
    if not np.all(answer_array == np.rint(answer_array)):
        raise ValueError("检查点查询答案不是整数计数")
    if (
        np.any(target_array < 0)
        or np.any(target_array > n_records)
        or np.any(answer_array < 0)
        or np.any(answer_array > n_records)
    ):
        raise ValueError("目标或检查点查询答案超出 [0, N] 计数范围")
    target_int = target_array.astype(np.int64)
    answer_int = answer_array.astype(np.int64)
    absolute = np.abs(target_int - answer_int)
    count_sum = int(sum(int(value) for value in absolute))
    squared_sum = sum(
        (int(target_value) - int(answer_value)) ** 2
        for target_value, answer_value in zip(target_int, answer_int, strict=True)
    )
    gap_sum = math.fsum(
        int(error) / max(int(target_value), 8)
        for error, target_value in zip(absolute, target_int, strict=True)
    )
    return {
        "absolute_count_error_sum": count_sum,
        "normalized_l1": float(count_sum / (len(absolute) * n_records)),
        "squared_loss": float(squared_sum / 2),
        "gap_e": float(gap_sum / len(absolute)),
    }


def _extract_checkpoint_artifact(
    task: joint.JointTrajectoryTask,
    diagnostics: dict[str, Any],
    target: np.ndarray,
    *,
    applied_rounds: int,
) -> dict[str, Any]:
    trace = diagnostics.get("stationarity_trace")
    if trace is None:
        raise RuntimeError(f"{task.task_id} 缺少 stationarity trace（状态轨迹）")
    spec = protocol.DATASETS[task.dataset]
    if trace.query_identity_sha256 != spec["trace_query_identity_sha256"]:
        raise RuntimeError(f"{task.task_id} 状态轨迹查询身份漂移")
    if trace.target_identity_sha256 != spec["trace_target_vector_sha256"]:
        raise RuntimeError(f"{task.task_id} 状态轨迹目标身份漂移")
    answers = np.asarray(trace.measured_query_answers, dtype=float)
    metrics = diagnostics["current_state_metrics_history"]
    expected_states = applied_rounds + 1
    if answers.shape != (expected_states, len(target)):
        raise RuntimeError(f"{task.task_id} 状态查询向量形状不完整")
    if len(metrics) != expected_states or len(trace.observations) != expected_states:
        raise RuntimeError(f"{task.task_id} 状态指标或观察数量不完整")

    def state_row(state_index: int, *, kind: str) -> dict[str, Any]:
        computed = _metrics_from_answers(
            target, answers[state_index], protocol.DATASETS[task.dataset]["n_records"]
        )
        recorded = metrics[state_index]
        if int(recorded["state_index"]) != state_index:
            raise RuntimeError(f"{task.task_id} 状态编号漂移")
        if not math.isclose(
            float(recorded["current_normalized_l1"]),
            computed["normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"{task.task_id} 检查点 L1 复算不一致")
        if float(recorded["current_squared_loss"]) != computed["squared_loss"]:
            raise RuntimeError(f"{task.task_id} 检查点平方误差复算不一致")
        return {
            "kind": kind,
            "state_index": state_index,
            "round": int(recorded["round"]),
            "phase": recorded["phase"],
            "query_answers": [int(value) for value in answers[state_index]],
            **computed,
        }

    fixed = [
        state_row(round_index, kind="fixed_checkpoint")
        for round_index in protocol.CHECKPOINT_ROUNDS
        if round_index <= applied_rounds
    ]
    terminal = state_row(applied_rounds, kind="terminal")
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "task_id": task.task_id,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "n_records": protocol.DATASETS[task.dataset]["n_records"],
        "query_count": len(target),
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": trace.query_identity_sha256,
        "trace_target_vector_sha256": trace.target_identity_sha256,
        "fixed_checkpoint_rounds_requested": list(protocol.CHECKPOINT_ROUNDS),
        "fixed_checkpoints": fixed,
        "terminal": terminal,
        "historical_best_included": False,
    }


def _single_attempts(
    task: joint.JointTrajectoryTask,
    nested: Sequence[Sequence[dict[str, Any]]],
    applied_rounds: int,
    label: str,
) -> list[dict[str, Any]]:
    if len(nested) != applied_rounds:
        raise RuntimeError(f"{task.task_id} {label} 诊断轮数不完整")
    flattened = []
    for round_index, attempts in enumerate(nested, start=1):
        if len(attempts) != 1:
            raise RuntimeError(f"{task.task_id} 第{round_index}轮不是唯一尝试")
        flattened.append(attempts[0])
    return flattened


def _extract_transition_audit(
    task: joint.JointTrajectoryTask,
    diagnostics: dict[str, Any],
    *,
    applied_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != applied_rounds:
        raise RuntimeError(f"{task.task_id} 转移时钟数量不完整")
    gap_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    gap_actual = 0
    gap_expected = 0
    nonfinite = 0

    if task.arm == protocol.ARM_GAP:
        flattened = _single_attempts(
            task,
            diagnostics["gap_l1_attempt_diagnostics_history"],
            applied_rounds,
            "缺口核",
        )
        for round_index, item in enumerate(flattened, start=1):
            if item.get("no_gate") is not True or int(item["n_sweeps"]) != 8:
                raise RuntimeError(f"{task.task_id} 缺口核无门控或扫描身份漂移")
            scan_applied = item.get("gap_l1_scan_applied") is True
            actual = int(item["gibbs_microsteps"])
            expected = (
                int(item["n_sweeps"]) * int(item["active_switches_k"])
                if scan_applied
                else 0
            )
            item_nonfinite = int(item.get("nonfinite_condition_count", 0))
            item_clip = int(item.get("clip_hit_count", 0))
            if actual != expected or item_nonfinite or item_clip:
                raise RuntimeError(f"{task.task_id} 缺口核第{round_index}轮护栏失败")
            gap_actual += actual
            gap_expected += expected
            nonfinite += item_nonfinite
            gap_rows.append(
                {
                    "round": round_index,
                    "scan_applied": scan_applied,
                    "active_switches_k": int(item["active_switches_k"]),
                    "n_sweeps": int(item["n_sweeps"]),
                    "microsteps": actual,
                    "expected_microsteps": expected,
                    "clip_hit_count": item_clip,
                    "nonfinite_condition_count": item_nonfinite,
                }
            )
        if gap_actual != int(diagnostics["gap_l1_microsteps"]):
            raise RuntimeError(f"{task.task_id} 缺口核累计微步不一致")
        if applied_rounds and diagnostics["gap_l1_reference_scale"] is None:
            raise RuntimeError(f"{task.task_id} 已推进但未建立缺口固定尺度")
    elif task.arm == "factor_b_s8":
        flattened = _single_attempts(
            task,
            diagnostics["factorized_gibbs_attempt_diagnostics_history"],
            applied_rounds,
            "因子核",
        )
        for round_index, item in enumerate(flattened, start=1):
            guard = item.get("factor_conditional_logit_diagnostics")
            if not isinstance(guard, dict) or guard.get("all_finite") is not True:
                raise RuntimeError(f"{task.task_id} 因子核第{round_index}轮非有限")
            factor_rows.append(
                {
                    "round": round_index,
                    "all_finite": True,
                    "gibbs_microsteps": int(item.get("gibbs_microsteps", 0)),
                }
            )

    compact_clocks = []
    for round_index, clock in enumerate(clocks, start=1):
        attempts = clock.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise RuntimeError(f"{task.task_id} 第{round_index}轮转移尝试数漂移")
        attempt = attempts[0]
        if int(clock["accepted_attempt"]) != 1:
            raise RuntimeError(
                f"{task.task_id} 第{round_index}轮不是无条件应用唯一候选"
            )
        compact_clocks.append(
            {
                "state_index": int(clock["state_index"]),
                "round": int(clock["round"]),
                "accepted_attempt": 1,
                "candidate_evaluation_count_cumulative": int(
                    clock["candidate_evaluation_count_cumulative"]
                ),
                "post_current_table_sha256": clock["post_current_table_sha256"],
                "primary_rng_state_sha256": clock["primary_rng_state_sha256"],
                "factorized_gibbs_rng_state_sha256": clock[
                    "factorized_gibbs_rng_state_sha256"
                ],
                "participating_rows": int(attempt["participating_rows"]),
                "changed_rows": int(attempt["changed_rows"]),
                "changed_cells": int(attempt["changed_cells"]),
                "changed_queries": int(attempt["changed_queries"]),
                "gibbs_microsteps": int(attempt["gibbs_microsteps"]),
            }
        )

    artifact = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "task_id": task.task_id,
        "round_count": applied_rounds,
        "all_applied_unconditionally": True,
        "clocks": compact_clocks,
        "gap_rounds": gap_rows,
        "factor_rounds": factor_rows,
    }
    summary = {
        "gap_microsteps": gap_actual,
        "gap_expected_microsteps": gap_expected,
        "gap_8k_identity": gap_actual == gap_expected,
        "gap_reference_scale_established": bool(
            diagnostics["gap_l1_reference_scale"] is not None
            if task.arm == protocol.ARM_GAP
            else False
        ),
        "gap_reference_scale": (
            float(diagnostics["gap_l1_reference_scale"])
            if task.arm == protocol.ARM_GAP
            and diagnostics["gap_l1_reference_scale"] is not None
            else None
        ),
        "gap_clip_hit_count": int(
            diagnostics["gap_l1_clip_hit_count"] if task.arm == protocol.ARM_GAP else 0
        ),
        "factorized_gibbs_microsteps": int(diagnostics["factorized_gibbs_microsteps"]),
        "factorized_gibbs_conditional_logit_evaluated_count": int(
            diagnostics["factorized_gibbs_conditional_logit_evaluated_count"]
        ),
        "factorized_gibbs_conditional_logit_clipped_count": int(
            diagnostics["factorized_gibbs_conditional_logit_clipped_count"]
        ),
        "direction_logit_evaluated_count": int(
            sum(diagnostics["direction_logit_evaluated_count_history"])
        ),
        "direction_logit_clipped_count": int(
            sum(diagnostics["direction_logit_clipped_count_history"])
        ),
        "nonfinite_count": nonfinite,
    }
    if (
        summary["gap_clip_hit_count"]
        or summary["factorized_gibbs_conditional_logit_clipped_count"]
        or summary["direction_logit_clipped_count"]
        or summary["nonfinite_count"]
    ):
        raise RuntimeError(f"{task.task_id} 正式无截断/有限值资格失败")
    return artifact, summary


def _call_generator_silently(
    task_id: str, **generator_arguments: Any
) -> tuple[Any, dict[str, Any]]:
    discard = io.StringIO()
    try:
        with redirect_stdout(discard), redirect_stderr(discard):
            return run_evolution(**generator_arguments)
    except Exception as exc:  # noqa: BLE001 - fail closed with task identity only
        raise RuntimeError(
            f"{task_id} 完整生成器执行失败：{type(exc).__name__}"
        ) from None


def _write_case_artifacts(
    task: joint.JointTrajectoryTask,
    staging_root: Path,
    final_table: Any,
    diagnostics: dict[str, Any],
    target: np.ndarray,
    *,
    elapsed_sec: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    execution_commit: str,
    execution_shard_id: str,
    generator_params_manifest: dict[str, Any],
) -> dict[str, Any]:
    if protocol.task_shard_id(task) != execution_shard_id:
        raise RuntimeError(f"{task.task_id} 被提交到错误执行分片")
    final_dir = staging_root / "cases" / task.task_id
    if final_dir.exists():
        raise FileExistsError(f"正式 case 已存在，不覆盖：{final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{task.task_id}.tmp-", dir=final_dir.parent)
    )
    try:
        accept_history = list(diagnostics["accept_history"])
        applied_rounds = len(accept_history)
        if accept_history != [True] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 出现拒绝或未应用候选")
        if diagnostics["proposal_attempts_history"] != [1] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 出现多次尝试")
        if diagnostics["accepted_attempt_history"] != [1] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 应用候选身份不是唯一尝试")
        termination_reason = str(diagnostics["termination_reason"])
        if termination_reason not in {"candidate_budget", "exact_residual"}:
            raise RuntimeError(f"{task.task_id} 非冻结终止原因：{termination_reason}")
        expected_diagnostic_rounds = (
            applied_rounds + 1
            if termination_reason == "exact_residual"
            else applied_rounds
        )
        if int(diagnostics["rounds_run"]) != expected_diagnostic_rounds:
            raise RuntimeError(f"{task.task_id} 生成器轮次与实际状态转移不一致")
        if (
            termination_reason == "candidate_budget"
            and applied_rounds != protocol.ROUNDS
        ):
            raise RuntimeError(f"{task.task_id} 未跑满冻结固定预算")

        terminal_sha = _frame_sha256(final_table)
        expected_terminal_sha = (
            diagnostics["transition_clock_history"][-1]["post_current_table_sha256"]
            if applied_rounds
            else diagnostics["initial_table_sha256"]
        )
        if terminal_sha != expected_terminal_sha:
            raise RuntimeError(f"{task.task_id} 最后当前表与状态时钟不一致")

        checkpoint_artifact = _extract_checkpoint_artifact(
            task, diagnostics, target, applied_rounds=applied_rounds
        )
        transition_artifact, kernel_summary = _extract_transition_audit(
            task, diagnostics, applied_rounds=applied_rounds
        )

        table_path = temporary / TERMINAL_TABLE
        final_table.reset_index(drop=True).to_csv(table_path, index=False)
        if protocol.file_sha256(table_path) != terminal_sha:
            raise RuntimeError(f"{task.task_id} 终表落盘哈希不一致")
        checkpoint_path = temporary / CHECKPOINT_ARTIFACT
        _write_json(checkpoint_path, _json_safe(checkpoint_artifact))
        transition_path = temporary / TRANSITION_AUDIT
        _write_json(transition_path, _json_safe(transition_artifact))

        relative_base = Path("cases") / task.task_id
        collection_row = {
            "task_id": task.task_id,
            "execution_shard_id": execution_shard_id,
            "dataset": task.dataset,
            "arm": task.arm,
            "seed": int(task.seed),
            "requested_rounds": int(task.rounds),
            "applied_rounds": applied_rounds,
            "termination_reason": termination_reason,
            "device": "cuda:0",
            "output_table_identity": "terminal_current",
            "all_applied_unconditionally": True,
            "proposal_attempt_count": int(diagnostics["candidate_evaluation_count"]),
            "query_identity_sha256": protocol.DATASETS[task.dataset][
                "query_identity_sha256"
            ],
            "target_vector_sha256": protocol.DATASETS[task.dataset][
                "target_vector_sha256"
            ],
            "trace_query_identity_sha256": diagnostics[
                "stationarity_trace"
            ].query_identity_sha256,
            "trace_target_vector_sha256": diagnostics[
                "stationarity_trace"
            ].target_identity_sha256,
            "initial_table_sha256": diagnostics["initial_table_sha256"],
            "primary_rng_post_initialization_sha256": diagnostics[
                "primary_rng_post_initialization_state_sha256"
            ],
            "primary_rng_endpoint_sha256": diagnostics["primary_rng_state_sha256"],
            "direction_reference_scale": (
                float(diagnostics["direction_reference_scale"])
                if diagnostics["direction_reference_scale"] is not None
                else None
            ),
            "terminal_table_path": str(relative_base / TERMINAL_TABLE),
            "terminal_table_sha256": terminal_sha,
            "checkpoint_artifact_path": str(relative_base / CHECKPOINT_ARTIFACT),
            "checkpoint_artifact_sha256": protocol.file_sha256(checkpoint_path),
            "transition_audit_path": str(relative_base / TRANSITION_AUDIT),
            "transition_audit_sha256": protocol.file_sha256(transition_path),
            "elapsed_sec": float(elapsed_sec),
            "peak_allocated_bytes": int(peak_allocated_bytes),
            "peak_reserved_bytes": int(peak_reserved_bytes),
            "state_evaluation_count": int(diagnostics["state_evaluation_count"]),
            "candidate_evaluation_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "distance_evaluation_count": int(diagnostics["distance_evaluation_count"]),
            "direction_evaluation_count": int(
                diagnostics["direction_evaluation_count"]
            ),
            **kernel_summary,
        }
        manifest = {
            "contract_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
            "execution_commit": execution_commit,
            "execution_shard_id": execution_shard_id,
            "generator_params": generator_params_manifest,
            "collection_row": collection_row,
            "raw_reference_data_accessed": False,
            "method_comparison_emitted": False,
        }
        _write_json(temporary / CASE_MANIFEST, manifest)
        os.replace(temporary, final_dir)
        return collection_row
    except Exception as exc:
        raise RuntimeError(
            f"{task.task_id} 生成后产物收口失败；临时目录已保留且不得自动重跑："
            f"{temporary}"
        ) from exc


def _execute_trajectory_task(
    task: joint.JointTrajectoryTask,
    *,
    repository_root: str,
    staging_root: str,
    execution_commit: str,
    execution_shard_id: str,
    generator_params_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = Path(repository_root)
    spec = protocol.DATASETS[task.dataset]
    try:
        generator_params_manifest = generator_params_by_task[task.task_id]
    except KeyError as exc:
        raise RuntimeError(f"正式任务缺少预检参数清单：{task.task_id}") from exc
    if generator_params_manifest != protocol.generator_params_manifest(
        task.dataset,
        task.arm,
        task.seed,
    ):
        raise RuntimeError(f"正式任务预检参数清单漂移：{task.task_id}")
    schema = load_schema(str(root / spec["schema"]))
    queries, target, _identity = _query_target_identity_audit(root, task.dataset)
    marginals = load_marginals(str(root / spec["marginals"]))

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("工作进程没有恰好一张可用显卡")
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    returned_table, diagnostics = _call_generator_silently(
        task.task_id,
        target=target,
        queries=queries,
        schema=schema,
        marginals=marginals,
        **protocol.task_generator_params(task.dataset, task.arm, task.seed),
    )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    if "final_table" not in diagnostics:
        raise RuntimeError(f"{task.task_id} 缺少最后当前表")
    final_table = diagnostics.pop("final_table")
    del returned_table
    return _write_case_artifacts(
        task,
        Path(staging_root),
        final_table,
        diagnostics,
        target,
        elapsed_sec=elapsed,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        execution_commit=execution_commit,
        execution_shard_id=execution_shard_id,
        generator_params_manifest=generator_params_manifest,
    )


def _validate_case_row(task: joint.JointTrajectoryTask, row: dict[str, Any]) -> None:
    applied_rounds = row.get("applied_rounds")
    termination_reason = row.get("termination_reason")
    if (
        row.get("task_id") != task.task_id
        or row.get("execution_shard_id") != protocol.task_shard_id(task)
        or row.get("dataset") != task.dataset
        or row.get("arm") != task.arm
        or row.get("seed") != task.seed
        or row.get("requested_rounds") != task.rounds
        or row.get("device") != "cuda:0"
        or row.get("output_table_identity") != "terminal_current"
        or row.get("all_applied_unconditionally") is not True
        or not isinstance(applied_rounds, int)
        or isinstance(applied_rounds, bool)
        or not 0 <= applied_rounds <= task.rounds
        or termination_reason not in {"candidate_budget", "exact_residual"}
        or row.get("proposal_attempt_count") != applied_rounds
        or row.get("candidate_evaluation_count") != applied_rounds
        or row.get("state_evaluation_count") != applied_rounds + 1
    ):
        raise RuntimeError(f"正式 case 行身份漂移：{task.task_id}")
    if termination_reason == "candidate_budget" and applied_rounds != task.rounds:
        raise RuntimeError(f"正式 case 未跑满冻结候选预算：{task.task_id}")


def _load_completed_case(
    staging: Path,
    task: joint.JointTrajectoryTask,
    execution_commit: str,
) -> dict[str, Any] | None:
    case_dir = staging / "cases" / task.task_id
    if not case_dir.exists():
        return None
    manifest = _load_json(case_dir / CASE_MANIFEST)
    if (
        manifest.get("contract_version") != protocol.PROTOCOL_VERSION
        or manifest.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or manifest.get("execution_commit") != execution_commit
        or manifest.get("execution_shard_id") != protocol.task_shard_id(task)
        or manifest.get("generator_params")
        != protocol.generator_params_manifest(task.dataset, task.arm, task.seed)
        or manifest.get("raw_reference_data_accessed") is not False
        or manifest.get("method_comparison_emitted") is not False
    ):
        raise RuntimeError(f"恢复 case 协议身份漂移：{task.task_id}")
    row = manifest.get("collection_row")
    if not isinstance(row, dict):
        raise TypeError(f"恢复 case 缺少 collection row：{task.task_id}")
    _validate_case_row(task, row)
    for path_key, sha_key in (
        ("terminal_table_path", "terminal_table_sha256"),
        ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
        ("transition_audit_path", "transition_audit_sha256"),
    ):
        artifact = staging / row[path_key]
        if protocol.file_sha256(artifact) != row[sha_key]:
            raise RuntimeError(f"恢复 case artifact 漂移：{task.task_id}/{path_key}")
    return row


def _partition_completed_cases(
    staging: Path,
    tasks: Sequence[joint.JointTrajectoryTask],
    execution_commit: str,
) -> tuple[dict[str, dict[str, Any]], list[joint.JointTrajectoryTask]]:
    """严格复核已完成案例，只返回尚未完成、允许执行的任务。"""

    completed: dict[str, dict[str, Any]] = {}
    pending: list[joint.JointTrajectoryTask] = []
    for task in tasks:
        row = _load_completed_case(staging, task, execution_commit)
        if row is None:
            pending.append(task)
        else:
            completed[task.task_id] = row
    if len(completed) + len(pending) != len(tasks):
        raise RuntimeError("正式恢复任务划分不完整")
    return completed, pending


def _validate_pairing_for_tasks(
    rows: Sequence[dict[str, Any]],
    tasks: Sequence[joint.JointTrajectoryTask],
) -> None:
    indexed = {(row["seed"], row["dataset"], row["arm"]): row for row in rows}
    if len(indexed) != len(tasks):
        raise RuntimeError("正式分片 case 地址不完整或重复")
    blocks = {(task.seed, task.dataset) for task in tasks}
    for seed, dataset in sorted(blocks):
        expected_arms = {
            task.arm for task in tasks if task.seed == seed and task.dataset == dataset
        }
        if expected_arms != set(joint.ARM_ORDER):
            raise RuntimeError(f"{seed}/{dataset} 冻结任务不是完整三方法配对")
        try:
            paired = [indexed[(seed, dataset, arm)] for arm in joint.ARM_ORDER]
        except KeyError as exc:
            raise RuntimeError(f"{seed}/{dataset} 三方法结果不完整") from exc
        for key in (
            "initial_table_sha256",
            "primary_rng_post_initialization_sha256",
            "direction_reference_scale",
            "query_identity_sha256",
            "target_vector_sha256",
            "trace_query_identity_sha256",
            "trace_target_vector_sha256",
        ):
            if len({row[key] for row in paired}) != 1:
                raise RuntimeError(f"{seed}/{dataset} 三方法配对身份不一致：{key}")


def _validate_pairing(rows: Sequence[dict[str, Any]]) -> None:
    tasks = protocol.task_plan().tasks
    _validate_pairing_for_tasks(rows, tasks)
    if len(rows) != len(tasks):
        raise RuntimeError("正式30条 case 地址不完整或重复")


def run_frozen_task_matrix(
    worker: Callable[[joint.JointTrajectoryTask], dict[str, Any]],
    tasks: Sequence[joint.JointTrajectoryTask],
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    results = run_ordered_process_tasks(
        worker,
        list(tasks),
        max_workers=max_workers,
    )
    if len(results) != len(tasks):
        raise RuntimeError("正式任务返回数量不完整")
    for task, row in zip(tasks, results):
        _validate_case_row(task, row)
    return results


def _find_or_create_staging(
    root: Path,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    shard_id: str,
) -> tuple[Path, bool]:
    tasks = protocol.tasks_for_shard(shard_id)
    destination = root / protocol.SHARD_OUTPUT_ROOT / shard_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        path
        for path in destination.parent.glob(f".{destination.name}.staging-*")
        if path.is_dir()
    )
    if len(candidates) > 1:
        raise RuntimeError("发现多个第 6D staging（暂存）目录，禁止猜测恢复目标")
    if candidates:
        staging = candidates[0]
        manifest = _load_json(staging / STAGING_MANIFEST)
        if manifest != {
            "contract_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
            "execution_commit": execution_commit,
            "runner_sha256": runner_sha256,
            "shard_id": shard_id,
            "shard_assignment_sha256": protocol.shard_assignment_sha256(),
            "generator_params_manifest_sha256": (generator_params_manifest_sha256),
            "task_ids": [task.task_id for task in tasks],
            "partial_quality_inspection_allowed": False,
        }:
            raise RuntimeError("第 6D staging（暂存）身份漂移，禁止恢复")
        cases_root = staging / "cases"
        if cases_root.exists():
            stale = sorted(
                path for path in cases_root.glob(".*.tmp-*") if path.is_dir()
            )
            if stale:
                raise RuntimeError(
                    "发现生成后未完成收口的临时 case；已保留且禁止自动删除或重跑："
                    + ", ".join(str(path) for path in stale)
                )
        return staging, True
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    _write_json(
        staging / STAGING_MANIFEST,
        {
            "contract_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
            "execution_commit": execution_commit,
            "runner_sha256": runner_sha256,
            "shard_id": shard_id,
            "shard_assignment_sha256": protocol.shard_assignment_sha256(),
            "generator_params_manifest_sha256": (generator_params_manifest_sha256),
            "task_ids": [task.task_id for task in tasks],
            "partial_quality_inspection_allowed": False,
        },
    )
    return staging, False


def _validate_numeric_rows(rows: Sequence[dict[str, Any]]) -> None:
    if any(
        row["gap_8k_identity"] is not True
        or row["gap_clip_hit_count"] != 0
        or row["factorized_gibbs_conditional_logit_clipped_count"] != 0
        or row["direction_logit_clipped_count"] != 0
        or row["nonfinite_count"] != 0
        for row in rows
    ):
        raise RuntimeError("正式 collection 数值/微步护栏失败")


def _shard_report(
    *,
    shard_id: str,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    input_sha256: dict[str, dict[str, str]],
    gpu: dict[str, Any],
    gpu_samples: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    resumed_case_count: int,
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    tasks = protocol.tasks_for_shard(shard_id)
    task_ids = [task.task_id for task in tasks]
    if [row["task_id"] for row in rows] != task_ids:
        raise RuntimeError(f"{shard_id} 行顺序与冻结分片任务顺序不一致")
    _validate_pairing_for_tasks(rows, tasks)
    _validate_numeric_rows(rows)
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "shard_id": shard_id,
        "shard_assignment_sha256": protocol.shard_assignment_sha256(),
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha256,
        "generator_params_manifest_sha256": generator_params_manifest_sha256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gpu": gpu,
        },
        "generation_input_sha256": input_sha256,
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec_this_invocation": float(elapsed_sec),
            "resumed_case_count": int(resumed_case_count),
            "new_case_count": len(rows) - int(resumed_case_count),
            "max_workers": protocol.EXECUTION_SHARDS[shard_id]["max_workers"],
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_arm_then_dataset",
        },
        "gpu_samples": list(gpu_samples),
        "gpu_summary": (
            gpu_helpers._summarize_gpu_samples(gpu_samples) if gpu_samples else None
        ),
        "case_count": len(rows),
        "paired_dataset_seed_count": len(protocol.SHARD_BLOCKS[shard_id]),
        "task_ids": task_ids,
        "raw_results": list(rows),
        "shard_audit": {
            "all_assigned_cases_present": len(rows) == len(tasks),
            "all_assigned_dataset_seed_triplets_paired": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "all_zero_clip_and_finite": True,
            "all_artifact_sha256_verified": True,
            "all_generator_params_preflighted_before_gpu": True,
            "completed_cases_resumed_without_rerun": True,
            "all_gpu_samples_match_shard_physical_index": True,
        },
        "formal_shard_complete": True,
        "raw_reference_data_accessed": False,
        "partial_shard_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_shard": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def _validate_shard_report(
    root: Path,
    shard_id: str,
    *,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    input_sha256: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    shard_root = root / protocol.SHARD_OUTPUT_ROOT / shard_id
    report_path = shard_root / protocol.SHARD_REPORT
    report_sha256 = protocol.file_sha256(report_path)
    report = _load_json(report_path)
    tasks = protocol.tasks_for_shard(shard_id)
    expected_task_ids = [task.task_id for task in tasks]
    expected_gpu = protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]
    gpu = report.get("environment", {}).get("gpu")
    expected_shard_audit = {
        "all_assigned_cases_present": True,
        "all_assigned_dataset_seed_triplets_paired": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_shard_physical_index": True,
    }
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("shard_id") != shard_id
        or report.get("shard_assignment_sha256") != protocol.shard_assignment_sha256()
        or report.get("execution_commit") != execution_commit
        or report.get("runner_sha256") != runner_sha256
        or report.get("generator_params_manifest_sha256")
        != generator_params_manifest_sha256
        or report.get("generation_input_sha256") != input_sha256
        or report.get("task_ids") != expected_task_ids
        or report.get("case_count") != len(tasks)
        or report.get("paired_dataset_seed_count")
        != len(protocol.SHARD_BLOCKS[shard_id])
        or report.get("shard_audit") != expected_shard_audit
        or report.get("formal_shard_complete") is not True
        or report.get("raw_reference_data_accessed") is not False
        or report.get("partial_shard_comparison_emitted") is not False
        or report.get("method_ranking_emitted") is not False
        or report.get("l1_results_published_by_shard") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
    ):
        raise RuntimeError(f"{shard_id} 正式分片报告身份或边界漂移")
    if not isinstance(gpu, dict):
        raise TypeError(f"{shard_id} 缺少显卡环境")
    if (
        gpu.get("shard_id") != shard_id
        or gpu.get("hostname") != protocol.EXECUTION_SHARDS[shard_id]["hostname"]
    ):
        raise RuntimeError(f"{shard_id} 主机环境漂移")
    for key, expected in expected_gpu.items():
        if gpu.get(key) != expected:
            raise RuntimeError(f"{shard_id} 显卡环境漂移：{key}")
    for key, expected in protocol.EXPECTED_SOFTWARE.items():
        if gpu.get(key) != expected:
            raise RuntimeError(f"{shard_id} 软件环境漂移：{key}")
    gpu_samples = report.get("gpu_samples")
    if (
        not isinstance(gpu_samples, list)
        or not gpu_samples
        or any(
            sample.get("physical_index") != expected_gpu["physical_index"]
            for sample in gpu_samples
        )
    ):
        raise RuntimeError(f"{shard_id} 显卡监控样本物理编号漂移")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError(f"{shard_id} 正式分片结果数量漂移")
    loaded_rows = []
    for task, row in zip(tasks, rows):
        loaded = _load_completed_case(shard_root, task, execution_commit)
        if loaded is None or loaded != row:
            raise RuntimeError(f"{shard_id}/{task.task_id} 完整案例与分片报告不一致")
        loaded_rows.append(loaded)
    _validate_pairing_for_tasks(loaded_rows, tasks)
    _validate_numeric_rows(loaded_rows)
    return report, loaded_rows, report_sha256


def _collection_report(
    *,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    input_sha256: dict[str, dict[str, str]],
    shard_reports: dict[str, dict[str, Any]],
    shard_report_sha256: dict[str, str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    task_ids = [task.task_id for task in protocol.task_plan().tasks]
    if [row["task_id"] for row in rows] != task_ids:
        raise RuntimeError("正式 collection 行顺序与冻结任务顺序不一致")
    _validate_pairing(rows)
    _validate_numeric_rows(rows)
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "protocol": protocol.frozen_protocol_manifest(),
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha256,
        "generator_params_manifest_sha256": generator_params_manifest_sha256,
        "shard_assignment_sha256": protocol.shard_assignment_sha256(),
        "shard_report_sha256": dict(shard_report_sha256),
        "shard_report_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / protocol.SHARD_REPORT),
                "sha256": shard_report_sha256[shard_id],
            }
            for shard_id in protocol.SHARD_ORDER
        },
        "environment": {
            "shards": {
                shard_id: shard_reports[shard_id]["environment"]
                for shard_id in protocol.SHARD_ORDER
            }
        },
        "generation_input_sha256": input_sha256,
        "execution": {
            "shard_order": list(protocol.SHARD_ORDER),
            "task_counts": {
                shard_id: len(protocol.tasks_for_shard(shard_id))
                for shard_id in protocol.SHARD_ORDER
            },
            "paired_block_counts": {
                shard_id: len(protocol.SHARD_BLOCKS[shard_id])
                for shard_id in protocol.SHARD_ORDER
            },
            "max_workers_by_shard": {
                shard_id: protocol.EXECUTION_SHARDS[shard_id]["max_workers"]
                for shard_id in protocol.SHARD_ORDER
            },
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_arm_then_dataset",
            "shard_execution": {
                shard_id: shard_reports[shard_id]["execution"]
                for shard_id in protocol.SHARD_ORDER
            },
        },
        "case_count": len(rows),
        "paired_dataset_seed_count": len(protocol.FORMAL_SEEDS) * 2,
        "raw_results": list(rows),
        "collection_audit": {
            "all_30_cases_present": len(rows) == 30,
            "all_10_dataset_seed_triplets_paired": True,
            "exact_21_9_shard_assignment": True,
            "all_triplets_single_shard": True,
            "both_shard_reports_verified_before_merge": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "all_zero_clip_and_finite": True,
            "all_artifact_sha256_verified": True,
            "all_generator_params_preflighted_before_gpu": True,
            "completed_cases_resumed_without_rerun": True,
            "all_gpu_samples_match_shard_physical_index": True,
        },
        "formal_result_valid": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def build_plan() -> dict[str, Any]:
    plan = protocol.build_plan(_repo_root())
    _manifests, generator_params_sha256 = _preflight_generator_param_manifests(
        protocol.task_plan().tasks
    )
    plan.update(
        {
            "collector_wired": True,
            "collector_sha256": protocol.file_sha256(Path(__file__)),
            "formal_collection_authorized": False,
            "resume_only_from_same_protocol_commit_and_artifacts": True,
            "generator_params_manifest_sha256": generator_params_sha256,
            "all_generator_params_preflighted_without_gpu": True,
            "completed_case_directories_resumed_without_rerun": True,
            "run_mode": "two_explicit_shards_then_verified_merge",
            "shard_assignment_sha256": protocol.shard_assignment_sha256(),
            "shard_task_counts": {
                shard_id: len(protocol.tasks_for_shard(shard_id))
                for shard_id in protocol.SHARD_ORDER
            },
            "merge_requires_both_complete_shards": True,
        }
    )
    return plan


def run_shard(confirmed_protocol_sha256: str, shard_id: str) -> Path:
    protocol.require_run_confirmation(confirmed_protocol_sha256)
    if shard_id not in protocol.SHARD_ORDER:
        raise ValueError(f"未知第 6D 执行分片：{shard_id!r}")
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    execution_commit = _assert_clean_worktree(root)
    destination = root / protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"第 6D 正式输出已存在，不覆盖：{destination}")
    input_sha256 = _generation_input_audit(root)
    runner_sha256 = protocol.file_sha256(Path(__file__))
    frozen_tasks = protocol.task_plan().tasks
    generator_params_by_task, generator_params_sha256 = (
        _preflight_generator_param_manifests(frozen_tasks)
    )
    shard_destination = root / protocol.SHARD_OUTPUT_ROOT / shard_id
    if shard_destination.exists():
        _report, _rows, _sha = _validate_shard_report(
            root,
            shard_id,
            execution_commit=execution_commit,
            runner_sha256=runner_sha256,
            generator_params_manifest_sha256=generator_params_sha256,
            input_sha256=input_sha256,
        )
        return shard_destination / protocol.SHARD_REPORT
    shard_tasks = protocol.tasks_for_shard(shard_id)
    staging, _resumed = _find_or_create_staging(
        root,
        execution_commit,
        runner_sha256,
        generator_params_sha256,
        shard_id,
    )

    completed, pending = _partition_completed_cases(
        staging,
        shard_tasks,
        execution_commit,
    )

    gpu = _gpu_idle_audit(shard_id)
    gpu_samples: list[dict[str, Any]] = []
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    if pending:
        monitor = _GpuMonitor(
            protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]["physical_index"]
        )
        monitor.start()
        try:
            worker = partial(
                _execute_trajectory_task,
                repository_root=str(root),
                staging_root=str(staging),
                execution_commit=execution_commit,
                execution_shard_id=shard_id,
                generator_params_by_task=generator_params_by_task,
            )
            new_rows = run_frozen_task_matrix(
                worker,
                pending,
                max_workers=protocol.EXECUTION_SHARDS[shard_id]["max_workers"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"第 6D/{shard_id} 正式采集中断；未发布部分结论，"
                f"可在同提交原命令恢复：{staging}"
            ) from exc
        finally:
            gpu_samples = monitor.finish()
        completed.update({row["task_id"]: row for row in new_rows})
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone().isoformat()
    rows = [completed[task.task_id] for task in shard_tasks]
    report = _shard_report(
        shard_id=shard_id,
        execution_commit=execution_commit,
        runner_sha256=runner_sha256,
        generator_params_manifest_sha256=generator_params_sha256,
        input_sha256=input_sha256,
        gpu=gpu,
        gpu_samples=gpu_samples,
        rows=rows,
        resumed_case_count=len(frozen_tasks) - len(pending),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=elapsed,
    )
    _write_json(staging / protocol.SHARD_REPORT, report)
    os.replace(staging, shard_destination)
    return shard_destination / protocol.SHARD_REPORT


def merge_shards(confirmed_protocol_sha256: str) -> Path:
    protocol.require_run_confirmation(confirmed_protocol_sha256)
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    execution_commit = _assert_clean_worktree(root)
    destination = root / protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"第 6D 正式输出已存在，不覆盖：{destination}")
    input_sha256 = _generation_input_audit(root)
    runner_sha256 = protocol.file_sha256(Path(__file__))
    generator_params_by_task, generator_params_sha256 = (
        _preflight_generator_param_manifests(protocol.task_plan().tasks)
    )
    if generator_params_by_task != protocol.generator_params_manifest_matrix():
        raise RuntimeError("合并前30条生成参数清单漂移")

    shard_reports: dict[str, dict[str, Any]] = {}
    shard_rows: dict[str, dict[str, dict[str, Any]]] = {}
    shard_report_sha256: dict[str, str] = {}
    for shard_id in protocol.SHARD_ORDER:
        report, rows, report_sha = _validate_shard_report(
            root,
            shard_id,
            execution_commit=execution_commit,
            runner_sha256=runner_sha256,
            generator_params_manifest_sha256=generator_params_sha256,
            input_sha256=input_sha256,
        )
        shard_reports[shard_id] = report
        shard_rows[shard_id] = {row["task_id"]: row for row in rows}
        shard_report_sha256[shard_id] = report_sha

    rows = [
        shard_rows[protocol.task_shard_id(task)][task.task_id]
        for task in protocol.task_plan().tasks
    ]
    report = _collection_report(
        execution_commit=execution_commit,
        runner_sha256=runner_sha256,
        generator_params_manifest_sha256=generator_params_sha256,
        input_sha256=input_sha256,
        shard_reports=shard_reports,
        shard_report_sha256=shard_report_sha256,
        rows=rows,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.merge-", dir=destination.parent)
    )
    try:
        for task in protocol.task_plan().tasks:
            shard_id = protocol.task_shard_id(task)
            source = (
                root / protocol.SHARD_OUTPUT_ROOT / shard_id / "cases" / task.task_id
            )
            target = staging / "cases" / task.task_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            loaded = _load_completed_case(staging, task, execution_commit)
            if loaded != shard_rows[shard_id][task.task_id]:
                raise RuntimeError(f"合并复制后案例身份漂移：{task.task_id}")
        for shard_id in protocol.SHARD_ORDER:
            source_report = (
                root / protocol.SHARD_OUTPUT_ROOT / shard_id / protocol.SHARD_REPORT
            )
            target_report = staging / "shards" / shard_id / protocol.SHARD_REPORT
            target_report.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_report, target_report)
            if protocol.file_sha256(target_report) != shard_report_sha256[shard_id]:
                raise RuntimeError(f"合并复制后分片报告漂移：{shard_id}")
        _write_json(staging / protocol.COLLECTION_REPORT, report)
        os.replace(staging, destination)
    except Exception as exc:
        raise RuntimeError(
            f"第 6D 双分片合并失败；源分片未改动，合并暂存已保留：{staging}"
        ) from exc
    return destination / protocol.COLLECTION_REPORT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    run_parser.add_argument("--shard", choices=protocol.SHARD_ORDER, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "run-shard":
        path = run_shard(args.confirm_protocol_sha, args.shard)
        print(f"第 6D 正式 shard（分片） -> {path}")
        print(f"shard SHA-256 -> {protocol.file_sha256(path)}")
        return
    path = merge_shards(args.confirm_protocol_sha)
    print(f"第 6D 正式 collection（采集） -> {path}")
    print(f"collection SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
