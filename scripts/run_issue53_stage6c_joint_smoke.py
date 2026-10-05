#!/usr/bin/env python
"""运行 Issue #53 第 6C 阶段冻结的两数据三方法联合测速。"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from scripts import issue53_stage6c_joint_smoke_protocol as protocol
from scripts import issue53_stage6c_joint_trajectories as joint
from table_diffevo.evolution import run_evolution
from table_diffevo.experiment_parallel import run_ordered_process_tasks
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


REPORT_FILENAME = "timing_report.json"
GPU_SAMPLE_INTERVAL_SEC = 0.25

TOP_LEVEL_REPORT_FIELDS = {
    "contract_version",
    "protocol_sha256",
    "execution_commit",
    "runner_sha256",
    "environment",
    "input_sha256",
    "execution",
    "gpu_samples",
    "gpu_summary",
    "tasks",
    "summary",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


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
        raise RuntimeError("联合测速要求包含未跟踪文件在内的干净工作树")
    return _git_text(root, "rev-parse", "HEAD")


def _frame_sha256(frame: Any) -> str:
    return hashlib.sha256(
        frame.reset_index(drop=True).to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _audit_inputs(root: Path) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for dataset, spec in protocol.DATASETS.items():
        observed[dataset] = {}
        for key in ("schema", "queries", "marginals"):
            digest = protocol.file_sha256(root / spec[key])
            if digest != spec["input_sha256"][key]:
                raise RuntimeError(f"{dataset}.{key} SHA-256 漂移")
            observed[dataset][key] = digest
    return observed


def _parse_csv_row(text: str, expected_columns: int) -> list[str]:
    values = [value.strip() for value in text.strip().split(",")]
    if len(values) != expected_columns:
        raise RuntimeError(f"nvidia-smi 返回列数错误：{text!r}")
    return values


def _nvidia_smi(*arguments: str) -> str:
    return subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_gpu_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    expected = protocol.EXPECTED_GPU
    for key in (
        "physical_index",
        "cuda_visible_devices",
        "uuid",
        "name",
        "process_device",
        "visible_device_count",
    ):
        if snapshot.get(key) != expected[key]:
            raise RuntimeError(
                f"物理 1 号显卡身份不一致：{key}="
                f"{snapshot.get(key)!r}，expected={expected[key]!r}"
            )
    if not snapshot.get("cuda_available"):
        raise RuntimeError("CUDA（显卡计算）不可用")
    if snapshot.get("deterministic_algorithms") is not True:
        raise RuntimeError("显卡确定性算法没有开启")
    return snapshot


def _gpu_preflight() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != protocol.EXPECTED_GPU["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES 必须精确等于 1")
    row = _parse_csv_row(
        _nvidia_smi(
            "--id=1",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        4,
    )

    import torch

    torch.use_deterministic_algorithms(True)
    snapshot = {
        "physical_index": int(row[0]),
        "cuda_visible_devices": visible,
        "uuid": row[1],
        "name": row[2],
        "memory_total_mib": int(row[3]),
        "process_device": "cuda:0",
        "visible_device_count": int(torch.cuda.device_count()),
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    return _validate_gpu_snapshot(snapshot)


def _sample_gpu() -> dict[str, Any]:
    row = _parse_csv_row(
        _nvidia_smi(
            "--id=1",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ),
        3,
    )
    return {
        "elapsed_sec": 0.0,
        "utilization_percent": int(row[0]),
        "memory_used_mib": int(row[1]),
        "memory_total_mib": int(row[2]),
    }


class _GpuMonitor:
    def __init__(self, interval_sec: float = GPU_SAMPLE_INTERVAL_SEC):
        self.interval_sec = float(interval_sec)
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self) -> None:
        try:
            sample = _sample_gpu()
            sample["elapsed_sec"] = float(time.perf_counter() - self._started)
            self.samples.append(sample)
        except Exception as exc:  # pragma: no cover - hardware failure path
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
            name="issue53-stage6c-gpu-monitor",
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
        return list(self.samples)


def _summarize_gpu_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("显卡样本不得为空")
    utilization = np.asarray(
        [sample["utilization_percent"] for sample in samples], dtype=float
    )
    memory = np.asarray(
        [sample["memory_used_mib"] for sample in samples], dtype=np.int64
    )
    totals = {int(sample["memory_total_mib"]) for sample in samples}
    if len(totals) != 1:
        raise RuntimeError("显卡总显存采样不一致")
    return {
        "sample_count": len(samples),
        "sample_interval_sec": GPU_SAMPLE_INTERVAL_SEC,
        "utilization_mean_percent": float(utilization.mean()),
        "utilization_max_percent": int(utilization.max()),
        "utilization_nonzero_fraction": float(np.mean(utilization > 0.0)),
        "memory_used_min_mib": int(memory.min()),
        "memory_used_max_mib": int(memory.max()),
        "memory_total_mib": totals.pop(),
    }


def _expected_gap_microsteps(
    nested_attempts: Sequence[Sequence[dict[str, Any]]],
) -> tuple[int, int]:
    actual = 0
    expected = 0
    nonfinite = 0
    for attempts in nested_attempts:
        if len(attempts) != 1:
            raise RuntimeError("无重试测速每轮必须恰好一个缺口核诊断")
        diagnostic = attempts[0]
        if diagnostic.get("no_gate") is not True:
            raise RuntimeError("缺口核诊断没有确认无门控")
        if int(diagnostic["n_sweeps"]) != int(
            joint.arm_kernel_params("gap_l1_global_s8")["gap_l1_sweeps"]
        ):
            raise RuntimeError("缺口核诊断的扫描次数不是冻结值")
        actual += int(diagnostic["gibbs_microsteps"])
        expected += int(diagnostic["n_sweeps"]) * int(
            diagnostic["active_switches_k"]
        )
        if diagnostic.get("gap_l1_scan_applied") is True:
            nonfinite += int(diagnostic["nonfinite_condition_count"])
        elif int(diagnostic["gibbs_microsteps"]) != 0:
            raise RuntimeError("未定尺缺口轮不得执行扫描微步")
    return actual, expected, nonfinite


def _assert_factor_diagnostics_finite(
    nested_attempts: Sequence[Sequence[dict[str, Any]]],
) -> int:
    nonfinite = 0
    for attempts in nested_attempts:
        if len(attempts) != 1:
            raise RuntimeError("无重试测速每轮必须恰好一个因子核诊断")
        guard = attempts[0].get("factor_conditional_logit_diagnostics")
        if not isinstance(guard, dict) or "all_finite" not in guard:
            raise RuntimeError("因子核诊断缺少有限值护栏")
        if guard["all_finite"] is not True:
            nonfinite += 1
    return nonfinite


def _build_task_report(
    task: joint.JointTrajectoryTask,
    returned_table: Any,
    diagnostics: dict[str, Any],
    *,
    elapsed_sec: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    if "final_table" not in diagnostics:
        raise RuntimeError(f"{task.task_id} 缺少最后当前表")
    final_table = diagnostics.pop("final_table")
    terminal_sha = _frame_sha256(final_table)
    diagnostic_rounds = int(diagnostics["rounds_run"])
    accept_history = list(diagnostics["accept_history"])
    clocks = list(diagnostics["transition_clock_history"])
    rounds_run = len(accept_history)
    termination_reason = str(diagnostics["termination_reason"])
    expected_diagnostic_rounds = (
        rounds_run + 1 if termination_reason == "exact_residual" else rounds_run
    )
    if diagnostic_rounds != expected_diagnostic_rounds:
        raise RuntimeError(f"{task.task_id} 生成器轮次与终止原因不一致")
    if len(clocks) != rounds_run:
        raise RuntimeError(f"{task.task_id} 轮次、应用记录与时钟数量不一致")
    if accept_history != [True] * rounds_run:
        raise RuntimeError(f"{task.task_id} 出现非无条件应用轮次")

    expected_terminal_sha = (
        clocks[-1]["post_current_table_sha256"]
        if clocks else diagnostics["initial_table_sha256"]
    )
    if terminal_sha != expected_terminal_sha:
        raise RuntimeError(f"{task.task_id} 最后当前表哈希与状态时钟不一致")

    # 独立 B/因子核的公共生成器历史主返回仍可能是历史最好表。本测速明确
    # 丢弃该返回，只使用 return_final_table 提供且由状态时钟核对的最后当前表。
    del returned_table

    gap_microsteps = 0
    gap_expected_microsteps = 0
    nonfinite_count = 0
    if task.arm == "gap_l1_global_s8":
        gap_attempts = diagnostics[
            "gap_l1_attempt_diagnostics_history"
        ]
        if len(gap_attempts) != rounds_run:
            raise RuntimeError(f"{task.task_id} 缺口核诊断轮数不完整")
        gap_microsteps, gap_expected_microsteps, nonfinite_count = (
            _expected_gap_microsteps(gap_attempts)
        )
        if gap_microsteps != int(diagnostics["gap_l1_microsteps"]):
            raise RuntimeError(f"{task.task_id} 缺口微步累计不一致")
        if rounds_run and diagnostics["gap_l1_reference_scale"] is None:
            raise RuntimeError(f"{task.task_id} 完成应用轮次后仍未建立缺口定尺")
    elif task.arm == "factor_b_s8":
        factor_attempts = diagnostics[
            "factorized_gibbs_attempt_diagnostics_history"
        ]
        if len(factor_attempts) != rounds_run:
            raise RuntimeError(f"{task.task_id} 因子核诊断轮数不完整")
        nonfinite_count = _assert_factor_diagnostics_finite(
            factor_attempts
        )
    if nonfinite_count:
        raise RuntimeError(f"{task.task_id} 出现非有限条件值")

    report = {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": int(task.seed),
        "requested_rounds": int(task.rounds),
        "rounds_run": rounds_run,
        "termination_reason": termination_reason,
        "completed": True,
        "device": "cuda:0",
        "elapsed_sec": float(elapsed_sec),
        "sec_per_round": (
            float(elapsed_sec / rounds_run) if rounds_run else 0.0
        ),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "output_table_identity": "terminal_current",
        "terminal_table_sha256": terminal_sha,
        "main_rng_endpoint_sha256": diagnostics["primary_rng_state_sha256"],
        "all_applied_unconditionally": True,
        "applied_round_count": rounds_run,
        "gap_reference_scale_established": bool(
            diagnostics["gap_l1_reference_scale"] is not None
            if task.arm == "gap_l1_global_s8" else False
        ),
        "gap_microsteps": int(gap_microsteps),
        "gap_expected_microsteps": int(gap_expected_microsteps),
        "gap_8k_identity": bool(gap_microsteps == gap_expected_microsteps),
        "gap_clip_hit_count": int(
            diagnostics["gap_l1_clip_hit_count"]
            if task.arm == "gap_l1_global_s8" else 0
        ),
        "nonfinite_count": int(nonfinite_count),
    }
    _validate_task_report(report)
    return report


def _call_generator_silently(
    task_id: str,
    **generator_arguments: Any,
) -> tuple[Any, dict[str, Any]]:
    """调用现有生成器，但不让其质量日志或异常值进入测速控制台。"""

    discard = io.StringIO()
    try:
        with redirect_stdout(discard), redirect_stderr(discard):
            return run_evolution(**generator_arguments)
    except Exception as exc:
        raise RuntimeError(
            f"{task_id} 完整生成器执行失败：{type(exc).__name__}"
        ) from None


def _execute_trajectory_task(
    task: joint.JointTrajectoryTask,
    *,
    repository_root: str,
) -> dict[str, Any]:
    root = Path(repository_root)
    spec = protocol.DATASETS[task.dataset]
    schema = load_schema(str(root / spec["schema"]))
    queries = load_queries(str(root / spec["queries"]))
    marginals = load_marginals(str(root / spec["marginals"]))
    if len(queries) != int(spec["query_count"]):
        raise RuntimeError(f"{task.dataset} 查询数漂移")
    target = np.asarray([query["result"] for query in queries], dtype=float)

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
        **protocol.task_generator_params(task.dataset, task.arm),
    )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    return _build_task_report(
        task,
        returned_table,
        diagnostics,
        elapsed_sec=elapsed,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
    )


def _frozen_tasks() -> tuple[joint.JointTrajectoryTask, ...]:
    return joint.build_joint_trajectory_plan(
        seeds=(protocol.SMOKE_SEED,),
        rounds_by_dataset={
            dataset: protocol.ROUNDS for dataset in joint.DATASET_ORDER
        },
    ).tasks


def run_frozen_task_matrix(
    worker: Callable[[joint.JointTrajectoryTask], Any],
) -> list[Any]:
    tasks = _frozen_tasks()
    results = run_ordered_process_tasks(
        worker,
        tasks,
        max_workers=protocol.MAX_WORKERS,
    )
    if len(results) != len(tasks):
        raise RuntimeError("联合测速任务返回数量不完整")
    for task, result in zip(tasks, results):
        if not isinstance(result, dict) or result.get("task_id") != task.task_id:
            raise RuntimeError("联合测速任务返回顺序或身份不一致")
    return results


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in protocol.FORBIDDEN_QUALITY_FIELDS:
                found.add(key)
            found.update(_forbidden_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _validate_task_report(report: dict[str, Any]) -> None:
    expected = set(protocol.ALLOWED_TASK_REPORT_FIELDS)
    if set(report) != expected:
        raise RuntimeError(
            "测速任务报告字段不符合白名单："
            f"missing={sorted(expected - set(report))}, "
            f"extra={sorted(set(report) - expected)}"
        )
    forbidden = _forbidden_keys(report)
    if forbidden:
        raise RuntimeError(f"测速任务报告泄漏质量字段：{sorted(forbidden)}")
    json.dumps(report, allow_nan=False)


def _validate_final_report(report: dict[str, Any]) -> None:
    if set(report) != TOP_LEVEL_REPORT_FIELDS:
        raise RuntimeError("联合测速顶层报告字段不符合冻结结构")
    forbidden = _forbidden_keys(report)
    if forbidden:
        raise RuntimeError(f"联合测速报告泄漏质量字段：{sorted(forbidden)}")
    expected_ids = [task.task_id for task in _frozen_tasks()]
    task_rows = report["tasks"]
    if [row.get("task_id") for row in task_rows] != expected_ids:
        raise RuntimeError("联合测速报告任务矩阵不完整或乱序")
    for row in task_rows:
        _validate_task_report(row)
    if report["summary"] != {
        "trajectory_count": 6,
        "completed_count": 6,
        "all_tasks_completed": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "total_nonfinite_count": 0,
        "quality_compared_or_emitted": False,
        "method_ranking_emitted": False,
        "formal_effect_claim_emitted": False,
    }:
        raise RuntimeError("联合测速汇总不符合冻结无质量结论")
    _strict_json_text(report)


def _write_report_atomically(destination: Path, report: dict[str, Any]) -> Path:
    _validate_final_report(report)
    if destination.exists():
        raise FileExistsError(f"联合测速输出已存在，不覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
    )
    try:
        report_path = temporary / REPORT_FILENAME
        report_path.write_text(_strict_json_text(report), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / REPORT_FILENAME


def build_plan() -> dict[str, Any]:
    plan = protocol.build_plan(_repo_root())
    if plan["runner_wired"] is not False:
        raise RuntimeError("协议冻结时执行器状态漂移")
    plan["runner_wired_at_protocol_freeze"] = False
    plan["runner_wired"] = True
    plan["runner_sha256"] = protocol.file_sha256(Path(__file__))
    plan["smoke_run_authorized"] = False
    return plan


def run(confirmed_protocol_sha256: str) -> Path:
    protocol.require_run_confirmation(confirmed_protocol_sha256)
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    execution_commit = _assert_clean_worktree(root)
    destination = root / protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"联合测速输出已存在，不覆盖：{destination}")
    gpu = _gpu_preflight()
    input_sha256 = _audit_inputs(root)
    runner_sha256 = protocol.file_sha256(Path(__file__))

    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    monitor = _GpuMonitor()
    monitor.start()
    try:
        worker = partial(
            _execute_trajectory_task,
            repository_root=str(root),
        )
        task_rows = run_frozen_task_matrix(worker)
    finally:
        gpu_samples = monitor.finish()
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone().isoformat()

    report = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gpu": gpu,
        },
        "input_sha256": input_sha256,
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": float(elapsed),
            "max_workers": protocol.MAX_WORKERS,
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_arm_then_dataset",
            "atomic_output": True,
        },
        "gpu_samples": gpu_samples,
        "gpu_summary": _summarize_gpu_samples(gpu_samples),
        "tasks": task_rows,
        "summary": {
            "trajectory_count": 6,
            "completed_count": 6,
            "all_tasks_completed": True,
            "all_terminal_current": all(
                row["output_table_identity"] == "terminal_current"
                for row in task_rows
            ),
            "all_applied_unconditionally": all(
                row["all_applied_unconditionally"] for row in task_rows
            ),
            "all_gap_8k_identity": all(
                row["gap_8k_identity"] for row in task_rows
            ),
            "total_nonfinite_count": sum(
                row["nonfinite_count"] for row in task_rows
            ),
            "quality_compared_or_emitted": False,
            "method_ranking_emitted": False,
            "formal_effect_claim_emitted": False,
        },
    }
    return _write_report_atomically(destination, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    print(run(args.confirm_protocol_sha))


if __name__ == "__main__":
    main()
