#!/usr/bin/env python3
"""单进程测量 Issue #53 NLTCS 剩余缺口核的时间去向。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import issue53_stage6c_joint_smoke_protocol as protocol
from scripts import run_issue53_stage6c_joint_smoke as gpu_helpers
from table_diffevo import evolution as evolution_module
from table_diffevo import gap_l1_diffusion as gap_module
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema

PROFILE_FORMAT = "issue53-gap-l1-nltcs-single-worker-profile-v1"
SEED = 9907
ROUNDS = 100
CANDIDATE_BUDGET = 100
DATASET = "nltcs"
ARM = "gap_l1_global_s8"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/issue53_gap_l1_nltcs_single_worker_profile_seed9907_100round_v1"
)
STAGE6C_REFERENCE_REPORT = Path(
    "outputs/issue53_stage6c_joint_smoke_seed9907_v1/timing_report.json"
)
FORBIDDEN_REPORT_KEYS = {
    "loss",
    "loss_history",
    "best_loss",
    "normalized_l1_error",
    "final_current_squared_loss",
    "final_current_normalized_l1",
    "query_counts",
    "query_residuals",
    "terminal_table",
    "final_table",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        raise RuntimeError("性能基线要求包含未跟踪文件在内的干净工作树")
    return _git_text(root, "rev-parse", "HEAD")


def _frame_sha256(frame: Any) -> str:
    return hashlib.sha256(
        frame.reset_index(drop=True).to_csv(index=False).encode("utf-8")
    ).hexdigest()


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


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("计时汇总要求非空有限一维数值")
    return {
        "count": len(array),
        "total": float(array.sum()),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def summarize_stage_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_keys: Sequence[str] = ("round",),
) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("逐轮计时不得为空")
    keys = set(rows[0]).difference(excluded_keys)
    if any(set(row).difference(excluded_keys) != keys for row in rows):
        raise ValueError("逐轮计时字段不一致")
    return {
        key: summarize_values(float(row[key]) for row in rows) for key in sorted(keys)
    }


class _TimingRegistry:
    """在不修改冻结实现源码的前提下记录函数调用与 CUDA 流跨度。"""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}
        self._cuda_event_pairs: dict[str, list[tuple[Any, Any]]] = {}

    def record(self, name: str, elapsed_sec: float) -> None:
        self.samples.setdefault(name, []).append(float(elapsed_sec))

    def wrap(
        self,
        name: str,
        function: Any,
        *,
        cuda_stream_span: bool = False,
    ) -> Any:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            start_event = None
            end_event = None
            if cuda_stream_span:
                import torch

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.record(
                    f"{name}.python_wall_sec",
                    time.perf_counter() - started,
                )
                if start_event is not None and end_event is not None:
                    end_event.record()
                    self._cuda_event_pairs.setdefault(
                        f"{name}.cuda_stream_span_sec", []
                    ).append((start_event, end_event))

        return measured

    def finalize_cuda_events(self) -> None:
        import torch

        torch.cuda.synchronize(0)
        for name, pairs in self._cuda_event_pairs.items():
            self.samples[name] = [
                float(start.elapsed_time(end) / 1000.0) for start, end in pairs
            ]

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            name: summarize_values(values)
            for name, values in sorted(self.samples.items())
            if values
        }


class _TimedGenerator(np.random.Generator):
    __slots__ = ("_timing_prefix", "_timing_registry")

    def __init__(
        self,
        bit_generator: Any,
        timing_registry: _TimingRegistry,
        timing_prefix: str,
    ) -> None:
        super().__init__(bit_generator)
        self._timing_registry = timing_registry
        self._timing_prefix = timing_prefix

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super().integers(*args, **kwargs)
        finally:
            self._timing_registry.record(
                f"{self._timing_prefix}.integers_wall_sec",
                time.perf_counter() - started,
            )

    def random(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super().random(*args, **kwargs)
        finally:
            self._timing_registry.record(
                f"{self._timing_prefix}.random_wall_sec",
                time.perf_counter() - started,
            )


@contextmanager
def _instrument_calls(registry: _TimingRegistry):
    """临时包裹现有入口；退出后所有绑定恢复。"""

    original_default_rng = np.random.default_rng
    generator_count = 0

    def timed_default_rng(seed: Any = None) -> _TimedGenerator:
        nonlocal generator_count
        base = original_default_rng(seed)
        prefix = "primary_rng" if generator_count == 0 else "gap_scan_rng"
        generator_count += 1
        return _TimedGenerator(base.bit_generator, registry, prefix)

    bindings = (
        (
            evolution_module,
            "evaluate_vectorized",
            "outer.evaluate_vectorized",
            True,
        ),
        (
            evolution_module,
            "pairwise_block_distance",
            "outer.pairwise_block_distance",
            True,
        ),
        (
            evolution_module,
            "compute_sampling_probs",
            "outer.compute_sampling_probs",
            True,
        ),
        (
            evolution_module,
            "sample_donors",
            "outer.sample_donors",
            True,
        ),
        (
            evolution_module,
            "_mean_selected_distance",
            "outer.mean_selected_distance",
            True,
        ),
        (
            evolution_module,
            "compute_copy_direction_scores",
            "outer.compute_copy_direction_scores",
            True,
        ),
        (
            evolution_module,
            "sample_update_random_plan",
            "outer.sample_update_random_plan",
            False,
        ),
        (
            evolution_module,
            "isolated_gap_l1_scores",
            "outer.isolated_gap_l1_scores",
            True,
        ),
        (
            evolution_module,
            "stable_nonzero_rms",
            "outer.stable_nonzero_rms",
            False,
        ),
        (
            evolution_module,
            "evolve_step_gap_l1_global",
            "outer.evolve_step_gap_l1_global",
            True,
        ),
        (
            evolution_module,
            "apply_planned_mutations",
            "outer.apply_planned_mutations",
            False,
        ),
        (
            gap_module,
            "_prepare_cuda_plan",
            "kernel.prepare_cuda_plan",
            True,
        ),
        (
            gap_module,
            "_cuda_probability_parameters",
            "kernel.cuda_probability_parameters",
            True,
        ),
        (
            gap_module,
            "_materialize_copy_table",
            "kernel.materialize_copy_table",
            False,
        ),
        (
            gap_module,
            "_full_query_recount_cuda",
            "kernel.full_query_recount_cuda",
            True,
        ),
        (
            gap_module,
            "_update_trace",
            "kernel.update_trace",
            False,
        ),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(evolution_module.np.random, "default_rng", timed_default_rng)
        )
        for module, attribute, name, cuda_span in bindings:
            original = getattr(module, attribute)
            stack.enter_context(
                patch.object(
                    module,
                    attribute,
                    registry.wrap(
                        name,
                        original,
                        cuda_stream_span=cuda_span,
                    ),
                )
            )
        yield


def _forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_REPORT_KEYS:
                paths.append(child_path)
            paths.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, f"{prefix}[{index}]"))
    return paths


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/gap_l1_diffusion.py"),
        Path("src/table_diffevo/_gap_l1_triton.py"),
        Path("src/table_diffevo/update.py"),
        Path("scripts/profile_issue53_gap_l1_nltcs.py"),
    )
    return {str(path): protocol.file_sha256(root / path) for path in paths}


def _input_hashes(root: Path) -> dict[str, str]:
    spec = protocol.DATASETS[DATASET]
    observed = {
        key: protocol.file_sha256(root / spec[key])
        for key in ("schema", "queries", "marginals")
    }
    if observed != spec["input_sha256"]:
        raise RuntimeError("NLTCS 输入 SHA-256 与已冻结 Stage 6C 输入不一致")
    return observed


def _generator_parameters() -> dict[str, Any]:
    params = protocol.task_generator_params(DATASET, ARM)
    params.update(
        {
            "n_rounds": ROUNDS,
            "candidate_budget": CANDIDATE_BUDGET,
            "seed": SEED,
            "log_every": ROUNDS + 1,
        }
    )
    return params


def _per_round_totals(
    samples: Sequence[float],
    counts: Sequence[int],
    *,
    name: str,
) -> list[float]:
    if len(samples) != sum(counts):
        raise RuntimeError(
            f"{name} 调用数 {len(samples)} 与微步总数 {sum(counts)} 不一致"
        )
    totals = []
    offset = 0
    for count in counts:
        totals.append(float(sum(samples[offset : offset + count])))
        offset += count
    return totals


def _stage6c_prefix_reference(root: Path) -> dict[str, Any]:
    path = root / STAGE6C_REFERENCE_REPORT
    report = json.loads(path.read_text(encoding="utf-8"))
    task_id = f"seed_{SEED}__{ARM}__{DATASET}"
    matches = [task for task in report["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise RuntimeError("Stage 6C 历史报告中缺少唯一 NLTCS 新核任务")
    task = matches[0]
    return {
        "report_path": str(STAGE6C_REFERENCE_REPORT),
        "report_sha256": protocol.file_sha256(path),
        "round": int(task["rounds_run"]),
        "terminal_table_sha256": task["terminal_table_sha256"],
        "primary_rng_endpoint_sha256": task["main_rng_endpoint_sha256"],
    }


def _extract_profile(
    root: Path,
    returned_table: Any,
    diagnostics: dict[str, Any],
    registry: _TimingRegistry,
) -> dict[str, Any]:
    final_table = diagnostics.pop("final_table")
    terminal_sha = _frame_sha256(final_table)
    if _frame_sha256(returned_table) != terminal_sha:
        raise RuntimeError("缺口核主返回不是最后当前表")
    rounds_run = len(diagnostics["accept_history"])
    if rounds_run != ROUNDS:
        raise RuntimeError(f"要求 {ROUNDS} 轮，实际应用 {rounds_run} 轮")
    if diagnostics["accept_history"] != [True] * ROUNDS:
        raise RuntimeError("单进程基线出现非无条件应用轮次")
    if int(diagnostics["candidate_evaluation_count"]) != CANDIDATE_BUDGET:
        raise RuntimeError("候选评价数与 100 轮固定基线不一致")

    attempts = diagnostics["gap_l1_attempt_diagnostics_history"]
    if len(attempts) != ROUNDS or any(len(row) != 1 for row in attempts):
        raise RuntimeError("每轮必须恰好有一个缺口核诊断")
    scans = [row[0] for row in attempts]
    if any(scan.get("gap_l1_scan_applied") is not True for scan in scans):
        raise RuntimeError("100 轮基线必须全部建立定尺并执行缺口扫描")
    expected_microsteps = sum(
        int(scan["n_sweeps"]) * int(scan["active_switches_k"]) for scan in scans
    )
    actual_microsteps = sum(int(scan["gibbs_microsteps"]) for scan in scans)
    if actual_microsteps != expected_microsteps:
        raise RuntimeError("缺口核微步不满足逐轮 8*K")
    if any(int(scan["nonfinite_condition_count"]) for scan in scans):
        raise RuntimeError("缺口核出现非有限条件数值")

    microsteps_by_round = [int(scan["gibbs_microsteps"]) for scan in scans]
    rng_integer_by_round = _per_round_totals(
        registry.samples.get("gap_scan_rng.integers_wall_sec", ()),
        microsteps_by_round,
        name="缺口坐标随机数",
    )
    rng_random_by_round = _per_round_totals(
        registry.samples.get("gap_scan_rng.random_wall_sec", ()),
        microsteps_by_round,
        name="缺口均匀随机数",
    )
    trace_by_round = _per_round_totals(
        registry.samples.get("kernel.update_trace.python_wall_sec", ()),
        microsteps_by_round,
        name="缺口轨迹哈希",
    )
    kernel_rows = []
    scan_cpu_rows = []
    combined_rounds = []
    for index, scan in enumerate(scans):
        coarse = {
            "prepare_wall_sec": float(scan["prepared_elapsed_sec_diagnostic_only"]),
            "scan_wall_sec": float(scan["scan_elapsed_sec_diagnostic_only"]),
            "materialize_wall_sec": float(
                scan["materialize_elapsed_sec_diagnostic_only"]
            ),
            "full_recount_wall_sec": float(
                scan["full_recount_elapsed_sec_diagnostic_only"]
            ),
            "kernel_total_wall_sec": float(scan["elapsed_sec_diagnostic_only"]),
        }
        measured_scan_cpu = {
            "coordinate_rng_wall_sec": rng_integer_by_round[index],
            "uniform_rng_wall_sec": rng_random_by_round[index],
            "trace_hash_wall_sec": trace_by_round[index],
        }
        measured_scan_cpu["remaining_scan_wall_sec"] = float(
            max(
                0.0,
                coarse["scan_wall_sec"] - sum(measured_scan_cpu.values()),
            )
        )
        kernel_rows.append(coarse)
        scan_cpu_rows.append(measured_scan_cpu)
        combined_rounds.append(
            {
                "round": index + 1,
                "active_switches_k": int(scan["active_switches_k"]),
                "gibbs_microsteps": int(scan["gibbs_microsteps"]),
                "kernel_coarse_stage_seconds": coarse,
                "scan_cpu_substage_seconds": measured_scan_cpu,
            }
        )

    kernel_summary = summarize_stage_rows(kernel_rows, excluded_keys=())
    scan_cpu_summary = summarize_stage_rows(scan_cpu_rows, excluded_keys=())
    kernel_total = kernel_summary["kernel_total_wall_sec"]["total"]
    kernel_shares = {
        key: float(summary["total"] / kernel_total)
        for key, summary in kernel_summary.items()
        if key != "kernel_total_wall_sec"
    }

    historical = _stage6c_prefix_reference(root)
    clocks = diagnostics["transition_clock_history"]
    prefix_index = int(historical["round"]) - 1
    if prefix_index >= len(clocks):
        raise RuntimeError("100 轮轨迹没有覆盖 Stage 6C 历史前缀")
    observed_prefix = {
        "terminal_table_sha256": clocks[prefix_index]["post_current_table_sha256"],
        "primary_rng_endpoint_sha256": clocks[prefix_index]["primary_rng_state_sha256"],
    }
    prefix_match = all(
        observed_prefix[key] == historical[key] for key in observed_prefix
    )
    if not prefix_match:
        raise RuntimeError("外置计时改变了已冻结 Stage 6C 前 20 轮轨迹")

    return {
        "trajectory_identity": {
            "terminal_table_sha256": terminal_sha,
            "primary_rng_endpoint_sha256": diagnostics["primary_rng_state_sha256"],
            "gap_trace_sha256_by_round": [
                scan["microstep_trace_sha256"] for scan in scans
            ],
        },
        "stage6c_prefix_equivalence": {
            "matched": True,
            "reference": historical,
            "observed": observed_prefix,
        },
        "contract_checks": {
            "rounds_run": rounds_run,
            "candidate_evaluations": int(diagnostics["candidate_evaluation_count"]),
            "termination_reason": diagnostics["termination_reason"],
            "all_rounds_applied_unconditionally": True,
            "gap_reference_scale_established": bool(
                diagnostics["gap_l1_reference_scale"] is not None
            ),
            "gap_microsteps": actual_microsteps,
            "gap_expected_microsteps": expected_microsteps,
            "gap_8k_identity": True,
            "gap_clip_hit_count": int(diagnostics["gap_l1_clip_hit_count"]),
            "nonfinite_condition_count": 0,
        },
        "timing": {
            "generator_loop_wall_sec": float(diagnostics["elapsed_sec"]),
            "generator_sec_per_round": float(diagnostics["sec_per_round"]),
            "workload_compile_wall_sec": float(
                diagnostics["gap_l1_workload_compile_elapsed_sec"]
            ),
            "direction_evaluation_wall_sec": float(
                diagnostics["direction_evaluation_elapsed_sec"]
            ),
            "function_call_summary": registry.summary(),
            "kernel_total_wall_sec": float(kernel_total),
            "kernel_coarse_stage_summary": kernel_summary,
            "kernel_coarse_stage_total_share_of_kernel_wall": kernel_shares,
            "scan_cpu_substage_summary": scan_cpu_summary,
            "active_switches_k": summarize_values(
                int(scan["active_switches_k"]) for scan in scans
            ),
            "gibbs_microsteps_per_round": summarize_values(
                int(scan["gibbs_microsteps"]) for scan in scans
            ),
            "per_round": combined_rounds,
        },
    }


def _build_report(
    root: Path,
    *,
    execution_commit: str,
    environment: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    wall_elapsed_sec: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    gpu_samples: Sequence[Mapping[str, Any]],
    extracted: Mapping[str, Any],
) -> dict[str, Any]:
    params = _generator_parameters()
    report = {
        "format": PROFILE_FORMAT,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "performance_attribution_only_not_quality_evaluation",
        "execution_commit": execution_commit,
        "source_sha256": _source_hashes(root),
        "input_sha256": dict(input_hashes),
        "environment": {
            **dict(environment),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "max_workers": 1,
        },
        "configuration": {
            "dataset": DATASET,
            "arm": ARM,
            "seed": SEED,
            "rounds": ROUNDS,
            "candidate_budget": CANDIDATE_BUDGET,
            "gap_l1_sweeps": int(params["gap_l1_sweeps"]),
            "rho": float(params["rho"]),
            "eta": float(params["eta"]),
            "mu": float(params["mu"]),
            "device": params["device"],
            "instrumentation": "external_temporary_function_wrappers",
            "verify_full_recount": True,
            "quality_output_allowed": False,
        },
        **dict(extracted),
        "process_wall_sec": float(wall_elapsed_sec),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "gpu_samples": [dict(sample) for sample in gpu_samples],
        "gpu_summary": gpu_helpers._summarize_gpu_samples(gpu_samples),
        "observer": {
            "cuda_stage_events_enabled": True,
            "explicit_stage_synchronization_enabled": False,
            "frozen_generator_sources_modified": False,
            "rng_bit_generators_preserved": True,
            "nvidia_smi_interval_sec": gpu_helpers.GPU_SAMPLE_INTERVAL_SEC,
            "timings_are_instrumented_and_diagnostic_only": True,
        },
    }
    forbidden = _forbidden_key_paths(report)
    if forbidden:
        raise RuntimeError(f"性能报告越过质量边界：{forbidden}")
    return report


def _write_report_atomically(
    root: Path,
    output_dir: Path,
    report: Mapping[str, Any],
) -> Path:
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    if destination.exists():
        raise FileExistsError(f"输出目录已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        temporary_path = Path(temporary)
        report_path = temporary_path / "profile_report.json"
        report_path.write_text(_strict_json_text(report), encoding="utf-8")
        os.replace(temporary_path, destination)
    return destination / "profile_report.json"


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    root = _repo_root()
    execution_commit = _assert_clean_worktree(root)
    input_hashes = _input_hashes(root)
    environment = gpu_helpers._gpu_preflight()
    spec = protocol.DATASETS[DATASET]
    schema = load_schema(str(root / spec["schema"]))
    queries = load_queries(str(root / spec["queries"]))
    marginals = load_marginals(str(root / spec["marginals"]))
    if len(queries) != int(spec["query_count"]):
        raise RuntimeError("NLTCS 查询数量漂移")
    target = np.asarray([query["result"] for query in queries], dtype=float)

    import torch

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    monitor = gpu_helpers._GpuMonitor()
    monitor.start()
    started = time.perf_counter()
    discarded_output = io.StringIO()
    registry = _TimingRegistry()
    try:
        with (
            _instrument_calls(registry),
            redirect_stdout(discarded_output),
            redirect_stderr(discarded_output),
        ):
            returned_table, diagnostics = run_evolution(
                target=target,
                queries=queries,
                schema=schema,
                marginals=marginals,
                **_generator_parameters(),
            )
        torch.cuda.synchronize(0)
        wall_elapsed_sec = time.perf_counter() - started
        registry.finalize_cuda_events()
    finally:
        gpu_samples = monitor.finish()
    extracted = _extract_profile(root, returned_table, diagnostics, registry)
    report = _build_report(
        root,
        execution_commit=execution_commit,
        environment=environment,
        input_hashes=input_hashes,
        wall_elapsed_sec=wall_elapsed_sec,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        gpu_samples=gpu_samples,
        extracted=extracted,
    )
    return _write_report_atomically(root, output_dir, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    report_path = run(arguments.output_dir)
    print(report_path)


if __name__ == "__main__":
    main()
