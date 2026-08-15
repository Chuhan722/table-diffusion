#!/usr/bin/env python
"""Issue #53 Stage 2B 封存 detector 验证轨迹采集。

默认 ``plan`` 不读取 validation workload 或启动生成。``collect`` 只接受冻结验证
协议 SHA-256，要求干净工作树和恰好一张可见 CUDA GPU，顺序采集固定的 20 条
8000 轮轨迹。采集过程不执行 detector replay、不输出部分验证分类，也不在线早停。
每条轨迹目录原子落盘；完整结束后再发布集合 manifest，支持中断后严格审计续跑。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Iterable, List, Sequence

try:
    from scripts import collect_issue53_stage2b_range_finding as collector
    from scripts import issue53_stage2b_validation_protocol as protocol
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import collect_issue53_stage2b_range_finding as collector
    import issue53_stage2b_validation_protocol as protocol

from table_diffevo.stationarity import (
    StationarityTrace,
    load_stationarity_trace,
    save_stationarity_trace,
)


VALIDATION_COLLECTION_CONTRACT_VERSION = (
    "issue53-stage2b-validation-collection-v1"
)
DEFAULT_OUTPUT_DIR = Path("outputs/issue53_stage2b_validation")

_RUN_MANIFEST_KEYS = {
    "contract_version",
    "validation_protocol_sha256",
    "mode",
    "formal_heldout_validation",
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


def _expected_cell_set() -> set[tuple[str, int, str]]:
    return {
        (row["dataset"], row["seed"], row["kernel"])
        for row in protocol.expected_validation_cells()
    }


def build_collection_plan(output_dir: Path) -> Dict[str, Any]:
    frozen = protocol.build_validation_plan()
    plan = {
        "contract_version": VALIDATION_COLLECTION_CONTRACT_VERSION,
        "validation_protocol_sha256": frozen["protocol_sha256"],
        "mode": "plan_only_no_generation_or_validation_read",
        "output_dir": str(output_dir),
        "cells": frozen["cells"],
        "trajectory_count": frozen["trajectory_count"],
        "round_budget_per_trajectory": frozen[
            "round_budget_per_trajectory"
        ],
        "total_round_budget": frozen["total_round_budget"],
        "requires_clean_worktree": True,
        "requires_exactly_one_visible_cuda_gpu": True,
        "detector_replay_during_collection": False,
        "validation_seed_accessed": False,
        "generation_started": False,
    }
    _strict_json_bytes(plan)
    return plan


def _single_visible_gpu_manifest() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise RuntimeError(
            "正式 validation 要求 CUDA_VISIBLE_DEVICES 显式指定一张卡"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("正式 validation 进程必须恰好看见一张可用 CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": visible,
        "torch_visible_device_count": int(torch.cuda.device_count()),
        "logical_device": "cuda:0",
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
    }


def formal_environment_manifest() -> Dict[str, Any]:
    environment = collector.environment_manifest()
    if not environment["git_worktree_clean_including_untracked"]:
        raise RuntimeError("正式 validation 采集要求干净工作树")
    return {
        **environment,
        "validation_collection_started_at": (
            datetime.now().astimezone().isoformat()
        ),
        "gpu": _single_visible_gpu_manifest(),
    }


def _run_destination(
    output_dir: Path,
    dataset: str,
    seed: int,
    kernel: str,
) -> Path:
    return output_dir / dataset / f"seed_{seed}" / kernel


def save_validation_run(
    output_dir: Path,
    *,
    workload: collector.LoadedWorkload,
    kernel: str,
    seed: int,
    preflight: Dict[str, Any],
    diagnostics: Dict[str, Any],
    trace: StationarityTrace,
    final_table: Any,
    elapsed_sec: float,
    environment: Dict[str, Any],
) -> Path:
    rounds = protocol.VALIDATION_ROUND_BUDGET
    destination = _run_destination(
        output_dir, workload.name, seed, kernel
    )
    if destination.exists():
        raise FileExistsError(f"validation run 已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=destination.parent,
        prefix=f".{kernel}.partial-",
    ))
    try:
        trace_paths = save_stationarity_trace(trace, temporary / "trace")
        trace_files = {
            "metadata": {
                "path": "trace/stationarity_trace.json",
                "sha256": _sha256_file(
                    Path(trace_paths["metadata_path"])
                ),
            },
            "query_array": {
                "path": "trace/measured_query_answers.npz",
                "sha256": _sha256_file(
                    Path(trace_paths["query_array_path"])
                ),
            },
        }
        if trace_files["query_array"]["sha256"] != trace_paths[
            "query_array_sha256"
        ]:
            raise RuntimeError("validation query array 写入后哈希不一致")
        manifest = {
            "contract_version": VALIDATION_COLLECTION_CONTRACT_VERSION,
            "validation_protocol_sha256": (
                protocol.validation_protocol_sha256()
            ),
            "mode": "validation",
            "formal_heldout_validation": True,
            "dataset": workload.name,
            "kernel": kernel,
            "seed": int(seed),
            "maximum_round_budget": rounds,
            "input_sha256": workload.input_sha256,
            "query_identity_sha256": workload.query_identity_sha256,
            "target_identity_sha256": workload.target_identity_sha256,
            "s0_preflight": preflight,
            "generator_params": diagnostics["params"],
            "reference_process_contract": diagnostics[
                "reference_process_contract"
            ],
            "run_summary": {
                "rounds_run": diagnostics["rounds_run"],
                "termination_reason": diagnostics["termination_reason"],
                "candidate_evaluation_count": diagnostics[
                    "candidate_evaluation_count"
                ],
                "initial_table_sha256": diagnostics[
                    "initial_table_sha256"
                ],
                "final_table_sha256": collector._frame_sha256(final_table),
                "primary_rng_state_sha256": diagnostics[
                    "primary_rng_state_sha256"
                ],
                "factorized_gibbs_rng_state_sha256": diagnostics[
                    "factorized_gibbs_rng_state_sha256"
                ],
                "clip_audit": collector._clip_audit(diagnostics),
                "elapsed_sec": float(elapsed_sec),
            },
            "trace_files": trace_files,
            "environment": environment,
        }
        _strict_json_bytes(manifest)
        _write_json_exclusive(temporary / "run_manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def audit_validation_run(
    destination: Path,
    *,
    expected_git_commit: str | None = None,
) -> Dict[str, Any]:
    manifest_path = destination / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 validation manifest：{destination}") from exc
    _validate_exact_keys(manifest, _RUN_MANIFEST_KEYS, "validation manifest")
    cell = (
        manifest["dataset"], manifest["seed"], manifest["kernel"]
    )
    if cell not in _expected_cell_set():
        raise RuntimeError(f"validation manifest 包含未知 cell：{cell}")
    expected_destination = _run_destination(
        destination.parents[2], cell[0], cell[1], cell[2]
    )
    if destination != expected_destination:
        raise RuntimeError("validation manifest 路径与 cell 身份不一致")
    if (
        manifest["contract_version"]
        != VALIDATION_COLLECTION_CONTRACT_VERSION
        or manifest["validation_protocol_sha256"]
        != protocol.validation_protocol_sha256()
        or manifest["mode"] != "validation"
        or manifest["formal_heldout_validation"] is not True
        or manifest["maximum_round_budget"]
        != protocol.VALIDATION_ROUND_BUDGET
    ):
        raise RuntimeError("validation manifest 协议身份不一致")
    if expected_git_commit is not None and manifest["environment"][
        "git_commit"
    ] != expected_git_commit:
        raise RuntimeError("validation 轨迹不是由同一执行提交生成")
    if manifest["environment"][
        "git_worktree_clean_including_untracked"
    ] is not True:
        raise RuntimeError("validation 轨迹不是从干净工作树生成")
    if manifest["environment"]["gpu"][
        "torch_visible_device_count"
    ] != 1:
        raise RuntimeError("validation 轨迹没有固定单卡可见性")

    specification = collector.DATASETS[cell[0]]
    if manifest["input_sha256"] != specification["sha256"]:
        raise RuntimeError("validation workload 输入哈希不一致")
    if int(manifest["generator_params"]["seed"]) != cell[1]:
        raise RuntimeError("validation generator seed 不一致")
    expected_kernel = collector.KERNELS[cell[2]]
    for key in (
        "factorized_gibbs_sweeps",
        "factorized_gibbs_max_order",
        "factorized_gibbs_logit_clip",
        "factorized_gibbs_use_compiled_workload",
    ):
        if manifest["generator_params"][key] != expected_kernel[key]:
            raise RuntimeError("validation kernel 参数不一致")

    _validate_exact_keys(
        manifest["trace_files"], {"metadata", "query_array"}, "trace_files"
    )
    for role, expected_path in {
        "metadata": "trace/stationarity_trace.json",
        "query_array": "trace/measured_query_answers.npz",
    }.items():
        info = manifest["trace_files"][role]
        _validate_exact_keys(info, {"path", "sha256"}, f"trace_files.{role}")
        if info["path"] != expected_path:
            raise RuntimeError("validation trace 相对路径不一致")
        if _sha256_file(destination / expected_path) != info["sha256"]:
            raise RuntimeError("validation trace 文件哈希不一致")

    trace = load_stationarity_trace(destination / "trace")
    rounds = protocol.VALIDATION_ROUND_BUDGET
    summary = manifest["run_summary"]
    if (
        trace.n_records != specification["n_records"]
        or trace.query_count != specification["query_count"]
        or trace.post_round_count != rounds
        or trace.state_count != rounds + 1
        or trace.termination_reason != "max_rounds"
        or summary["rounds_run"] != rounds
        or summary["termination_reason"] != "max_rounds"
        or summary["candidate_evaluation_count"] != rounds
    ):
        raise RuntimeError("validation 轨迹没有跑满冻结预算")
    if (
        trace.query_identity_sha256 != manifest["query_identity_sha256"]
        or trace.target_identity_sha256 != manifest["target_identity_sha256"]
    ):
        raise RuntimeError("validation trace 查询或 target 身份不一致")
    if not all(
        observation["proposal_accepted"]
        for observation in trace.observations[1:]
    ):
        raise RuntimeError("validation no-gate 轨迹出现未应用 proposal")
    if trace.observations[-1]["current_table_sha256"] != summary[
        "final_table_sha256"
    ]:
        raise RuntimeError("validation trace 终态哈希不一致")
    if trace.observations[-1][
        "candidate_evaluation_count_cumulative"
    ] != rounds:
        raise RuntimeError("validation 候选评价累计数不一致")
    return manifest


def _audit_pair_bindings(manifests: Sequence[Dict[str, Any]]) -> None:
    bindings: Dict[tuple[str, int], tuple[Any, ...]] = {}
    for manifest in manifests:
        key = (manifest["dataset"], manifest["seed"])
        binding = (
            manifest["s0_preflight"]["direction_reference_scale"],
            manifest["run_summary"]["initial_table_sha256"],
            manifest["s0_preflight"][
                "primary_rng_post_initialization_state_sha256"
            ],
        )
        if key in bindings and bindings[key] != binding:
            raise RuntimeError("validation 配对核没有共享 s0/S0/初始化后 RNG")
        bindings[key] = binding
    if len(bindings) != len(collector.DATASETS) * len(protocol.VALIDATION_SEEDS):
        raise RuntimeError("validation dataset×seed 配对绑定不完整")


def _write_collection_manifest(
    output_dir: Path,
    manifests: Sequence[Dict[str, Any]],
    elapsed_sec: float,
) -> Path:
    path = output_dir / "collection_manifest.json"
    if path.exists():
        raise FileExistsError("validation collection manifest 已存在")
    run_hashes = {}
    for manifest in manifests:
        cell = (
            f"{manifest['dataset']}/seed_{manifest['seed']}/"
            f"{manifest['kernel']}"
        )
        run_hashes[cell] = _sha256_file(
            _run_destination(
                output_dir,
                manifest["dataset"],
                manifest["seed"],
                manifest["kernel"],
            ) / "run_manifest.json"
        )
    summary = {
        "contract_version": VALIDATION_COLLECTION_CONTRACT_VERSION,
        "validation_protocol_sha256": protocol.validation_protocol_sha256(),
        "formal_validation_collection_complete": True,
        "trajectory_count": len(manifests),
        "rounds_per_trajectory": protocol.VALIDATION_ROUND_BUDGET,
        "total_round_count": (
            len(manifests) * protocol.VALIDATION_ROUND_BUDGET
        ),
        "detector_replay_performed": False,
        "partial_validation_classification_read": False,
        "run_manifest_sha256": dict(sorted(run_hashes.items())),
        "collection_elapsed_sec": float(elapsed_sec),
    }
    _strict_json_bytes(summary)
    temporary_dir = Path(tempfile.mkdtemp(
        dir=output_dir,
        prefix=".collection-manifest.partial-",
    ))
    try:
        temporary_path = temporary_dir / path.name
        _write_json_exclusive(temporary_path, summary)
        os.replace(temporary_path, path)
        temporary_dir.rmdir()
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return path


def run_frozen_validation_collection(
    output_dir: Path,
    confirmed_protocol_sha256: str,
) -> Path:
    expected_sha = protocol.validation_protocol_sha256()
    if confirmed_protocol_sha256 != expected_sha:
        raise ValueError("必须显式确认完整冻结 validation protocol SHA-256")
    environment = formal_environment_manifest()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "collection_manifest.json").exists():
        raise FileExistsError("validation collection 已经正式完成")

    overall_start = time.perf_counter()
    manifests: List[Dict[str, Any]] = []
    completed = 0
    total = len(protocol.expected_validation_cells())
    for dataset in collector.DATASETS:
        workload = collector.load_public_workload(dataset)
        for seed in protocol.VALIDATION_SEEDS:
            s0, preflight = collector.derive_workload_seed_s0(workload, seed)
            for kernel in collector.KERNELS:
                destination = _run_destination(
                    output_dir, dataset, seed, kernel
                )
                if destination.exists():
                    manifest = audit_validation_run(
                        destination,
                        expected_git_commit=environment["git_commit"],
                    )
                    completed += 1
                    print(
                        f"[validation resume {completed}/{total}] "
                        f"{dataset} seed={seed} {kernel} audited",
                        flush=True,
                    )
                    manifests.append(manifest)
                    continue

                start = time.perf_counter()
                final_table, diagnostics, trace = collector.collect_one_trajectory(
                    workload,
                    kernel,
                    seed,
                    protocol.VALIDATION_ROUND_BUDGET,
                    s0,
                    preflight,
                )
                elapsed = time.perf_counter() - start
                save_validation_run(
                    output_dir,
                    workload=workload,
                    kernel=kernel,
                    seed=seed,
                    preflight=preflight,
                    diagnostics=diagnostics,
                    trace=trace,
                    final_table=final_table,
                    elapsed_sec=elapsed,
                    environment=environment,
                )
                manifest = audit_validation_run(
                    destination,
                    expected_git_commit=environment["git_commit"],
                )
                completed += 1
                print(
                    f"[validation {completed}/{total}] {dataset} "
                    f"seed={seed} {kernel} complete "
                    f"({elapsed / 60.0:.2f} min)",
                    flush=True,
                )
                manifests.append(manifest)

    if len(manifests) != total:
        raise RuntimeError("validation collection 没有覆盖全部 20 条轨迹")
    _audit_pair_bindings(manifests)
    collection_manifest = _write_collection_manifest(
        output_dir,
        manifests,
        time.perf_counter() - overall_start,
    )
    return collection_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "collect"), default="plan"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--confirm-protocol-sha")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "plan":
        print(json.dumps(
            build_collection_plan(args.output_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    if args.confirm_protocol_sha is None:
        raise RuntimeError("collect 模式必须显式传入 --confirm-protocol-sha")
    destination = run_frozen_validation_collection(
        args.output_dir, args.confirm_protocol_sha
    )
    print(f"validation collection -> {destination}", flush=True)


if __name__ == "__main__":
    main()
