#!/usr/bin/env python
"""Issue #53 Stage 2B 初始量程轨迹采集。

本脚本只采集 exact、no-gate、fixed-parameter 的 current-state 长轨迹；
不选择窗口或阈值，也不执行在线早停。开发 seed 与封存验证 seed 在代码中分离，
验证 seed 在 detector 配置冻结前不能由本入口运行。

默认 ``plan`` 只打印计划。正式 development 入口严格要求使用用户确认并冻结的
``DEVELOPMENT_ROUND_BUDGET``。``smoke`` 仅允许未保留 seed、至多 3 轮和小数据集。
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
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.reference_process import (
    derive_fixed_direction_reference_scale,
    run_stationarity_calibration_evolution,
)
from table_diffevo.schema import Schema, load_schema
from table_diffevo.stationarity import (
    StationarityTrace,
    ordered_query_identity_sha256,
    save_stationarity_trace,
    target_answer_identity_sha256,
)


STAGE2B_RANGE_FINDING_PROTOCOL_VERSION = (
    "issue53-stage2b-range-finding-v1"
)
DEVELOPMENT_SEEDS = (200, 201, 202)
SEALED_VALIDATION_SEEDS = (220, 221, 222, 223, 224)
SMOKE_DEFAULT_SEED = 999

# 2026-08-15 已在正式实验前向用户报告 8000 轮最大观察预算及单卡预计开销，
# 并得到明确确认。它只是固定采集上限，不是收敛阈值，也不触发在线早停。
DEVELOPMENT_ROUND_BUDGET: int | None = 8000
S0_PREFLIGHT_MAX_ROUNDS = 8
SMOKE_MAX_ROUNDS = 3

FIXED_ALPHA = 16.0
FIXED_RHO = 0.01
FIXED_ETA = 0.5
FIXED_MU = 0.01
FIXED_DIRECTION_STRENGTH = 2.0
FIXED_DIRECTION_LOGIT_CLIP = 30.0
FIXED_GIBBS_LOGIT_CLIP = 30.0

SHARED_GENERATOR_PARAMS: Dict[str, Any] = {
    "beta": 1.0,
    "h": 0.8,
    "lambda_param": 0.5,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
    "init_method": "marginal",
    "selection_scale_invariant": True,
    "selection_scale_invariant_min_spread": 1e-3,
    "exclude_self": True,
    "candidate_budget": None,
    "eval_method": "vectorized",
    "batch_size": 256,
}

KERNELS: Dict[str, Dict[str, Any]] = {
    "independent": {
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": FIXED_GIBBS_LOGIT_CLIP,
        "factorized_gibbs_use_compiled_workload": False,
    },
    "factorized_gibbs": {
        "factorized_gibbs_sweeps": 8,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": FIXED_GIBBS_LOGIT_CLIP,
        "factorized_gibbs_use_compiled_workload": True,
    },
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "query_count": 50,
        "device": "numpy",
        "sha256": {
            "schema": (
                "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
            ),
            "queries": (
                "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88"
            ),
            "marginals": (
                "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
            ),
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        # 历史文件名写 1000，但冻结文件实际含 1001 条；不得静默删减。
        "query_count": 1001,
        "device": "cuda",
        "sha256": {
            "schema": (
                "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4"
            ),
            "queries": (
                "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f"
            ),
            "marginals": (
                "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
            ),
        },
    },
}


@dataclass(frozen=True)
class LoadedWorkload:
    name: str
    schema: Schema
    queries: List[Dict[str, Any]]
    marginals: Dict[str, Any]
    target: np.ndarray
    n_records: int
    query_identity_sha256: str
    target_identity_sha256: str
    input_sha256: Dict[str, str]


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _validate_unique_known(
    values: Sequence[Any],
    allowed: Iterable[Any],
    name: str,
) -> List[Any]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} 不能为空")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} 不得重复")
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise ValueError(f"{name} 包含未冻结值：{unknown}")
    return result


def validate_development_seeds(seeds: Sequence[int]) -> List[int]:
    """Only expose development seeds; held-out seeds remain sealed."""
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("development seeds 必须是显式整数")
    return _validate_unique_known(
        seeds,
        DEVELOPMENT_SEEDS,
        "development seeds",
    )


def validate_smoke_request(
    datasets: Sequence[str],
    seeds: Sequence[int],
    rounds: int,
) -> None:
    if list(datasets) != ["test_300x10"]:
        raise ValueError("smoke 只允许 test_300x10")
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or not 1 <= rounds <= SMOKE_MAX_ROUNDS
    ):
        raise ValueError(f"smoke rounds 必须位于 [1, {SMOKE_MAX_ROUNDS}]")
    reserved = set(DEVELOPMENT_SEEDS).union(SEALED_VALIDATION_SEEDS)
    if not seeds or any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed in reserved
        for seed in seeds
    ):
        raise ValueError("smoke 必须使用未保留的显式整数 seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("smoke seeds 不得重复")


def _validate_frozen_input_hashes(name: str) -> Dict[str, str]:
    specification = DATASETS[name]
    observed = {
        key: _sha256_file(specification[key])
        for key in ("schema", "queries", "marginals")
    }
    if observed != specification["sha256"]:
        raise RuntimeError(
            f"[{name}] 公开输入 SHA-256 与冻结协议不一致"
        )
    return observed


def load_public_workload(name: str) -> LoadedWorkload:
    """Load only schema, measured answers and marginal initialization data."""
    if name not in DATASETS:
        raise ValueError(f"未知数据集：{name!r}")
    specification = DATASETS[name]
    hashes = _validate_frozen_input_hashes(name)

    with specification["queries"].open(encoding="utf-8") as handle:
        query_payload = json.load(handle)
    if not isinstance(query_payload, dict):
        raise RuntimeError(f"[{name}] 查询文件必须带冻结 metadata")
    if query_payload.get("record_count") != specification["n_records"]:
        raise RuntimeError(f"[{name}] 查询 record_count 与协议不一致")
    if query_payload.get("query_count") != specification["query_count"]:
        raise RuntimeError(f"[{name}] 查询 query_count 与协议不一致")

    schema = load_schema(str(specification["schema"]))
    queries = load_queries(str(specification["queries"]))
    marginals = load_marginals(str(specification["marginals"]))
    if len(queries) != specification["query_count"]:
        raise RuntimeError(f"[{name}] 实际查询条数与协议不一致")
    if marginals.get("n_records") != specification["n_records"]:
        raise RuntimeError(f"[{name}] marginals n_records 与协议不一致")

    target = np.asarray([query["result"] for query in queries], dtype=float)
    if (
        target.shape != (specification["query_count"],)
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or np.any(target > specification["n_records"])
    ):
        raise RuntimeError(f"[{name}] measured target 非法")
    return LoadedWorkload(
        name=name,
        schema=schema,
        queries=queries,
        marginals=marginals,
        target=target,
        n_records=specification["n_records"],
        query_identity_sha256=ordered_query_identity_sha256(queries),
        target_identity_sha256=target_answer_identity_sha256(target),
        input_sha256=hashes,
    )


def frozen_protocol_manifest() -> Dict[str, Any]:
    """Return the decisions frozen before any development trajectory."""
    manifest = {
        "contract_version": STAGE2B_RANGE_FINDING_PROTOCOL_VERSION,
        "purpose": "stationarity_detector_range_finding_only",
        "data_boundary": {
            "measured_answers": "exact_no_noise",
            "source_table_generation_access": "forbidden",
            "heldout_quality_queries": "excluded",
            "differential_privacy_noise": "excluded_for_now",
        },
        "scope": {
            "datasets": list(DATASETS),
            "kernels": list(KERNELS),
            "one_common_detector_required": True,
            "per_cell_detector_tuning": False,
        },
        "seed_split": {
            "development": list(DEVELOPMENT_SEEDS),
            "validation": list(SEALED_VALIDATION_SEEDS),
            "validation_status": "sealed_until_detector_config_frozen",
            "same_seed_initial_state_paired_across_kernels": True,
        },
        "generator": {
            "fixed_alpha": FIXED_ALPHA,
            "rho": FIXED_RHO,
            "eta": FIXED_ETA,
            "mu": FIXED_MU,
            "diffusion_direction_strength_tau": (
                FIXED_DIRECTION_STRENGTH
            ),
            "diffusion_direction_logit_clip": (
                FIXED_DIRECTION_LOGIT_CLIP
            ),
            "shared": {
                **SHARED_GENERATOR_PARAMS,
                "winsorize_quantiles": list(
                    SHARED_GENERATOR_PARAMS["winsorize_quantiles"]
                ),
            },
            "kernels": KERNELS,
            "factor_builder": {
                "independent": "not_used",
                "factorized_gibbs": "compiled_batch",
                "compiled_batch_role": (
                    "output_equivalent_performance_implementation"
                ),
            },
            "s0": {
                "method": "independent_initial_rms_preflight_then_restart",
                "preflight_max_rounds": S0_PREFLIGHT_MAX_ROUNDS,
                "shared_within_dataset_seed_across_kernels": True,
                "preflight_states_in_stationarity_trace": False,
            },
            "no_gate": True,
            "retries": 0,
            "self_cooling": False,
            "rho_annealing": False,
            "online_stationarity_stop": False,
        },
        "collection": {
            "development_round_budget": DEVELOPMENT_ROUND_BUDGET,
            "budget_status": (
                "locked_pending_user_notification"
                if DEVELOPMENT_ROUND_BUDGET is None else "frozen"
            ),
            "output_state": "final_current",
            "trace": "every_initial_and_post_round_current_state",
        },
        "datasets": {
            name: {
                "schema": str(specification["schema"]),
                "queries": str(specification["queries"]),
                "marginals": str(specification["marginals"]),
                "n_records": specification["n_records"],
                "query_count": specification["query_count"],
                "device": specification["device"],
                "sha256": specification["sha256"],
            }
            for name, specification in DATASETS.items()
        },
        "detector": {
            "window_and_thresholds": "not_yet_calibrated",
            "validation_may_run": False,
        },
    }
    _strict_json_bytes(manifest)
    return manifest


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(frozen_protocol_manifest()))


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def environment_manifest() -> Dict[str, Any]:
    status = _git_output("status", "--porcelain")
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": status == "",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "started_at": datetime.now().astimezone().isoformat(),
    }


def _clip_audit(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    direction_evaluated = int(sum(
        diagnostics["direction_logit_evaluated_count_history"]
    ))
    direction_clipped = int(sum(
        diagnostics["direction_logit_clipped_count_history"]
    ))
    gibbs_evaluated = int(
        diagnostics[
            "factorized_gibbs_conditional_logit_evaluated_count"
        ]
    )
    gibbs_clipped = int(
        diagnostics[
            "factorized_gibbs_conditional_logit_clipped_count"
        ]
    )
    if direction_clipped > direction_evaluated:
        raise RuntimeError("方向 clip 计数超过评价数")
    if gibbs_clipped > gibbs_evaluated:
        raise RuntimeError("Gibbs clip 计数超过评价数")
    return {
        "direction": {
            "evaluated_count": direction_evaluated,
            "clipped_count": direction_clipped,
            "clipped_rate": (
                direction_clipped / direction_evaluated
                if direction_evaluated else 0.0
            ),
        },
        "gibbs_conditional": {
            "evaluated_count": gibbs_evaluated,
            "clipped_count": gibbs_clipped,
            "clipped_rate": (
                gibbs_clipped / gibbs_evaluated
                if gibbs_evaluated else 0.0
            ),
        },
    }


def derive_workload_seed_s0(
    workload: LoadedWorkload,
    seed: int,
    *,
    log_every: int = 100_000,
) -> tuple[float, Dict[str, Any]]:
    return derive_fixed_direction_reference_scale(
        target=workload.target,
        queries=workload.queries,
        schema=workload.schema,
        n_records=workload.n_records,
        seed=seed,
        fixed_alpha=FIXED_ALPHA,
        rho=FIXED_RHO,
        eta=FIXED_ETA,
        mu=FIXED_MU,
        diffusion_direction_strength=FIXED_DIRECTION_STRENGTH,
        diffusion_direction_logit_clip=FIXED_DIRECTION_LOGIT_CLIP,
        max_rounds=S0_PREFLIGHT_MAX_ROUNDS,
        marginals=workload.marginals,
        device=DATASETS[workload.name]["device"],
        log_every=log_every,
        **SHARED_GENERATOR_PARAMS,
    )


def _validate_completed_run(
    workload: LoadedWorkload,
    kernel: str,
    seed: int,
    rounds: int,
    final_table: pd.DataFrame,
    diagnostics: Dict[str, Any],
    trace: StationarityTrace,
    preflight: Dict[str, Any],
) -> None:
    if (
        preflight["seed"] != seed
        or preflight["n_records"] != workload.n_records
        or preflight["query_identity_sha256"]
        != workload.query_identity_sha256
        or preflight["target_identity_sha256"]
        != workload.target_identity_sha256
    ):
        raise RuntimeError("s0 preflight 身份与正式 workload 不一致")
    if diagnostics["rounds_run"] != rounds:
        raise RuntimeError("轨迹没有跑满固定最大预算")
    if diagnostics["termination_reason"] != "max_rounds":
        raise RuntimeError("量程轨迹出现非预算终止")
    if diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError("no-gate 轨迹出现未应用 proposal")
    if trace.post_round_count != rounds or trace.state_count != rounds + 1:
        raise RuntimeError("stationarity trace 状态数与轮数不一致")
    if trace.query_identity_sha256 != workload.query_identity_sha256:
        raise RuntimeError("trace 查询身份不一致")
    if trace.target_identity_sha256 != workload.target_identity_sha256:
        raise RuntimeError("trace target 身份不一致")
    if diagnostics["initial_table_sha256"] != preflight[
        "initial_table_sha256"
    ]:
        raise RuntimeError("正式轨迹没有从 preflight 的同一 S0 重启")
    if diagnostics[
        "primary_rng_post_initialization_state_sha256"
    ] != preflight["primary_rng_post_initialization_state_sha256"]:
        raise RuntimeError("正式轨迹初始化后的主 RNG 与 preflight 不一致")
    if _frame_sha256(final_table) != trace.observations[-1][
        "current_table_sha256"
    ]:
        raise RuntimeError("最终 current table 与 trace 终态不一致")
    expected_sweeps = KERNELS[kernel]["factorized_gibbs_sweeps"]
    if diagnostics["params"]["factorized_gibbs_sweeps"] != expected_sweeps:
        raise RuntimeError("kernel sweeps 与冻结协议不一致")
    expected_compiled = KERNELS[kernel][
        "factorized_gibbs_use_compiled_workload"
    ]
    if diagnostics["params"][
        "factorized_gibbs_use_compiled_workload"
    ] is not expected_compiled:
        raise RuntimeError("factor workload builder 与冻结协议不一致")
    gibbs_evaluated = diagnostics[
        "factorized_gibbs_conditional_logit_evaluated_count"
    ]
    if kernel == "independent" and gibbs_evaluated != 0:
        raise RuntimeError("独立核不应评价 Gibbs 条件 logit")
    if kernel == "factorized_gibbs" and gibbs_evaluated != diagnostics[
        "factorized_gibbs_microsteps"
    ]:
        raise RuntimeError("Gibbs 条件 logit 评价数与微步数不一致")
    if diagnostics["params"][
        "diffusion_direction_reference_scale"
    ] != preflight["direction_reference_scale"]:
        raise RuntimeError("正式轨迹未固定使用 preflight s0")
    if int(diagnostics["params"]["seed"]) != seed:
        raise RuntimeError("diagnostics seed 与运行 seed 不一致")


def collect_one_trajectory(
    workload: LoadedWorkload,
    kernel: str,
    seed: int,
    rounds: int,
    s0: float,
    preflight: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any], StationarityTrace]:
    if kernel not in KERNELS:
        raise ValueError(f"未知 kernel：{kernel!r}")
    if s0 != preflight.get("direction_reference_scale"):
        raise ValueError("s0 必须与绑定的 preflight 结果完全一致")
    final_table, diagnostics, trace = (
        run_stationarity_calibration_evolution(
            target=workload.target,
            queries=workload.queries,
            schema=workload.schema,
            n_records=workload.n_records,
            n_rounds=rounds,
            seed=seed,
            fixed_alpha=FIXED_ALPHA,
            rho=FIXED_RHO,
            eta=FIXED_ETA,
            mu=FIXED_MU,
            diffusion_direction_strength=FIXED_DIRECTION_STRENGTH,
            diffusion_direction_reference_scale=s0,
            diffusion_direction_logit_clip=FIXED_DIRECTION_LOGIT_CLIP,
            marginals=workload.marginals,
            device=DATASETS[workload.name]["device"],
            log_every=100,
            **SHARED_GENERATOR_PARAMS,
            **KERNELS[kernel],
        )
    )
    _validate_completed_run(
        workload,
        kernel,
        seed,
        rounds,
        final_table,
        diagnostics,
        trace,
        preflight,
    )
    return final_table, diagnostics, trace


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


def save_collected_run(
    output_dir: Path,
    *,
    mode: str,
    workload: LoadedWorkload,
    kernel: str,
    seed: int,
    rounds: int,
    preflight: Dict[str, Any],
    diagnostics: Dict[str, Any],
    trace: StationarityTrace,
    final_table: pd.DataFrame,
    elapsed_sec: float,
    environment: Dict[str, Any],
) -> Path:
    destination = output_dir / workload.name / f"seed_{seed}" / kernel
    if destination.exists():
        raise FileExistsError(f"run 输出已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=destination.parent,
        prefix=f".{kernel}.partial-",
    ))
    try:
        trace_paths = save_stationarity_trace(trace, temporary / "trace")
        trace_files = {
            "metadata": {
                "path": str(
                    Path(trace_paths["metadata_path"]).relative_to(temporary)
                ),
                "sha256": _sha256_file(
                    Path(trace_paths["metadata_path"])
                ),
            },
            "query_array": {
                "path": str(
                    Path(trace_paths["query_array_path"]).relative_to(
                        temporary
                    )
                ),
                "sha256": _sha256_file(
                    Path(trace_paths["query_array_path"])
                ),
            }
        }
        if trace_files["query_array"]["sha256"] != trace_paths[
            "query_array_sha256"
        ]:
            raise RuntimeError("query array SHA-256 写入后不一致")
        manifest = {
            "contract_version": STAGE2B_RANGE_FINDING_PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256(),
            "mode": mode,
            "formal_development_calibration": (
                mode == "development"
                and environment["git_worktree_clean_including_untracked"]
                and DEVELOPMENT_ROUND_BUDGET == rounds
                and seed in DEVELOPMENT_SEEDS
            ),
            "dataset": workload.name,
            "kernel": kernel,
            "seed": seed,
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
                "final_table_sha256": _frame_sha256(final_table),
                "primary_rng_state_sha256": diagnostics[
                    "primary_rng_state_sha256"
                ],
                "factorized_gibbs_rng_state_sha256": diagnostics[
                    "factorized_gibbs_rng_state_sha256"
                ],
                "clip_audit": _clip_audit(diagnostics),
                "elapsed_sec": elapsed_sec,
            },
            "trace_files": trace_files,
            "environment": environment,
        }
        _write_json_exclusive(temporary / "run_manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def run_collection(
    *,
    mode: str,
    datasets: Sequence[str],
    kernels: Sequence[str],
    seeds: Sequence[int],
    rounds: int,
    output_dir: Path,
) -> List[Path]:
    if mode not in {"smoke", "development"}:
        raise ValueError("run_collection mode 必须是 smoke 或 development")
    selected_datasets = _validate_unique_known(
        datasets, DATASETS, "datasets"
    )
    selected_kernels = _validate_unique_known(kernels, KERNELS, "kernels")
    if mode == "development":
        selected_seeds = validate_development_seeds(seeds)
        if DEVELOPMENT_ROUND_BUDGET is None:
            raise RuntimeError(
                "development 正式运行仍被轮数硬锁保护；"
                "必须先向用户报告轮数与预计开销"
            )
        if rounds != DEVELOPMENT_ROUND_BUDGET:
            raise ValueError("rounds 与冻结 development 预算不一致")
    else:
        validate_smoke_request(selected_datasets, seeds, rounds)
        selected_seeds = list(seeds)

    environment = environment_manifest()
    if mode == "development" and not environment[
        "git_worktree_clean_including_untracked"
    ]:
        raise RuntimeError("development 正式运行要求干净工作树")

    outputs: List[Path] = []
    for dataset in selected_datasets:
        workload = load_public_workload(dataset)
        for seed in selected_seeds:
            s0, preflight = derive_workload_seed_s0(workload, seed)
            for kernel in selected_kernels:
                start = time.perf_counter()
                final_table, diagnostics, trace = collect_one_trajectory(
                    workload,
                    kernel,
                    seed,
                    rounds,
                    s0,
                    preflight,
                )
                elapsed = time.perf_counter() - start
                destination = save_collected_run(
                    output_dir,
                    mode=mode,
                    workload=workload,
                    kernel=kernel,
                    seed=seed,
                    rounds=rounds,
                    preflight=preflight,
                    diagnostics=diagnostics,
                    trace=trace,
                    final_table=final_table,
                    elapsed_sec=elapsed,
                    environment=environment,
                )
                outputs.append(destination)
                print(
                    f"[{mode} {dataset} seed={seed} {kernel}] "
                    f"{rounds} rounds -> {destination}",
                    flush=True,
                )
    return outputs


def build_execution_plan(
    datasets: Sequence[str],
    kernels: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    selected_datasets = _validate_unique_known(
        datasets, DATASETS, "datasets"
    )
    selected_kernels = _validate_unique_known(kernels, KERNELS, "kernels")
    selected_seeds = validate_development_seeds(seeds)
    cells = [
        {"dataset": dataset, "seed": seed, "kernel": kernel}
        for dataset in selected_datasets
        for seed in selected_seeds
        for kernel in selected_kernels
    ]
    return {
        "protocol_sha256": protocol_sha256(),
        "mode": "plan_only_no_generation",
        "development_round_budget": DEVELOPMENT_ROUND_BUDGET,
        "development_execution_locked": (
            DEVELOPMENT_ROUND_BUDGET is None
        ),
        "cells": cells,
        "trajectory_count": len(cells),
        "validation_seeds_touched": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "smoke", "development"),
        default="plan",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=None,
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=tuple(KERNELS),
        default=list(KERNELS),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/issue53_stage2b_range_finding"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = args.datasets or (
        ["test_300x10"]
        if args.mode == "smoke" else list(DATASETS)
    )
    if args.mode == "plan":
        seeds = args.seeds or list(DEVELOPMENT_SEEDS)
        print(json.dumps(
            build_execution_plan(datasets, args.kernels, seeds),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    if args.mode == "smoke":
        seeds = args.seeds or [SMOKE_DEFAULT_SEED]
        rounds = args.rounds if args.rounds is not None else 2
    else:
        seeds = args.seeds or list(DEVELOPMENT_SEEDS)
        rounds = (
            args.rounds
            if args.rounds is not None else DEVELOPMENT_ROUND_BUDGET
        )
        if rounds is None:
            raise RuntimeError(
                "development round budget 尚未冻结；先运行 plan 并向用户报告"
            )
    run_collection(
        mode=args.mode,
        datasets=datasets,
        kernels=args.kernels,
        seeds=seeds,
        rounds=rounds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
