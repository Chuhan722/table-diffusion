#!/usr/bin/env python3
"""Collect the frozen Issue #53 Stage 5 independent/factor kernel A/B."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__:
    from scripts import issue53_stage5_protocol as protocol
    from scripts import run_issue53_fixed_alpha_calibration as fixed_collection
else:
    import issue53_stage5_protocol as protocol
    import run_issue53_fixed_alpha_calibration as fixed_collection


PROTOCOL_VERSION = protocol.PROTOCOL_VERSION
FROZEN_PROTOCOL_SHA256 = protocol.FROZEN_PROTOCOL_SHA256
PROTOCOL_DOC = protocol.PROTOCOL_DOC
PROTOCOL_DOC_SHA256 = protocol.PROTOCOL_DOC_SHA256
OUTPUT_DIR = protocol.OUTPUT_DIR
SMOKE_OUTPUT_DIR = protocol.SMOKE_OUTPUT_DIR
COLLECTION_REPORT = protocol.COLLECTION_REPORT
SMOKE_REPORT = "smoke_report.json"
DATASETS = protocol.DATASETS
DATASET_ORDER = protocol.DATASET_ORDER
ARM_INDEPENDENT = protocol.ARM_INDEPENDENT
ARM_FACTOR = protocol.ARM_FACTOR
ARMS = protocol.ARMS
SEEDS = protocol.FORMAL_SEEDS
SMOKE_SEED = protocol.SMOKE_SEED


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    return protocol.file_sha256(path)


def _git_text(root: Path, *arguments: str) -> str:
    return fixed_collection._git_text(root, *arguments)


def _load_json(path: Path) -> dict[str, Any]:
    return fixed_collection._load_json(path)


def _strict_json_bytes(value: Any) -> bytes:
    return protocol._strict_json_bytes(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def _json_safe(value: Any, _path: tuple[str, ...] = ()) -> Any:
    """Normalize diagnostics to strict JSON without silently dropping fields."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"strict JSON 禁止 NaN：path={'.'.join(_path)}")
        if math.isinf(value):
            if value > 0 and _path and _path[-1] == "tol":
                return "positive_infinity"
            raise ValueError(
                f"非 tol 字段禁止 infinity：path={'.'.join(_path)}"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, _path + (str(key),))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, _path + (str(index),))
            for index, item in enumerate(value)
        ]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), _path)
    if hasattr(value, "item"):
        return _json_safe(value.item(), _path)
    raise TypeError(f"diagnostics 含不可序列化类型：{type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            _json_safe(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _arm_label(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm 不在冻结矩阵中：{arm!r}")
    return arm


def _safe_artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact path 必须是相对字符串")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise RuntimeError(f"artifact path 越界：{relative!r}")
    return candidate


def generator_params(
    dataset: str,
    seed: int,
    arm: str,
    *,
    mode: str = "formal",
) -> dict[str, Any]:
    return protocol.generator_params(dataset, seed, arm, mode=mode)


def frozen_protocol_manifest() -> dict[str, Any]:
    return protocol.frozen_protocol_manifest()


def protocol_sha256() -> str:
    return protocol.protocol_sha256()


def assert_frozen_protocol_identity() -> str:
    return protocol.assert_frozen_protocol_identity()


def build_plan() -> dict[str, Any]:
    return protocol.build_plan()


def _audit_dataset(root: Path, name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise ValueError(f"未知数据集：{name!r}")
    observed = fixed_collection._audit_dataset(root, name)
    expected = DATASETS[name]
    if (
        observed["sha256"] != expected["sha256"]
        or observed["query_count"] != expected["query_count"]
        or observed["order_counts"] != expected["order_counts"]
        or observed["query_identity_sha256"]
        != expected["query_identity_sha256"]
        or observed["target_vector_sha256"]
        != expected["target_vector_sha256"]
    ):
        raise RuntimeError(f"{name} Stage 5 输入身份漂移")
    return observed


def _audit_inputs(root: Path) -> dict[str, Any]:
    if _sha256_file(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("Stage 5 protocol 文档 SHA 漂移")
    return {
        name: {
            key: value
            for key, value in _audit_dataset(root, name).items()
            if key not in {"queries", "targets"}
        }
        for name in DATASET_ORDER
    }


def _load_runtime() -> SimpleNamespace:
    return fixed_collection._load_runtime()


def _environment(root: Path, runtime: SimpleNamespace) -> dict[str, Any]:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Stage 5 执行要求包含 untracked 在内的干净工作树")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise RuntimeError("每个 Stage 5 seed shard 必须显式且只暴露一张 GPU")
    if not runtime.torch.cuda.is_available() or runtime.torch.cuda.device_count() != 1:
        raise RuntimeError("nltcs CUDA 路径要求进程内恰好一张可用 GPU")
    return {
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "worktree_clean_including_untracked": True,
        "hostname": platform.node(),
        "python": sys.version,
        "numpy": runtime.np.__version__,
        "pandas": runtime.pd.__version__,
        "torch": runtime.torch.__version__,
        "cuda_visible_devices": visible,
        "cuda_device_name": runtime.torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@contextlib.contextmanager
def _exclusive_gpu_shard_lock(environment: dict[str, Any]):
    """Prevent two Stage 5 processes from sharing one visible physical GPU."""

    import fcntl

    identity = (
        f"{environment['hostname']}\0"
        f"{environment['cuda_visible_devices']}\0"
        f"{environment['cuda_device_name']}"
    )
    label = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / (
        f"table-diffevo-issue53-stage5-gpu-{label}.lock"
    )
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "同一 host/visible GPU 已有 Stage 5 shard 在运行；"
                "协议禁止同一物理 GPU 并发两个 shard"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _applied_rows(clock: dict[str, Any]) -> int:
    return fixed_collection._applied_rows(clock)


def _concentration_summary(
    diagnostics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    if int(diagnostics["rounds_run"]) == 0:
        return {
            "row_max_prob_mean": None,
            "row_max_prob_max": None,
            "effective_donors_mean": None,
            "effective_donor_fraction": None,
            "donor_top_share": None,
            "tail_window_rounds": 0,
            "effective_donor_fraction_denominator": n_records - 1,
        }
    return fixed_collection._concentration_summary(diagnostics, n_records)


def _observed_param_expectation(
    dataset: str,
    seed: int,
    arm: str,
    *,
    mode: str,
    compiled_override: bool | None,
) -> dict[str, Any]:
    params = generator_params(dataset, seed, arm, mode=mode)
    compiled = params["factorized_gibbs_use_compiled_workload"]
    if compiled_override is not None:
        compiled = compiled_override
    return {
        "n_records": DATASETS[dataset]["n_records"],
        "n_rounds": params["n_rounds"],
        "seed": seed,
        "beta": params["beta"],
        "h": params["h"],
        "rho": params["rho"],
        "eta": params["eta"],
        "mu": params["mu"],
        "tol": params["tol"],
        "device": DATASETS[dataset]["device"],
        "eval_method": params["eval_method"],
        "batch_size": params["batch_size"],
        "init_method": "marginal",
        "distance_mode": params["distance_mode"],
        "lambda": params["lambda_param"],
        "alpha_schedule_mode": params["alpha_schedule_mode"],
        "fixed_alpha": params["fixed_alpha"],
        "delta": params["delta"],
        "winsorize_quantiles": params["winsorize_quantiles"],
        "exclude_self": params["exclude_self"],
        "max_retries": params["max_retries"],
        "residual_directed_diffusion": params["residual_directed_diffusion"],
        "diffusion_direction_strength": params["diffusion_direction_strength"],
        "diffusion_direction_normalization": params[
            "diffusion_direction_normalization"
        ],
        "diffusion_direction_reference_scale": None,
        "diffusion_direction_logit_clip": params[
            "diffusion_direction_logit_clip"
        ],
        "factorized_gibbs_sweeps": params["factorized_gibbs_sweeps"],
        "factorized_gibbs_max_order": params["factorized_gibbs_max_order"],
        "factorized_gibbs_logit_clip": params["factorized_gibbs_logit_clip"],
        "factorized_gibbs_use_compiled_workload": compiled,
        "candidate_budget": params["candidate_budget"],
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": protocol.PATIENCE_TICKS,
        "horizon_invariant": False,
    }


def _audit_generator_identity(
    diagnostics: dict[str, Any],
    dataset: str,
    seed: int,
    arm: str,
    *,
    mode: str,
    compiled_override: bool | None,
) -> None:
    observed = diagnostics.get("params")
    if not isinstance(observed, dict):
        raise RuntimeError("generator params diagnostics 缺失")
    expected = _observed_param_expectation(
        dataset,
        seed,
        arm,
        mode=mode,
        compiled_override=compiled_override,
    )
    drift = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if drift:
        raise RuntimeError(
            f"{dataset}/{arm}/seed{seed} generator 参数漂移：{drift}"
        )
    rounds = int(diagnostics["rounds_run"])
    if diagnostics["alpha_history"] != [protocol.FIXED_ALPHA] * rounds:
        raise RuntimeError("fixed alpha 历史漂移")
    if diagnostics.get("stationarity_trace") is not None:
        raise RuntimeError("collection 不得记录 stationarity trace")
    if diagnostics.get("natural_work_snapshots") is not None:
        raise RuntimeError("collection 不得记录 natural-work snapshots")


def _exact_count_error_sum(target: Any, answers: Any, runtime: Any) -> int:
    target_array = runtime.np.asarray(target)
    answer_array = runtime.np.asarray(answers)
    if target_array.shape != answer_array.shape:
        raise RuntimeError("measured target/answer 形状漂移")
    if not runtime.np.all(runtime.np.isfinite(answer_array)):
        raise RuntimeError("measured answer 含非有限值")
    rounded_target = runtime.np.rint(target_array)
    rounded_answer = runtime.np.rint(answer_array)
    if not runtime.np.array_equal(target_array, rounded_target):
        raise RuntimeError("measured target 不是整数 count")
    if not runtime.np.array_equal(answer_array, rounded_answer):
        raise RuntimeError("measured answer 不是整数 count")
    return int(runtime.np.abs(rounded_target - rounded_answer).sum())


def _terminal_behavior(
    diagnostics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    clocks = diagnostics["transition_clock_history"]
    cumulative = 0
    work_by_state = {0: 0.0}
    for clock in clocks:
        cumulative += _applied_rows(clock)
        work_by_state[int(clock["state_index"])] = cumulative / n_records
    decision = diagnostics["inner_early_stopping"]["last_decision"]
    best_index = int(decision["best_state_index_diagnostic_only"])
    final_work = float(decision["normalized_work"])
    if best_index not in work_by_state:
        raise RuntimeError("best state 的 natural-work 地址缺失")
    return {
        "terminal_state_index": int(decision["state_index"]),
        "best_state_index_diagnostic_only": best_index,
        "completed_work_ticks": int(decision["completed_work_ticks"]),
        "cumulative_applied_participating_rows": int(
            decision["cumulative_participating_rows"]
        ),
        "normalized_work_at_stop": final_work,
        "natural_work_since_last_strict_best": float(
            final_work - work_by_state[best_index]
        ),
        "consecutive_no_progress_ticks": int(
            decision["consecutive_no_progress_ticks"]
        ),
        "resource_cap_source_diagnostic_only": diagnostics[
            "inner_early_stopping"
        ]["resource_cap_source_diagnostic_only"],
    }


def _cost_summary(diagnostics: dict[str, Any], case_elapsed: float) -> dict[str, Any]:
    return {
        "raw_rounds": int(diagnostics["rounds_run"]),
        "candidate_evaluation_count": int(
            diagnostics["candidate_evaluation_count"]
        ),
        "state_evaluation_count": int(diagnostics["state_evaluation_count"]),
        "distance_evaluation_count": int(
            diagnostics["distance_evaluation_count"]
        ),
        "direction_evaluation_count": int(
            diagnostics["direction_evaluation_count"]
        ),
        "direction_evaluation_elapsed_sec": float(
            diagnostics["direction_evaluation_elapsed_sec"]
        ),
        "factorized_gibbs_workload_compile_elapsed_sec": float(
            diagnostics["factorized_gibbs_workload_compile_elapsed_sec"]
        ),
        "factorized_gibbs_factor_build_elapsed_sec": float(
            diagnostics["factorized_gibbs_factor_build_elapsed_sec"]
        ),
        "factorized_gibbs_sample_elapsed_sec": float(
            diagnostics["factorized_gibbs_sample_elapsed_sec"]
        ),
        "factorized_gibbs_active_rows": int(
            diagnostics["factorized_gibbs_active_rows"]
        ),
        "factorized_gibbs_active_blocks": int(
            diagnostics["factorized_gibbs_active_blocks"]
        ),
        "factorized_gibbs_factor_count": int(
            diagnostics["factorized_gibbs_factor_count"]
        ),
        "factorized_gibbs_factor_table_entries": int(
            diagnostics["factorized_gibbs_factor_table_entries"]
        ),
        "factorized_gibbs_microsteps": int(
            diagnostics["factorized_gibbs_microsteps"]
        ),
        "factorized_gibbs_conditional_logit_evaluated_count": int(
            diagnostics[
                "factorized_gibbs_conditional_logit_evaluated_count"
            ]
        ),
        "factorized_gibbs_conditional_logit_clipped_count": int(
            diagnostics[
                "factorized_gibbs_conditional_logit_clipped_count"
            ]
        ),
        "generator_elapsed_sec": float(diagnostics["elapsed_sec"]),
        "case_wall_clock_elapsed_sec": float(case_elapsed),
    }


def _audit_kernel_counters(
    diagnostics: dict[str, Any], arm: str, rounds: int
) -> None:
    direction_evaluated = diagnostics["direction_logit_evaluated_count_history"]
    direction_clipped = diagnostics["direction_logit_clipped_count_history"]
    if len(direction_evaluated) != rounds or len(direction_clipped) != rounds:
        raise RuntimeError("direction clip 诊断长度漂移")
    if any(
        int(clipped) < 0 or int(clipped) > int(evaluated)
        for clipped, evaluated in zip(direction_clipped, direction_evaluated)
    ):
        raise RuntimeError("direction clip 计数非法")
    microsteps = int(diagnostics["factorized_gibbs_microsteps"])
    conditional = int(
        diagnostics["factorized_gibbs_conditional_logit_evaluated_count"]
    )
    active_blocks = int(diagnostics["factorized_gibbs_active_blocks"])
    if arm == ARM_INDEPENDENT:
        if (
            microsteps != 0
            or conditional != 0
            or active_blocks != 0
            or diagnostics["factorized_gibbs_initial_rng_state_sha256"]
            is not None
            or diagnostics["factorized_gibbs_rng_state_sha256"] is not None
        ):
            raise RuntimeError("independent 臂意外实例化或执行 Gibbs")
        return
    if (
        microsteps != protocol.FACTOR_SWEEPS * active_blocks
        or conditional != microsteps
        or not isinstance(
            diagnostics["factorized_gibbs_initial_rng_state_sha256"], str
        )
        or not isinstance(diagnostics["factorized_gibbs_rng_state_sha256"], str)
    ):
        raise RuntimeError("factor Gibbs microstep / RNG 身份漂移")


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    dataset: str,
    arm: str,
    seed: int,
    protocol_sha: str,
    git_commit: str,
    environment: dict[str, Any],
    runtime: Any,
    mode: str,
    execution_order_index: int,
    compiled_override: bool | None = None,
    artifact_label: str | None = None,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    input_audit = _audit_dataset(root, dataset)
    schema = runtime.load_schema(str(root / spec["schema"]))
    queries = runtime.load_queries(str(root / spec["queries"]))
    marginals = runtime.load_marginals(str(root / spec["marginals"]))
    target = runtime.np.asarray(input_audit["targets"], dtype=float)
    params = generator_params(dataset, seed, arm, mode=mode)
    if compiled_override is not None:
        if arm != ARM_FACTOR:
            raise ValueError("compiled override 只允许 smoke factor replay")
        params["factorized_gibbs_use_compiled_workload"] = compiled_override

    started = time.perf_counter()
    table, diagnostics = runtime.run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=spec["n_records"],
        marginals=marginals,
        device=spec["device"],
        init_method="marginal",
        **params,
    )
    case_elapsed = time.perf_counter() - started
    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} 输出不是 terminal current")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError("output_table_identity 漂移")
    _audit_generator_identity(
        diagnostics,
        dataset,
        seed,
        arm,
        mode=mode,
        compiled_override=compiled_override,
    )

    reason = diagnostics["termination_reason"]
    if reason not in {*protocol.NORMAL_REASONS, protocol.RESOURCE_CAP_REASON}:
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} 未返回冻结 A/B/C")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if (
        len(clocks) != rounds
        or diagnostics["accept_history"] != [True] * rounds
        or diagnostics["proposal_attempts_history"] != [1] * rounds
        or diagnostics["accepted_attempt_history"] != [1] * rounds
        or diagnostics["candidate_evaluation_count"] != rounds
    ):
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} no-gate/one-candidate 审计失败")
    _audit_kernel_counters(diagnostics, arm, rounds)

    answers = runtime.np.asarray(runtime.evaluate_table(final_table, queries))
    count_error_sum = _exact_count_error_sum(target, answers, runtime)
    loss = float(runtime.compute_squared_loss(target, answers))
    normalized_l1 = float(
        runtime.compute_normalized_l1(target, answers, spec["n_records"])
    )
    if (
        loss != diagnostics["final_current_squared_loss"]
        or not math.isclose(
            normalized_l1,
            diagnostics["final_current_normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or count_error_sum
        != int(round(normalized_l1 * spec["n_records"] * spec["query_count"]))
    ):
        raise RuntimeError("terminal measured 指标复算漂移")

    behavior = _terminal_behavior(diagnostics, spec["n_records"])
    terminal_frame_sha = fixed_collection._frame_sha256(final_table)
    expected_terminal_state_sha = (
        diagnostics["initial_table_sha256"]
        if rounds == 0
        else clocks[-1]["post_current_table_sha256"]
    )
    if (
        terminal_frame_sha != expected_terminal_state_sha
        or behavior["terminal_state_index"] != rounds
        or len(diagnostics["current_state_metrics_history"]) != rounds + 1
        or int(
            diagnostics["current_state_metrics_history"][-1]["state_index"]
        )
        != rounds
    ):
        raise RuntimeError("terminal current 状态索引或 table SHA 漂移")
    cost = _cost_summary(diagnostics, case_elapsed)
    role = artifact_label or _arm_label(arm)
    case_dir = shard_dir / dataset / role
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    terminal_file_sha = _sha256_file(table_path)
    if terminal_file_sha != terminal_frame_sha:
        raise RuntimeError("terminal CSV bytes 与 canonical table SHA 漂移")
    diagnostics_path = case_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    relative_table = str(table_path.relative_to(shard_dir))
    relative_diagnostics = str(diagnostics_path.relative_to(shard_dir))
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "kernel_arm"
            if compiled_override is None
            else "compiled_equivalence_rowwise_replay"
        ),
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "execution_order_index": execution_order_index,
        "hostname": environment["hostname"],
        "device": spec["device"],
        "cuda_device_name": environment["cuda_device_name"],
        "query_file": str(spec["queries"]),
        "query_count": spec["query_count"],
        "query_order_counts": {
            str(order): count for order, count in input_audit["order_counts"].items()
        },
        "query_identity_sha256": input_audit["query_identity_sha256"],
        "target_vector_sha256": input_audit["target_vector_sha256"],
        "generator_params": _json_safe(diagnostics["params"]),
        "termination_reason": reason,
        "inner_complete": bool(diagnostics["inner_complete"]),
        "output_table_identity": "terminal_current",
        "terminal_behavior": behavior,
        "measured": {
            "absolute_count_error_sum": count_error_sum,
            "normalized_l1_mean": normalized_l1,
            "squared_loss": loss,
            "best_squared_loss_diagnostic_only": float(
                diagnostics["best_loss_diagnostic_only"]
            ),
            "normalized_l1_at_best_squared_loss_diagnostic_only": float(
                diagnostics[
                    "normalized_l1_at_best_squared_loss_diagnostic_only"
                ]
            ),
            "terminal_minus_best_squared_loss": float(
                loss - diagnostics["best_loss_diagnostic_only"]
            ),
            "terminal_minus_best_normalized_l1": float(
                normalized_l1
                - diagnostics[
                    "normalized_l1_at_best_squared_loss_diagnostic_only"
                ]
            ),
        },
        "cost": cost,
        "direction_reference_scale": float(
            diagnostics["direction_reference_scale"]
        ),
        "direction_logit_evaluated_count": int(
            sum(diagnostics["direction_logit_evaluated_count_history"])
        ),
        "direction_logit_clipped_count": int(
            sum(diagnostics["direction_logit_clipped_count_history"])
        ),
        "initial_table_sha256": diagnostics["initial_table_sha256"],
        "primary_rng_post_initialization_state_sha256": diagnostics[
            "primary_rng_post_initialization_state_sha256"
        ],
        "primary_rng_state_sha256_by_round": [
            clock["primary_rng_state_sha256"] for clock in clocks
        ],
        "primary_rng_state_sha256": diagnostics["primary_rng_state_sha256"],
        "factorized_gibbs_initial_rng_state_sha256": diagnostics[
            "factorized_gibbs_initial_rng_state_sha256"
        ],
        "factorized_gibbs_rng_state_sha256": diagnostics[
            "factorized_gibbs_rng_state_sha256"
        ],
        "donor_concentration": _concentration_summary(
            diagnostics, spec["n_records"]
        ),
        "terminal_table_path": relative_table,
        "terminal_table_sha256": terminal_file_sha,
        "diagnostics_path": relative_diagnostics,
        "diagnostics_sha256": _sha256_file(diagnostics_path),
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    result_path = case_dir / "result.json"
    result["result_path"] = str(result_path.relative_to(shard_dir))
    _write_json(result_path, result)
    returned = dict(result)
    returned["result_artifact_sha256"] = _sha256_file(result_path)
    print(
        f"[{mode} seed={seed} {dataset}/{role}] complete: "
        f"reason={reason}, rounds={rounds}",
        flush=True,
    )
    return returned


def _assert_seed_pairing(
    rows: Sequence[dict[str, Any]], seed: int, *, mode: str = "formal"
) -> dict[str, Any]:
    expected_order = protocol.case_order_for_seed(seed, mode=mode)
    if len(rows) != len(expected_order):
        raise RuntimeError(f"seed {seed} 不是完整四 case shard")
    observed_order = tuple((row["dataset"], row["arm"]) for row in rows)
    if observed_order != expected_order:
        raise RuntimeError(f"seed {seed} case 顺序漂移")
    if [row["execution_order_index"] for row in rows] != list(
        range(len(expected_order))
    ):
        raise RuntimeError(f"seed {seed} execution order index 漂移")
    result = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset"] == dataset]
        if len(selected) != 2 or {row["arm"] for row in selected} != set(ARMS):
            raise RuntimeError(f"seed {seed}/{dataset} 不是完整 paired arms")
        initial = {row["initial_table_sha256"] for row in selected}
        primary = {
            row["primary_rng_post_initialization_state_sha256"]
            for row in selected
        }
        scales = {row["direction_reference_scale"] for row in selected}
        if (
            len(initial) != 1
            or len(primary) != 1
            or len(scales) != 1
            or not all(math.isfinite(value) and value > 0.0 for value in scales)
        ):
            raise RuntimeError(f"seed {seed}/{dataset} S0/RNG/direction scale 未配对")
        traces = [row["primary_rng_state_sha256_by_round"] for row in selected]
        shared_rounds = min(len(trace) for trace in traces)
        if traces[0][:shared_rounds] != traces[1][:shared_rounds]:
            raise RuntimeError(
                f"seed {seed}/{dataset} primary RNG common prefix 漂移"
            )
        shared_prefix = traces[0][:shared_rounds]
        if len({row["hostname"] for row in selected}) != 1 or len(
            {row["cuda_device_name"] for row in selected}
        ) != 1:
            raise RuntimeError(f"seed {seed}/{dataset} host/device 未配对")
        result[dataset] = {
            "initial_table_sha256": next(iter(initial)),
            "primary_rng_post_initialization_state_sha256": next(iter(primary)),
            "direction_reference_scale": next(iter(scales)),
            "primary_rng_common_prefix_round_count": shared_rounds,
            "primary_rng_common_prefix_sha256": _canonical_sha256(
                shared_prefix
            ),
            "hostname": selected[0]["hostname"],
            "cuda_device_name": selected[0]["cuda_device_name"],
        }
    return result


def _non_timing_factor_identity(row: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "terminal_table_sha256": row["terminal_table_sha256"],
        "termination_reason": row["termination_reason"],
        "terminal_behavior": row["terminal_behavior"],
        "measured": row["measured"],
        "initial_table_sha256": row["initial_table_sha256"],
        "primary_rng_post_initialization_state_sha256": row[
            "primary_rng_post_initialization_state_sha256"
        ],
        "primary_rng_state_sha256": row["primary_rng_state_sha256"],
        "factorized_gibbs_initial_rng_state_sha256": row[
            "factorized_gibbs_initial_rng_state_sha256"
        ],
        "factorized_gibbs_rng_state_sha256": row[
            "factorized_gibbs_rng_state_sha256"
        ],
        "direction_reference_scale": row["direction_reference_scale"],
        "current_state_metrics_history": diagnostics[
            "current_state_metrics_history"
        ],
        "transition_clock_history": diagnostics["transition_clock_history"],
        "loss_history": diagnostics["loss_history"],
        "accept_history": diagnostics["accept_history"],
        "rho_schedule_history": diagnostics["rho_schedule_history"],
        "alpha_history": diagnostics["alpha_history"],
        "direction_reference_scale_history": diagnostics[
            "direction_reference_scale_history"
        ],
        "direction_logit_evaluated_count_history": diagnostics[
            "direction_logit_evaluated_count_history"
        ],
        "direction_logit_clipped_count_history": diagnostics[
            "direction_logit_clipped_count_history"
        ],
        "factorized_gibbs_active_rows": diagnostics[
            "factorized_gibbs_active_rows"
        ],
        "factorized_gibbs_active_blocks": diagnostics[
            "factorized_gibbs_active_blocks"
        ],
        "factorized_gibbs_factor_count": diagnostics[
            "factorized_gibbs_factor_count"
        ],
        "factorized_gibbs_factor_table_entries": diagnostics[
            "factorized_gibbs_factor_table_entries"
        ],
        "factorized_gibbs_microsteps": diagnostics[
            "factorized_gibbs_microsteps"
        ],
        "factorized_gibbs_conditional_logit_evaluated_count": diagnostics[
            "factorized_gibbs_conditional_logit_evaluated_count"
        ],
        "factorized_gibbs_conditional_logit_clipped_count": diagnostics[
            "factorized_gibbs_conditional_logit_clipped_count"
        ],
    }


def _load_case_diagnostics(shard_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    return _load_json(_safe_artifact_path(shard_root, row["diagnostics_path"]))


def _compiled_equivalence(
    shard_root: Path,
    compiled: dict[str, Any],
    rowwise: dict[str, Any],
) -> dict[str, Any]:
    compiled_identity = _non_timing_factor_identity(
        compiled, _load_case_diagnostics(shard_root, compiled)
    )
    rowwise_identity = _non_timing_factor_identity(
        rowwise, _load_case_diagnostics(shard_root, rowwise)
    )
    compiled_sha = _canonical_sha256(compiled_identity)
    rowwise_sha = _canonical_sha256(rowwise_identity)
    if compiled_sha != rowwise_sha:
        raise RuntimeError(
            f"{compiled['dataset']} compiled/rowwise 非计时结果不等价"
        )
    return {
        "dataset": compiled["dataset"],
        "non_timing_identity_sha256": compiled_sha,
        "compiled_equals_rowwise": True,
        "compiled_result_artifact_sha256": compiled[
            "result_artifact_sha256"
        ],
        "rowwise_result_artifact_sha256": rowwise[
            "result_artifact_sha256"
        ],
    }


def run_smoke(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 Stage 5 protocol SHA-256")
    root = _repo_root()
    destination = root / SMOKE_OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"smoke 输出已存在，不覆盖：{destination}")
    runtime = _load_runtime()
    environment = _environment(root, runtime)
    input_audit = _audit_inputs(root)
    git_commit = environment["git_commit"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_gpu_shard_lock(environment), tempfile.TemporaryDirectory(
        prefix=".issue53-stage5-smoke.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        official_rows = [
            _run_case(
                root,
                temporary,
                dataset=dataset,
                arm=arm,
                seed=SMOKE_SEED,
                protocol_sha=expected,
                git_commit=git_commit,
                environment=environment,
                runtime=runtime,
                mode="smoke",
                execution_order_index=index,
            )
            for index, (dataset, arm) in enumerate(
                protocol.case_order_for_seed(SMOKE_SEED, mode="smoke")
            )
        ]
        replay_rows = []
        equivalence = []
        for replay_index, dataset in enumerate(DATASET_ORDER, start=len(official_rows)):
            rowwise = _run_case(
                root,
                temporary,
                dataset=dataset,
                arm=ARM_FACTOR,
                seed=SMOKE_SEED,
                protocol_sha=expected,
                git_commit=git_commit,
                environment=environment,
                runtime=runtime,
                mode="smoke",
                execution_order_index=replay_index,
                compiled_override=False,
                artifact_label="factor_random_scan_s8_rowwise_equivalence_replay",
            )
            replay_rows.append(rowwise)
            compiled = next(
                row
                for row in official_rows
                if row["dataset"] == dataset and row["arm"] == ARM_FACTOR
            )
            equivalence.append(_compiled_equivalence(temporary, compiled, rowwise))

        pairing = _assert_seed_pairing(official_rows, SMOKE_SEED, mode="smoke")
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "mode": "pipeline_smoke_real_inputs_formal_result_invalid",
            "formal_result_valid": False,
            "seed": SMOKE_SEED,
            "round_cap": protocol.SMOKE_ROUND_CAP,
            "execution_git_commit": git_commit,
            "environment": environment,
            "input_audit": input_audit,
            "case_count": len(official_rows),
            "rowwise_equivalence_replay_count": len(replay_rows),
            "official_results": official_rows,
            "rowwise_equivalence_replays": replay_rows,
            "initial_state_pairing": pairing,
            "compiled_equivalence": equivalence,
            "all_compiled_equivalence_passed": all(
                row["compiled_equals_rowwise"] for row in equivalence
            ),
            "raw_reference_data_accessed": False,
            "offline_classification_emitted": False,
            "privacy_budget_consumed": False,
        }
        _write_json(temporary / SMOKE_REPORT, report)
        os.replace(temporary, destination)
    return destination / SMOKE_REPORT


def _audit_result_artifacts(shard_root: Path, row: dict[str, Any]) -> None:
    table_path = _safe_artifact_path(shard_root, row["terminal_table_path"])
    diagnostics_path = _safe_artifact_path(shard_root, row["diagnostics_path"])
    result_path = _safe_artifact_path(shard_root, row["result_path"])
    if _sha256_file(table_path) != row["terminal_table_sha256"]:
        raise RuntimeError(f"terminal table artifact SHA 漂移：{table_path}")
    if _sha256_file(diagnostics_path) != row["diagnostics_sha256"]:
        raise RuntimeError(f"diagnostics artifact SHA 漂移：{diagnostics_path}")
    if _sha256_file(result_path) != row["result_artifact_sha256"]:
        raise RuntimeError(f"result artifact SHA 漂移：{result_path}")
    expected_payload = {
        key: value for key, value in row.items() if key != "result_artifact_sha256"
    }
    if _load_json(result_path) != expected_payload:
        raise RuntimeError(f"result artifact 内容与 manifest 漂移：{result_path}")


def _audit_smoke_gate(root: Path, execution_commit: str) -> dict[str, Any]:
    report_path = root / SMOKE_OUTPUT_DIR / SMOKE_REPORT
    report = _load_json(report_path)
    if (
        report.get("contract_version") != PROTOCOL_VERSION
        or report.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256
        or report.get("formal_result_valid") is not False
        or report.get("seed") != SMOKE_SEED
        or report.get("execution_git_commit") != execution_commit
        or report.get("case_count") != 4
        or report.get("rowwise_equivalence_replay_count") != 2
        or report.get("all_compiled_equivalence_passed") is not True
    ):
        raise RuntimeError("formal collection 的 smoke gate 身份漂移或未通过")
    smoke_root = report_path.parent
    all_rows = report["official_results"] + report["rowwise_equivalence_replays"]
    for row in all_rows:
        _audit_result_artifacts(smoke_root, row)
    _assert_seed_pairing(report["official_results"], SMOKE_SEED, mode="smoke")
    for dataset in DATASET_ORDER:
        compiled = next(
            row
            for row in report["official_results"]
            if row["dataset"] == dataset and row["arm"] == ARM_FACTOR
        )
        rowwise = next(
            row
            for row in report["rowwise_equivalence_replays"]
            if row["dataset"] == dataset
        )
        _compiled_equivalence(smoke_root, compiled, rowwise)
    return {
        "smoke_report_path": str(SMOKE_OUTPUT_DIR / SMOKE_REPORT),
        "smoke_report_sha256": _sha256_file(report_path),
        "execution_git_commit": execution_commit,
        "compiled_equivalence_passed": True,
        "formal_result_valid": False,
    }


def run_shard(confirmed_protocol_sha256: str, shard_index: int) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 Stage 5 protocol SHA-256")
    if shard_index not in range(len(SEEDS)):
        raise ValueError(f"shard_index 非法：{shard_index!r}")
    root = _repo_root()
    seed = SEEDS[shard_index]
    destination = root / OUTPUT_DIR / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(f"formal seed shard 已存在，不覆盖：{destination}")
    runtime = _load_runtime()
    environment = _environment(root, runtime)
    git_commit = environment["git_commit"]
    smoke_gate = _audit_smoke_gate(root, git_commit)
    input_audit = _audit_inputs(root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_gpu_shard_lock(environment), tempfile.TemporaryDirectory(
        prefix=f".seed_{seed}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        rows = [
            _run_case(
                root,
                temporary,
                dataset=dataset,
                arm=arm,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
                environment=environment,
                runtime=runtime,
                mode="formal",
                execution_order_index=index,
            )
            for index, (dataset, arm) in enumerate(
                protocol.case_order_for_seed(seed)
            )
        ]
        pairing = _assert_seed_pairing(rows, seed)
        manifest = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "mode": "formal_result_blind_collection_seed_shard",
            "formal_result_valid": True,
            "shard_index": shard_index,
            "seed": seed,
            "execution_git_commit": git_commit,
            "environment": environment,
            "smoke_gate": smoke_gate,
            "input_audit": input_audit,
            "case_order": [
                {"dataset": dataset, "arm": arm}
                for dataset, arm in protocol.case_order_for_seed(seed)
            ],
            "initial_state_pairing": pairing,
            "case_count": len(rows),
            "results": rows,
            "raw_reference_data_accessed": False,
            "partial_matrix_comparison_emitted": False,
            "offline_classification_emitted": False,
            "privacy_budget_consumed": False,
        }
        _write_json(temporary / "shard_manifest.json", manifest)
        os.replace(temporary, destination)
    return destination / "shard_manifest.json"


def _pairing_from_unordered_rows(
    rows: Sequence[dict[str, Any]], seed: int
) -> dict[str, Any]:
    order = protocol.case_order_for_seed(seed)
    indexed = {(row["dataset"], row["arm"]): row for row in rows}
    if set(indexed) != set(order):
        raise RuntimeError(f"seed {seed} case identity 不完整")
    return _assert_seed_pairing([indexed[key] for key in order], seed)


def aggregate(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 Stage 5 protocol SHA-256")
    root = _repo_root()
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Stage 5 aggregate 要求干净工作树")
    current_commit = _git_text(root, "rev-parse", "HEAD")
    smoke_gate = _audit_smoke_gate(root, current_commit)
    destination = root / OUTPUT_DIR
    report_path = destination / COLLECTION_REPORT
    if report_path.exists():
        raise FileExistsError(f"collection report 已存在，不覆盖：{report_path}")

    rows = []
    shard_sha = {}
    pairing = {}
    environments = {}
    for index, seed in enumerate(SEEDS):
        shard_root = destination / f"seed_{seed}"
        shard_path = shard_root / "shard_manifest.json"
        payload = _load_json(shard_path)
        if (
            payload.get("contract_version") != PROTOCOL_VERSION
            or payload.get("protocol_sha256") != expected
            or payload.get("formal_result_valid") is not True
            or payload.get("shard_index") != index
            or payload.get("seed") != seed
            or payload.get("execution_git_commit") != current_commit
            or payload.get("case_order")
            != [
                {"dataset": dataset, "arm": arm}
                for dataset, arm in protocol.case_order_for_seed(seed)
            ]
            or payload.get("raw_reference_data_accessed") is not False
            or payload.get("partial_matrix_comparison_emitted") is not False
            or payload.get("offline_classification_emitted") is not False
        ):
            raise RuntimeError(f"seed {seed} shard 身份/信息边界漂移")
        seed_rows = payload.get("results")
        if not isinstance(seed_rows, list) or len(seed_rows) != 4:
            raise RuntimeError(f"seed {seed} shard case 不完整")
        for row in seed_rows:
            _audit_result_artifacts(shard_root, row)
        pairing[str(seed)] = _assert_seed_pairing(seed_rows, seed)
        if payload.get("initial_state_pairing") != pairing[str(seed)]:
            raise RuntimeError(f"seed {seed} pairing manifest 漂移")
        rows.extend(seed_rows)
        shard_sha[str(seed)] = _sha256_file(shard_path)
        environments[str(seed)] = payload["environment"]

    expected_identities = {
        (seed, dataset, arm)
        for seed in SEEDS
        for dataset in DATASET_ORDER
        for arm in ARMS
    }
    identities = {(row["seed"], row["dataset"], row["arm"]) for row in rows}
    if len(rows) != 40 or identities != expected_identities:
        raise RuntimeError("formal collection 40-case 矩阵不完整或重复")
    for seed in SEEDS:
        _pairing_from_unordered_rows(
            [row for row in rows if row["seed"] == seed], seed
        )

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    arm_rank = {name: index for index, name in enumerate(ARMS)}
    report = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": expected,
        "protocol": frozen_protocol_manifest(),
        "mode": "complete_result_blind_collection_after_artifact_audit",
        "formal_result_valid": True,
        "execution_git_commit": current_commit,
        "smoke_gate": smoke_gate,
        "case_count": len(rows),
        "paired_dataset_seed_count": len(SEEDS) * len(DATASET_ORDER),
        "shard_manifest_sha256_by_seed": shard_sha,
        "environment_by_seed": environments,
        "initial_state_pairing_by_seed": pairing,
        "raw_results": sorted(
            rows,
            key=lambda row: (
                row["seed"],
                dataset_rank[row["dataset"]],
                arm_rank[row["arm"]],
            ),
        ),
        "collection_audit": {
            "all_40_cases_present": True,
            "all_20_pairs_present": True,
            "all_artifact_sha256_verified": True,
            "all_initial_state_pairing_verified": True,
            "all_case_order_verified": True,
            "single_clean_execution_commit": True,
            "compiled_equivalence_smoke_gate_passed": True,
        },
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "offline_classification_emitted": False,
        "privacy_budget_consumed": False,
        "claim_scope": "collection_identity_only_before_offline_evaluation",
    }
    _atomic_write_json(report_path, report)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("--confirm-protocol-sha", required=True)
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--confirm-protocol-sha", required=True)
    shard.add_argument(
        "--shard-index", required=True, type=int, choices=range(len(SEEDS))
    )
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    if args.command == "run-smoke":
        path = run_smoke(args.confirm_protocol_sha)
        print(f"Stage 5 pipeline smoke -> {path}")
        print(f"smoke report SHA-256 -> {_sha256_file(path)}")
        return
    if args.command == "run-shard":
        path = run_shard(args.confirm_protocol_sha, args.shard_index)
        print(f"Stage 5 formal shard -> {path}")
        return
    path = aggregate(args.confirm_protocol_sha)
    print(f"Stage 5 collection -> {path}")
    print(f"collection SHA-256 -> {_sha256_file(path)}")


if __name__ == "__main__":
    main()
