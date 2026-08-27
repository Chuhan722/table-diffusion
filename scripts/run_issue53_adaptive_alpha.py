#!/usr/bin/env python3
"""采集 Issue #53 两档自适应 alpha 的冻结三臂矩阵。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import run_issue53_fixed_alpha_calibration as fixed_collection
else:
    import run_issue53_fixed_alpha_calibration as fixed_collection


PROTOCOL_VERSION = "issue53-adaptive-alpha-collection-v1"
FROZEN_PROTOCOL_SHA256 = (
    "5a88ddc5077df82528b7fda3cd12a4fb79c1b8e5c027d6d555d4a50e869e911e"
)
PROTOCOL_DOC = Path("docs/设计/Issue53_两档自适应alpha结果前冻结协议.md")
PROTOCOL_DOC_SHA256 = (
    "24913ea3faafb6202f90ec2cdb082d21195af98fa83ffd31412dc48eefdf9d1e"
)
PROTOCOL_DOC_COMMIT = "ec4b07520f45bee69324f41070b917361c4d3092"

OUTPUT_DIR = Path("outputs/issue53_adaptive_alpha_v1")
COLLECTION_REPORT = "collection_report.json"
SEEDS = (328, 329, 330, 331, 332)
ARM_FIXED_16 = "fixed_alpha_16"
ARM_FIXED_12 = "fixed_alpha_12"
ARM_ADAPTIVE = "adaptive_alpha_16_12"
ARMS = (ARM_FIXED_16, ARM_FIXED_12, ARM_ADAPTIVE)
DATASET_ORDER = ("test_300x10", "nltcs")
CASE_ORDER = tuple(
    (dataset, arm) for dataset in DATASET_ORDER for arm in ARMS
)
DATASETS = fixed_collection.DATASETS

ADAPTIVE_SCHEDULE_MODE = "stall_escape_16_12"
ADAPTIVE_CONFIG = {
    "normal_alpha": 16.0,
    "escape_alpha": 12.0,
    "stall_trigger_ticks": 2,
    "escape_duration_ticks": 2,
}
PATIENCE_TICKS = 6
RHO = 0.01
ROUND_CAP = 6000
CANDIDATE_BUDGET = 6000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return fixed_collection._sha256_file(path)


def _frame_sha256(frame: Any) -> str:
    return fixed_collection._frame_sha256(frame)


def _git_text(root: Path, *arguments: str) -> str:
    return fixed_collection._git_text(root, *arguments)


def _load_json(path: Path) -> dict[str, Any]:
    return fixed_collection._load_json(path)


def _arm_label(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm 不在冻结矩阵中：{arm!r}")
    return arm


def generator_params(seed: int, arm: str) -> dict[str, Any]:
    """返回三臂之一的精确生成配置。"""

    if seed not in SEEDS:
        raise ValueError(f"seed 不在冻结矩阵中：{seed!r}")
    _arm_label(arm)
    fixed_alpha = {
        ARM_FIXED_16: 16.0,
        ARM_FIXED_12: 12.0,
        ARM_ADAPTIVE: None,
    }[arm]
    return {
        "n_rounds": ROUND_CAP,
        "seed": seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": RHO,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "eval_method": "vectorized",
        "batch_size": 256,
        "log_every": 100,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": 16.0,
        "alpha_max": 16.0,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": (
            ADAPTIVE_SCHEDULE_MODE if arm == ARM_ADAPTIVE else "fixed"
        ),
        "fixed_alpha": fixed_alpha,
        "record_transition_clocks": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, float) and math.isinf(value):
            value = "positive_infinity"
        result[key] = value
    return result


def _common_generator_params() -> dict[str, Any]:
    params = generator_params(SEEDS[0], ARM_FIXED_16).copy()
    for key in (
        "seed",
        "alpha_min",
        "alpha_max",
        "alpha_schedule_mode",
        "fixed_alpha",
    ):
        params.pop(key)
    return _jsonable_params(params)


def _arm_generator_identity(arm: str) -> dict[str, Any]:
    params = generator_params(SEEDS[0], arm)
    return {
        "alpha_schedule_mode": params["alpha_schedule_mode"],
        "fixed_alpha": params["fixed_alpha"],
        "adaptive_alpha_config": (
            ADAPTIVE_CONFIG if arm == ARM_ADAPTIVE else None
        ),
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "two_level_stall_escape_alpha_evaluation",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "datasets": {
            name: {
                "schema": str(spec["schema"]),
                "queries": str(spec["queries"]),
                "marginals": str(spec["marginals"]),
                "n_records": spec["n_records"],
                "device": spec["device"],
                "query_count": spec["query_count"],
                "order_counts": {
                    str(order): count
                    for order, count in spec["order_counts"].items()
                },
                "query_identity_sha256": spec["query_identity_sha256"],
                "target_vector_sha256": spec["target_vector_sha256"],
                "input_sha256": spec["sha256"],
            }
            for name, spec in DATASETS.items()
        },
        "dataset_order": list(DATASET_ORDER),
        "arms": {
            arm: _arm_generator_identity(arm) for arm in ARMS
        },
        "arm_order": list(ARMS),
        "seeds": list(SEEDS),
        "case_order_within_seed": [
            {"dataset": dataset, "arm": arm}
            for dataset, arm in CASE_ORDER
        ],
        "trajectory_count": len(SEEDS) * len(CASE_ORDER),
        "common_generator": _common_generator_params(),
        "primary_comparison": f"{ARM_ADAPTIVE}_vs_{ARM_FIXED_16}",
        "mechanism_comparison": f"{ARM_FIXED_12}_vs_{ARM_FIXED_16}",
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "output_identity": "terminal_current",
        "initial_table_pairing": "same_sha_across_three_arms_per_dataset_seed",
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "parameter_retuning_allowed": False,
        "formal_generation_started": False,
        "execution": {
            "one_visible_gpu_per_seed_shard": True,
            "cases_within_seed_shard_serial": True,
            "same_dataset_seed_arms_share_host_and_backend": True,
            "cross_machine_merge_requires_short_prefix_equivalence": True,
        },
    }


def protocol_sha256() -> str:
    return hashlib.sha256(_strict_json_bytes(frozen_protocol_manifest())).hexdigest()


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol 身份漂移：expected={FROZEN_PROTOCOL_SHA256}, "
            f"observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": frozen_protocol_manifest(),
        "shards": [
            {
                "shard_index": index,
                "seed": seed,
                "case_count": len(CASE_ORDER),
            }
            for index, seed in enumerate(SEEDS)
        ],
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }


def _audit_dataset(root: Path, name: str) -> dict[str, Any]:
    return fixed_collection._audit_dataset(root, name)


def _audit_inputs(root: Path) -> dict[str, Any]:
    if _sha256_file(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("adaptive alpha protocol 文档 SHA 漂移")
    return {
        name: {
            key: value
            for key, value in _audit_dataset(root, name).items()
            if key not in {"queries", "targets"}
        }
        for name in DATASET_ORDER
    }


def _load_runtime():
    return fixed_collection._load_runtime()


def _environment(root: Path, runtime: Any) -> dict[str, Any]:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式采集要求包含 untracked 在内的干净工作树")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise RuntimeError("每个 seed shard 必须且只能显式暴露一张 GPU")
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


def _applied_rows(clock: dict[str, Any]) -> int:
    return fixed_collection._applied_rows(clock)


def _concentration_summary(
    diagnostics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    return fixed_collection._concentration_summary(diagnostics, n_records)


def _optional_series_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        return {
            "count": 0,
            "first": None,
            "final": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(normalized),
        "first": normalized[0],
        "final": normalized[-1],
        "mean": _mean(normalized),
        "median": _median(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def _concentration_window(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "row_max_prob_mean",
        "row_max_prob_max",
        "effective_donors_mean",
        "effective_donor_fraction",
        "donor_top_share",
    )
    return {
        "round_count": len(rows),
        "state_index_first": rows[0]["state_index"] if rows else None,
        "state_index_last": rows[-1]["state_index"] if rows else None,
        "metrics": {
            field: _optional_series_summary([row[field] for row in rows])
            for field in fields
        },
    }


def _observation_point(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_index": int(observation["state_index"]),
        "round_index": int(observation["state_index"]),
        "completed_work_ticks": int(observation["completed_work_ticks"]),
        "cumulative_participating_rows": int(
            observation["cumulative_participating_rows"]
        ),
        "normalized_work": float(observation["normalized_work"]),
    }


def _adaptive_phase_diagnostics(
    diagnostics: dict[str, Any],
    observations: Sequence[dict[str, Any]],
    n_records: int,
) -> dict[str, Any]:
    rounds = int(diagnostics["rounds_run"])
    histories = {
        "row_max_prob_mean": diagnostics["row_max_prob_mean_history"],
        "row_max_prob_max": diagnostics["row_max_prob_max_history"],
        "effective_donors_mean": diagnostics[
            "effective_donors_mean_history"
        ],
        "donor_top_share": diagnostics["donor_top_share_history"],
    }
    if any(len(values) != rounds for values in histories.values()):
        raise RuntimeError("自适应供体集中度诊断长度与轮数不一致")

    denominator = n_records - 1
    rows = []
    for index, observation in enumerate(observations):
        rows.append(
            {
                "state_index": int(observation["state_index"]),
                "phase": observation["phase_before"],
                "escape_index": observation["escape_index_observed"],
                "row_max_prob_mean": float(
                    histories["row_max_prob_mean"][index]
                ),
                "row_max_prob_max": float(
                    histories["row_max_prob_max"][index]
                ),
                "effective_donors_mean": float(
                    histories["effective_donors_mean"][index]
                ),
                "effective_donor_fraction": float(
                    histories["effective_donors_mean"][index] / denominator
                ),
                "donor_top_share": float(
                    histories["donor_top_share"][index]
                ),
            }
        )

    starts = [
        observation
        for observation in observations
        if "escape_started" in observation["events"]
    ]
    completions = {
        int(observation["escape_index_observed"]): observation
        for observation in observations
        if "escape_completed" in observation["events"]
    }
    segments = []
    for position, trigger in enumerate(starts):
        escape_index = int(trigger["escape_index_observed"])
        escape_rows = [
            row for row in rows if row["escape_index"] == escape_index
            and row["phase"] == "escape"
        ]
        escape_observations = [
            observation
            for observation in observations
            if observation["escape_index_observed"] == escape_index
            and observation["phase_before"] == "escape"
        ]
        completion = completions.get(escape_index)
        next_trigger_state = (
            int(starts[position + 1]["state_index"])
            if position + 1 < len(starts)
            else None
        )
        recovery_observations = []
        recovery_rows = []
        if completion is not None:
            completion_state = int(completion["state_index"])
            recovery_observations = [
                observation
                for observation in observations
                if observation["phase_before"] == "normal"
                and int(observation["state_index"]) > completion_state
                and (
                    next_trigger_state is None
                    or int(observation["state_index"]) <= next_trigger_state
                )
            ]
            recovery_state_indices = {
                int(observation["state_index"])
                for observation in recovery_observations
            }
            recovery_rows = [
                row
                for row in rows
                if row["state_index"] in recovery_state_indices
            ]
        best_during = [
            observation
            for observation in escape_observations
            if observation["best_updated"]
        ]
        best_after = [
            observation
            for observation in recovery_observations
            if observation["best_updated"]
        ]
        first_location = (
            "during_escape"
            if best_during
            else "after_restore"
            if best_after
            else "none_observed"
        )
        segments.append(
            {
                "escape_index": escape_index,
                "progress_epoch_at_trigger": int(
                    trigger["progress_epoch_before"]
                ),
                "trigger": _observation_point(trigger),
                "first_escape_state": (
                    _observation_point(escape_observations[0])
                    if escape_observations
                    else None
                ),
                "completion": (
                    _observation_point(completion)
                    if completion is not None
                    else None
                ),
                "completed": completion is not None,
                "new_best_during_escape_state_indices": [
                    int(observation["state_index"])
                    for observation in best_during
                ],
                "first_new_best_after_restore_state_index": (
                    int(best_after[0]["state_index"]) if best_after else None
                ),
                "first_new_best_location": first_location,
                "donor_concentration": {
                    "trigger_round_alpha16": _concentration_window(
                        [rows[int(trigger["state_index"]) - 1]]
                    ),
                    "escape_alpha12": _concentration_window(escape_rows),
                    "restored_alpha16_until_next_escape": (
                        _concentration_window(recovery_rows)
                    ),
                },
            }
        )

    first_trigger_state = (
        int(starts[0]["state_index"]) if starts else None
    )
    completed_states = [
        int(observation["state_index"])
        for observation in completions.values()
    ]
    return {
        "effective_donor_fraction_denominator": denominator,
        "phase_windows_are_descriptive_only": True,
        "phase_concentration": {
            "normal_alpha16_all": _concentration_window(
                [row for row in rows if row["phase"] == "normal"]
            ),
            "escape_alpha12_all": _concentration_window(
                [row for row in rows if row["phase"] == "escape"]
            ),
            "normal_alpha16_before_first_escape": _concentration_window(
                [
                    row
                    for row in rows
                    if row["phase"] == "normal"
                    and (
                        first_trigger_state is None
                        or row["state_index"] <= first_trigger_state
                    )
                ]
            ),
            "restored_alpha16_after_first_completed_escape": (
                _concentration_window(
                    [
                        row
                        for row in rows
                        if row["phase"] == "normal"
                        and completed_states
                        and row["state_index"] > min(completed_states)
                    ]
                )
            ),
        },
        "escape_segments": segments,
    }


def _adaptive_summary(
    diagnostics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    adaptive = diagnostics["adaptive_alpha"]
    observations = adaptive["observation_history"]
    rounds = int(diagnostics["rounds_run"])
    if len(observations) != rounds:
        raise RuntimeError("adaptive alpha 观察长度与轮数不一致")
    for index, (alpha, observation) in enumerate(
        zip(diagnostics["alpha_history"], observations),
        start=1,
    ):
        if (
            observation["state_index"] != index
            or observation["alpha_used"] != alpha
        ):
            raise RuntimeError("adaptive alpha 逐轮身份漂移")
    starts = [
        row for row in observations if "escape_started" in row["events"]
    ]
    completions = [
        row for row in observations if "escape_completed" in row["events"]
    ]
    if len(starts) != adaptive["escape_count"]:
        raise RuntimeError("adaptive alpha 触发计数漂移")
    alpha12_rows = sum(
        int(row["applied_participating_rows"])
        for row in observations
        if row["alpha_used"] == 12.0
    )
    return {
        "enabled": True,
        "escape_count": int(adaptive["escape_count"]),
        "triggered": bool(starts),
        "trigger_completed_work_ticks": [
            int(row["completed_work_ticks"]) for row in starts
        ],
        "trigger_state_indices": [int(row["state_index"]) for row in starts],
        "escape_completed_count": len(completions),
        "new_best_during_escape_count": sum(
            row["phase_before"] == "escape" and row["best_updated"]
            for row in observations
        ),
        "alpha12_round_count": sum(
            alpha == 12.0 for alpha in diagnostics["alpha_history"]
        ),
        "alpha12_normalized_work": float(alpha12_rows / n_records),
        "phase_diagnostics": _adaptive_phase_diagnostics(
            diagnostics, observations, n_records
        ),
    }


def _audit_schedule(
    diagnostics: dict[str, Any], arm: str, n_records: int
) -> dict[str, Any]:
    rounds = int(diagnostics["rounds_run"])
    observed = diagnostics["params"]
    common_ok = (
        observed["selection_scale_invariant"] is True
        and observed["residual_geometry"] == "relative"
        and observed["residual_geometry_floor"] == 8.0
        and observed["factorized_gibbs_sweeps"] == 0
        and observed["inner_early_stopping_patience_ticks"] == 6
    )
    if not common_ok:
        raise RuntimeError(f"{arm} 生成参数漂移")

    if arm == ARM_ADAPTIVE:
        if (
            observed["alpha_schedule_mode"] != ADAPTIVE_SCHEDULE_MODE
            or observed["fixed_alpha"] is not None
            or observed["adaptive_alpha_config"] != ADAPTIVE_CONFIG
            or diagnostics["adaptive_alpha"]["enabled"] is not True
            or diagnostics["adaptive_alpha"]["config"] != ADAPTIVE_CONFIG
            or any(alpha not in {12.0, 16.0} for alpha in diagnostics["alpha_history"])
            or (rounds > 0 and diagnostics["alpha_history"][0] != 16.0)
        ):
            raise RuntimeError("adaptive alpha 状态机身份漂移")
        return _adaptive_summary(diagnostics, n_records)

    expected = 16.0 if arm == ARM_FIXED_16 else 12.0
    if (
        observed["alpha_schedule_mode"] != "fixed"
        or observed["fixed_alpha"] != expected
        or diagnostics["alpha_history"] != [expected] * rounds
        or diagnostics["adaptive_alpha"]["enabled"] is not False
    ):
        raise RuntimeError(f"{arm} 固定 alpha 历史漂移")
    return {
        "enabled": False,
        "escape_count": 0,
        "triggered": False,
        "trigger_completed_work_ticks": [],
        "trigger_state_indices": [],
        "escape_completed_count": 0,
        "new_best_during_escape_count": 0,
        "alpha12_round_count": rounds if expected == 12.0 else 0,
        "alpha12_normalized_work": None,
    }


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    dataset: str,
    arm: str,
    seed: int,
    protocol_sha: str,
    git_commit: str,
    runtime: Any,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    audit = _audit_dataset(root, dataset)
    schema = runtime.load_schema(str(root / spec["schema"]))
    queries = runtime.load_queries(str(root / spec["queries"]))
    marginals = runtime.load_marginals(str(root / spec["marginals"]))
    target = runtime.np.asarray(audit["targets"], dtype=float)
    params = generator_params(seed, arm)

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
    elapsed = time.perf_counter() - started

    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} 不是终态当前表")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError("输出身份不是 terminal current")
    adaptive_summary = _audit_schedule(diagnostics, arm, spec["n_records"])

    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} 未返回 A/B/C")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{dataset}/{arm}/seed{seed} no-gate 审计失败")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError("候选评估数与轮数不一致")

    answers = runtime.np.asarray(runtime.evaluate_table(final_table, queries))
    loss = float(runtime.compute_squared_loss(target, answers))
    l1 = float(runtime.compute_normalized_l1(target, answers, spec["n_records"]))
    if (
        loss != diagnostics["final_current_squared_loss"]
        or not math.isclose(
            l1,
            diagnostics["final_current_normalized_l1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("终态测量指标复算不一致")
    work = sum(_applied_rows(clock) for clock in clocks) / spec["n_records"]
    concentration = _concentration_summary(diagnostics, spec["n_records"])

    case_dir = shard_dir / dataset / _arm_label(arm)
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "device": spec["device"],
        "query_file": str(spec["queries"]),
        "query_count": spec["query_count"],
        "query_order_counts": {
            str(order): count for order, count in audit["order_counts"].items()
        },
        "query_identity_sha256": audit["query_identity_sha256"],
        "target_vector_sha256": audit["target_vector_sha256"],
        "alpha_schedule_mode": diagnostics["params"]["alpha_schedule_mode"],
        "fixed_alpha": diagnostics["params"]["fixed_alpha"],
        "alpha_history": diagnostics["alpha_history"],
        "adaptive_alpha": diagnostics["adaptive_alpha"],
        "adaptive_alpha_summary": adaptive_summary,
        "termination_reason": reason,
        "inner_complete": bool(diagnostics["inner_complete"]),
        "output_table_identity": "terminal_current",
        "rounds_run": rounds,
        "candidate_evaluations": int(diagnostics["candidate_evaluation_count"]),
        "normalized_work_at_stop": float(work),
        "terminal_current_squared_loss": loss,
        "terminal_current_normalized_l1": l1,
        "best_loss_diagnostic_only": float(diagnostics["best_loss"]),
        "elapsed_sec": float(elapsed),
        "initial_table_sha256": diagnostics["initial_table_sha256"],
        "primary_rng_post_initialization_state_sha256": diagnostics[
            "primary_rng_post_initialization_state_sha256"
        ],
        "terminal_table_sha256": _frame_sha256(final_table),
        "donor_concentration": concentration,
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[seed={seed} {dataset}/{arm}] {reason} rounds={rounds} "
        f"work={work:.4f} L1={l1:.10f} "
        f"escapes={adaptive_summary['escape_count']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _assert_seed_pairing(
    rows: Sequence[dict[str, Any]], seed: int
) -> dict[str, dict[str, str]]:
    if len(rows) != len(CASE_ORDER):
        raise RuntimeError(f"seed {seed} 不是完整六臂")
    pairing = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset"] == dataset]
        if len(selected) != len(ARMS) or {row["arm"] for row in selected} != set(ARMS):
            raise RuntimeError(f"seed {seed}/{dataset} 不是完整三臂")
        table_hashes = {row["initial_table_sha256"] for row in selected}
        rng_hashes = {
            row["primary_rng_post_initialization_state_sha256"]
            for row in selected
        }
        if len(table_hashes) != 1 or len(rng_hashes) != 1:
            raise RuntimeError(f"seed {seed}/{dataset} 初始状态未配对")
        pairing[dataset] = {
            "initial_table_sha256": next(iter(table_hashes)),
            "primary_rng_post_initialization_state_sha256": next(iter(rng_hashes)),
        }
    return pairing


def run_shard(confirmed_protocol_sha256: str, shard_index: int) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    if shard_index not in range(len(SEEDS)):
        raise ValueError(f"shard_index 非法：{shard_index!r}")

    root = _repo_root()
    seed = SEEDS[shard_index]
    destination = root / OUTPUT_DIR / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(f"seed shard 已存在，不覆盖：{destination}")
    runtime = _load_runtime()
    environment = _environment(root, runtime)
    input_audit = _audit_inputs(root)
    git_commit = environment["git_commit"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".seed_{seed}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_case(
                root,
                temporary,
                dataset=dataset,
                arm=arm,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
                runtime=runtime,
            )
            for dataset, arm in CASE_ORDER
        ]
        pairing = _assert_seed_pairing(results, seed)
        manifest = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "shard_index": shard_index,
            "seed": seed,
            "git_commit": git_commit,
            "environment": environment,
            "input_audit": input_audit,
            "case_order": [
                {"dataset": dataset, "arm": arm}
                for dataset, arm in CASE_ORDER
            ],
            "initial_state_pairing": pairing,
            "results": results,
        }
        (temporary / "shard_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "shard_manifest.json"


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = {dataset: {} for dataset in DATASET_ORDER}
    for dataset in DATASET_ORDER:
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["arm"] == arm
            ]
            if len(selected) != len(SEEDS):
                raise RuntimeError(f"{dataset}/{arm} case 数漂移")
            cells[dataset][arm] = {
                "case_count": len(selected),
                "termination_counts": dict(
                    sorted(Counter(row["termination_reason"] for row in selected).items())
                ),
                "terminal_normalized_l1_mean": _mean(
                    [row["terminal_current_normalized_l1"] for row in selected]
                ),
                "terminal_normalized_l1_median": _median(
                    [row["terminal_current_normalized_l1"] for row in selected]
                ),
                "terminal_squared_loss_mean": _mean(
                    [row["terminal_current_squared_loss"] for row in selected]
                ),
                "rounds_mean": _mean([row["rounds_run"] for row in selected]),
                "normalized_work_mean": _mean(
                    [row["normalized_work_at_stop"] for row in selected]
                ),
                "escape_count_mean": _mean(
                    [row["adaptive_alpha_summary"]["escape_count"] for row in selected]
                ),
                "triggered_case_count": sum(
                    row["adaptive_alpha_summary"]["triggered"] for row in selected
                ),
            }
    return {
        "cells": cells,
        "resource_cap_case_count": sum(
            row["termination_reason"] == "resource_cap_reached" for row in rows
        ),
        "normal_completion_case_count": sum(
            row["termination_reason"] in {"fit_target_reached", "early_stopped"}
            for row in rows
        ),
    }


def aggregate(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    root = _repo_root()
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("aggregate 要求包含 untracked 在内的干净工作树")
    destination = root / OUTPUT_DIR
    report_path = destination / COLLECTION_REPORT
    if report_path.exists():
        raise FileExistsError(f"采集报告已存在，不覆盖：{report_path}")

    rows = []
    commits = set()
    pairing = {}
    for index, seed in enumerate(SEEDS):
        shard_path = destination / f"seed_{seed}" / "shard_manifest.json"
        payload = _load_json(shard_path)
        if (
            payload.get("protocol_sha256") != expected
            or payload.get("shard_index") != index
            or payload.get("seed") != seed
        ):
            raise RuntimeError(f"seed {seed} shard 身份漂移")
        commits.add(payload["git_commit"])
        seed_rows = payload.get("results")
        if not isinstance(seed_rows, list):
            raise TypeError(f"seed {seed} results 缺失")
        pairing[str(seed)] = _assert_seed_pairing(seed_rows, seed)
        for row in seed_rows:
            table_path = (
                destination
                / f"seed_{seed}"
                / row["dataset"]
                / _arm_label(row["arm"])
                / "terminal_current.csv"
            )
            if _sha256_file(table_path) != row["terminal_table_sha256"]:
                raise RuntimeError(f"seed {seed}/{row['dataset']}/{row['arm']} SHA 漂移")
            rows.append(row)

    identities = {(row["seed"], row["dataset"], row["arm"]) for row in rows}
    expected_identities = {
        (seed, dataset, arm)
        for seed in SEEDS
        for dataset, arm in CASE_ORDER
    }
    if len(rows) != 30 or identities != expected_identities:
        raise RuntimeError("30-case 矩阵不完整或重复")
    if len(commits) != 1:
        raise RuntimeError(f"shard Git commit 不一致：{sorted(commits)}")

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    arm_rank = {value: index for index, value in enumerate(ARMS)}
    report = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": expected,
        "protocol": frozen_protocol_manifest(),
        "execution_git_commit": next(iter(commits)),
        "case_count": len(rows),
        "raw_results": sorted(
            rows,
            key=lambda row: (
                row["seed"],
                dataset_rank[row["dataset"]],
                arm_rank[row["arm"]],
            ),
        ),
        "initial_state_pairing_by_seed": pairing,
        "summary": _summarize(rows),
        "claim_scope": "adaptive_alpha_collection_before_offline_evaluation",
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".collection-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
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
    if args.command == "run-shard":
        path = run_shard(args.confirm_protocol_sha, args.shard_index)
        print(f"adaptive alpha shard -> {path}")
        return
    path = aggregate(args.confirm_protocol_sha)
    print(f"adaptive alpha collection -> {path}")
    print(f"collection SHA-256 -> {_sha256_file(path)}")


if __name__ == "__main__":
    main()
