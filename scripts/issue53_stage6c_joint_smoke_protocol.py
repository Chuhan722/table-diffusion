"""Issue #53 第 6C 阶段两数据三方法联合测速冻结协议。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__:
    from scripts import issue53_stage6c_joint_trajectories as joint
else:
    import issue53_stage6c_joint_trajectories as joint


PROTOCOL_VERSION = "issue53-stage6c-joint-two-dataset-smoke-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage6C两数据三方法联合长轨迹测速接线协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "35b97ae43120724e6125732ad17b2e006c032465fa1876f858477d0d362b4aff"
)

# 不写入 manifest（冻结清单），避免自指；清单确定后填入。
FROZEN_PROTOCOL_SHA256 = (
    "aae6bee94f5872b61b3af3263a8f9bf8bcc87b0fb64812b1610069ea29787924"
)

SMOKE_SEED = 9907
ROUNDS = 20
CANDIDATE_BUDGET = 20
MAX_WORKERS = 2
OUTPUT_DIR = Path("outputs/issue53_stage6c_joint_smoke_seed9907_v1")

EXPECTED_GPU = {
    "physical_index": 1,
    "cuda_visible_devices": "1",
    "uuid": "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02",
    "name": "NVIDIA GeForce RTX 4090",
    "process_device": "cuda:0",
    "visible_device_count": 1,
}

DATASETS: dict[str, dict[str, Any]] = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path(
            "configs/test_300x10/measured_50query_30_15_5.json"
        ),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "query_count": 50,
        "max_factor_order": 4,
        "input_sha256": {
            "schema": (
                "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
            ),
            "queries": (
                "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
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
        "n_records": 16_181,
        "query_count": 1_001,
        "max_factor_order": 3,
        "input_sha256": {
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

IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "cb0cb9fd19f7172c957cbd28543a9a80630d68f3ffe97e4ccc079601a0a8801b"
        ),
    },
    "shared_update_plan": {
        "path": Path("src/table_diffevo/update.py"),
        "sha256": (
            "505c6ce582dfbc6cf597305ca54eb384fe9a460e0abc0da5efbf50414fcae541"
        ),
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "8c0fd1f8233edf8c756d9036e0ebce7289f45c5f7c5ece100fd7acc585d83b47"
        ),
    },
    "joint_task_matrix": {
        "path": Path("scripts/issue53_stage6c_joint_trajectories.py"),
        "sha256": (
            "37070be03d0b2d8121553f49f01079e37744eb418143f380160b058c9671a194"
        ),
    },
}

ALLOWED_TASK_REPORT_FIELDS = (
    "task_id",
    "dataset",
    "arm",
    "seed",
    "requested_rounds",
    "rounds_run",
    "termination_reason",
    "completed",
    "device",
    "elapsed_sec",
    "sec_per_round",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "output_table_identity",
    "terminal_table_sha256",
    "main_rng_endpoint_sha256",
    "all_applied_unconditionally",
    "applied_round_count",
    "gap_reference_scale_established",
    "gap_microsteps",
    "gap_expected_microsteps",
    "gap_8k_identity",
    "gap_clip_hit_count",
    "nonfinite_count",
)

FORBIDDEN_QUALITY_FIELDS = (
    "loss",
    "loss_history",
    "best_loss",
    "normalized_l1_error",
    "normalized_l1_history",
    "final_current_squared_loss",
    "final_current_normalized_l1",
    "query_counts",
    "query_residuals",
    "quality_rank",
    "winner",
    "terminal_table",
    "final_table",
)


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def common_generator_params() -> dict[str, Any]:
    """返回除数据集和方法核以外的固定完整生成参数。"""

    return {
        "n_rounds": ROUNDS,
        "seed": SMOKE_SEED,
        "beta": 1.0,
        "h": 0.8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "device": "cuda",
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "log_every": ROUNDS + 1,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": 16.0,
        "alpha_max": 16.0,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "retry_rho_decay": 0.5,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_logit_clip": 30.0,
        "candidate_budget": CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 16.0,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": None,
    }


def task_generator_params(dataset: str, arm: str) -> dict[str, Any]:
    """返回一条冻结测速任务传给现有生成器的参数。"""

    if dataset not in DATASETS:
        raise ValueError(f"未知测速数据集：{dataset!r}")
    params = common_generator_params()
    params.update(joint.arm_kernel_params(arm))
    params.update(
        {
            "n_records": int(DATASETS[dataset]["n_records"]),
            "factorized_gibbs_max_order": int(
                DATASETS[dataset]["max_factor_order"]
            ),
        }
    )
    return params


def _task_plan() -> joint.JointTrajectoryPlan:
    return joint.build_joint_trajectory_plan(
        seeds=(SMOKE_SEED,),
        rounds_by_dataset={dataset: ROUNDS for dataset in joint.DATASET_ORDER},
    )


def frozen_protocol_manifest() -> dict[str, Any]:
    task_plan = _task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6C_joint_full_trajectory_wiring_and_timing_smoke",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "purpose": "wiring_and_resource_timing_only_not_method_evaluation",
        "task_matrix": {
            "datasets": list(joint.DATASET_ORDER),
            "arms": list(joint.ARM_ORDER),
            "seeds": list(task_plan.seeds),
            "rounds_by_dataset": dict(task_plan.rounds_by_dataset),
            "task_order": "seed_then_arm_then_dataset",
            "task_ids": [task.task_id for task in task_plan.tasks],
            "trajectory_count": len(task_plan.tasks),
            "all_tasks_declared_before_any_result": True,
            "partial_dataset_or_arm_plan_allowed": False,
            "result_dependent_task_removal_allowed": False,
            "smoke_seed_excluded_from_formal_effect_seeds": True,
        },
        "datasets": {
            name: {
                "schema": str(spec["schema"]),
                "queries": str(spec["queries"]),
                "marginals": str(spec["marginals"]),
                "n_records": spec["n_records"],
                "query_count": spec["query_count"],
                "max_factor_order": spec["max_factor_order"],
                "device": "cuda",
                "input_sha256": dict(spec["input_sha256"]),
            }
            for name, spec in DATASETS.items()
        },
        "generator_common": _jsonable(common_generator_params()),
        "arm_kernel_params": {
            arm: _jsonable(joint.arm_kernel_params(arm))
            for arm in joint.ARM_ORDER
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "no_gate_contract": {
            "tol_positive_infinity": True,
            "max_retries": 0,
            "accept_or_reject": False,
            "rollback": False,
            "winner_selection": False,
            "terminal_current_output": True,
            "exact_zero_pre_donor_stop_preserved": True,
            "gap_sweeps": 8,
            "gap_microsteps_identity": "8*K",
        },
        "execution": {
            "output_dir": str(OUTPUT_DIR),
            "max_workers": MAX_WORKERS,
            "multiprocessing_start_method": "spawn",
            "task_reordering": False,
            "adaptive_concurrency": False,
            "automatic_cpu_fallback": False,
            "atomic_output_creation": True,
            "expected_gpu": dict(EXPECTED_GPU),
        },
        "report_boundary": {
            "allowed_task_fields": list(ALLOWED_TASK_REPORT_FIELDS),
            "forbidden_quality_fields": list(FORBIDDEN_QUALITY_FIELDS),
            "quality_values_emitted": False,
            "method_ranking_or_classification_emitted": False,
            "raw_reference_or_heldout_evaluation": False,
            "generated_table_contents_persisted": False,
        },
        "authorization_boundary": {
            "protocol_frozen": True,
            "runner_wired_at_protocol_freeze": False,
            "smoke_run_authorized_at_protocol_freeze": False,
            "formal_effect_protocol_frozen": False,
            "formal_run_authorized": False,
            "public_default_kernel_changed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("第 6C 联合测速协议文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"第 6C 联合测速实现源码漂移：{name}")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "第 6C 联合测速协议清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_protocol_identity(repository_root)
    task_plan = _task_plan()
    return {
        "mode": "plan_only_no_dataset_read_no_generation",
        "protocol_sha256": protocol_sha,
        "task_ids": [task.task_id for task in task_plan.tasks],
        "trajectory_count": len(task_plan.tasks),
        "max_workers": MAX_WORKERS,
        "output_dir": str(OUTPUT_DIR),
        "quality_values_emitted": False,
        "runner_wired": False,
        "generation_started": False,
        "confirmation_consumed": False,
    }


def require_run_confirmation(confirmed_protocol_sha256: str | None) -> None:
    if confirmed_protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError("联合测速运行需要用户另行授权并确认完整协议 SHA-256")
