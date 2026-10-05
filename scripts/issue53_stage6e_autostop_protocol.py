"""Issue #53 Stage 6E 恢复既有自动停止的三核正式比较协议。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as stage6d


PROTOCOL_VERSION = "issue53-stage6e-autostop-three-kernel-formal-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage6E自动停止三核正式比较结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "78e01e6503c666daca7f6c34d9f2f6443ffa92e044bea32d315a04002e5512b2"
)

# 该常量不进入清单，避免自指。
FROZEN_PROTOCOL_SHA256 = (
    "fec396940db1d071cab9030ac49631cd18029f250754037c9b503c47efff36c4"
)

FORMAL_SEEDS = tuple(range(353, 358))
ROUND_CAP = 6_000
ROUNDS = ROUND_CAP
CANDIDATE_BUDGET = 6_000
PATIENCE_TICKS = 6
CHECKPOINT_ROUNDS = (0, 100, 250, 500, 1_000, 2_000, 4_000, 6_000)
MAX_WORKERS = 2
STABLE_WIN_MINIMUM = 4
LOWER_RISK_RATIO_MAX = 1.05
HIGHER_QUALITY_RATIO_MIN = 0.95
NORMAL_TERMINATION_REASONS = (
    "fit_target_reached",
    "early_stopped",
    "resource_cap_reached",
)

OUTPUT_DIR = Path("outputs/issue53_stage6e_autostop_three_kernel_formal_v1")
SHARD_OUTPUT_ROOT = Path(
    "outputs/issue53_stage6e_autostop_three_kernel_formal_v1_shards"
)
COLLECTION_REPORT = "collection_report.json"
SHARD_REPORT = "shard_report.json"
EVALUATION_REPORT = "evaluation_report.json"
L1_RESULTS_CSV = "l1_results.csv"
AUDIT_REPORT = "independent_audit.json"
POSITIVE_INFINITY_MANIFEST_SENTINEL = "positive_infinity"

ARM_GAP = "gap_l1_global_s8"
BASELINE_ARMS = ("factor_b_s8", "independent_b_s0")

EXPECTED_SOFTWARE = {
    "python_major_minor": "3.11",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "torch": "2.13.0+cu130",
    "cuda_runtime": "13.0",
}

LOCAL_SHARD = "local_rtx4090"
SHARD_ORDER = (LOCAL_SHARD,)
SHARD_BLOCKS = {
    LOCAL_SHARD: tuple(
        (seed, dataset)
        for seed in FORMAL_SEEDS
        for dataset in joint.DATASET_ORDER
    )
}
EXECUTION_SHARDS = {
    LOCAL_SHARD: {
        "hostname": "linyao-system",
        "task_count": 30,
        "paired_block_count": 10,
        "max_workers": MAX_WORKERS,
        "expected_gpu": {
            "physical_index": 1,
            "cuda_visible_devices": "1",
            "uuid": "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_mib": 24564,
            "process_device": "cuda:0",
            "visible_device_count": 1,
        },
    }
}

# Stage 6D 已冻结的数据、查询和离线评价身份不变。
DATASETS = stage6d.DATASETS
TEST_IDENTITY_ARTIFACT = stage6d.TEST_IDENTITY_ARTIFACT
TEST_IDENTITY_ARTIFACT_SHA256 = stage6d.TEST_IDENTITY_ARTIFACT_SHA256
TEST_GROUP_ORDER = stage6d.TEST_GROUP_ORDER
TEST_GROUP_COUNTS = stage6d.TEST_GROUP_COUNTS
TEST_GROUP_IDENTITIES = stage6d.TEST_GROUP_IDENTITIES
NLTCS_GROUP_COUNTS = stage6d.NLTCS_GROUP_COUNTS
NLTCS_GROUP_IDENTITIES = stage6d.NLTCS_GROUP_IDENTITIES

# 实现完成后填入字节身份。协议文件自身不列入，避免自指。
IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": "cb0cb9fd19f7172c957cbd28543a9a80630d68f3ffe97e4ccc079601a0a8801b",
    },
    "shared_update_plan": {
        "path": Path("src/table_diffevo/update.py"),
        "sha256": "505c6ce582dfbc6cf597305ca54eb384fe9a460e0abc0da5efbf50414fcae541",
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": "c166aca47d87fcb789b39a5fccef6f43eeec863960154591db4c5f79e3fa12ae",
    },
    "gap_triton_kernel": {
        "path": Path("src/table_diffevo/_gap_l1_triton.py"),
        "sha256": "418103e86b70d835bb5845fda29ae72d226e0fcc9007f0d3b02329ae5554b439",
    },
    "inner_early_stopping": {
        "path": Path("src/table_diffevo/inner_early_stopping.py"),
        "sha256": "a1dc68d9077eb98cf587281a63149e41ad22f2a7de7b3e76420cd3b27ecda211",
    },
    "joint_task_matrix": {
        "path": Path("scripts/issue53_stage6c_joint_trajectories.py"),
        "sha256": "37070be03d0b2d8121553f49f01079e37744eb418143f380160b058c9671a194",
    },
    "gpu_runtime_helpers": {
        "path": Path("scripts/run_issue53_stage6c_joint_smoke.py"),
        "sha256": "a676a9f2c29a9481dda66e842abeece9e3a46ad2f67d4bb91ce7df520446a864",
    },
    "parallel_execution": {
        "path": Path("src/table_diffevo/experiment_parallel.py"),
        "sha256": "44d6fd8a56903b2952b78621029126ce07c9e5e2f35ad3101233c5d243bac137",
    },
    "stationarity_trace": {
        "path": Path("src/table_diffevo/stationarity.py"),
        "sha256": "a280c18e630beb8f5342fa17a743b3847426e821cf66b2ca7a4f69a8cb0b2152",
    },
    "result_blind_query_identity": {
        "path": Path("scripts/freeze_issue53_test_query_workload_ab.py"),
        "sha256": "cfaf56569999438798d2673eee3d282814b714b08b7d192c046a443bdfd56410",
    },
    "offline_evaluation_helpers": {
        "path": Path("scripts/evaluate_issue53_fixed_alpha_calibration.py"),
        "sha256": "df41d09ec23e9272af762ae43c4379dda1190f1d5fae2a0ca3700569309afc3e",
    },
    "collector": {
        "path": Path("scripts/run_issue53_stage6e_autostop.py"),
        "sha256": "7130223c2dbfae5723f6d445e92ae7154588b4aebd631121a7f79f325f4b51d1",
    },
    "evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6e_autostop.py"),
        "sha256": "bced5d11806e6dc8e082660acd96193e739a77bef43698644ef3f62041c6508e",
    },
    "independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6e_autostop.py"),
        "sha256": "9ee100b7ec552fb17f57b9a5807524789de3ad3708a939f5f1d470194d4dbc60",
    },
}


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
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return (
            POSITIVE_INFINITY_MANIFEST_SENTINEL
            if value > 0
            else "negative_infinity"
        )
    return value


def common_generator_params() -> dict[str, Any]:
    """返回除数据、方法核和种子外的共同生成参数。"""

    return {
        "n_rounds": ROUND_CAP,
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
        "log_every": ROUND_CAP + 1,
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
        # 记录查询向量只用于事后提取固定检查点，不改变随机流。
        "record_stationarity_trace": True,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def task_generator_params(dataset: str, arm: str, seed: int) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"未知正式数据集：{dataset!r}")
    if arm not in joint.ARM_ORDER:
        raise ValueError(f"未知正式方法：{arm!r}")
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"随机种子不在正式集合：{seed!r}")
    params = common_generator_params()
    params.update(joint.arm_kernel_params(arm))
    params.update(
        {
            "seed": int(seed),
            "n_records": int(DATASETS[dataset]["n_records"]),
            "factorized_gibbs_max_order": int(
                DATASETS[dataset]["max_factor_order"]
            ),
        }
    )
    return params


def generator_params_manifest(dataset: str, arm: str, seed: int) -> dict[str, Any]:
    params = task_generator_params(dataset, arm, seed)
    nonfinite = [
        key
        for key, value in params.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if nonfinite != ["tol"] or params["tol"] != float("inf"):
        raise RuntimeError("正式生成参数只允许 tol=+inf")
    manifest = _jsonable(params)
    if manifest["tol"] != POSITIVE_INFINITY_MANIFEST_SENTINEL:
        raise RuntimeError("正式无门控 tol 清单编码漂移")
    _strict_json_bytes(manifest)
    return manifest


def task_plan() -> joint.JointTrajectoryPlan:
    return joint.build_joint_trajectory_plan(
        seeds=FORMAL_SEEDS,
        rounds_by_dataset={dataset: ROUND_CAP for dataset in joint.DATASET_ORDER},
    )


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    return {
        task.task_id: generator_params_manifest(task.dataset, task.arm, task.seed)
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def tasks_for_shard(shard_id: str) -> tuple[joint.JointTrajectoryTask, ...]:
    if shard_id != LOCAL_SHARD:
        raise ValueError(f"未知 Stage 6E 执行分片：{shard_id!r}")
    tasks = task_plan().tasks
    if len(tasks) != 30:
        raise RuntimeError("Stage 6E 正式任务数量漂移")
    return tasks


def task_shard_id(task: joint.JointTrajectoryTask) -> str:
    if task.task_id not in {item.task_id for item in task_plan().tasks}:
        raise RuntimeError(f"任务不在 Stage 6E 冻结矩阵：{task.task_id}")
    return LOCAL_SHARD


def shard_assignment_manifest() -> dict[str, Any]:
    tasks = tasks_for_shard(LOCAL_SHARD)
    return {
        LOCAL_SHARD: {
            "hostname": EXECUTION_SHARDS[LOCAL_SHARD]["hostname"],
            "task_count": len(tasks),
            "paired_block_count": len(SHARD_BLOCKS[LOCAL_SHARD]),
            "max_workers": MAX_WORKERS,
            "expected_gpu": dict(
                EXECUTION_SHARDS[LOCAL_SHARD]["expected_gpu"]
            ),
            "paired_blocks": [
                {"seed": seed, "dataset": dataset}
                for seed, dataset in SHARD_BLOCKS[LOCAL_SHARD]
            ],
            "task_ids_in_global_order": [task.task_id for task in tasks],
        }
    }


def shard_assignment_sha256() -> str:
    return canonical_sha256(shard_assignment_manifest())


def frozen_protocol_manifest() -> dict[str, Any]:
    tasks = task_plan().tasks
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6E_existing_autostop_three_kernel_formal",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "scientific_question": (
            "kernel_only_comparison_under_existing_complete_autostop_flow"
        ),
        "prior_fixed_2500_run_role": "diagnostic_only_not_reused_as_stage6e",
        "task_matrix": {
            "datasets": list(joint.DATASET_ORDER),
            "arms": list(joint.ARM_ORDER),
            "candidate_arm": ARM_GAP,
            "baseline_arms": list(BASELINE_ARMS),
            "seeds": list(FORMAL_SEEDS),
            "seeds_are_unseen_claimed": False,
            "round_cap": ROUND_CAP,
            "candidate_budget": CANDIDATE_BUDGET,
            "checkpoint_rounds_diagnostic_only": list(CHECKPOINT_ROUNDS),
            "trajectory_count": len(tasks),
            "task_order": "seed_then_arm_then_dataset",
            "task_ids": [task.task_id for task in tasks],
            "all_tasks_declared_before_any_result": True,
            "result_dependent_task_removal_allowed": False,
        },
        "datasets": {
            name: {
                "schema": str(spec["schema"]),
                "queries": str(spec["queries"]),
                "marginals": str(spec["marginals"]),
                "reference_evaluation_only": str(spec["reference"]),
                "n_records": spec["n_records"],
                "query_count": spec["query_count"],
                "order_counts": spec["order_counts"],
                "max_factor_order": spec["max_factor_order"],
                "query_identity_sha256": spec["query_identity_sha256"],
                "target_vector_sha256": spec["target_vector_sha256"],
                "trace_query_identity_sha256": spec[
                    "trace_query_identity_sha256"
                ],
                "trace_target_vector_sha256": spec[
                    "trace_target_vector_sha256"
                ],
                "input_sha256": dict(spec["input_sha256"]),
            }
            for name, spec in DATASETS.items()
        },
        "generator_common": _jsonable(common_generator_params()),
        "arm_kernel_params": {
            arm: _jsonable(joint.arm_kernel_params(arm))
            for arm in joint.ARM_ORDER
        },
        "kernel_only_difference_contract": {
            "all_non_kernel_generator_params_shared": True,
            "same_initial_table_and_primary_rng_after_initialization": True,
            "factor_b_s8": {
                "factorized_gibbs_sweeps": 8,
                "gap_l1_sweeps": 0,
            },
            "independent_b_s0": {
                "factorized_gibbs_sweeps": 0,
                "gap_l1_sweeps": 0,
            },
            "gap_l1_global_s8": {
                "factorized_gibbs_sweeps": 0,
                "gap_l1_sweeps": 8,
            },
        },
        "stopping_contract": {
            "patience_ticks": PATIENCE_TICKS,
            "natural_work_formula": "cumulative_participating_rows/n_records",
            "priority": list(NORMAL_TERMINATION_REASONS),
            "resource_round_cap": ROUND_CAP,
            "resource_candidate_cap": CANDIDATE_BUDGET,
            "terminal_current_output": True,
            "historical_best_output_allowed": False,
        },
        "no_gate_contract": {
            "tol_positive_infinity": True,
            "max_retries": 0,
            "one_candidate_per_round": True,
            "accept_or_reject": False,
            "rollback": False,
            "winner_selection": False,
            "proposal_applied_before_stopping_check": True,
            "gap_microsteps_identity": "8*K",
        },
        "offline_evaluation": {
            "generation_may_read_reference": False,
            "evaluation_reads_reference_after_complete_collection": True,
            "test_identity_artifact": {
                "path": str(TEST_IDENTITY_ARTIFACT),
                "sha256": TEST_IDENTITY_ARTIFACT_SHA256,
            },
            "test_groups": {
                "order": list(TEST_GROUP_ORDER),
                "counts": dict(TEST_GROUP_COUNTS),
                "identity_sha256": dict(TEST_GROUP_IDENTITIES),
            },
            "nltcs_groups": {
                "counts": dict(NLTCS_GROUP_COUNTS),
                "identity_sha256": dict(NLTCS_GROUP_IDENTITIES),
            },
            "stable_win_minimum": STABLE_WIN_MINIMUM,
            "lower_is_better_ratio_max": LOWER_RISK_RATIO_MAX,
            "higher_is_better_ratio_min": HIGHER_QUALITY_RATIO_MIN,
            "valid_row_rate_required": 1.0,
            "checkpoint_selection_allowed": False,
        },
        "timing_reporting": {
            "per_case_total_elapsed_sec": True,
            "per_case_average_sec_per_applied_round": True,
            "per_dataset_arm_total_elapsed_sec": True,
            "per_dataset_arm_total_applied_rounds": True,
            "per_dataset_arm_average_sec_per_applied_round": True,
            "wall_clock_role": "diagnostic_only",
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "execution": {
            "output_dir": str(OUTPUT_DIR),
            "shard_output_root": str(SHARD_OUTPUT_ROOT),
            "shard_order": list(SHARD_ORDER),
            "shard_assignment": shard_assignment_manifest(),
            "shard_assignment_sha256": shard_assignment_sha256(),
            "max_workers": MAX_WORKERS,
            "expected_software": dict(EXPECTED_SOFTWARE),
            "multiprocessing_start_method": "spawn",
            "generator_params_manifest_sha256": (
                generator_params_manifest_sha256()
            ),
            "all_cases_same_physical_gpu": True,
            "exclusive_idle_gpu_required": True,
            "formal_output_overwrite_allowed": False,
            "completed_cases_resumed_without_rerun": True,
        },
        "information_flow": {
            "collection_emits_partial_comparison": False,
            "collection_reads_raw_reference": False,
            "evaluation_requires_complete_collection_sha_confirmation": True,
            "auditor_reruns_generation": False,
            "post_result_threshold_retuning_allowed": False,
        },
        "authorization_boundary": {
            "preparation_may_run_generator": False,
            "formal_collection_authorized_at_freeze": False,
            "final_user_confirmation_required_before_generation": True,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("Stage 6E 正式协议文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"Stage 6E 正式实现源码漂移：{name}")
    if file_sha256(root / TEST_IDENTITY_ARTIFACT) != TEST_IDENTITY_ARTIFACT_SHA256:
        raise RuntimeError("Stage 6E test 离线查询身份文件漂移")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "Stage 6E 正式协议清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_protocol_identity(repository_root)
    tasks = task_plan().tasks
    return {
        "mode": "plan_only_no_dataset_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "task_ids": [task.task_id for task in tasks],
        "trajectory_count": len(tasks),
        "seeds": list(FORMAL_SEEDS),
        "round_cap": ROUND_CAP,
        "candidate_budget": CANDIDATE_BUDGET,
        "patience_ticks": PATIENCE_TICKS,
        "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
        "max_workers": MAX_WORKERS,
        "output_dir": str(OUTPUT_DIR),
        "shard_output_root": str(SHARD_OUTPUT_ROOT),
        "shard_assignment_sha256": shard_assignment_sha256(),
        "shards": shard_assignment_manifest(),
        "generation_started": False,
        "formal_collection_authorized": False,
    }


def require_run_confirmation(confirmed_protocol_sha256: str | None) -> None:
    if confirmed_protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            "Stage 6E 正式采集需要最终确认完整协议 SHA-256"
        )


def require_collection_confirmation(confirmed_sha256: str | None) -> None:
    if not isinstance(confirmed_sha256, str) or len(confirmed_sha256) != 64:
        raise PermissionError("正式评价需要确认完整 collection 报告 SHA-256")


def require_evaluation_confirmation(confirmed_sha256: str | None) -> None:
    if not isinstance(confirmed_sha256, str) or len(confirmed_sha256) != 64:
        raise PermissionError("独立复核需要确认完整 evaluation 报告 SHA-256")
