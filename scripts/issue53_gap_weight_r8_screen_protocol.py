"""Issue #53 B+C 问题一：R8 有界查询权重单种子配对筛查协议。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from scripts import issue53_stage6d_formal_protocol as stage6d


PROTOCOL_VERSION = "issue53-gap-weight-r8-paired-screen-v1"
PROTOCOL_DOC = Path(
    "docs/设计/"
    "Issue53_BC问题一R8有界查询权重单种子配对筛查结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "c15409d28e25690a70b9acf3ebeecd70bd4fdb86b0d2bdc914c0215799fbf118"
)

# 不进入冻结清单，避免自指。
FROZEN_PROTOCOL_SHA256 = (
    "50e49ab9786db1dd74476363e0489eeedf76bbdc44a1aa6940a1c7f80e34147e"
)

DEVELOPMENT_SEED = 9908
RESERVED_CONFIRMATION_SEEDS = tuple(range(358, 363))
ROUND_CAP = 6_000
CANDIDATE_BUDGET = 6_000
PATIENCE_TICKS = 6
CHECKPOINT_ROUNDS = (0, 100, 250, 500, 1_000, 2_000, 4_000, 6_000)
MAX_WORKERS = 2
MAX_WEIGHT_RATIO = 8.0
LEGACY_WEIGHTING = "legacy_relative"
BOUNDED_WEIGHTING = "bounded_relative"
POSITIVE_INFINITY_MANIFEST_SENTINEL = "positive_infinity"

OUTPUT_DIR = Path(
    "outputs/issue53_gap_weight_r8_paired_screen_seed9908_v1"
)

Dataset = Literal["test_300x10", "nltcs"]
Arm = Literal["gap_legacy_s8", "gap_bounded_r8_s8"]
DATASET_ORDER: tuple[Dataset, ...] = ("test_300x10", "nltcs")
ARM_ORDER: tuple[Arm, ...] = (
    "gap_legacy_s8",
    "gap_bounded_r8_s8",
)

# 复用已冻结的数据、生成查询和离线评价身份；
# 新协议另行绑定其来源文件。
DATASETS = stage6d.DATASETS
TEST_IDENTITY_ARTIFACT = stage6d.TEST_IDENTITY_ARTIFACT
TEST_IDENTITY_ARTIFACT_SHA256 = stage6d.TEST_IDENTITY_ARTIFACT_SHA256
TEST_GROUP_ORDER = stage6d.TEST_GROUP_ORDER
TEST_GROUP_COUNTS = stage6d.TEST_GROUP_COUNTS
TEST_GROUP_IDENTITIES = stage6d.TEST_GROUP_IDENTITIES
NLTCS_GROUP_COUNTS = stage6d.NLTCS_GROUP_COUNTS
NLTCS_GROUP_IDENTITIES = stage6d.NLTCS_GROUP_IDENTITIES

TARGET_COUNT_BINS: dict[str, tuple[dict[str, Any], ...]] = {
    "test_300x10": (
        {
            "name": "target_0_8",
            "lower_inclusive": 0,
            "upper_inclusive": 8,
            "query_count": 14,
            "membership_sha256": (
                "236c262d3b1d06c6e8c338d271dc58d71c0a828f0f93db5906b99a3b062f294f"
            ),
        },
        {
            "name": "target_9_25",
            "lower_inclusive": 9,
            "upper_inclusive": 25,
            "query_count": 16,
            "membership_sha256": (
                "49a6378e43aa14adc641007ea965eb69ab6ad4db255d6f30e7c3c4eed1a14d6f"
            ),
        },
        {
            "name": "target_26_50",
            "lower_inclusive": 26,
            "upper_inclusive": 50,
            "query_count": 11,
            "membership_sha256": (
                "c5984bc8445b59135f8049f54bd43f1d344c62ddfea34db1f3511f99c598fb22"
            ),
        },
        {
            "name": "target_51_300",
            "lower_inclusive": 51,
            "upper_inclusive": 300,
            "query_count": 9,
            "membership_sha256": (
                "8cef23ac55c3e9c0dcd9bfd1800d627a716124933accb6852f32a79ebe49a216"
            ),
        },
    ),
    "nltcs": (
        {
            "name": "target_0_50",
            "lower_inclusive": 0,
            "upper_inclusive": 50,
            "query_count": 34,
            "membership_sha256": (
                "6e360dfce7137966ef36aaef4ce2fb9d231a8245be54f8e266a904a3c53bc099"
            ),
        },
        {
            "name": "target_51_500",
            "lower_inclusive": 51,
            "upper_inclusive": 500,
            "query_count": 368,
            "membership_sha256": (
                "ce407fbcc6a5e61186b0bf2691bfc0e47f1a06ad425e8a1ed502fa5f43e5a791"
            ),
        },
        {
            "name": "target_501_2000",
            "lower_inclusive": 501,
            "upper_inclusive": 2_000,
            "query_count": 210,
            "membership_sha256": (
                "0e49aeed14ee47d94c670b0adbbbb402e07a242705da8950af14e2011c5f3745"
            ),
        },
        {
            "name": "target_2001_16181",
            "lower_inclusive": 2_001,
            "upper_inclusive": 16_181,
            "query_count": 389,
            "membership_sha256": (
                "49073c6c762cf25c15a8a460909f9ea47bcb83c215f2e00ef5f04f92819b1433"
            ),
        },
    ),
}

NLTCS_COMMON_BIN = "target_2001_16181"
NLTCS_RARE_BIN = "target_0_50"
NLTCS_RARE_RATIO_MAX = 1.25
TEST_OVERALL_RATIO_MAX = 1.05
ONE_WAY_RATIO_MAX = 1.05

EXPECTED_SOFTWARE = {
    "python_major_minor": "3.11",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "torch": "2.13.0+cu130",
    "cuda_runtime": "13.0",
}

EXPECTED_GPU = {
    "physical_index": 1,
    "cuda_visible_devices": "1",
    "uuid": "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02",
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mib": 24_564,
    "process_device": "cuda:0",
    "visible_device_count": 1,
}

IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "57ba2ade7c30ebfd34083c7a8686c4b01d68d07ab89ca885ba1b423dd1a86c95"
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
            "fff58706188690ea2c2ebed9ded0ee0ff9502749611c10ed715e46e0ad95c4ff"
        ),
    },
    "gap_triton_kernel": {
        "path": Path("src/table_diffevo/_gap_l1_triton.py"),
        "sha256": (
            "418103e86b70d835bb5845fda29ae72d226e0fcc9007f0d3b02329ae5554b439"
        ),
    },
    "inner_early_stopping": {
        "path": Path("src/table_diffevo/inner_early_stopping.py"),
        "sha256": (
            "a1dc68d9077eb98cf587281a63149e41ad22f2a7de7b3e76420cd3b27ecda211"
        ),
    },
    "stationarity_trace": {
        "path": Path("src/table_diffevo/stationarity.py"),
        "sha256": (
            "a280c18e630beb8f5342fa17a743b3847426e821cf66b2ca7a4f69a8cb0b2152"
        ),
    },
    "dataset_identity_protocol": {
        "path": Path("scripts/issue53_stage6d_formal_protocol.py"),
        "sha256": (
            "33bbf388a130a332970bc851137d51edcea94434404984498f58b9f91b186d2e"
        ),
    },
    "result_blind_query_identity": {
        "path": Path("scripts/freeze_issue53_test_query_workload_ab.py"),
        "sha256": (
            "cfaf56569999438798d2673eee3d282814b714b08b7d192c046a443bdfd56410"
        ),
    },
    "offline_evaluation_helpers": {
        "path": Path("scripts/evaluate_issue53_fixed_alpha_calibration.py"),
        "sha256": (
            "df41d09ec23e9272af762ae43c4379dda1190f1d5fae2a0ca3700569309afc3e"
        ),
    },
}


@dataclass(frozen=True)
class WeightingScreenTask:
    dataset: Dataset
    arm: Arm
    seed: int
    rounds: int

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.arm}__{self.dataset}"


@dataclass(frozen=True)
class WeightingScreenPlan:
    seed: int
    tasks: tuple[WeightingScreenTask, ...]
    protocol_frozen: bool = True
    runner_wired: bool = False
    generation_started: bool = False


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
    """返回除数据、种子和权重臂以外的完整共同生成参数。"""

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
        "record_stationarity_trace": True,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def arm_kernel_params(arm: str) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise ValueError(f"未知 R8 筛查方法：{arm!r}")
    weighting = (
        LEGACY_WEIGHTING if arm == "gap_legacy_s8" else BOUNDED_WEIGHTING
    )
    return {
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "gap_l1_sweeps": 8,
        "gap_l1_weighting": weighting,
        "gap_l1_max_weight_ratio": (
            None if weighting == LEGACY_WEIGHTING else MAX_WEIGHT_RATIO
        ),
    }


def task_generator_params(dataset: str, arm: str) -> dict[str, Any]:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"未知 R8 筛查数据集：{dataset!r}")
    params = common_generator_params()
    params.update(arm_kernel_params(arm))
    params.update(
        {
            "seed": DEVELOPMENT_SEED,
            "n_records": int(DATASETS[dataset]["n_records"]),
            "factorized_gibbs_max_order": int(
                DATASETS[dataset]["max_factor_order"]
            ),
        }
    )
    return params


def generator_params_manifest(dataset: str, arm: str) -> dict[str, Any]:
    params = task_generator_params(dataset, arm)
    nonfinite = [
        key
        for key, value in params.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if nonfinite != ["tol"] or params["tol"] != float("inf"):
        raise RuntimeError("R8 筛查生成参数只允许 tol=+inf")
    manifest = _jsonable(params)
    if manifest["tol"] != POSITIVE_INFINITY_MANIFEST_SENTINEL:
        raise RuntimeError("R8 筛查无门控 tol 清单编码漂移")
    _strict_json_bytes(manifest)
    return manifest


def task_plan() -> WeightingScreenPlan:
    tasks = tuple(
        WeightingScreenTask(
            dataset=dataset,
            arm=arm,
            seed=DEVELOPMENT_SEED,
            rounds=ROUND_CAP,
        )
        for dataset in DATASET_ORDER
        for arm in ARM_ORDER
    )
    if len(tasks) != 4 or len({task.task_id for task in tasks}) != 4:
        raise RuntimeError("R8 筛查任务矩阵不完整")
    return WeightingScreenPlan(seed=DEVELOPMENT_SEED, tasks=tasks)


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    return {
        task.task_id: generator_params_manifest(task.dataset, task.arm)
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def _validated_ratio(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or math.isnan(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} 必须是非负数或正无穷")
    return float(value)


def classify_screen(
    *,
    execution_valid: bool,
    nltcs_measured_l1_ratio: Any,
    nltcs_common_bin_ratio: Any,
    nltcs_rare_bin_ratio: Any,
    test_measured_l1_ratio: Any,
    one_way_ratio_by_dataset: Mapping[str, Any],
) -> str:
    """按结果前冻结的顺序给出单种子开发筛查分类。"""

    if not isinstance(execution_valid, (bool, np.bool_)):
        raise ValueError("execution_valid 必须是布尔值")
    if not bool(execution_valid):
        return "execution_invalid"
    if not isinstance(one_way_ratio_by_dataset, Mapping):
        raise ValueError("one_way_ratio_by_dataset 必须是映射")
    if set(one_way_ratio_by_dataset) != set(DATASET_ORDER):
        raise ValueError("one_way_ratio_by_dataset 必须恰好覆盖两个数据集")
    ratios = {
        "nltcs_measured_l1_ratio": _validated_ratio(
            nltcs_measured_l1_ratio, "nltcs_measured_l1_ratio"
        ),
        "nltcs_common_bin_ratio": _validated_ratio(
            nltcs_common_bin_ratio, "nltcs_common_bin_ratio"
        ),
        "nltcs_rare_bin_ratio": _validated_ratio(
            nltcs_rare_bin_ratio, "nltcs_rare_bin_ratio"
        ),
        "test_measured_l1_ratio": _validated_ratio(
            test_measured_l1_ratio, "test_measured_l1_ratio"
        ),
    }
    one_way = {
        dataset: _validated_ratio(
            one_way_ratio_by_dataset[dataset],
            f"one_way_ratio_by_dataset[{dataset!r}]",
        )
        for dataset in DATASET_ORDER
    }
    if (
        ratios["nltcs_measured_l1_ratio"] >= 1.0
        or ratios["nltcs_common_bin_ratio"] >= 1.0
    ):
        return "bounded_weighting_mechanism_not_supported"
    if ratios["nltcs_rare_bin_ratio"] > NLTCS_RARE_RATIO_MAX:
        return "common_gain_with_rare_query_regression"
    if (
        ratios["test_measured_l1_ratio"] > TEST_OVERALL_RATIO_MAX
        or any(value > ONE_WAY_RATIO_MAX for value in one_way.values())
    ):
        return "measured_gain_with_safety_risk"
    return "advance_to_five_seed_confirmation"


def frozen_protocol_manifest() -> dict[str, Any]:
    plan = task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "gap_weight_problem_1_r8_single_seed_paired_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "scientific_question": (
            "does_bounded_r8_restore_common_query_error_without_"
            "unacceptable_rare_or_one_way_harm"
        ),
        "claim_boundary": {
            "development_screen_only": True,
            "formal_superiority_claim_allowed": False,
            "public_default_change_allowed": False,
            "comparison_to_independent_or_factor_b_in_scope": False,
        },
        "task_matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(ARM_ORDER),
            "seed": DEVELOPMENT_SEED,
            "seed_used_by_prior_weight_screen": False,
            "reserved_confirmation_seeds": list(
                RESERVED_CONFIRMATION_SEEDS
            ),
            "round_cap": ROUND_CAP,
            "candidate_budget": CANDIDATE_BUDGET,
            "trajectory_count": len(plan.tasks),
            "task_order": "seed_then_dataset_then_arm",
            "task_ids": [task.task_id for task in plan.tasks],
            "all_tasks_declared_before_any_result": True,
            "partial_matrix_allowed": False,
            "result_dependent_task_removal_allowed": False,
        },
        "datasets": {
            name: {
                "schema": str(DATASETS[name]["schema"]),
                "queries": str(DATASETS[name]["queries"]),
                "marginals": str(DATASETS[name]["marginals"]),
                "reference_evaluation_only": str(
                    DATASETS[name]["reference"]
                ),
                "n_records": DATASETS[name]["n_records"],
                "query_count": DATASETS[name]["query_count"],
                "order_counts": DATASETS[name]["order_counts"],
                "max_factor_order": DATASETS[name]["max_factor_order"],
                "query_identity_sha256": DATASETS[name][
                    "query_identity_sha256"
                ],
                "target_vector_sha256": DATASETS[name][
                    "target_vector_sha256"
                ],
                "trace_query_identity_sha256": DATASETS[name][
                    "trace_query_identity_sha256"
                ],
                "trace_target_vector_sha256": DATASETS[name][
                    "trace_target_vector_sha256"
                ],
                "input_sha256": dict(DATASETS[name]["input_sha256"]),
            }
            for name in DATASET_ORDER
        },
        "generator_common": _jsonable(common_generator_params()),
        "arm_kernel_params": {
            arm: _jsonable(arm_kernel_params(arm)) for arm in ARM_ORDER
        },
        "generator_params_manifest_sha256": (
            generator_params_manifest_sha256()
        ),
        "single_variable_contract": {
            "only_explicit_parameter_differences": [
                "gap_l1_weighting",
                "gap_l1_max_weight_ratio",
            ],
            "derived_reference_scale_may_differ": True,
            "derived_reference_scale_reason": (
                "each_weight_geometry_uses_its_own_first_nonzero_isolated_score_rms"
            ),
            "same_seed_and_initialization": True,
            "same_rng_domain_initialization": True,
            "post_divergence_state_dependent_rng_consumption_may_differ": True,
            "factorized_gibbs_sweeps": 0,
            "gap_l1_sweeps": 8,
            "gap_microsteps_identity": "8*K",
        },
        "bounded_weighting": {
            "ratio_cap": MAX_WEIGHT_RATIO,
            "smoothing_formula": "max(8,N/(R-1))",
            "denominator_formula": "clip(target,0,N)+smoothing",
            "division_dtype": "float64_no_integer_rounding",
            "smoothing_by_dataset": {
                name: max(
                    8.0,
                    float(DATASETS[name]["n_records"])
                    / (MAX_WEIGHT_RATIO - 1.0),
                )
                for name in DATASET_ORDER
            },
        },
        "stopping_contract": {
            "patience_ticks": PATIENCE_TICKS,
            "natural_work_formula": "cumulative_participating_rows/n_records",
            "priority": [
                "fit_target_reached",
                "early_stopped",
                "resource_cap_reached",
            ],
            "resource_round_cap": ROUND_CAP,
            "resource_candidate_cap": CANDIDATE_BUDGET,
            "terminal_current_output": True,
            "historical_best_output_allowed": False,
            "checkpoint_rounds_diagnostic_only": list(CHECKPOINT_ROUNDS),
            "checkpoint_selection_allowed": False,
        },
        "no_gate_contract": {
            "tol_positive_infinity": True,
            "max_retries": 0,
            "one_candidate_per_round": True,
            "accept_or_reject": False,
            "rollback": False,
            "winner_selection": False,
        },
        "target_count_bins": _jsonable(TARGET_COUNT_BINS),
        "offline_evaluation": {
            "generation_may_read_reference_or_unseen_queries": False,
            "evaluation_after_complete_collection_only": True,
            "measured_normalized_l1_formula": (
                "sum_abs_error/(query_count*n_records)"
            ),
            "target_bin_metric": "mean_absolute_count_error_per_query",
            "query_order_metric": "mean_absolute_count_error_per_query",
            "report_both_proxy_objectives": True,
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
            "ratio_direction": "bounded_r8_over_legacy_lower_is_better",
            "zero_baseline_rule": (
                "both_zero_ratio_one_else_positive_infinity"
            ),
            "timing_role": "diagnostic_only",
        },
        "screen_decision": {
            "precedence": [
                "execution_invalid",
                "bounded_weighting_mechanism_not_supported",
                "common_gain_with_rare_query_regression",
                "measured_gain_with_safety_risk",
                "advance_to_five_seed_confirmation",
            ],
            "nltcs_measured_l1_ratio_strictly_less_than": 1.0,
            "nltcs_common_bin": NLTCS_COMMON_BIN,
            "nltcs_common_bin_ratio_strictly_less_than": 1.0,
            "nltcs_rare_bin": NLTCS_RARE_BIN,
            "nltcs_rare_bin_ratio_max": NLTCS_RARE_RATIO_MAX,
            "test_measured_l1_ratio_max": TEST_OVERALL_RATIO_MAX,
            "one_way_ratio_max_each_dataset": ONE_WAY_RATIO_MAX,
            "post_result_threshold_retuning_allowed": False,
            "automatic_other_ratio_sweep_allowed": False,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "execution": {
            "output_dir": str(OUTPUT_DIR),
            "max_workers": MAX_WORKERS,
            "expected_software": dict(EXPECTED_SOFTWARE),
            "expected_gpu": dict(EXPECTED_GPU),
            "multiprocessing_start_method": "spawn",
            "exclusive_idle_gpu_required": True,
            "atomic_output_creation_required": True,
            "existing_output_overwrite_allowed": False,
            "runner_wired_at_protocol_freeze": False,
        },
        "information_flow": {
            "collection_emits_partial_quality": False,
            "collection_reads_raw_reference": False,
            "evaluation_requires_complete_collection_sha_confirmation": True,
            "independent_audit_reruns_generation": False,
            "partial_dataset_decision_allowed": False,
        },
        "authorization_boundary": {
            "protocol_frozen": True,
            "preparation_may_run_generator": False,
            "runner_wiring_authorized_at_freeze": False,
            "screen_generation_authorized_at_freeze": False,
            "matching_protocol_hash_alone_authorizes_generation": False,
            "final_user_confirmation_required_after_runner_review": True,
            "passing_screen_authorizes_confirmation_run": False,
            "pr69_change_allowed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("R8 单种子筛查协议文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"R8 单种子筛查实现源码漂移：{name}")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "R8 单种子筛查协议清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_protocol_identity(repository_root)
    plan = task_plan()
    return {
        "mode": "plan_only_no_dataset_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "task_ids": [task.task_id for task in plan.tasks],
        "trajectory_count": len(plan.tasks),
        "seed": DEVELOPMENT_SEED,
        "reserved_confirmation_seeds": list(RESERVED_CONFIRMATION_SEEDS),
        "round_cap": ROUND_CAP,
        "candidate_budget": CANDIDATE_BUDGET,
        "patience_ticks": PATIENCE_TICKS,
        "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
        "max_workers": MAX_WORKERS,
        "output_dir": str(OUTPUT_DIR),
        "generator_params_manifest_sha256": (
            generator_params_manifest_sha256()
        ),
        "runner_wired": False,
        "generation_started": False,
        "screen_generation_authorized": False,
    }


def require_run_confirmation(_confirmed_protocol_sha256: str | None) -> None:
    raise PermissionError(
        "R8 单种子筛查执行器尚未接线；"
        "协议哈希不能代替真实生成的单独授权"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan",))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "plan":
        print(
            json.dumps(
                build_plan(Path(__file__).resolve().parents[1]),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
