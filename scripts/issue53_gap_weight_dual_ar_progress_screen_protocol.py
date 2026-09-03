"""Issue #53 B+C 问题一：A/R 相对初始进度单种子开发筛查协议。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from scripts import issue53_gap_weight_dual_ar_screen_protocol as prior


PROTOCOL_VERSION = "issue53-gap-weight-dual-ar-progress-max-screen-v1"
PROTOCOL_DOC = Path(
    "docs/设计/"
    "Issue53_BC问题一AR相对初始进度单种子开发筛查结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "8b5ffb1d4745aeee62a1ca6359cb908d4f059d711edc203ee4a6b59468805004"
)
FROZEN_PROTOCOL_SHA256 = (
    "0c523145426e559c4536f66a1b97c0947e0819754874c232b7f0f036056029ef"
)

DESIGN_DOC = Path(
    "docs/设计/Issue53_BC问题一AR相对初始进度max设计冻结.md"
)
DESIGN_DOC_SHA256 = (
    "d993bdafd740585034b9985268ec8a0da919e9b2bad2a9f74ec5ab37faa90319"
)
OFFLINE_AUDIT_DOC = Path(
    "docs/设计/Issue53_BC问题一AR相对初始进度历史轨迹离线审计.md"
)
OFFLINE_AUDIT_DOC_SHA256 = (
    "074301610da56e209dbed1cef2a4ff8517753c87a12808b4d00c17e5e782a0ec"
)

DEVELOPMENT_SEED = prior.DEVELOPMENT_SEED
ROUND_CAP = prior.ROUND_CAP
CANDIDATE_BUDGET = prior.CANDIDATE_BUDGET
PATIENCE_TICKS = prior.PATIENCE_TICKS
CHECKPOINT_ROUNDS = prior.CHECKPOINT_ROUNDS
MAX_WORKERS = prior.MAX_WORKERS

DUAL_WEIGHTING = "dual_abs_relative_progress_max"
CANDIDATE_ARM = "gap_dual_abs_relative_progress_max_s8"
DATASET_ORDER = prior.DATASET_ORDER
ARM_ORDER = (CANDIDATE_ARM,)
DATASETS = prior.DATASETS
TARGET_COUNT_BINS = prior.TARGET_COUNT_BINS
NLTCS_COMMON_BIN = prior.NLTCS_COMMON_BIN
NLTCS_RARE_BIN = prior.NLTCS_RARE_BIN
EXPECTED_WEIGHT_AUDIT = prior.EXPECTED_WEIGHT_AUDIT

EXPECTED_INITIAL_CHANNEL_REFERENCES = {
    "test_300x10": {
        "absolute_initial": 0.023733333333333332,
        "relative_initial": 0.011553523550340313,
    },
    "nltcs": {
        "absolute_initial": 0.063173770793819,
        "relative_initial": 0.03520355244885822,
    },
}

# 单种子开发筛查只决定是否值得进入新种子确认，
# 不支持正式优越性结论。
MEASURED_L1_RATIO_MAX = 1.0
NLTCS_COMMON_RATIO_MAX = 1.0
NLTCS_RARE_RATIO_MAX = 1.05
TEST_MEASURED_L1_RATIO_MAX = 1.05
ONE_WAY_RATIO_MAX = 1.05

OUTPUT_DIR = Path(
    "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v1"
)

PRIOR_CANDIDATE_ARTIFACTS = {
    "collection": {
        "path": prior.OUTPUT_DIR / "collection_report.json",
        "sha256": (
            "3b53fcac678924205a4bd1038ec15f404b5245b09894d2b065d28794e615a790"
        ),
    },
    "evaluation": {
        "path": prior.OUTPUT_DIR / "evaluation_report.json",
        "sha256": (
            "df9c64430be28f45c0a354dff0861ba501b4c3e429e66a85956119961f5b5c56"
        ),
    },
    "metrics_csv": {
        "path": prior.OUTPUT_DIR / "screen_metrics.csv",
        "sha256": (
            "3517bb58884f9aab2533a807a7e3996745de0e5b03154871489a76fa9fb4680e"
        ),
    },
}
BASELINE_ARTIFACTS = {
    **prior.BASELINE_ARTIFACTS,
    "prior_dual_ar_max": PRIOR_CANDIDATE_ARTIFACTS,
}

IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "7fffd0330bf7c00dfb352112b4d15675d06e9942ccc80a8895947197fdf600ee"
        ),
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "31237867ee9d6b7aeac810e94e4912c84eda29a79a443e1b863e2d04c269f590"
        ),
    },
    "offline_auditor": {
        "path": Path(
            "scripts/audit_issue53_gap_weight_dual_ar_progress_offline.py"
        ),
        "sha256": (
            "c601abbcb7656539477ddc9ee8c2d77b69f4d083f13def301c5864711d98f8e5"
        ),
    },
}

Dataset = Literal["test_300x10", "nltcs"]


@dataclass(frozen=True)
class DualArProgressScreenTask:
    dataset: Dataset
    arm: str
    seed: int
    rounds: int

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.arm}__{self.dataset}"


@dataclass(frozen=True)
class DualArProgressScreenPlan:
    seed: int
    tasks: tuple[DualArProgressScreenTask, ...]
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


def strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


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
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def task_plan() -> DualArProgressScreenPlan:
    tasks = tuple(
        DualArProgressScreenTask(
            dataset=dataset,
            arm=CANDIDATE_ARM,
            seed=DEVELOPMENT_SEED,
            rounds=ROUND_CAP,
        )
        for dataset in DATASET_ORDER
    )
    if len(tasks) != 2 or len({task.task_id for task in tasks}) != 2:
        raise RuntimeError("A/R 相对初始进度筛查任务矩阵不完整")
    return DualArProgressScreenPlan(seed=DEVELOPMENT_SEED, tasks=tasks)


def common_generator_params() -> dict[str, Any]:
    return prior.common_generator_params()


def arm_kernel_params(arm: str) -> dict[str, Any]:
    if arm != CANDIDATE_ARM:
        raise ValueError(f"未知 A/R 相对初始进度筛查方法：{arm!r}")
    return {
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "gap_l1_sweeps": 8,
        "gap_l1_weighting": DUAL_WEIGHTING,
        "gap_l1_max_weight_ratio": None,
    }


def task_generator_params(dataset: str, arm: str) -> dict[str, Any]:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"未知 A/R 相对初始进度筛查数据集：{dataset!r}")
    params = common_generator_params()
    params.update(arm_kernel_params(arm))
    params.update({
        "seed": DEVELOPMENT_SEED,
        "n_records": int(DATASETS[dataset]["n_records"]),
        "factorized_gibbs_max_order": int(
            DATASETS[dataset]["max_factor_order"]
        ),
    })
    return params


def generator_params_manifest(dataset: str, arm: str) -> dict[str, Any]:
    params = task_generator_params(dataset, arm)
    nonfinite = [
        key
        for key, value in params.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if nonfinite != ["tol"] or params["tol"] != float("inf"):
        raise RuntimeError("A/R 相对初始进度生成参数只允许 tol=+inf")
    manifest = _jsonable(params)
    _strict_json_bytes(manifest)
    return manifest


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    return {
        task.task_id: generator_params_manifest(task.dataset, task.arm)
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def target_weight_audit(repository_root: str | Path) -> dict[str, Any]:
    return prior.target_weight_audit(repository_root)


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
    relative_channel_activated_by_dataset: Mapping[str, Any],
    measured_l1_ratio_by_dataset: Mapping[str, Any],
    nltcs_common_bin_ratio: Any,
    nltcs_rare_bin_ratio: Any,
    one_way_ratio_by_dataset: Mapping[str, Any],
) -> str:
    if not isinstance(execution_valid, (bool, np.bool_)):
        raise ValueError("execution_valid 必须是布尔值")
    if not bool(execution_valid):
        return "execution_invalid"
    expected = set(DATASET_ORDER)
    for value, name in (
        (relative_channel_activated_by_dataset, "relative_channel_activated"),
        (measured_l1_ratio_by_dataset, "measured_l1_ratio"),
        (one_way_ratio_by_dataset, "one_way_ratio"),
    ):
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"{name} 必须恰好覆盖两个数据集")
    if any(
        not isinstance(relative_channel_activated_by_dataset[name], (bool, np.bool_))
        for name in DATASET_ORDER
    ):
        raise ValueError("relative_channel_activated 必须是布尔映射")
    if not all(
        bool(relative_channel_activated_by_dataset[name])
        for name in DATASET_ORDER
    ):
        return "relative_channel_not_activated"
    measured = {
        name: _validated_ratio(
            measured_l1_ratio_by_dataset[name], f"measured_l1_ratio[{name}]"
        )
        for name in DATASET_ORDER
    }
    one_way = {
        name: _validated_ratio(
            one_way_ratio_by_dataset[name], f"one_way_ratio[{name}]"
        )
        for name in DATASET_ORDER
    }
    common = _validated_ratio(nltcs_common_bin_ratio, "nltcs_common_bin_ratio")
    rare = _validated_ratio(nltcs_rare_bin_ratio, "nltcs_rare_bin_ratio")
    if (
        measured["nltcs"] >= MEASURED_L1_RATIO_MAX
        or common >= NLTCS_COMMON_RATIO_MAX
    ):
        return "common_query_improvement_not_recovered"
    if rare > NLTCS_RARE_RATIO_MAX:
        return "rare_query_safety_risk"
    if (
        measured["test_300x10"] > TEST_MEASURED_L1_RATIO_MAX
        or any(value > ONE_WAY_RATIO_MAX for value in one_way.values())
    ):
        return "measured_or_one_way_safety_risk"
    return "advance_to_fresh_seed_confirmation"


def _flatten_baselines() -> dict[str, dict[str, Any]]:
    return {
        f"{family}_{name}": value
        for family, artifacts in BASELINE_ARTIFACTS.items()
        for name, value in artifacts.items()
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    plan = task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "gap_weight_problem_1_dual_ar_progress_single_seed_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "design_document": {
            "path": str(DESIGN_DOC),
            "sha256": DESIGN_DOC_SHA256,
        },
        "offline_audit_document": {
            "path": str(OFFLINE_AUDIT_DOC),
            "sha256": OFFLINE_AUDIT_DOC_SHA256,
        },
        "claim_boundary": {
            "development_screen_only": True,
            "formal_superiority_claim_allowed": False,
            "public_default_change_allowed": False,
        },
        "formula": {
            "weighting": DUAL_WEIGHTING,
            "absolute": "sum_all(abs(target-count))/(N*J)",
            "relative": (
                "sum_positive(abs(target-count)/target)"
                "/(N*sum_positive(1/target))"
            ),
            "aggregation": "max(A/A_init,R/R_init)",
            "reference_source": "initial_current_before_round_1",
            "reference_recomputed_by_round": False,
            "zero_target_policy": "absolute_channel_only",
            "empty_positive_target_policy": "A/A_init_only",
            "zero_positive_reference_policy": "fail_closed",
            "floor_applied": False,
            "max_weight_ratio": None,
            "smoothing_count": None,
            "tunable_parameter_count": 0,
        },
        "expected_initial_channel_references": (
            EXPECTED_INITIAL_CHANNEL_REFERENCES
        ),
        "task_matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(ARM_ORDER),
            "seed": DEVELOPMENT_SEED,
            "round_cap": ROUND_CAP,
            "candidate_budget": CANDIDATE_BUDGET,
            "trajectory_count": len(plan.tasks),
            "task_ids": [task.task_id for task in plan.tasks],
            "prior_arms_rerun_allowed": False,
        },
        "baseline_artifacts": _jsonable(BASELINE_ARTIFACTS),
        "datasets": {
            name: {
                "schema": str(DATASETS[name]["schema"]),
                "queries": str(DATASETS[name]["queries"]),
                "marginals": str(DATASETS[name]["marginals"]),
                "reference_evaluation_only": str(DATASETS[name]["reference"]),
                "n_records": DATASETS[name]["n_records"],
                "query_count": DATASETS[name]["query_count"],
                "query_identity_sha256": DATASETS[name]["query_identity_sha256"],
                "target_vector_sha256": DATASETS[name]["target_vector_sha256"],
                "generation_input_sha256": {
                    key: DATASETS[name]["input_sha256"][key]
                    for key in ("schema", "queries", "marginals")
                },
                "expected_weight_audit": EXPECTED_WEIGHT_AUDIT[name],
            }
            for name in DATASET_ORDER
        },
        "target_count_bins": _jsonable(TARGET_COUNT_BINS),
        "generator_params_by_task": generator_params_manifest_matrix(),
        "generator_params_manifest_sha256": (
            generator_params_manifest_sha256()
        ),
        "implementation_sources": _jsonable(IMPLEMENTATION_SOURCES),
        "invariants": {
            "only_c_channel_aggregation_changes": True,
            "b_residual_channel_unchanged": True,
            "zero_target_policy_unchanged": True,
            "gap_l1_sweeps": 8,
            "microsteps": "8*K",
            "one_unconditional_candidate_per_round": True,
            "patience_ticks": PATIENCE_TICKS,
            "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
            "terminal_current_output": True,
            "historical_weighting_compatibility_required": True,
        },
        "screen_gates": {
            "relative_channel_activated_each_dataset": True,
            "nltcs_measured_l1_ratio_vs_prior_strictly_less_than": 1.0,
            "nltcs_common_bin_ratio_vs_prior_strictly_less_than": 1.0,
            "nltcs_rare_bin_ratio_vs_prior_max": NLTCS_RARE_RATIO_MAX,
            "test_measured_l1_ratio_vs_prior_max": TEST_MEASURED_L1_RATIO_MAX,
            "one_way_ratio_vs_prior_each_dataset_max": ONE_WAY_RATIO_MAX,
            "classification_order": [
                "execution_invalid",
                "relative_channel_not_activated",
                "common_query_improvement_not_recovered",
                "rare_query_safety_risk",
                "measured_or_one_way_safety_risk",
                "advance_to_fresh_seed_confirmation",
            ],
        },
        "information_flow": {
            "generation_reads_reference_or_unseen_queries": False,
            "complete_candidate_collection_before_quality_evaluation": True,
            "automatic_generation_allowed": False,
            "automatic_evaluation_allowed": False,
            "automatic_parameter_search_allowed": False,
        },
    }


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    for path, expected in (
        (PROTOCOL_DOC, PROTOCOL_DOC_SHA256),
        (DESIGN_DOC, DESIGN_DOC_SHA256),
        (OFFLINE_AUDIT_DOC, OFFLINE_AUDIT_DOC_SHA256),
    ):
        if file_sha256(root / path) != expected:
            raise RuntimeError(f"冻结文件 SHA-256 漂移：{path}")
    for spec in IMPLEMENTATION_SOURCES.values():
        if file_sha256(root / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"实现源码 SHA-256 漂移：{spec['path']}")
    for artifact in _flatten_baselines().values():
        if file_sha256(root / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"复用基线 SHA-256 漂移：{artifact['path']}")
    observed = canonical_sha256(frozen_protocol_manifest())
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "A/R 相对初始进度冻结协议清单 SHA-256 漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )


def require_run_confirmation(value: str) -> None:
    if value != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            "必须显式确认当前 A/R 相对初始进度冻结协议 SHA-256"
        )


def build_plan() -> dict[str, Any]:
    plan = task_plan()
    return {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol": frozen_protocol_manifest(),
        "seed": plan.seed,
        "tasks": [task.__dict__ | {"task_id": task.task_id} for task in plan.tasks],
        "runner_wired": plan.runner_wired,
        "generation_started": plan.generation_started,
        "raw_reference_data_accessed": False,
        "candidate_results_accessed": False,
    }
