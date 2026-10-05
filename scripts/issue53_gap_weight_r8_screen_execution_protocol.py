"""Issue #53 R8 单种子筛查的执行接线差量协议。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_screen_protocol as scientific


PROTOCOL_VERSION = "issue53-gap-weight-r8-paired-screen-execution-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_BC问题一R8筛查执行接线差量协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "447f42db914ad1e170f30a28c3e5efa93862d9f2b4831bf91f1fd2d26673d729"
)

# 不进入清单，避免自指；实现全部完成后冻结。
FROZEN_PROTOCOL_SHA256 = (
    "8c662f6d0568525b62a6002f3be4913a3334b432bde91faaa70d5b58e6606a4a"
)
SCIENTIFIC_PROTOCOL_SHA256 = scientific.FROZEN_PROTOCOL_SHA256

DEVELOPMENT_SEED = scientific.DEVELOPMENT_SEED
FORMAL_SEEDS = (DEVELOPMENT_SEED,)
RESERVED_CONFIRMATION_SEEDS = scientific.RESERVED_CONFIRMATION_SEEDS
ROUND_CAP = scientific.ROUND_CAP
ROUNDS = ROUND_CAP
CANDIDATE_BUDGET = scientific.CANDIDATE_BUDGET
PATIENCE_TICKS = scientific.PATIENCE_TICKS
CHECKPOINT_ROUNDS = scientific.CHECKPOINT_ROUNDS
MAX_WORKERS = scientific.MAX_WORKERS
MAX_WEIGHT_RATIO = scientific.MAX_WEIGHT_RATIO
LEGACY_WEIGHTING = scientific.LEGACY_WEIGHTING
BOUNDED_WEIGHTING = scientific.BOUNDED_WEIGHTING
NORMAL_TERMINATION_REASONS = (
    "fit_target_reached",
    "early_stopped",
    "resource_cap_reached",
)

DATASET_ORDER = scientific.DATASET_ORDER
ARM_ORDER = scientific.ARM_ORDER
DATASETS = scientific.DATASETS
TARGET_COUNT_BINS = scientific.TARGET_COUNT_BINS
NLTCS_COMMON_BIN = scientific.NLTCS_COMMON_BIN
NLTCS_RARE_BIN = scientific.NLTCS_RARE_BIN
NLTCS_RARE_RATIO_MAX = scientific.NLTCS_RARE_RATIO_MAX
TEST_OVERALL_RATIO_MAX = scientific.TEST_OVERALL_RATIO_MAX
ONE_WAY_RATIO_MAX = scientific.ONE_WAY_RATIO_MAX
TEST_IDENTITY_ARTIFACT = scientific.TEST_IDENTITY_ARTIFACT
TEST_IDENTITY_ARTIFACT_SHA256 = scientific.TEST_IDENTITY_ARTIFACT_SHA256
TEST_GROUP_ORDER = scientific.TEST_GROUP_ORDER
TEST_GROUP_COUNTS = scientific.TEST_GROUP_COUNTS
TEST_GROUP_IDENTITIES = scientific.TEST_GROUP_IDENTITIES
NLTCS_GROUP_COUNTS = scientific.NLTCS_GROUP_COUNTS
NLTCS_GROUP_IDENTITIES = scientific.NLTCS_GROUP_IDENTITIES
EXPECTED_SOFTWARE = scientific.EXPECTED_SOFTWARE
EXPECTED_GPU = scientific.EXPECTED_GPU
POSITIVE_INFINITY_MANIFEST_SENTINEL = (
    scientific.POSITIVE_INFINITY_MANIFEST_SENTINEL
)

# 旧基础设施需要一个 ARM_GAP 名称；R8 collector 会覆盖其单臂判断，
# 禁止
# 用这个兼容常量决定哪一臂接受 gap 审计。
ARM_GAP = "gap_bounded_r8_s8"
BASELINE_ARMS = ("gap_legacy_s8",)

OUTPUT_DIR = scientific.OUTPUT_DIR
SHARD_OUTPUT_ROOT = Path(
    "outputs/issue53_gap_weight_r8_paired_screen_seed9908_v1_shards"
)
COLLECTION_REPORT = "collection_report.json"
SHARD_REPORT = "shard_report.json"
EVALUATION_REPORT = "evaluation_report.json"
L1_RESULTS_CSV = "screen_metrics.csv"
AUDIT_REPORT = "independent_audit.json"

LOCAL_SHARD = "local_rtx4090"
SHARD_ORDER = (LOCAL_SHARD,)
SHARD_BLOCKS = {
    LOCAL_SHARD: tuple(
        (DEVELOPMENT_SEED, dataset) for dataset in DATASET_ORDER
    )
}
EXECUTION_SHARDS = {
    LOCAL_SHARD: {
        "hostname": "linyao-system",
        "task_count": 4,
        "paired_block_count": 2,
        "max_workers": MAX_WORKERS,
        "expected_gpu": dict(EXPECTED_GPU),
    }
}

# 新接线源文件在完成后绑定字节身份。
# 科学实现仍由 scientific 协议绑定。
IMPLEMENTATION_SOURCES = {
    "scientific_protocol": {
        "path": Path("scripts/issue53_gap_weight_r8_screen_protocol.py"),
        "sha256": "c5f1d0cc35b267ef9d40162289c9284ee865aa6cab0fb646f840e17aad08b5e6",
    },
    "stage6d_collection_infrastructure": {
        "path": Path("scripts/run_issue53_stage6d_formal.py"),
        "sha256": "d020648c2ee9c3808e370ddab7fca74a26ddd412ea28c4643e3f9b280071f88b",
    },
    "stage6e_autostop_adapter": {
        "path": Path("scripts/run_issue53_stage6e_autostop.py"),
        "sha256": "7130223c2dbfae5723f6d445e92ae7154588b4aebd631121a7f79f325f4b51d1",
    },
    "offline_evaluation_primitives": {
        "path": Path("scripts/evaluate_issue53_fixed_alpha_calibration.py"),
        "sha256": "df41d09ec23e9272af762ae43c4379dda1190f1d5fae2a0ca3700569309afc3e",
    },
    "collector": {
        "path": Path("scripts/run_issue53_gap_weight_r8_screen.py"),
        "sha256": "3a3151a09043bbcaa0684a18b0abc6b41325c2e3d59e3328d18912fcce6bb756",
    },
    "evaluator": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen.py"),
        "sha256": "287779a7e66466d3ed2e04862d2aded7ce3e8f084277de72893600e71c6f7445",
    },
    "independent_auditor": {
        "path": Path("scripts/audit_issue53_gap_weight_r8_screen.py"),
        "sha256": "7a7f991fd9ef066771c58c36f1b22afa13f78ab27dd021cb8c13bb2ddbd4ad03",
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
    return scientific.file_sha256(path)


def common_generator_params() -> dict[str, Any]:
    return scientific.common_generator_params()


def arm_kernel_params(arm: str) -> dict[str, Any]:
    return scientific.arm_kernel_params(arm)


def task_generator_params(
    dataset: str, arm: str, seed: int
) -> dict[str, Any]:
    if seed != DEVELOPMENT_SEED:
        raise ValueError(f"R8 筛查种子不是 9908：{seed!r}")
    return scientific.task_generator_params(dataset, arm)


def generator_params_manifest(
    dataset: str, arm: str, seed: int
) -> dict[str, Any]:
    if seed != DEVELOPMENT_SEED:
        raise ValueError(f"R8 筛查种子不是 9908：{seed!r}")
    return scientific.generator_params_manifest(dataset, arm)


def task_plan():
    return scientific.task_plan()


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    return {
        task.task_id: generator_params_manifest(
            task.dataset, task.arm, task.seed
        )
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def tasks_for_shard(shard_id: str):
    if shard_id != LOCAL_SHARD:
        raise ValueError(f"未知 R8 筛查执行分片：{shard_id!r}")
    tasks = task_plan().tasks
    if len(tasks) != 4:
        raise RuntimeError("R8 筛查任务数量漂移")
    return tasks


def task_shard_id(task: Any) -> str:
    if task.task_id not in {item.task_id for item in task_plan().tasks}:
        raise RuntimeError(f"任务不在 R8 筛查冻结矩阵：{task.task_id}")
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


def smoothing_count(dataset: str) -> float:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"未知 R8 筛查数据集：{dataset!r}")
    return float(
        max(
            8.0,
            float(DATASETS[dataset]["n_records"])
            / (MAX_WEIGHT_RATIO - 1.0),
        )
    )


def classify_screen(**values: Any) -> str:
    return scientific.classify_screen(**values)


def frozen_protocol_manifest() -> dict[str, Any]:
    tasks = task_plan().tasks
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "gap_weight_problem_1_r8_execution_wiring_delta",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "scientific_protocol": {
            "version": scientific.PROTOCOL_VERSION,
            "sha256": SCIENTIFIC_PROTOCOL_SHA256,
            "source_path": "scripts/issue53_gap_weight_r8_screen_protocol.py",
            "scientific_fields_modified_by_delta": [],
        },
        "task_matrix_identity": {
            "task_ids": [task.task_id for task in tasks],
            "trajectory_count": 4,
            "paired_block_count": 2,
            "task_order": "seed_then_dataset_then_arm",
            "generator_params_manifest_sha256": (
                generator_params_manifest_sha256()
            ),
        },
        "collector": {
            "reused_infrastructure": [
                "stage6d_atomic_case_artifacts_and_resume",
                "stage6d_input_and_gpu_identity_audit",
                "stage6d_spawn_execution_and_verified_merge",
                "stage6e_p6_autostop_terminal_current_artifacts",
            ],
            "r8_specific_overrides": [
                "two_arm_pairing_without_shared_derived_reference_scale",
                "both_arms_gap_8k_transition_audit",
                "legacy_vs_bounded_weight_diagnostic_guard",
                "four_case_two_block_collection_report",
            ],
            "collection_reads_raw_reference": False,
            "partial_quality_emission_allowed": False,
        },
        "evaluation": {
            "requires_complete_collection_sha_confirmation": True,
            "terminal_and_checkpoint_metrics": [
                "measured_normalized_l1",
                "legacy_relative_gap_proxy",
                "bounded_r8_gap_proxy",
                "target_count_bins",
                "query_orders",
            ],
            "terminal_only_metrics": [
                "frozen_offline_query_groups",
                "schema_validity",
                "diversity",
                "reference_support",
            ],
            "checkpoint_selection_allowed": False,
            "post_result_gate_addition_allowed": False,
        },
        "independent_audit": {
            "requires_collection_and_evaluation_sha_confirmation": True,
            "imports_evaluator_arithmetic": False,
            "recomputes_generation": False,
            "recomputes_terminal_checkpoint_and_classification": True,
        },
        "execution": {
            "output_dir": str(OUTPUT_DIR),
            "shard_output_root": str(SHARD_OUTPUT_ROOT),
            "shard_order": list(SHARD_ORDER),
            "shard_assignment": shard_assignment_manifest(),
            "shard_assignment_sha256": shard_assignment_sha256(),
            "max_workers": MAX_WORKERS,
            "multiprocessing_start_method": "spawn",
            "exclusive_idle_gpu_required": True,
            "atomic_output_creation_required": True,
            "existing_output_overwrite_allowed": False,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "authorization_boundary": {
            "runner_wired": True,
            "generation_authorized_at_execution_freeze": False,
            "matching_hash_alone_authorizes_generation": False,
            "explicit_later_user_confirmation_required": True,
            "current_wiring_work_may_run_generator": False,
            "pr69_change_allowed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_scientific_inheritance() -> None:
    if scientific.FROZEN_PROTOCOL_SHA256 != SCIENTIFIC_PROTOCOL_SHA256:
        raise RuntimeError("R8 科学协议 SHA-256 继承漂移")
    if task_plan().tasks != scientific.task_plan().tasks:
        raise RuntimeError("R8 执行层任务矩阵改变了科学协议")
    if generator_params_manifest_sha256() != (
        scientific.generator_params_manifest_sha256()
    ):
        raise RuntimeError("R8 执行层生成参数改变了科学协议")
    if TARGET_COUNT_BINS != scientific.TARGET_COUNT_BINS:
        raise RuntimeError("R8 执行层目标分箱改变了科学协议")


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    scientific.assert_frozen_protocol_identity(root)
    assert_scientific_inheritance()
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("R8 筛查执行接线文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"R8 筛查执行接线源码漂移：{name}")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "R8 筛查执行接线清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_protocol_identity(repository_root)
    tasks = task_plan().tasks
    return {
        "mode": "execution_plan_only_no_generator_or_reference_access",
        "protocol_sha256": protocol_sha,
        "scientific_protocol_sha256": SCIENTIFIC_PROTOCOL_SHA256,
        "task_ids": [task.task_id for task in tasks],
        "trajectory_count": len(tasks),
        "paired_block_count": len(SHARD_BLOCKS[LOCAL_SHARD]),
        "output_dir": str(OUTPUT_DIR),
        "shard_output_root": str(SHARD_OUTPUT_ROOT),
        "shard_assignment_sha256": shard_assignment_sha256(),
        "generator_params_manifest_sha256": (
            generator_params_manifest_sha256()
        ),
        "collector_wired": True,
        "generation_started": False,
        "screen_generation_authorized": False,
        "requires_explicit_later_user_confirmation": True,
    }


def require_run_confirmation(confirmed_protocol_sha256: str | None) -> None:
    if confirmed_protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            "R8 筛查真实采集需要后续用户明确授权并确认完整"
            "执行协议 SHA-256"
        )


def _require_artifact_sha(value: str | None, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{label}需要确认完整 SHA-256")


def require_collection_confirmation(confirmed_sha256: str | None) -> None:
    _require_artifact_sha(confirmed_sha256, "R8 筛查离线评价")


def require_evaluation_confirmation(confirmed_sha256: str | None) -> None:
    _require_artifact_sha(confirmed_sha256, "R8 筛查独立审计")
