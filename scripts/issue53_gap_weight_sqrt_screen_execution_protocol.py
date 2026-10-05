"""平方根查询权重单种子筛查的执行接线差量协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_screen_protocol as r8
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific


PROTOCOL_VERSION = "issue53-gap-weight-sqrt-target-screen-execution-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_BC问题一平方根查询权重筛查执行接线协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "aa6df17358169fec9c8c988e6e77486bb38e02f46b8b2cabb123a74b6d6dfa4f"
)
FROZEN_PROTOCOL_SHA256 = (
    "3330da8dd3bc9fbe264848491414893143f17da13cf42e3438611c24041d359a"
)
SCIENTIFIC_PROTOCOL_SHA256 = scientific.FROZEN_PROTOCOL_SHA256

DEVELOPMENT_SEED = scientific.DEVELOPMENT_SEED
FORMAL_SEEDS = (DEVELOPMENT_SEED,)
ROUND_CAP = scientific.ROUND_CAP
ROUNDS = ROUND_CAP
CANDIDATE_BUDGET = scientific.CANDIDATE_BUDGET
PATIENCE_TICKS = scientific.PATIENCE_TICKS
CHECKPOINT_ROUNDS = scientific.CHECKPOINT_ROUNDS
MAX_WORKERS = scientific.MAX_WORKERS
SQRT_WEIGHTING = scientific.SQRT_WEIGHTING
NORMAL_TERMINATION_REASONS = (
    "fit_target_reached",
    "early_stopped",
    "resource_cap_reached",
)

DATASET_ORDER = scientific.DATASET_ORDER
ARM_ORDER = scientific.ARM_ORDER
DATASETS = scientific.DATASETS
TARGET_COUNT_BINS = scientific.TARGET_COUNT_BINS
EXPECTED_WEIGHT_AUDIT = scientific.EXPECTED_WEIGHT_AUDIT
EXPECTED_SOFTWARE = r8.EXPECTED_SOFTWARE
EXPECTED_GPU = r8.EXPECTED_GPU

# Stage 6D 的基础转移审计用该名字识别 gap 核臂。本筛查只有这一臂。
ARM_GAP = scientific.CANDIDATE_ARM

OUTPUT_DIR = scientific.OUTPUT_DIR
SHARD_OUTPUT_ROOT = Path(
    "outputs/issue53_gap_weight_sqrt_target_screen_seed9908_v1_shards"
)
COLLECTION_REPORT = "collection_report.json"
SHARD_REPORT = "shard_report.json"

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
        "task_count": 2,
        "dataset_block_count": 2,
        "max_workers": MAX_WORKERS,
        "expected_gpu": dict(EXPECTED_GPU),
    }
}

IMPLEMENTATION_SOURCES = {
    "scientific_protocol": {
        "path": Path("scripts/issue53_gap_weight_sqrt_screen_protocol.py"),
        "sha256": (
            "3a5c18d56c097a99440946410f13eaeb9c3e73e6c4a417798a9897b63a2bef4f"
        ),
    },
    "stage6d_collection_infrastructure": {
        "path": Path("scripts/run_issue53_stage6d_formal.py"),
        "sha256": (
            "d020648c2ee9c3808e370ddab7fca74a26ddd412ea28c4643e3f9b280071f88b"
        ),
    },
    "stage6e_autostop_adapter": {
        "path": Path("scripts/run_issue53_stage6e_autostop.py"),
        "sha256": (
            "7130223c2dbfae5723f6d445e92ae7154588b4aebd631121a7f79f325f4b51d1"
        ),
    },
    "collector": {
        "path": Path("scripts/run_issue53_gap_weight_sqrt_screen.py"),
        "sha256": (
            "6ad50b81c1c7c7d9043124b50da11b3540cde712d47546a5ce9c809631ff12e4"
        ),
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
        raise ValueError(f"平方根权重筛查种子不是 9908：{seed!r}")
    return scientific.task_generator_params(dataset, arm)


def generator_params_manifest(
    dataset: str, arm: str, seed: int
) -> dict[str, Any]:
    if seed != DEVELOPMENT_SEED:
        raise ValueError(f"平方根权重筛查种子不是 9908：{seed!r}")
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
        raise ValueError(f"未知平方根权重筛查执行分片：{shard_id!r}")
    tasks = task_plan().tasks
    if len(tasks) != 2:
        raise RuntimeError("平方根权重筛查任务数量漂移")
    return tasks


def task_shard_id(task: Any) -> str:
    if task.task_id not in {item.task_id for item in task_plan().tasks}:
        raise RuntimeError(f"任务不在平方根权重筛查矩阵：{task.task_id}")
    return LOCAL_SHARD


def shard_assignment_manifest() -> dict[str, Any]:
    tasks = tasks_for_shard(LOCAL_SHARD)
    return {
        LOCAL_SHARD: {
            "hostname": EXECUTION_SHARDS[LOCAL_SHARD]["hostname"],
            "task_count": len(tasks),
            "dataset_block_count": len(SHARD_BLOCKS[LOCAL_SHARD]),
            "max_workers": MAX_WORKERS,
            "expected_gpu": dict(
                EXECUTION_SHARDS[LOCAL_SHARD]["expected_gpu"]
            ),
            "dataset_blocks": [
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
        "stage": "gap_weight_problem_1_sqrt_target_execution_wiring",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "scientific_protocol": {
            "version": scientific.PROTOCOL_VERSION,
            "sha256": SCIENTIFIC_PROTOCOL_SHA256,
            "source_path": (
                "scripts/issue53_gap_weight_sqrt_screen_protocol.py"
            ),
            "scientific_fields_modified_by_delta": [],
        },
        "task_matrix_identity": {
            "task_ids": [task.task_id for task in tasks],
            "trajectory_count": 2,
            "dataset_block_count": 2,
            "task_order": "seed_then_dataset",
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
            "sqrt_specific_overrides": [
                "single_candidate_arm_transition_audit",
                "sqrt_weight_diagnostic_guard",
                "two_case_candidate_only_collection_report",
            ],
            "collection_reads_raw_reference": False,
            "collection_reads_reused_baseline": False,
            "partial_quality_emission_allowed": False,
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
        "weighting_guard": {
            "mode": SQRT_WEIGHTING,
            "max_weight_ratio": None,
            "smoothing_count": None,
            "target_count_quantum": 1.0,
            "actual_weight_ratio_by_dataset": {
                dataset: EXPECTED_WEIGHT_AUDIT[dataset][
                    "actual_weight_ratio"
                ]
                for dataset in DATASET_ORDER
            },
            "microsteps": "8*K",
            "zero_clip_required": True,
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
            "quality_evaluation_separately_authorized": True,
            "pr69_change_allowed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_scientific_inheritance() -> None:
    if scientific.FROZEN_PROTOCOL_SHA256 != SCIENTIFIC_PROTOCOL_SHA256:
        raise RuntimeError("平方根权重科学协议 SHA-256 继承漂移")
    if task_plan().tasks != scientific.task_plan().tasks:
        raise RuntimeError("执行层任务矩阵改变了科学协议")
    if generator_params_manifest_sha256() != (
        scientific.generator_params_manifest_sha256()
    ):
        raise RuntimeError("执行层生成参数改变了科学协议")
    if TARGET_COUNT_BINS != scientific.TARGET_COUNT_BINS:
        raise RuntimeError("执行层目标分箱改变了科学协议")


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root).resolve()
    scientific.assert_frozen_protocol_identity(root)
    assert_scientific_inheritance()
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("平方根权重执行接线文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"平方根权重执行接线源码漂移：{name}")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "平方根权重执行接线清单漂移："
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
        "dataset_block_count": len(SHARD_BLOCKS[LOCAL_SHARD]),
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
            "平方根权重筛查真实采集需要后续用户明确授权并确认完整"
            "执行协议 SHA-256"
        )
