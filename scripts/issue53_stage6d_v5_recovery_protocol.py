"""Issue #53 Stage 6D（阶段 6D）第五版生成后收口恢复冻结协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_stage6d_formal_protocol as generation_protocol

RECOVERY_PROTOCOL_VERSION = "issue53-stage6d-v5-postcollection-recovery-v1"
RECOVERY_DOC = Path("docs/设计/Issue53_Stage6D第五版生成后收口恢复协议.md")
RECOVERY_DOC_SHA256 = "34465710a0584d37970c128e1c2de5dc3ab685634691eb8a9ea6298649796c19"
RECOVERY_INVENTORY = Path("configs/issue53_stage6d_v5_recovery_inventory.json")
RECOVERY_INVENTORY_SHA256 = (
    "1c8b0e790f19de9dd83b1f5cb1a2a0ac9c76fbb704a99523cf388c8458d765e9"
)

# 清单不包含该常量，避免自指；实现、测试与文档审查完成后填入。
FROZEN_RECOVERY_PROTOCOL_SHA256 = (
    "44527c0a7663a03ac0f5cf54bec11bf22610c3806de84d9cc11b6c6dc0edfbca"
)

SOURCE_GENERATION_COMMIT = "b8bd59f26acc5528165dcb54850f101249563715"
SOURCE_PROTOCOL_SHA256 = generation_protocol.FROZEN_PROTOCOL_SHA256
SOURCE_PROTOCOL_FILE_SHA256 = (
    "33bbf388a130a332970bc851137d51edcea94434404984498f58b9f91b186d2e"
)
SOURCE_RUNNER_SHA256 = generation_protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256 = (
    "771b27775ddc9bde562c169e3c3cb991c2635d00b0087f90d1795847951e8e9d"
)
SOURCE_SHARD_ASSIGNMENT_SHA256 = (
    "627f39e59f8eb0d80a194719a8995c2db123fa1d1f1375de81d4f8ebc057d789"
)

OUTPUT_DIR = generation_protocol.OUTPUT_DIR
SHARD_OUTPUT_ROOT = generation_protocol.SHARD_OUTPUT_ROOT
SHARD_REPORT = generation_protocol.SHARD_REPORT
COLLECTION_REPORT = generation_protocol.COLLECTION_REPORT
EVALUATION_REPORT = generation_protocol.EVALUATION_REPORT
L1_RESULTS_CSV = generation_protocol.L1_RESULTS_CSV
AUDIT_REPORT = generation_protocol.AUDIT_REPORT

MONITORING_LIMITATION_CODE = "gpu_samples_lost_before_original_shard_report_write"
MONITORING_LIMITATION_REASON = (
    "旧收集器在全部工作进程结束后执行了错误的状态评价计数校验；"
    "异常发生在分片报告写盘前，父进程内存中的显卡监控样本随退出丢失"
)

# 哈希在三个恢复入口实现稳定后填入。原第五版文件绝不修改。
IMPLEMENTATION_SOURCES = {
    "source_generation_protocol": {
        "path": Path("scripts/issue53_stage6d_formal_protocol.py"),
        "sha256": SOURCE_PROTOCOL_FILE_SHA256,
    },
    "source_generation_collector": {
        "path": Path("scripts/run_issue53_stage6d_formal.py"),
        "sha256": SOURCE_RUNNER_SHA256,
    },
    "recovery_collector": {
        "path": Path("scripts/recover_issue53_stage6d_v5.py"),
        "sha256": "b3e6b3985dc4f749e5c869c4bb3b6662b770a66bf47963049a51cb49a653554f",
    },
    "recovery_evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6d_v5_recovered.py"),
        "sha256": "622f91680895a06c392f7d028d162d0c61912bdeb69e947722f04157d0912e0e",
    },
    "recovery_independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6d_v5_recovered.py"),
        "sha256": "7ce502738329cc1b532517769e434fa01d9e923ef1d7fb996dfe1904801d6957",
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
    return generation_protocol.file_sha256(path)


def _load_inventory_unchecked(repository_root: str | Path) -> dict[str, Any]:
    path = Path(repository_root) / RECOVERY_INVENTORY
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("恢复输入清单根必须是对象")
    return value


def recovery_inventory(repository_root: str | Path) -> dict[str, Any]:
    """读取并结构化复核结果前冻结的恢复输入清单。"""

    root = Path(repository_root)
    path = root / RECOVERY_INVENTORY
    if file_sha256(path) != RECOVERY_INVENTORY_SHA256:
        raise RuntimeError("恢复输入清单 SHA-256 漂移")
    value = _load_inventory_unchecked(root)
    expected_top = {
        "inventory_version": "issue53-stage6d-v5-postcollection-recovery-inventory-v1",
        "source_generation_commit": SOURCE_GENERATION_COMMIT,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_generator_params_manifest_sha256": (
            SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "source_shard_assignment_sha256": SOURCE_SHARD_ASSIGNMENT_SHA256,
    }
    if any(value.get(key) != expected for key, expected in expected_top.items()):
        raise RuntimeError("恢复输入清单来源身份漂移")
    shards = value.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(
        generation_protocol.SHARD_ORDER
    ):
        raise RuntimeError("恢复输入清单必须恰好包含两个冻结分片")
    for shard_id in generation_protocol.SHARD_ORDER:
        shard = shards[shard_id]
        if not isinstance(shard, dict):
            raise TypeError(f"恢复输入清单分片必须是对象：{shard_id}")
        expected_tasks = generation_protocol.tasks_for_shard(shard_id)
        expected_ids = {task.task_id for task in expected_tasks}
        case_hashes = shard.get("case_manifest_sha256")
        if (
            shard.get("source_hostname")
            != generation_protocol.EXECUTION_SHARDS[shard_id]["hostname"]
            or not isinstance(shard.get("source_staging_basename"), str)
            or not shard["source_staging_basename"].startswith(f".{shard_id}.staging-")
            or not _is_sha256(shard.get("staging_manifest_sha256"))
            or not isinstance(case_hashes, dict)
            or set(case_hashes) != expected_ids
            or any(not _is_sha256(digest) for digest in case_hashes.values())
        ):
            raise RuntimeError(f"恢复输入清单分片身份漂移：{shard_id}")
    if (
        sum(
            len(shards[shard_id]["case_manifest_sha256"])
            for shard_id in generation_protocol.SHARD_ORDER
        )
        != 30
    ):
        raise RuntimeError("恢复输入清单不是完整 21/9 共 30 条")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def recovery_protocol_manifest(repository_root: str | Path) -> dict[str, Any]:
    inventory = recovery_inventory(repository_root)
    return {
        "contract_version": RECOVERY_PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6D_v5_postcollection_recovery",
        "recovery_document": {
            "path": str(RECOVERY_DOC),
            "sha256": RECOVERY_DOC_SHA256,
        },
        "recovery_inventory": {
            "path": str(RECOVERY_INVENTORY),
            "sha256": RECOVERY_INVENTORY_SHA256,
            "inventory_version": inventory["inventory_version"],
            "staging_manifest_sha256": {
                shard_id: inventory["shards"][shard_id]["staging_manifest_sha256"]
                for shard_id in generation_protocol.SHARD_ORDER
            },
            "case_manifest_count": 30,
        },
        "source_generation": {
            "contract_version": generation_protocol.PROTOCOL_VERSION,
            "protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "execution_commit": SOURCE_GENERATION_COMMIT,
            "runner_sha256": SOURCE_RUNNER_SHA256,
            "generator_params_manifest_sha256": (
                SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
            ),
            "shard_assignment_sha256": SOURCE_SHARD_ASSIGNMENT_SHA256,
            "shard_task_counts": {
                shard_id: len(generation_protocol.tasks_for_shard(shard_id))
                for shard_id in generation_protocol.SHARD_ORDER
            },
        },
        "only_validation_correction": {
            "termination_reason": "candidate_budget",
            "applied_rounds": generation_protocol.ROUNDS,
            "proposal_attempt_count": generation_protocol.ROUNDS,
            "candidate_evaluation_count": generation_protocol.ROUNDS,
            "state_evaluation_count": generation_protocol.ROUNDS,
            "logical_state_count": generation_protocol.ROUNDS + 1,
            "terminal_state_already_counted_as_candidate_evaluation": True,
        },
        "recovery_execution": {
            "new_generation_allowed": False,
            "gpu_access_allowed": False,
            "raw_reference_access_allowed": False,
            "quality_metric_access_allowed": False,
            "case_file_rewrite_allowed": False,
            "new_generation_case_count": 0,
            "resumed_case_counts": {
                shard_id: len(generation_protocol.tasks_for_shard(shard_id))
                for shard_id in generation_protocol.SHARD_ORDER
            },
            "exact_21_9_assignment_required": True,
            "atomic_shard_closeout": True,
            "merge_requires_both_recovered_shards": True,
        },
        "monitoring_evidence_limitation": {
            "code": MONITORING_LIMITATION_CODE,
            "gpu_samples_persisted": False,
            "gpu_samples_may_be_reconstructed": False,
            "reason": MONITORING_LIMITATION_REASON,
            "generation_result_bytes_affected": False,
            "quality_arithmetic_affected": False,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "authorization_boundary": {
            "recovery_execution_authorized_at_freeze": False,
            "recovery_requires_full_protocol_sha_confirmation": True,
            "evaluation_requires_recovered_collection_sha_confirmation": True,
        },
    }


def recovery_protocol_sha256(repository_root: str | Path) -> str:
    return canonical_sha256(recovery_protocol_manifest(repository_root))


def assert_frozen_recovery_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    generation_protocol.assert_frozen_protocol_identity(root)
    if file_sha256(root / RECOVERY_DOC) != RECOVERY_DOC_SHA256:
        raise RuntimeError("恢复协议文档 SHA-256 漂移")
    recovery_inventory(root)
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"恢复实现源码漂移：{name}")
    observed = recovery_protocol_sha256(root)
    if observed != FROZEN_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError(
            "恢复协议清单漂移："
            f"expected={FROZEN_RECOVERY_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_recovery_identity(repository_root)
    return {
        "contract_version": RECOVERY_PROTOCOL_VERSION,
        "mode": "plan_only_no_artifact_reference_gpu_or_generation_access",
        "recovery_protocol_sha256": protocol_sha,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_generation_commit": SOURCE_GENERATION_COMMIT,
        "shard_task_counts": {
            shard_id: len(generation_protocol.tasks_for_shard(shard_id))
            for shard_id in generation_protocol.SHARD_ORDER
        },
        "new_generation_case_count": 0,
        "gpu_access_allowed": False,
        "raw_reference_access_allowed": False,
        "l1_evaluation_allowed": False,
        "recovery_authorized": False,
    }


def require_recovery_confirmation(confirmed_sha256: str | None) -> None:
    if confirmed_sha256 != FROZEN_RECOVERY_PROTOCOL_SHA256:
        raise PermissionError("恢复收口需要用户另行授权并确认完整恢复协议 SHA-256")


def require_collection_confirmation(confirmed_sha256: str | None) -> None:
    if not _is_sha256(confirmed_sha256):
        raise PermissionError("恢复版评价需要显式确认完整 collection 报告 SHA-256")
