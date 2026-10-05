"""Issue #53 Stage 6E 生成后收口勘误恢复冻结协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_stage6e_autostop_protocol as generation_protocol


RECOVERY_PROTOCOL_VERSION = "issue53-stage6e-postcollection-erratum-recovery-v1"
RECOVERY_DOC = Path("docs/设计/Issue53_Stage6E生成后收口勘误恢复协议.md")
RECOVERY_DOC_SHA256 = "ae6bfb148b68ce6a763e5a64aac437aba01705210cbc8c87d4869abe905ce25b"
RECOVERY_INVENTORY = Path("configs/issue53_stage6e_recovery_inventory.json")
RECOVERY_INVENTORY_SHA256 = (
    "c0fc109490087ee6a1900752d6c3a76914a4d6115f39e642f8634ddbb5dd2dc2"
)

# 该常量不进入清单，避免自指；实现与测试稳定后填入。
FROZEN_RECOVERY_PROTOCOL_SHA256 = (
    "3b6e26761cb3c976c59821266119ca0fcc025527304f218c629ccd8287133ef1"
)

SOURCE_GENERATION_COMMIT = "04904b2cdf1ae2125a5ff7d347f4888506ed7cac"
SOURCE_PROTOCOL_SHA256 = generation_protocol.FROZEN_PROTOCOL_SHA256
SOURCE_PROTOCOL_FILE_SHA256 = (
    "327d6e6734c8e6b314d276780424389f1fa4abd49bcd933a9934f20505acdf6c"
)
SOURCE_RUNNER_SHA256 = generation_protocol.IMPLEMENTATION_SOURCES["collector"][
    "sha256"
]
SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256 = (
    "f331da081717dadf2efabbb8f4dd1e1c90e1d59e2586c2b79c7fe90e341e9c92"
)
SOURCE_SHARD_ASSIGNMENT_SHA256 = (
    "a6a7bd2a54ebd985987de4029c004d07e835cf3bc89f3794478582d351e95d40"
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
    "原收集器在全部工作进程结束后执行了错误的状态评价计数校验；"
    "异常发生在分片报告写盘前，父进程内存中的显卡监控样本随退出丢失"
)

IMPLEMENTATION_SOURCES = {
    "source_generation_protocol": {
        "path": Path("scripts/issue53_stage6e_autostop_protocol.py"),
        "sha256": SOURCE_PROTOCOL_FILE_SHA256,
    },
    "source_generation_collector": {
        "path": Path("scripts/run_issue53_stage6e_autostop.py"),
        "sha256": SOURCE_RUNNER_SHA256,
    },
    "recovery_collector": {
        "path": Path("scripts/recover_issue53_stage6e.py"),
        "sha256": "ab2b205798dfa63a3c983017f7714848edb19c11d2ee8533fe23f1d4b6b43fa9",
    },
    "recovery_evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6e_recovered.py"),
        "sha256": "712f8de4820fb6e88f8bcd1ad6742c4d8f2826418d008e8235b757ac373d974a",
    },
    "recovery_independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6e_recovered.py"),
        "sha256": "54763f7eaeca7cb678134dcb6c89677281472257058d6409621f83e67bd70326",
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


def json_roundtrip(value: Any) -> Any:
    """返回写入 JSON 后的规范形态，统一整数键等兼容细节。"""

    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def recovery_inventory(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root)
    path = root / RECOVERY_INVENTORY
    if file_sha256(path) != RECOVERY_INVENTORY_SHA256:
        raise RuntimeError("Stage 6E 恢复输入清单 SHA-256 漂移")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    expected_top = {
        "inventory_version": "issue53-stage6e-postcollection-recovery-inventory-v1",
        "source_generation_commit": SOURCE_GENERATION_COMMIT,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_generator_params_manifest_sha256": (
            SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "source_shard_assignment_sha256": SOURCE_SHARD_ASSIGNMENT_SHA256,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in expected_top.items()
    ):
        raise RuntimeError("Stage 6E 恢复输入清单来源身份漂移")
    shards = value.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(
        generation_protocol.SHARD_ORDER
    ):
        raise RuntimeError("Stage 6E 恢复清单必须恰好包含冻结单分片")
    shard_id = generation_protocol.LOCAL_SHARD
    shard = shards[shard_id]
    expected_ids = {
        task.task_id for task in generation_protocol.tasks_for_shard(shard_id)
    }
    case_hashes = shard.get("case_manifest_sha256")
    if (
        shard.get("source_hostname")
        != generation_protocol.EXECUTION_SHARDS[shard_id]["hostname"]
        or shard.get("source_staging_basename")
        != ".local_rtx4090.staging-vuwmu1zf"
        or not _is_sha256(shard.get("staging_manifest_sha256"))
        or not isinstance(case_hashes, dict)
        or set(case_hashes) != expected_ids
        or len(case_hashes) != 30
        or any(not _is_sha256(digest) for digest in case_hashes.values())
    ):
        raise RuntimeError("Stage 6E 恢复输入清单 case 身份漂移")
    return value


def recovery_protocol_manifest(repository_root: str | Path) -> dict[str, Any]:
    inventory = recovery_inventory(repository_root)
    shard_id = generation_protocol.LOCAL_SHARD
    return {
        "contract_version": RECOVERY_PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6E_postcollection_erratum_recovery",
        "recovery_document": {
            "path": str(RECOVERY_DOC),
            "sha256": RECOVERY_DOC_SHA256,
        },
        "recovery_inventory": {
            "path": str(RECOVERY_INVENTORY),
            "sha256": RECOVERY_INVENTORY_SHA256,
            "inventory_version": inventory["inventory_version"],
            "staging_manifest_sha256": inventory["shards"][shard_id][
                "staging_manifest_sha256"
            ],
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
            "shard_id": shard_id,
            "task_count": 30,
        },
        "only_validation_correction": {
            "old_rejected_identity": (
                "state_evaluation_count == applied_rounds + 1"
            ),
            "correct_observed_identity": (
                "state_evaluation_count == max(1, applied_rounds)"
            ),
            "logical_state_count_identity": (
                "logical_state_count == applied_rounds + 1"
            ),
            "terminal_state_already_counted_as_candidate_evaluation": True,
            "all_other_generation_guards_unchanged": True,
        },
        "recovery_execution": {
            "new_generation_allowed": False,
            "gpu_access_allowed": False,
            "raw_reference_access_allowed": False,
            "quality_metric_access_allowed": False,
            "case_file_rewrite_allowed": False,
            "new_generation_case_count": 0,
            "resumed_case_count": 30,
            "atomic_shard_closeout": True,
            "complete_single_shard_required_before_merge": True,
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
            "audit_requires_recovered_evaluation_sha_confirmation": True,
        },
    }


def recovery_protocol_sha256(repository_root: str | Path) -> str:
    return canonical_sha256(recovery_protocol_manifest(repository_root))


def assert_frozen_recovery_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    generation_protocol.assert_frozen_protocol_identity(root)
    if file_sha256(root / RECOVERY_DOC) != RECOVERY_DOC_SHA256:
        raise RuntimeError("Stage 6E 恢复协议文档 SHA-256 漂移")
    recovery_inventory(root)
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"Stage 6E 恢复实现源码漂移：{name}")
    observed = recovery_protocol_sha256(root)
    if observed != FROZEN_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError(
            "Stage 6E 恢复协议清单漂移："
            f"expected={FROZEN_RECOVERY_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    recovery_sha = assert_frozen_recovery_identity(repository_root)
    inventory = recovery_inventory(repository_root)
    shard_id = generation_protocol.LOCAL_SHARD
    return {
        "mode": "plan_only_no_case_reference_gpu_or_generation_access",
        "recovery_protocol_sha256": recovery_sha,
        "source_generation_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_generation_commit": SOURCE_GENERATION_COMMIT,
        "source_staging_basename": inventory["shards"][shard_id][
            "source_staging_basename"
        ],
        "source_case_count": 30,
        "output_dir": str(OUTPUT_DIR),
        "new_generation_allowed": False,
        "gpu_access_allowed": False,
        "raw_reference_access_allowed": False,
        "case_file_rewrite_allowed": False,
        "recovery_started": False,
    }


def require_recovery_confirmation(confirmed_sha256: str | None) -> None:
    if confirmed_sha256 != FROZEN_RECOVERY_PROTOCOL_SHA256:
        raise PermissionError("Stage 6E 恢复收口需要确认完整恢复协议 SHA-256")


def require_collection_confirmation(confirmed_sha256: str | None) -> None:
    if not _is_sha256(confirmed_sha256):
        raise PermissionError("Stage 6E 恢复版评价需要确认完整 collection SHA-256")


def require_evaluation_confirmation(confirmed_sha256: str | None) -> None:
    if not _is_sha256(confirmed_sha256):
        raise PermissionError("Stage 6E 恢复版审计需要确认完整 evaluation SHA-256")
