"""Issue #53 Stage 6D（阶段 6D）第五版评价 JSON 键兼容冻结协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol

COMPATIBILITY_PROTOCOL_VERSION = "issue53-stage6d-v5-evaluation-json-key-compat-v1"
COMPATIBILITY_DOC = Path("docs/设计/Issue53_Stage6D第五版评价JSON键兼容协议.md")
COMPATIBILITY_DOC_SHA256 = (
    "4d50ec80fa614eb113d3afce846ff822ad2b9e0de9545eaaf238112c84c3ac2e"
)

# 清单不包含该常量，避免自指。
FROZEN_COMPATIBILITY_PROTOCOL_SHA256 = (
    "d22eb6237fc5ef31fd51e28c6e905ce86a615a349199cca472b653c62699110b"
)

CONFIRMED_COLLECTION_REPORT_SHA256 = (
    "f81a18e0ae16726aa99ce4a95882c21d8531a66df35beaf683c738fa7df5771b"
)
SOURCE_GENERATION_PROTOCOL_SHA256 = generation_protocol.FROZEN_PROTOCOL_SHA256
SOURCE_RECOVERY_PROTOCOL_SHA256 = recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
SOURCE_GENERATION_COMMIT = recovery_protocol.SOURCE_GENERATION_COMMIT
SOURCE_RECOVERY_COMMIT = "0562a702fb36a8480d5ed03aac6a7f3733a50ad0"

OUTPUT_DIR = recovery_protocol.OUTPUT_DIR
COLLECTION_REPORT = recovery_protocol.COLLECTION_REPORT
EVALUATION_REPORT = recovery_protocol.EVALUATION_REPORT
L1_RESULTS_CSV = recovery_protocol.L1_RESULTS_CSV
AUDIT_REPORT = recovery_protocol.AUDIT_REPORT

NORMALIZED_KEY_PATHS = (
    "datasets.test_300x10.order_counts",
    "datasets.nltcs.order_counts",
)

# 原恢复第一版文件保持字节不变；新入口哈希在实现稳定后填入。
IMPLEMENTATION_SOURCES = {
    "source_generation_protocol": {
        "path": Path("scripts/issue53_stage6d_formal_protocol.py"),
        "sha256": "33bbf388a130a332970bc851137d51edcea94434404984498f58b9f91b186d2e",
    },
    "source_recovery_protocol": {
        "path": Path("scripts/issue53_stage6d_v5_recovery_protocol.py"),
        "sha256": "79e4040e61e5e40103dd23ae577309eeacf1e9c72bfc6e896f4a7d0165b75628",
    },
    "source_recovery_collector": {
        "path": Path("scripts/recover_issue53_stage6d_v5.py"),
        "sha256": "b3e6b3985dc4f749e5c869c4bb3b6662b770a66bf47963049a51cb49a653554f",
    },
    "source_recovery_evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6d_v5_recovered.py"),
        "sha256": "622f91680895a06c392f7d028d162d0c61912bdeb69e947722f04157d0912e0e",
    },
    "source_recovery_independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6d_v5_recovered.py"),
        "sha256": "7ce502738329cc1b532517769e434fa01d9e923ef1d7fb996dfe1904801d6957",
    },
    "compatibility_validator": {
        "path": Path("scripts/validate_issue53_stage6d_v5_recovered_collection.py"),
        "sha256": "c9ffb91f8c9c18514b8043dae3483e4ae61b6d3e761f2ad3a25ca5ea4984aeb5",
    },
    "compatibility_evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6d_v5_recovered_v2.py"),
        "sha256": "b8970d38da9c80d4942077988a0a6a4db0ae6b65bc9f6a4b13f31669a4f58a15",
    },
    "compatibility_independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6d_v5_recovered_v2.py"),
        "sha256": "a5fb2f3df576cad6e994b0d140f5395c8999f0e760272ad179c9c17783bc781f",
    },
}


def file_sha256(path: str | Path) -> str:
    return recovery_protocol.file_sha256(path)


def compatibility_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": COMPATIBILITY_PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6D_v5_recovered_evaluation_json_key_compatibility",
        "compatibility_document": {
            "path": str(COMPATIBILITY_DOC),
            "sha256": COMPATIBILITY_DOC_SHA256,
        },
        "fixed_input": {
            "collection_report": str(OUTPUT_DIR / COLLECTION_REPORT),
            "collection_report_sha256": CONFIRMED_COLLECTION_REPORT_SHA256,
            "source_generation_protocol_sha256": (SOURCE_GENERATION_PROTOCOL_SHA256),
            "source_recovery_protocol_sha256": SOURCE_RECOVERY_PROTOCOL_SHA256,
            "source_generation_commit": SOURCE_GENERATION_COMMIT,
            "source_recovery_commit": SOURCE_RECOVERY_COMMIT,
            "case_count": 30,
            "paired_dataset_seed_count": 10,
        },
        "only_compatibility_correction": {
            "cause": "json_object_keys_are_strings_after_roundtrip",
            "normalization_primitive": "json_dumps_then_json_loads",
            "normalized_key_paths": list(NORMALIZED_KEY_PATHS),
            "expected_integer_keys_before_roundtrip": {
                NORMALIZED_KEY_PATHS[0]: [2, 3, 4],
                NORMALIZED_KEY_PATHS[1]: [2, 3],
            },
            "expected_string_keys_after_roundtrip": {
                NORMALIZED_KEY_PATHS[0]: ["2", "3", "4"],
                NORMALIZED_KEY_PATHS[1]: ["2", "3"],
            },
            "all_values_unchanged": True,
            "all_other_paths_unchanged": True,
            "normalized_manifest_sha256_must_equal_source_protocol": True,
            "unknown_difference_allowed": False,
            "collection_report_rewrite_allowed": False,
        },
        "reuse_contract": {
            "collection_validation_reuses_recovery_v1_after_exact_adapter": True,
            "quality_metrics_reuse_recovery_v1_evaluator_primitives": True,
            "independent_metrics_reuse_recovery_v1_auditor_primitives": True,
            "metric_formula_rewrite_allowed": False,
            "generation_allowed": False,
        },
        "information_boundary": {
            "plan_reads_collection": False,
            "plan_reads_raw_reference": False,
            "compatibility_validation_reads_raw_reference": False,
            "compatibility_validation_interprets_quality": False,
            "evaluation_requires_both_hash_confirmations": True,
            "evaluation_authorized_at_freeze": False,
            "parameter_retuning_allowed": False,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
    }


def compatibility_protocol_sha256() -> str:
    return recovery_protocol.canonical_sha256(compatibility_protocol_manifest())


def assert_frozen_compatibility_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    recovery_protocol.assert_frozen_recovery_identity(root)
    if file_sha256(root / COMPATIBILITY_DOC) != COMPATIBILITY_DOC_SHA256:
        raise RuntimeError("评价兼容协议文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"评价兼容实现源码漂移：{name}")
    observed = compatibility_protocol_sha256()
    if observed != FROZEN_COMPATIBILITY_PROTOCOL_SHA256:
        raise RuntimeError(
            "评价兼容协议清单漂移："
            f"expected={FROZEN_COMPATIBILITY_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_compatibility_identity(repository_root)
    return {
        "contract_version": COMPATIBILITY_PROTOCOL_VERSION,
        "mode": "plan_only_no_collection_reference_or_evaluation_access",
        "compatibility_protocol_sha256": protocol_sha,
        "collection_report_sha256": CONFIRMED_COLLECTION_REPORT_SHA256,
        "source_recovery_protocol_sha256": SOURCE_RECOVERY_PROTOCOL_SHA256,
        "normalized_key_paths": list(NORMALIZED_KEY_PATHS),
        "collection_report_rewrite_allowed": False,
        "new_generation_allowed": False,
        "raw_reference_access_allowed": False,
        "l1_evaluation_authorized": False,
    }


def require_evaluation_confirmation(
    confirmed_collection_sha256: str | None,
    confirmed_compatibility_sha256: str | None,
) -> None:
    if confirmed_collection_sha256 != CONFIRMED_COLLECTION_REPORT_SHA256:
        raise PermissionError("正式评价需要用户确认固定完整采集报告 SHA-256")
    if confirmed_compatibility_sha256 != FROZEN_COMPATIBILITY_PROTOCOL_SHA256:
        raise PermissionError("正式评价需要用户确认完整评价兼容协议 SHA-256")
