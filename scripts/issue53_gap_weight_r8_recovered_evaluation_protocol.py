"""Issue #53 R8 恢复 collection 的离线评价适配冻结协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_screen_execution_protocol as source
from scripts import issue53_gap_weight_r8_screen_recovery_protocol as recovery


ADAPTER_VERSION = "issue53-gap-weight-r8-recovered-evaluation-adapter-v1"
ADAPTER_DOC = Path("docs/设计/Issue53_BC问题一R8恢复版离线评价适配协议.md")
ADAPTER_DOC_SHA256 = (
    "b32915f8577b3d7a03ac13cae521e63f9a06f56aebe458e9b1ac7128093d4189"
)
FROZEN_ADAPTER_SHA256 = (
    "35d93e9e77c5d3e28085bc20d76852bec63eb128b21f8c64d316675e609e4744"
)

SOURCE_COLLECTION_SHA256 = (
    "35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8"
)
SOURCE_COLLECTION_PATH = source.OUTPUT_DIR / source.COLLECTION_REPORT
SOURCE_RECOVERY_PROTOCOL_SHA256 = recovery.FROZEN_RECOVERY_SHA256
SOURCE_EVALUATOR_VERSION = "issue53-gap-weight-r8-paired-screen-evaluation-v1"
SOURCE_EVALUATOR_SHA256 = source.IMPLEMENTATION_SOURCES["evaluator"]["sha256"]

IMPLEMENTATION_SOURCES = {
    "source_recovery_protocol": {
        "path": Path("scripts/issue53_gap_weight_r8_screen_recovery_protocol.py"),
        "sha256": "3d55d0309b1f0eeca38d2e6d9fb3bccef8f9cb615af85d68df929361a9996580",
    },
    "source_recovery_loader": {
        "path": Path("scripts/recover_issue53_gap_weight_r8_screen.py"),
        "sha256": "d11b41d3d7f3f60678ca51e71f632f4df4c58923c20e7b64a3c5fa2f0e50e1b8",
    },
    "source_evaluator": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_EVALUATOR_SHA256,
    },
    "recovered_evaluation_adapter": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen_recovered.py"),
        "sha256": "2336261f0421db7f8b9d23a033422c26c3655633956dccc5871602b0c280c495",
    },
}

DELEGATED_SOURCE_FUNCTIONS = (
    "_evaluate_cases",
    "_summary_and_decision",
    "_csv_rows",
    "_publish",
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
    return source.file_sha256(path)


def frozen_adapter_manifest() -> dict[str, Any]:
    return {
        "contract_version": ADAPTER_VERSION,
        "issue": 53,
        "stage": "r8_recovered_collection_offline_evaluation_adapter",
        "document": {
            "path": str(ADAPTER_DOC),
            "sha256": ADAPTER_DOC_SHA256,
        },
        "source_collection": {
            "path": str(SOURCE_COLLECTION_PATH),
            "sha256": SOURCE_COLLECTION_SHA256,
            "case_count": 4,
            "paired_dataset_seed_count": 2,
            "generation_protocol_sha256": recovery.SOURCE_PROTOCOL_SHA256,
            "postcollection_recovery_protocol_sha256": (
                SOURCE_RECOVERY_PROTOCOL_SHA256
            ),
            "execution_monitoring_evidence_complete": False,
            "monitoring_limitation_code": (
                "runner_samples_lost_before_shard_report_write"
            ),
        },
        "single_adapter_change": {
            "replaced_source_entry": "source_evaluator._audit_collection",
            "replacement_entry": "recovery_loader.load_collection",
            "accepted_state_evaluation_identity": (
                "state_evaluation_count == max(1, applied_rounds)"
            ),
            "other_collection_validation_changes": [],
        },
        "source_evaluator": {
            "contract_version": SOURCE_EVALUATOR_VERSION,
            "sha256": SOURCE_EVALUATOR_SHA256,
            "delegated_functions": list(DELEGATED_SOURCE_FUNCTIONS),
            "metric_arithmetic_modified": False,
            "target_bins_modified": False,
            "query_groups_modified": False,
            "ratio_rules_modified": False,
            "classification_modified": False,
            "csv_schema_modified": False,
            "publication_logic_modified": False,
        },
        "output": {
            "directory": str(source.OUTPUT_DIR),
            "evaluation_report": source.EVALUATION_REPORT,
            "screen_metrics_csv": source.L1_RESULTS_CSV,
            "existing_output_overwrite_allowed": False,
            "atomic_source_publication_reused": True,
        },
        "result_before_information_boundary": {
            "raw_reference_access_allowed": False,
            "terminal_or_checkpoint_evaluation_allowed": False,
            "quality_metric_computation_allowed": False,
            "method_comparison_allowed": False,
            "evaluation_output_publication_allowed": False,
            "generator_call_allowed": False,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "authorization_boundary": {
            "evaluation_authorized_at_adapter_freeze": False,
            "matching_hashes_alone_authorize_evaluation": False,
            "explicit_later_user_instruction_required": True,
            "adapter_sha_confirmation_required": True,
            "collection_sha_confirmation_required": True,
            "automatic_independent_audit_allowed": False,
            "automatic_five_seed_confirmation_allowed": False,
        },
    }


def adapter_sha256() -> str:
    return canonical_sha256(frozen_adapter_manifest())


def assert_frozen_adapter_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if recovery.FROZEN_RECOVERY_SHA256 != SOURCE_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError("R8 恢复版评价继承的恢复协议身份漂移")
    recovery.assert_frozen_recovery_identity(root)
    if file_sha256(root / ADAPTER_DOC) != ADAPTER_DOC_SHA256:
        raise RuntimeError("R8 恢复版评价适配文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"R8 恢复版评价实现源码漂移：{name}")
    if file_sha256(root / SOURCE_COLLECTION_PATH) != SOURCE_COLLECTION_SHA256:
        raise RuntimeError("R8 恢复版评价绑定的 collection 漂移")
    observed = adapter_sha256()
    if observed != FROZEN_ADAPTER_SHA256:
        raise RuntimeError(
            "R8 恢复版评价适配清单漂移："
            f"expected={FROZEN_ADAPTER_SHA256}, observed={observed}"
        )
    return observed


def require_evaluation_confirmation(
    confirmed_adapter_sha256: str | None,
    confirmed_collection_sha256: str | None,
) -> None:
    if confirmed_adapter_sha256 != FROZEN_ADAPTER_SHA256:
        raise PermissionError("R8 恢复版评价需要确认完整适配协议 SHA-256")
    if confirmed_collection_sha256 != SOURCE_COLLECTION_SHA256:
        raise PermissionError("R8 恢复版评价需要确认冻结 collection SHA-256")
