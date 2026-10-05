"""Issue #53 R8 恢复版独立审计适配冻结协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_recovered_evaluation_protocol as evaluation
from scripts import issue53_gap_weight_r8_screen_execution_protocol as source


AUDIT_ADAPTER_VERSION = "issue53-gap-weight-r8-recovered-independent-audit-v1"
AUDIT_DOC = Path("docs/设计/Issue53_BC问题一R8恢复版独立审计适配协议.md")
AUDIT_DOC_SHA256 = (
    "a3fe1229413192ec3d8b82279c6908c7ad05b885d1c35ff37a11235fe601aa8d"
)
FROZEN_AUDIT_ADAPTER_SHA256 = (
    "0b1393a3f504e123028a8c6c08d15f51c7e2802530caf48f4a7be9e2c33daed1"
)

SOURCE_COLLECTION_SHA256 = evaluation.SOURCE_COLLECTION_SHA256
SOURCE_EVALUATION_SHA256 = (
    "548b611b3762cd5489fcbd97d787ea9e91f987c5a94bed0b73e6782f6108bc4c"
)
SOURCE_METRICS_CSV_SHA256 = (
    "b6d1efff24202a74003cee01eb05568db64b071aab3a47a86e3cafa4bc27b660"
)
SOURCE_EVALUATION_COMMIT = "5c0ebbbfcd1806a455fb6057052d9ff99a2eb775"
SOURCE_EVALUATION_ADAPTER_SHA256 = evaluation.FROZEN_ADAPTER_SHA256
SOURCE_AUDITOR_VERSION = (
    "issue53-gap-weight-r8-paired-screen-independent-audit-v1"
)
SOURCE_AUDITOR_SHA256 = source.IMPLEMENTATION_SOURCES[
    "independent_auditor"
]["sha256"]

SOURCE_EVALUATION_PATH = source.OUTPUT_DIR / source.EVALUATION_REPORT
SOURCE_METRICS_CSV_PATH = source.OUTPUT_DIR / source.L1_RESULTS_CSV
AUDIT_OUTPUT_PATH = source.OUTPUT_DIR / source.AUDIT_REPORT

IMPLEMENTATION_SOURCES = {
    "source_evaluation_adapter_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_r8_recovered_evaluation_protocol.py"
        ),
        "sha256": "f47be90a5147e8a7ef44186f8e2dfcaf04de98367d3f4e6a6d185a6c8eec5e23",
    },
    "source_recovered_evaluator": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen_recovered.py"),
        "sha256": "2336261f0421db7f8b9d23a033422c26c3655633956dccc5871602b0c280c495",
    },
    "source_recovery_loader": {
        "path": Path("scripts/recover_issue53_gap_weight_r8_screen.py"),
        "sha256": "d11b41d3d7f3f60678ca51e71f632f4df4c58923c20e7b64a3c5fa2f0e50e1b8",
    },
    "source_independent_auditor": {
        "path": Path("scripts/audit_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_AUDITOR_SHA256,
    },
    "recovered_independent_audit_adapter": {
        "path": Path("scripts/audit_issue53_gap_weight_r8_screen_recovered.py"),
        "sha256": "989677266d7cdbee9ff1896d12e5912045280c0293870d69211395128212d91a",
    },
}

DELEGATED_SOURCE_FUNCTIONS = (
    "_recompute_cases",
    "_summary_independent",
    "_csv_rows_independent",
    "_assert_equal",
    "_audit_csv",
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


def frozen_audit_adapter_manifest() -> dict[str, Any]:
    return {
        "contract_version": AUDIT_ADAPTER_VERSION,
        "issue": 53,
        "stage": "r8_recovered_evaluation_independent_audit_adapter",
        "document": {"path": str(AUDIT_DOC), "sha256": AUDIT_DOC_SHA256},
        "frozen_artifacts": {
            "collection": {
                "path": str(evaluation.SOURCE_COLLECTION_PATH),
                "sha256": SOURCE_COLLECTION_SHA256,
            },
            "evaluation": {
                "path": str(SOURCE_EVALUATION_PATH),
                "sha256": SOURCE_EVALUATION_SHA256,
                "evaluation_commit": SOURCE_EVALUATION_COMMIT,
                "adapter_protocol_sha256": SOURCE_EVALUATION_ADAPTER_SHA256,
            },
            "metrics_csv": {
                "path": str(SOURCE_METRICS_CSV_PATH),
                "sha256": SOURCE_METRICS_CSV_SHA256,
            },
        },
        "adapter_changes": {
            "collection_contract_loader": (
                "recovery_loader.load_collection"
            ),
            "evaluation_top_level_contract": (
                "recovered_evaluation_adapter_v1"
            ),
            "metric_arithmetic_changes": [],
            "comparison_or_classification_changes": [],
        },
        "source_independent_auditor": {
            "contract_version": SOURCE_AUDITOR_VERSION,
            "sha256": SOURCE_AUDITOR_SHA256,
            "delegated_functions": list(DELEGATED_SOURCE_FUNCTIONS),
            "imports_primary_r8_evaluator_arithmetic": False,
            "terminal_and_checkpoint_arithmetic_modified": False,
            "target_bins_or_query_groups_modified": False,
            "ratio_or_classification_modified": False,
            "csv_recomputation_modified": False,
            "float_comparison_tolerance_modified": False,
        },
        "pass_condition": {
            "all_four_cases_exact": True,
            "all_terminal_and_checkpoint_metrics_exact": True,
            "all_summaries_exact": True,
            "classification_exact": True,
            "csv_columns_rows_values_and_sha_exact": True,
            "any_mismatch_fails_without_pass_report": True,
        },
        "monitoring_limitation": {
            "execution_monitoring_evidence_complete": False,
            "code": "runner_samples_lost_before_shard_report_write",
            "must_propagate_to_audit_report": True,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "authorization_boundary": {
            "user_instructed_continue_after_audit_explanation": True,
            "clean_committed_adapter_required_before_run": True,
            "audit_adapter_sha_confirmation_required": True,
            "collection_sha_confirmation_required": True,
            "evaluation_sha_confirmation_required": True,
            "generation_rerun_allowed": False,
            "evaluation_artifact_modification_allowed": False,
            "automatic_five_seed_or_new_kernel_run_allowed": False,
        },
    }


def audit_adapter_sha256() -> str:
    return canonical_sha256(frozen_audit_adapter_manifest())


def assert_frozen_audit_adapter_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if evaluation.FROZEN_ADAPTER_SHA256 != SOURCE_EVALUATION_ADAPTER_SHA256:
        raise RuntimeError("R8 恢复版独立审计继承的评价适配协议漂移")
    evaluation.assert_frozen_adapter_identity(root)
    if file_sha256(root / AUDIT_DOC) != AUDIT_DOC_SHA256:
        raise RuntimeError("R8 恢复版独立审计适配文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"R8 恢复版独立审计实现源码漂移：{name}")
    for path, expected, name in (
        (SOURCE_EVALUATION_PATH, SOURCE_EVALUATION_SHA256, "evaluation"),
        (SOURCE_METRICS_CSV_PATH, SOURCE_METRICS_CSV_SHA256, "metrics CSV"),
    ):
        if file_sha256(root / path) != expected:
            raise RuntimeError(f"R8 恢复版独立审计绑定的 {name} 漂移")
    observed = audit_adapter_sha256()
    if observed != FROZEN_AUDIT_ADAPTER_SHA256:
        raise RuntimeError(
            "R8 恢复版独立审计适配清单漂移："
            f"expected={FROZEN_AUDIT_ADAPTER_SHA256}, observed={observed}"
        )
    return observed


def require_audit_confirmation(
    confirmed_audit_adapter_sha256: str | None,
    confirmed_collection_sha256: str | None,
    confirmed_evaluation_sha256: str | None,
) -> None:
    if confirmed_audit_adapter_sha256 != FROZEN_AUDIT_ADAPTER_SHA256:
        raise PermissionError("R8 恢复版独立审计需要确认完整适配协议 SHA-256")
    if confirmed_collection_sha256 != SOURCE_COLLECTION_SHA256:
        raise PermissionError("R8 恢复版独立审计需要确认 collection SHA-256")
    if confirmed_evaluation_sha256 != SOURCE_EVALUATION_SHA256:
        raise PermissionError("R8 恢复版独立审计需要确认 evaluation SHA-256")
