"""Issue #53 平方根权重恢复版独立审计适配冻结协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_sqrt_recovered_evaluation_protocol as evaluation
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific


AUDIT_ADAPTER_VERSION = (
    "issue53-gap-weight-sqrt-target-recovered-independent-audit-v1"
)
AUDIT_DOC = Path(
    "docs/设计/Issue53_BC问题一平方根权重恢复版独立审计适配协议.md"
)
AUDIT_DOC_SHA256 = (
    "cb7f436d8df5c72a38a45d43e34df0b8f4106f9738efb8944a0a7dac0550f10e"
)
FROZEN_AUDIT_ADAPTER_SHA256 = (
    "5b7d2425938e7dfc2c8480dc3dc15f7396c498d47a2fcb901c080110ca6c2400"
)

SOURCE_COLLECTION_SHA256 = evaluation.SOURCE_COLLECTION_SHA256
SOURCE_EVALUATION_SHA256 = (
    "e6d8110f521e366130949b4b4d3ac460b34eb7109d3e4a8f5518d96e92df3fad"
)
SOURCE_METRICS_CSV_SHA256 = (
    "b5dd01a653544bb7a3a98317ba0e708d2e89a10b70679c57c6046cf339b072e6"
)
SOURCE_EVALUATION_COMMIT = "1b84a5fb3fcb24cd0f21f2ac58f2f0d2e9437a7d"
SOURCE_EVALUATION_ADAPTER_SHA256 = evaluation.FROZEN_ADAPTER_SHA256
SOURCE_AUDITOR_VERSION = (
    "issue53-gap-weight-r8-paired-screen-independent-audit-v1"
)
SOURCE_AUDITOR_SHA256 = (
    "7a7f991fd9ef066771c58c36f1b22afa13f78ab27dd021cb8c13bb2ddbd4ad03"
)

SOURCE_EVALUATION_PATH = evaluation.OUTPUT_DIR / evaluation.EVALUATION_REPORT
SOURCE_METRICS_CSV_PATH = evaluation.OUTPUT_DIR / evaluation.L1_RESULTS_CSV
AUDIT_OUTPUT_PATH = evaluation.OUTPUT_DIR / "independent_audit.json"

IMPLEMENTATION_SOURCES = {
    "source_evaluation_adapter_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_sqrt_recovered_evaluation_protocol.py"
        ),
        "sha256": (
            "f7254224e75dcbf80b4a5919d796a5399a546b547972205da39b759530c71a50"
        ),
    },
    "source_recovered_evaluator": {
        "path": Path(
            "scripts/evaluate_issue53_gap_weight_sqrt_screen_recovered.py"
        ),
        "sha256": (
            "17c522c9ad37b0199c0ca2b4f540279cb5062df34112b19279fcebfe00177a59"
        ),
    },
    "source_recovery_loader": {
        "path": Path("scripts/recover_issue53_gap_weight_sqrt_screen.py"),
        "sha256": (
            "4f9168887539074fefec380a22a99880914b2919e37f294a201fa6000db12f77"
        ),
    },
    "source_independent_auditor": {
        "path": Path("scripts/audit_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_AUDITOR_SHA256,
    },
    "recovered_independent_audit_adapter": {
        "path": Path(
            "scripts/audit_issue53_gap_weight_sqrt_screen_recovered.py"
        ),
        "sha256": (
            "b0a496977b002e8d9849a530ed425274ae55f8233dae0556fa71de7b3356d868"
        ),
    },
}

DELEGATED_SOURCE_FUNCTIONS = (
    "_recompute_cases",
    "_nested",
    "_independent_ratio",
    "_ratio_value",
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
    return evaluation.file_sha256(path)


def frozen_audit_adapter_manifest() -> dict[str, Any]:
    return {
        "contract_version": AUDIT_ADAPTER_VERSION,
        "issue": 53,
        "stage": "sqrt_target_recovered_evaluation_independent_audit_adapter",
        "document": {"path": str(AUDIT_DOC), "sha256": AUDIT_DOC_SHA256},
        "frozen_artifacts": {
            "candidate_collection": {
                "path": str(evaluation.SOURCE_COLLECTION_PATH),
                "sha256": SOURCE_COLLECTION_SHA256,
            },
            "candidate_evaluation": {
                "path": str(SOURCE_EVALUATION_PATH),
                "sha256": SOURCE_EVALUATION_SHA256,
                "evaluation_commit": SOURCE_EVALUATION_COMMIT,
                "adapter_protocol_sha256": SOURCE_EVALUATION_ADAPTER_SHA256,
            },
            "candidate_metrics_csv": {
                "path": str(SOURCE_METRICS_CSV_PATH),
                "sha256": SOURCE_METRICS_CSV_SHA256,
            },
            "audited_baseline": {
                name: {"path": str(item["path"]), "sha256": item["sha256"]}
                for name, item in evaluation.BASELINE_ARTIFACTS.items()
            },
        },
        "candidate_recomputation": {
            "collection_contract_loader": "recovery_loader.load_collection",
            "source_protocol_runtime_binding": (
                "sqrt_recovered_evaluation_protocol"
            ),
            "source_independent_auditor_version": SOURCE_AUDITOR_VERSION,
            "source_independent_auditor_sha256": SOURCE_AUDITOR_SHA256,
            "delegated_functions": list(DELEGATED_SOURCE_FUNCTIONS),
            "imports_primary_sqrt_evaluator_arithmetic": False,
            "imports_primary_r8_evaluator_arithmetic": False,
            "terminal_or_checkpoint_metric_arithmetic_modified": False,
            "target_bins_or_query_groups_modified": False,
            "csv_arithmetic_modified": False,
            "float_comparison_tolerance_modified": False,
        },
        "independent_summary_adapter": {
            "baseline_cases_recomputed": False,
            "baseline_metrics_reused_only_after_frozen_independent_audit": True,
            "candidate_vs_legacy_and_r8_structure_rebuilt": True,
            "ratio_zero_denominator_source": (
                "source_independent_auditor._independent_ratio"
            ),
            "six_gate_records_rebuilt": True,
            "classification_explicitly_reimplemented": True,
            "scientific_classifier_called": False,
            "primary_summary_function_called": False,
            "screen_gates": scientific.frozen_protocol_manifest()[
                "screen_gates"
            ],
            "observed_metric_values_hardcoded": False,
            "observed_classification_hardcoded": False,
        },
        "pass_condition": {
            "both_candidate_cases_exact": True,
            "all_terminal_and_checkpoint_metrics_exact": True,
            "all_summaries_exact": True,
            "all_six_gate_records_exact": True,
            "classification_and_precedence_exact": True,
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
            "audit_authorized_at_adapter_freeze": False,
            "clean_committed_adapter_required_before_run": True,
            "audit_adapter_sha_confirmation_required": True,
            "collection_sha_confirmation_required": True,
            "evaluation_sha_confirmation_required": True,
            "generation_rerun_allowed": False,
            "evaluation_artifact_modification_allowed": False,
            "automatic_fresh_seed_or_new_kernel_run_allowed": False,
            "automatic_parameter_search_allowed": False,
        },
    }


def audit_adapter_sha256() -> str:
    return canonical_sha256(frozen_audit_adapter_manifest())


def assert_frozen_audit_adapter_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if evaluation.FROZEN_ADAPTER_SHA256 != SOURCE_EVALUATION_ADAPTER_SHA256:
        raise RuntimeError("平方根独立审计继承的评价适配协议漂移")
    evaluation.assert_frozen_adapter_identity(root)
    if file_sha256(root / AUDIT_DOC) != AUDIT_DOC_SHA256:
        raise RuntimeError("平方根独立审计适配文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"平方根独立审计实现源码漂移：{name}")
    for path, expected, name in (
        (SOURCE_EVALUATION_PATH, SOURCE_EVALUATION_SHA256, "evaluation"),
        (SOURCE_METRICS_CSV_PATH, SOURCE_METRICS_CSV_SHA256, "metrics CSV"),
    ):
        if file_sha256(root / path) != expected:
            raise RuntimeError(f"平方根独立审计绑定的 {name} 漂移")
    observed = audit_adapter_sha256()
    if observed != FROZEN_AUDIT_ADAPTER_SHA256:
        raise RuntimeError(
            "平方根独立审计适配清单漂移："
            f"expected={FROZEN_AUDIT_ADAPTER_SHA256}, observed={observed}"
        )
    return observed


def require_audit_confirmation(
    confirmed_audit_adapter_sha256: str | None,
    confirmed_collection_sha256: str | None,
    confirmed_evaluation_sha256: str | None,
) -> None:
    if confirmed_audit_adapter_sha256 != FROZEN_AUDIT_ADAPTER_SHA256:
        raise PermissionError("平方根独立审计需要确认完整适配协议 SHA-256")
    if confirmed_collection_sha256 != SOURCE_COLLECTION_SHA256:
        raise PermissionError("平方根独立审计需要确认 collection SHA-256")
    if confirmed_evaluation_sha256 != SOURCE_EVALUATION_SHA256:
        raise PermissionError("平方根独立审计需要确认 evaluation SHA-256")
