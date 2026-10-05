"""Issue #53 平方根权重恢复 collection 的离线评价适配协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_screen_execution_protocol as baseline
from scripts import issue53_gap_weight_sqrt_screen_execution_protocol as source
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific
from scripts import issue53_gap_weight_sqrt_screen_recovery_protocol as recovery


ADAPTER_VERSION = (
    "issue53-gap-weight-sqrt-target-recovered-evaluation-adapter-v1"
)
ADAPTER_DOC = Path(
    "docs/设计/Issue53_BC问题一平方根权重恢复版离线评价适配协议.md"
)
ADAPTER_DOC_SHA256 = (
    "e238ca9c2c66a6bb5f0f83012618970fe6ed5abf6187c4eb40727d5ab0a69865"
)
FROZEN_ADAPTER_SHA256 = (
    "1c0ec85b711356a98cd8540af8c8b5deedd7820b5a8f76f0e5a6f55f5101489d"
)

SOURCE_COLLECTION_SHA256 = (
    "50f0c3561fa9614abc85eacb014566e3354e987a647784f27453419ea3e882f4"
)
SOURCE_COLLECTION_PATH = source.OUTPUT_DIR / source.COLLECTION_REPORT
SOURCE_RECOVERY_PROTOCOL_SHA256 = recovery.FROZEN_RECOVERY_SHA256
SOURCE_EVALUATOR_VERSION = "issue53-gap-weight-r8-paired-screen-evaluation-v1"
SOURCE_EVALUATOR_SHA256 = (
    "287779a7e66466d3ed2e04862d2aded7ce3e8f084277de72893600e71c6f7445"
)

BASELINE_ARTIFACTS = scientific.BASELINE_ARTIFACTS
BASELINE_EVALUATION_PATH = BASELINE_ARTIFACTS["evaluation"]["path"]
BASELINE_EVALUATION_SHA256 = BASELINE_ARTIFACTS["evaluation"]["sha256"]
BASELINE_AUDIT_PATH = BASELINE_ARTIFACTS["independent_audit"]["path"]
BASELINE_AUDIT_SHA256 = BASELINE_ARTIFACTS["independent_audit"]["sha256"]
BASELINE_EVALUATION_ADAPTER_SHA256 = (
    "35d93e9e77c5d3e28085bc20d76852bec63eb128b21f8c64d316675e609e4744"
)
BASELINE_AUDIT_ADAPTER_SHA256 = (
    "0b1393a3f504e123028a8c6c08d15f51c7e2802530caf48f4a7be9e2c33daed1"
)

EVALUATION_REPORT = "evaluation_report.json"
L1_RESULTS_CSV = "screen_metrics.csv"

# 以下是原 R8 evaluator 运行时兼容面；不改科学协议。
PROTOCOL_VERSION = source.PROTOCOL_VERSION
FROZEN_PROTOCOL_SHA256 = source.FROZEN_PROTOCOL_SHA256
SCIENTIFIC_PROTOCOL_SHA256 = source.SCIENTIFIC_PROTOCOL_SHA256
OUTPUT_DIR = source.OUTPUT_DIR
DATASET_ORDER = scientific.DATASET_ORDER
DATASETS = scientific.DATASETS
TARGET_COUNT_BINS = scientific.TARGET_COUNT_BINS
CHECKPOINT_ROUNDS = scientific.CHECKPOINT_ROUNDS
DEVELOPMENT_SEED = scientific.DEVELOPMENT_SEED
CANDIDATE_ARM = scientific.CANDIDATE_ARM
SQRT_WEIGHTING = scientific.SQRT_WEIGHTING
TEST_GROUP_ORDER = baseline.TEST_GROUP_ORDER
NLTCS_GROUP_COUNTS = baseline.NLTCS_GROUP_COUNTS
NLTCS_COMMON_BIN = scientific.NLTCS_COMMON_BIN
NLTCS_RARE_BIN = scientific.NLTCS_RARE_BIN

DELEGATED_SOURCE_FUNCTIONS = (
    "_evaluate_cases",
    "_nested",
    "_ratio_record",
    "_decision_ratio",
    "_csv_rows",
    "_publish",
)

IMPLEMENTATION_SOURCES = {
    "scientific_protocol": {
        "path": Path("scripts/issue53_gap_weight_sqrt_screen_protocol.py"),
        "sha256": (
            "3a5c18d56c097a99440946410f13eaeb9c3e73e6c4a417798a9897b63a2bef4f"
        ),
    },
    "candidate_recovery_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_sqrt_screen_recovery_protocol.py"
        ),
        "sha256": (
            "8312a45031d894544e4c90cf78613d8f9de6c086b3525d90e1c0ef5a7f98b330"
        ),
    },
    "candidate_recovery_loader": {
        "path": Path("scripts/recover_issue53_gap_weight_sqrt_screen.py"),
        "sha256": (
            "4f9168887539074fefec380a22a99880914b2919e37f294a201fa6000db12f77"
        ),
    },
    "source_evaluator": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_EVALUATOR_SHA256,
    },
    "evaluation_adapter": {
        "path": Path(
            "scripts/evaluate_issue53_gap_weight_sqrt_screen_recovered.py"
        ),
        "sha256": (
            "17c522c9ad37b0199c0ca2b4f540279cb5062df34112b19279fcebfe00177a59"
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
    return source.file_sha256(path)


def task_plan():
    return source.task_plan()


def smoothing_count(dataset: str) -> float:
    return baseline.smoothing_count(dataset)


def frozen_adapter_manifest() -> dict[str, Any]:
    return {
        "contract_version": ADAPTER_VERSION,
        "issue": 53,
        "stage": "sqrt_target_recovered_collection_offline_evaluation",
        "document": {
            "path": str(ADAPTER_DOC),
            "sha256": ADAPTER_DOC_SHA256,
        },
        "candidate_collection": {
            "path": str(SOURCE_COLLECTION_PATH),
            "sha256": SOURCE_COLLECTION_SHA256,
            "case_count": 2,
            "dataset_seed_count": 2,
            "generation_protocol_sha256": recovery.SOURCE_PROTOCOL_SHA256,
            "postcollection_recovery_protocol_sha256": (
                SOURCE_RECOVERY_PROTOCOL_SHA256
            ),
            "execution_monitoring_evidence_complete": False,
            "monitoring_limitation_code": (
                "runner_samples_lost_before_shard_report_write"
            ),
        },
        "reused_baseline_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in BASELINE_ARTIFACTS.items()
        },
        "baseline_audit_binding": {
            "evaluation_adapter_sha256": (
                BASELINE_EVALUATION_ADAPTER_SHA256
            ),
            "independent_audit_adapter_sha256": (
                BASELINE_AUDIT_ADAPTER_SHA256
            ),
            "baseline_cases_recomputed": False,
            "frozen_metrics_reused": True,
        },
        "source_evaluator": {
            "contract_version": SOURCE_EVALUATOR_VERSION,
            "sha256": SOURCE_EVALUATOR_SHA256,
            "delegated_functions": list(DELEGATED_SOURCE_FUNCTIONS),
            "candidate_terminal_metric_arithmetic_modified": False,
            "candidate_checkpoint_metric_arithmetic_modified": False,
            "target_bins_modified": False,
            "query_groups_modified": False,
            "ratio_zero_denominator_rule_modified": False,
            "csv_schema_modified": False,
            "atomic_publication_modified": False,
        },
        "comparison_and_decision": {
            "candidate_arm": CANDIDATE_ARM,
            "baseline_arms": ["gap_legacy_s8", "gap_bounded_r8_s8"],
            "screen_gates": scientific.frozen_protocol_manifest()[
                "screen_gates"
            ],
            "classifier": (
                "scientific_protocol.classify_screen"
            ),
            "checkpoint_selection_allowed": False,
            "posthoc_gate_changes_allowed": False,
        },
        "output": {
            "directory": str(OUTPUT_DIR),
            "evaluation_report": EVALUATION_REPORT,
            "screen_metrics_csv": L1_RESULTS_CSV,
            "candidate_cases_in_csv": 2,
            "baseline_csv_reused_by_hash": True,
            "existing_output_overwrite_allowed": False,
            "atomic_source_publication_reused": True,
        },
        "result_before_information_boundary": {
            "raw_reference_access_allowed": False,
            "candidate_terminal_or_checkpoint_evaluation_allowed": False,
            "baseline_quality_value_access_allowed": False,
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
            "automatic_fresh_seed_confirmation_allowed": False,
            "automatic_parameter_search_allowed": False,
        },
    }


def adapter_sha256() -> str:
    return canonical_sha256(frozen_adapter_manifest())


def assert_frozen_adapter_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if recovery.FROZEN_RECOVERY_SHA256 != SOURCE_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError("平方根评价继承的恢复协议身份漂移")
    recovery.assert_frozen_recovery_identity(root)
    scientific.assert_frozen_protocol_identity(root)
    if file_sha256(root / ADAPTER_DOC) != ADAPTER_DOC_SHA256:
        raise RuntimeError("平方根评价适配文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"平方根评价实现源码漂移：{name}")
    if file_sha256(root / SOURCE_COLLECTION_PATH) != SOURCE_COLLECTION_SHA256:
        raise RuntimeError("平方根评价绑定的 collection 漂移")
    observed = adapter_sha256()
    if observed != FROZEN_ADAPTER_SHA256:
        raise RuntimeError(
            "平方根评价适配清单漂移："
            f"expected={FROZEN_ADAPTER_SHA256}, observed={observed}"
        )
    return observed


def require_evaluation_confirmation(
    confirmed_adapter_sha256: str | None,
    confirmed_collection_sha256: str | None,
) -> None:
    if confirmed_adapter_sha256 != FROZEN_ADAPTER_SHA256:
        raise PermissionError("平方根评价需要确认完整适配协议 SHA-256")
    if confirmed_collection_sha256 != SOURCE_COLLECTION_SHA256:
        raise PermissionError("平方根评价需要确认冻结 collection SHA-256")
