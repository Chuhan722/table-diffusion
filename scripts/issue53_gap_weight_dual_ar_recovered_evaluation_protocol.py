"""Issue #53 A/R 双通道恢复 collection 的离线评价适配协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as source
from scripts import issue53_gap_weight_dual_ar_screen_protocol as scientific
from scripts import issue53_gap_weight_dual_ar_screen_recovery_protocol as recovery
from scripts import issue53_gap_weight_r8_screen_execution_protocol as baseline


ADAPTER_VERSION = "issue53-gap-weight-dual-ar-recovered-evaluation-adapter-v1"
ADAPTER_DOC = Path(
    "docs/设计/Issue53_BC问题一AR双通道恢复版离线评价适配协议.md"
)
ADAPTER_DOC_SHA256 = (
    "ed2b0d519c0fc254589f0e0599f7b87075dcfb61e3f1b272f3b5ed09b46d43de"
)
FROZEN_ADAPTER_SHA256 = (
    "ebb1d6726b99f0f486934cb3c8b7bff8c561d594d02895ede68ed711dc422cb4"
)

SOURCE_COLLECTION_SHA256 = (
    "3b53fcac678924205a4bd1038ec15f404b5245b09894d2b065d28794e615a790"
)
SOURCE_COLLECTION_PATH = source.OUTPUT_DIR / source.COLLECTION_REPORT
SOURCE_RECOVERY_PROTOCOL_SHA256 = recovery.FROZEN_RECOVERY_SHA256
SOURCE_EVALUATOR_VERSION = "issue53-gap-weight-r8-paired-screen-evaluation-v1"
SOURCE_EVALUATOR_SHA256 = (
    "287779a7e66466d3ed2e04862d2aded7ce3e8f084277de72893600e71c6f7445"
)

BASELINE_ARTIFACTS = scientific.BASELINE_ARTIFACTS
R8_BASELINE = BASELINE_ARTIFACTS["legacy_and_r8"]
SQRT_BASELINE = BASELINE_ARTIFACTS["sqrt"]
R8_EVALUATION_ADAPTER_SHA256 = (
    "35d93e9e77c5d3e28085bc20d76852bec63eb128b21f8c64d316675e609e4744"
)
R8_AUDIT_ADAPTER_SHA256 = (
    "0b1393a3f504e123028a8c6c08d15f51c7e2802530caf48f4a7be9e2c33daed1"
)
SQRT_EVALUATION_ADAPTER_SHA256 = (
    "1c0ec85b711356a98cd8540af8c8b5deedd7820b5a8f76f0e5a6f55f5101489d"
)
SQRT_AUDIT_ADAPTER_SHA256 = (
    "5b7d2425938e7dfc2c8480dc3dc15f7396c498d47a2fcb901c080110ca6c2400"
)

EVALUATION_REPORT = "evaluation_report.json"
L1_RESULTS_CSV = "screen_metrics.csv"

# 冻结 R8 evaluator 的运行时兼容面。
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
DUAL_WEIGHTING = scientific.DUAL_WEIGHTING
TEST_GROUP_ORDER = baseline.TEST_GROUP_ORDER
NLTCS_GROUP_COUNTS = baseline.NLTCS_GROUP_COUNTS
NLTCS_COMMON_BIN = scientific.NLTCS_COMMON_BIN
NLTCS_RARE_BIN = scientific.NLTCS_RARE_BIN

IMPLEMENTATION_SOURCES = {
    "scientific_protocol": {
        "path": Path("scripts/issue53_gap_weight_dual_ar_screen_protocol.py"),
        "sha256": "54f0833c9e53ff5b3e42485805f039b95a67f88dedb69fc5b3aa5aa4ec0a4810",
    },
    "candidate_recovery_protocol": {
        "path": Path("scripts/issue53_gap_weight_dual_ar_screen_recovery_protocol.py"),
        "sha256": "530f85732fa3916ef9af736a7ebb175c50ab5158ce79161b7dc9b3401bb61729",
    },
    "candidate_recovery_loader": {
        "path": Path("scripts/recover_issue53_gap_weight_dual_ar_screen.py"),
        "sha256": "5b56588f27d41b34a69a4d17fbd6dc9b61204bef0922e565b1715cbb199dde5c",
    },
    "source_evaluator": {
        "path": Path("scripts/evaluate_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_EVALUATOR_SHA256,
    },
    "r8_baseline_loader": {
        "path": Path("scripts/evaluate_issue53_gap_weight_sqrt_screen_recovered.py"),
        "sha256": "17c522c9ad37b0199c0ca2b4f540279cb5062df34112b19279fcebfe00177a59",
    },
    "evaluation_adapter": {
        "path": Path("scripts/evaluate_issue53_gap_weight_dual_ar_screen_recovered.py"),
        "sha256": "5a54f9662d36f77a71b03d3e0c8c6a3744ab5da2182a987a5885ef4cfd56311d",
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


def _flatten_baselines() -> dict[str, dict[str, Any]]:
    return {
        f"{family}_{name}": item
        for family, artifacts in BASELINE_ARTIFACTS.items()
        for name, item in artifacts.items()
    }


def frozen_adapter_manifest() -> dict[str, Any]:
    return {
        "contract_version": ADAPTER_VERSION,
        "issue": 53,
        "stage": "dual_ar_recovered_collection_offline_evaluation",
        "document": {"path": str(ADAPTER_DOC), "sha256": ADAPTER_DOC_SHA256},
        "candidate_collection": {
            "path": str(SOURCE_COLLECTION_PATH),
            "sha256": SOURCE_COLLECTION_SHA256,
            "case_count": 2,
            "dataset_seed_count": 2,
            "generation_protocol_sha256": recovery.SOURCE_PROTOCOL_SHA256,
            "postcollection_recovery_protocol_sha256": (
                SOURCE_RECOVERY_PROTOCOL_SHA256
            ),
            "source_case_files_rewritten": False,
            "generator_reinvoked_by_recovery": False,
        },
        "reused_baseline_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in _flatten_baselines().items()
        },
        "source_evaluator": {
            "contract_version": SOURCE_EVALUATOR_VERSION,
            "sha256": SOURCE_EVALUATOR_SHA256,
            "terminal_metric_arithmetic_modified": False,
            "offline_group_arithmetic_modified": False,
            "target_bins_modified": False,
            "query_groups_modified": False,
            "ratio_zero_denominator_rule_modified": False,
            "csv_schema_modified": False,
            "atomic_publication_modified": False,
            "checkpoint_gap_e_legacy_diagnostic_preserved": True,
            "dual_kernel_identity_source": "recovered_transition_audit",
        },
        "comparison_and_decision": {
            "candidate_arm": CANDIDATE_ARM,
            "baseline_arms": [
                "gap_legacy_s8",
                "gap_bounded_r8_s8",
                "gap_sqrt_target_s8",
            ],
            "screen_gates": scientific.frozen_protocol_manifest()["screen_gates"],
            "classifier": "scientific_protocol.classify_screen",
            "checkpoint_selection_allowed": False,
            "posthoc_gate_changes_allowed": False,
        },
        "output": {
            "directory": str(OUTPUT_DIR),
            "evaluation_report": EVALUATION_REPORT,
            "screen_metrics_csv": L1_RESULTS_CSV,
            "candidate_cases_in_csv": 2,
            "baseline_metrics_reused_by_hash": True,
            "existing_output_overwrite_allowed": False,
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
            "explicit_user_evaluation_authorization_received": True,
            "adapter_sha_confirmation_required": True,
            "collection_sha_confirmation_required": True,
            "automatic_independent_audit_allowed": False,
            "automatic_fresh_seed_confirmation_allowed": False,
            "automatic_parameter_search_allowed": False,
        },
    }


def adapter_sha256() -> str:
    return canonical_sha256(frozen_adapter_manifest())


def assert_frozen_adapter_identity(root: str | Path) -> str:
    repository = Path(root)
    if recovery.FROZEN_RECOVERY_SHA256 != SOURCE_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError("A/R 评价继承的恢复协议身份漂移")
    recovery.assert_frozen_recovery_identity(repository)
    scientific.assert_frozen_protocol_identity(repository)
    if file_sha256(repository / ADAPTER_DOC) != ADAPTER_DOC_SHA256:
        raise RuntimeError("A/R 评价适配文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(repository / item["path"]) != item["sha256"]:
            raise RuntimeError(f"A/R 评价实现源码漂移：{name}")
    if file_sha256(repository / SOURCE_COLLECTION_PATH) != SOURCE_COLLECTION_SHA256:
        raise RuntimeError("A/R 评价绑定的 collection 漂移")
    for name, item in _flatten_baselines().items():
        if file_sha256(repository / item["path"]) != item["sha256"]:
            raise RuntimeError(f"A/R 评价冻结基线漂移：{name}")
    observed = adapter_sha256()
    if observed != FROZEN_ADAPTER_SHA256:
        raise RuntimeError(
            "A/R 评价适配清单漂移："
            f"expected={FROZEN_ADAPTER_SHA256}, observed={observed}"
        )
    return observed


def require_evaluation_confirmation(
    adapter_sha256_value: str | None,
    collection_sha256_value: str | None,
) -> None:
    if (
        adapter_sha256_value != FROZEN_ADAPTER_SHA256
        or collection_sha256_value != SOURCE_COLLECTION_SHA256
    ):
        raise PermissionError("A/R 离线评价需要同时确认适配协议和 collection SHA-256")
