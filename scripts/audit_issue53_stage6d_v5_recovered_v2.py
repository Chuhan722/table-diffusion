#!/usr/bin/env python3
"""独立复核经 JSON 键兼容验证的 Stage 6D（阶段 6D）第五版评价。"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts import audit_issue53_stage6d_v5_recovered as recovery_auditor_v1
from scripts import evaluate_issue53_stage6d_v5_recovered_v2 as compat_evaluator
from scripts import issue53_stage6d_v5_evaluation_compat_protocol as compat_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6d_v5 as recovery_collection
from scripts import validate_issue53_stage6d_v5_recovered_collection as compat_validator

AUDIT_VERSION = "issue53-stage6d-v5-recovered-independent-audit-json-key-compat-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_plan() -> dict[str, Any]:
    compatibility_sha = compat_protocol.assert_frozen_compatibility_identity(
        _repo_root()
    )
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "compatibility_protocol_sha256": compatibility_sha,
        "recovery_protocol_sha256": compat_protocol.SOURCE_RECOVERY_PROTOCOL_SHA256,
        "collection_report": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.COLLECTION_REPORT
        ),
        "collection_report_sha256": (
            compat_protocol.CONFIRMED_COLLECTION_REPORT_SHA256
        ),
        "evaluation_report": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.EVALUATION_REPORT
        ),
        "l1_results_csv": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.L1_RESULTS_CSV
        ),
        "audit_report": str(compat_protocol.OUTPUT_DIR / compat_protocol.AUDIT_REPORT),
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_l1_csv": True,
        "recompute_frozen_classification": True,
        "verify_json_key_compatibility": True,
        "rerun_generation": False,
        "audit_started": False,
    }


@contextmanager
def _temporary_v2_evaluation_identity() -> Iterator[None]:
    """让第一版指标复核器严格核对第二版入口身份，随后恢复模块状态。"""

    source_evaluator = recovery_auditor_v1.recovery_evaluator
    source_binding = recovery_auditor_v1.recovery_protocol.IMPLEMENTATION_SOURCES[
        "recovery_evaluator"
    ]
    original_version = source_evaluator.EVALUATION_VERSION
    original_sha256 = source_binding["sha256"]
    source_evaluator.EVALUATION_VERSION = compat_evaluator.EVALUATION_VERSION
    source_binding["sha256"] = compat_protocol.IMPLEMENTATION_SOURCES[
        "compatibility_evaluator"
    ]["sha256"]
    try:
        yield
    finally:
        source_evaluator.EVALUATION_VERSION = original_version
        source_binding["sha256"] = original_sha256


def _audit_evaluation_v2(
    root: Path,
    confirmed_evaluation_sha: str,
    confirmed_collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> dict[str, Any]:
    """先核对第二版兼容证据，再复用第一版独立指标与 CSV 复核。"""

    path = root / compat_protocol.OUTPUT_DIR / compat_protocol.EVALUATION_REPORT
    report = recovery_auditor_v1._load_json(path)
    expected_compatibility = compat_validator.compatibility_evidence()
    if (
        report.get("compatibility_protocol_sha256")
        != compat_protocol.FROZEN_COMPATIBILITY_PROTOCOL_SHA256
        or report.get("evaluation_compatibility_protocol_sha256")
        != compat_protocol.FROZEN_COMPATIBILITY_PROTOCOL_SHA256
        or report.get("compatibility_validator_sha256")
        != compat_protocol.IMPLEMENTATION_SOURCES["compatibility_validator"]["sha256"]
        or report.get("json_key_compatibility") != expected_compatibility
        or report.get("collection_report_sha256") != confirmed_collection_sha
    ):
        raise RuntimeError("独立复核第二版评价兼容身份或证据漂移")
    with _temporary_v2_evaluation_identity():
        audit = recovery_auditor_v1._audit_evaluation(
            root,
            confirmed_evaluation_sha,
            confirmed_collection_sha,
            collection_report,
            cases,
            identities,
        )
    audit.update(
        {
            "evaluation_json_key_compatibility_exactly_verified": True,
            "compatibility_protocol_sha256": (
                compat_protocol.FROZEN_COMPATIBILITY_PROTOCOL_SHA256
            ),
            "compatibility_validator_sha256": compat_protocol.IMPLEMENTATION_SOURCES[
                "compatibility_validator"
            ]["sha256"],
        }
    )
    return audit


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_compatibility_protocol_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    """独立重算指标，不生成数据、不调参，并原子写入复核报告。"""

    root = _repo_root()
    compat_protocol.assert_frozen_compatibility_identity(root)
    compat_protocol.require_evaluation_confirmation(
        confirmed_collection_report_sha256,
        confirmed_compatibility_protocol_sha256,
    )
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("第二版独立复核要求包含未跟踪文件在内的干净工作树")
    output = root / compat_protocol.OUTPUT_DIR / compat_protocol.AUDIT_REPORT
    if output.exists():
        raise FileExistsError(f"第二版独立复核报告已存在，不覆盖：{output}")
    collection_report, indexed = compat_validator.audit_collection_independently(
        root,
        confirmed_collection_report_sha256,
        confirmed_compatibility_protocol_sha256,
    )
    cases, identities = recovery_auditor_v1.legacy_auditor._recompute_cases(
        root, indexed
    )
    evaluation_audit = _audit_evaluation_v2(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report,
        cases,
        identities,
    )
    report = {
        **build_plan(),
        "mode": "independent_recompute_after_json_key_compatible_collection_validation",
        "audit_started": True,
        "audit_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovery_independent_auditor_sha256": compat_protocol.file_sha256(
            Path(__file__)
        ),
        "evaluation_compatibility_protocol_sha256": (
            confirmed_compatibility_protocol_sha256
        ),
        "compatibility_validator_sha256": compat_protocol.IMPLEMENTATION_SOURCES[
            "compatibility_validator"
        ]["sha256"],
        "json_key_compatibility": compat_validator.compatibility_evidence(),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "evaluation_report_sha256": confirmed_evaluation_report_sha256,
        "collection_generation_commit": collection_report[
            "generation_execution_commit"
        ],
        "collection_recovery_commit": collection_report["recovery_commit"],
        "collection_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "query_and_reference_identity_audit": identities,
        "recomputed_case_count": len(cases),
        "independent_classification": (
            recovery_auditor_v1.legacy_auditor._classification_independent(cases)
        ),
        "evaluation_audit": evaluation_audit,
        "overall_pass": True,
        "new_generation_performed_by_auditor": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "parameter_retuning_performed": False,
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"第二版独立复核临时文件已存在：{temporary}")
    try:
        temporary.write_text(
            recovery_collection._strict_json_text(report), encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--confirm-collection-sha", required=True)
    audit_parser.add_argument("--confirm-compatibility-sha", required=True)
    audit_parser.add_argument("--confirm-evaluation-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    path = audit(
        args.confirm_collection_sha,
        args.confirm_compatibility_sha,
        args.confirm_evaluation_sha,
    )
    print(f"第二版 independent audit（独立复核） -> {path}")
    print(f"复核报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
