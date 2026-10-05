#!/usr/bin/env python3
"""严格适配并验证第 6D 第五版恢复 collection（采集）报告的 JSON 键。"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts import audit_issue53_stage6d_v5_recovered as recovery_auditor_v1
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_evaluation_compat_protocol as compat_protocol
from scripts import recover_issue53_stage6d_v5 as recovery_collection_v1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_json_roundtrip(value: Any) -> Any:
    """使用真实 JSON 语义规范对象键，不做自定义递归类型转换。"""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def normalized_generation_protocol_manifest() -> dict[str, Any]:
    original = generation_protocol.frozen_protocol_manifest()
    normalized = canonical_json_roundtrip(original)
    if not isinstance(normalized, dict):
        raise TypeError("JSON 键兼容后的生成协议必须是对象")
    expected_after_only_allowed_changes = copy.deepcopy(original)
    expected_keys = {
        "test_300x10": (2, 3, 4),
        "nltcs": (2, 3),
    }
    for dataset, integer_keys in expected_keys.items():
        counts = expected_after_only_allowed_changes["datasets"][dataset][
            "order_counts"
        ]
        if (
            not isinstance(counts, dict)
            or tuple(sorted(counts)) != integer_keys
            or any(isinstance(key, bool) or not isinstance(key, int) for key in counts)
        ):
            raise RuntimeError(f"{dataset} 原始阶数计数键不再是冻结整数集合")
        expected_after_only_allowed_changes["datasets"][dataset]["order_counts"] = {
            str(key): value for key, value in counts.items()
        }
    if original == normalized:
        raise RuntimeError("预期的 JSON 整数键兼容差异没有出现")
    if expected_after_only_allowed_changes != normalized:
        raise RuntimeError("生成协议包含冻结两处 JSON 键以外的差异")
    observed_sha = recovery_collection_v1.recovery_protocol.canonical_sha256(normalized)
    if observed_sha != compat_protocol.SOURCE_GENERATION_PROTOCOL_SHA256:
        raise RuntimeError("JSON 键兼容后的生成协议 SHA-256 漂移")
    return normalized


@contextmanager
def _temporary_normalized_generation_manifest() -> Iterator[dict[str, Any]]:
    """仅在第一版验证调用期间提供 JSON 读回形态，随后无条件恢复。"""

    original_function = generation_protocol.frozen_protocol_manifest
    normalized = normalized_generation_protocol_manifest()

    def frozen_normalized_manifest() -> dict[str, Any]:
        return copy.deepcopy(normalized)

    generation_protocol.frozen_protocol_manifest = frozen_normalized_manifest
    try:
        yield normalized
    finally:
        generation_protocol.frozen_protocol_manifest = original_function


def _verify_embedded_manifest(
    report: dict[str, Any], normalized: dict[str, Any]
) -> None:
    embedded = report.get("source_generation_protocol")
    if embedded != normalized:
        raise RuntimeError("采集报告内嵌生成协议不等于唯一允许的 JSON 读回形态")
    if (
        recovery_collection_v1.recovery_protocol.canonical_sha256(embedded)
        != compat_protocol.SOURCE_GENERATION_PROTOCOL_SHA256
    ):
        raise RuntimeError("采集报告内嵌生成协议规范 SHA-256 漂移")


def load_collection(
    root: Path,
    confirmed_collection_sha256: str,
    confirmed_compatibility_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """复用恢复第一版采集器的完整验证，只适配冻结的两处 JSON 键。"""

    compat_protocol.assert_frozen_compatibility_identity(root)
    compat_protocol.require_evaluation_confirmation(
        confirmed_collection_sha256, confirmed_compatibility_sha256
    )
    with _temporary_normalized_generation_manifest() as normalized:
        report, indexed = recovery_collection_v1.load_collection(
            root, confirmed_collection_sha256
        )
    _verify_embedded_manifest(report, normalized)
    return report, indexed


def audit_collection_independently(
    root: Path,
    confirmed_collection_sha256: str,
    confirmed_compatibility_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """复用恢复第一版独立采集验证，只适配冻结的两处 JSON 键。"""

    compat_protocol.assert_frozen_compatibility_identity(root)
    compat_protocol.require_evaluation_confirmation(
        confirmed_collection_sha256, confirmed_compatibility_sha256
    )
    with _temporary_normalized_generation_manifest() as normalized:
        report, indexed = recovery_auditor_v1._audit_collection_independently(
            root, confirmed_collection_sha256
        )
    _verify_embedded_manifest(report, normalized)
    return report, indexed


def compatibility_evidence() -> dict[str, Any]:
    normalized = normalized_generation_protocol_manifest()
    return {
        "cause": "json_object_keys_are_strings_after_roundtrip",
        "normalization_primitive": "json_dumps_then_json_loads",
        "normalized_key_paths": list(compat_protocol.NORMALIZED_KEY_PATHS),
        "all_values_unchanged": True,
        "all_other_paths_unchanged": True,
        "normalized_generation_protocol_sha256": (
            recovery_collection_v1.recovery_protocol.canonical_sha256(normalized)
        ),
        "collection_report_rewritten": False,
    }


def build_plan() -> dict[str, Any]:
    plan = compat_protocol.build_plan(_repo_root())
    plan.update(
        {
            "compatibility_validator_wired": True,
            "collection_validation_started": False,
            "quality_metrics_interpreted": False,
        }
    )
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection_v1._strict_json_text(build_plan()), end="")


if __name__ == "__main__":
    main()
