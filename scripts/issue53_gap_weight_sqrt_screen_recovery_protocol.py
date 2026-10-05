"""平方根权重单种子筛查两 case 采集后勘误恢复协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_sqrt_screen_execution_protocol as source


RECOVERY_VERSION = (
    "issue53-gap-weight-sqrt-target-postcollection-erratum-recovery-v1"
)
RECOVERY_DOC = Path(
    "docs/设计/Issue53_BC问题一平方根权重筛查采集后勘误恢复协议.md"
)
RECOVERY_DOC_SHA256 = (
    "484c6cfbda8040352d28a9f870d31f0f06b6d05448441f049f55abeb93370173"
)
FROZEN_RECOVERY_SHA256 = (
    "0f6420a0501fbedf792720716b54735d141aafef90261e37560a6a312a4642ac"
)

SOURCE_PROTOCOL_SHA256 = (
    "3330da8dd3bc9fbe264848491414893143f17da13cf42e3438611c24041d359a"
)
SOURCE_GENERATION_COMMIT = "1ad75f9778969b8d0592f02601a0b596dbaa20e5"
SOURCE_RUNNER_SHA256 = (
    "6ad50b81c1c7c7d9043124b50da11b3540cde712d47546a5ce9c809631ff12e4"
)
SOURCE_STAGING_BASENAME = ".local_rtx4090.staging-iencmg3i"
SOURCE_STAGING_MANIFEST_SHA256 = (
    "fe29604c2ab5e6db71577686cd8d4b3d8b02198086cd1279df32196d30cfa07d"
)

ARTIFACT_INVENTORY = {
    "seed_9908__gap_sqrt_target_s8__test_300x10": {
        "case_manifest.json": {
            "size": 5519,
            "sha256": (
                "b028b21047409b1c65977eebf9577aac55dd817ea6df33f935116ee6b870a810"
            ),
        },
        "terminal_current.csv": {
            "size": 18675,
            "sha256": (
                "4540e72e653c75e3d8f3b3ba3f4a631bd86e4e2f6456682f2da76325e309a256"
            ),
        },
        "checkpoint_query_answers.json": {
            "size": 6103,
            "sha256": (
                "47ab1bfa32131f5f540d0030cf13d313986d8d71c8b2382bd972a6cc63358acb"
            ),
        },
        "transition_audit.json": {
            "size": 1970471,
            "sha256": (
                "94119b2ee8f05408374280c28d212700de54ee15bc1dc269a80116d0af01000c"
            ),
        },
    },
    "seed_9908__gap_sqrt_target_s8__nltcs": {
        "case_manifest.json": {
            "size": 5515,
            "sha256": (
                "133ba1defeb99cd249beeac34048165ddc1e78d31c8acacb6efa95545fce4a4f"
            ),
        },
        "terminal_current.csv": {
            "size": 517911,
            "sha256": (
                "cd2aee0149c20501c4b2d0b9e31d73d58541badbce3323db7302e42a2886a1ff"
            ),
        },
        "checkpoint_query_answers.json": {
            "size": 95635,
            "sha256": (
                "21564a0e5d15b8115387d467214de4269a1e8673d97eada26fc8f8c875de2c00"
            ),
        },
        "transition_audit.json": {
            "size": 2790393,
            "sha256": (
                "357136838c3df3b3540178f29b9ab652af12eb9228b67a1f46cb2c70c3efd6d4"
            ),
        },
    },
}

SUPERVISING_GPU_SAMPLES = (
    {
        "elapsed": "00:02:40",
        "physical_index": 1,
        "utilization_percent": 92,
        "memory_used_mib": 11405,
        "temperature_c": 51,
    },
    {
        "elapsed": "00:06:27",
        "physical_index": 1,
        "utilization_percent": 44,
        "memory_used_mib": 11407,
        "temperature_c": 55,
    },
    {
        "elapsed": "00:11:31",
        "physical_index": 1,
        "utilization_percent": 92,
        "memory_used_mib": 11407,
        "temperature_c": 57,
    },
    {
        "elapsed": "00:17:21",
        "physical_index": 1,
        "utilization_percent": 99,
        "memory_used_mib": 11407,
        "temperature_c": 57,
    },
    {
        "elapsed": "00:23:16",
        "physical_index": 1,
        "utilization_percent": 99,
        "memory_used_mib": 11407,
        "temperature_c": 58,
    },
    {
        "elapsed": "00:25:36",
        "physical_index": 1,
        "utilization_percent": 0,
        "memory_used_mib": 11407,
        "temperature_c": 50,
    },
)

IMPLEMENTATION_SOURCES = {
    "source_execution_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_sqrt_screen_execution_protocol.py"
        ),
        "sha256": (
            "934276e55db2c50ab95557bbbbdefce35908cef03452e8e69e54b4cb128c445d"
        ),
    },
    "source_collector": {
        "path": Path("scripts/run_issue53_gap_weight_sqrt_screen.py"),
        "sha256": SOURCE_RUNNER_SHA256,
    },
    "recovery_collector": {
        "path": Path("scripts/recover_issue53_gap_weight_sqrt_screen.py"),
        "sha256": (
            "4f9168887539074fefec380a22a99880914b2919e37f294a201fa6000db12f77"
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


def frozen_recovery_manifest() -> dict[str, Any]:
    return {
        "contract_version": RECOVERY_VERSION,
        "issue": 53,
        "stage": "sqrt_target_postcollection_state_count_erratum",
        "document": {
            "path": str(RECOVERY_DOC),
            "sha256": RECOVERY_DOC_SHA256,
        },
        "source": {
            "protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "generation_commit": SOURCE_GENERATION_COMMIT,
            "runner_sha256": SOURCE_RUNNER_SHA256,
            "staging_basename": SOURCE_STAGING_BASENAME,
            "staging_manifest_sha256": SOURCE_STAGING_MANIFEST_SHA256,
            "artifact_inventory": ARTIFACT_INVENTORY,
        },
        "single_erratum": {
            "rejected_old_assertion": (
                "state_evaluation_count == applied_rounds + 1"
            ),
            "accepted_generator_contract": (
                "state_evaluation_count == max(1, applied_rounds)"
            ),
            "other_validation_changes": [],
        },
        "monitoring_evidence": {
            "source_runner_samples_persisted": False,
            "limitation_code": (
                "runner_samples_lost_before_shard_report_write"
            ),
            "supervising_samples": list(SUPERVISING_GPU_SAMPLES),
            "supervising_samples_are_source_runner_samples": False,
        },
        "information_boundary": {
            "generator_call_allowed": False,
            "case_deletion_or_rerun_allowed": False,
            "raw_reference_access_allowed": False,
            "quality_metric_interpretation_allowed": False,
            "only_structural_reports_published": True,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
    }


def recovery_sha256() -> str:
    return canonical_sha256(frozen_recovery_manifest())


def assert_frozen_recovery_identity(root: str | Path) -> str:
    repository = Path(root)
    if source.FROZEN_PROTOCOL_SHA256 != SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("平方根恢复源执行协议身份漂移")
    source.assert_frozen_protocol_identity(repository)
    if file_sha256(repository / RECOVERY_DOC) != RECOVERY_DOC_SHA256:
        raise RuntimeError("平方根恢复协议文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(repository / item["path"]) != item["sha256"]:
            raise RuntimeError(f"平方根恢复实现源码漂移：{name}")
    observed = recovery_sha256()
    if observed != FROZEN_RECOVERY_SHA256:
        raise RuntimeError(
            "平方根恢复清单漂移："
            f"expected={FROZEN_RECOVERY_SHA256}, observed={observed}"
        )
    return observed


def require_confirmation(value: str | None) -> None:
    if value != FROZEN_RECOVERY_SHA256:
        raise PermissionError("平方根结果盲恢复需要确认完整恢复协议 SHA-256")
