"""A/R 双通道单种子筛查两 case 的采集后计数勘误恢复协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as source


RECOVERY_VERSION = "issue53-gap-weight-dual-ar-postcollection-erratum-recovery-v1"
RECOVERY_DOC = Path(
    "docs/设计/Issue53_BC问题一AR双通道筛查采集后勘误恢复协议.md"
)
RECOVERY_DOC_SHA256 = (
    "7090c949ddd21c1282ffcdca489a185ea2445ff4263f504f75bef89634990da5"
)
FROZEN_RECOVERY_SHA256 = (
    "2132a4392740114398064b62bfc604cb97b3226e4b758d654dbd6a71dcc8c4b3"
)

SOURCE_PROTOCOL_SHA256 = (
    "4ccf7bbe953d2523fca76d6a7e0ef1410e8773ebdb9a81fd50f8ed8f230b9386"
)
SOURCE_GENERATION_COMMIT = "a3ba71fa2e84512fc6c8ba1bc818ba03e908cc71"
SOURCE_RUNNER_SHA256 = (
    "f446def6c9a1d5cc3022cf9a7e7ead926a220047df9903b4cc1e339dafa64f3a"
)
SOURCE_STAGING_BASENAME = ".local_rtx4090.staging-m3bnqwn9"
SOURCE_STAGING_MANIFEST_SHA256 = (
    "de7f74ff717ad58b19f0836c81e575fe6f79faab796d8980854c9afdff8c6cd6"
)

ARTIFACT_INVENTORY = {
    "seed_9908__gap_dual_abs_relative_max_s8__test_300x10": {
        "case_manifest.json": {
            "size": 5920,
            "sha256": "60f26ec2be66d94718e089fd0a665aeec58fcd5853f829ce828a8eb6fcefbde2",
        },
        "terminal_current.csv": {
            "size": 18658,
            "sha256": "ce51ac10c5ccdcae2c5cf208ef8a9690777f33f24c9bc904b9b03f9149fa2086",
        },
        "checkpoint_query_answers.json": {
            "size": 6109,
            "sha256": "a9b0db3c7c7ccd6c929616f96eebc61c1ca2979d959c86beb3543826acba21aa",
        },
        "transition_audit.json": {
            "size": 2524319,
            "sha256": "2e5f681ca7b3878661df46ed7ea6d1b7b3017e4fcae6125494c90f1cefa52b73",
        },
    },
    "seed_9908__gap_dual_abs_relative_max_s8__nltcs": {
        "case_manifest.json": {
            "size": 5922,
            "sha256": "e0fe120793afb9cffe3f1ac96bf520c138b9b48bc2e273fa4964262208064080",
        },
        "terminal_current.csv": {
            "size": 517911,
            "sha256": "180f1b52083b0b39097b78806282c380d34741560150e933c378c08705391628",
        },
        "checkpoint_query_answers.json": {
            "size": 95674,
            "sha256": "6159d809ad42d34d109e8b7076a9179af81ee152f1fe74577010e52fc3116bc7",
        },
        "transition_audit.json": {
            "size": 4773945,
            "sha256": "4571861429d3002d48ca19d25fa5ee607e7b36a514bb859fc9380472f8451dc5",
        },
    },
}

# 源 runner 未能写 shard 报告；以下仅为监督进程的 nvidia-smi 观察。
SUPERVISING_GPU_SAMPLES = (
    {
        "elapsed": "00:01:30",
        "physical_index": 1,
        "utilization_percent": 94,
        "memory_used_mib": 11407,
    },
    {
        "elapsed": "00:25:00",
        "physical_index": 1,
        "utilization_percent": 69,
        "memory_used_mib": 11407,
    },
    {
        "elapsed": "00:45:00",
        "physical_index": 1,
        "utilization_percent": 70,
        "memory_used_mib": 11407,
    },
    {
        "elapsed": "00:58:22",
        "physical_index": 1,
        "utilization_percent": 70,
        "memory_used_mib": 11407,
    },
    {
        "elapsed": "01:25:00",
        "physical_index": 1,
        "utilization_percent": 70,
        "memory_used_mib": 11407,
    },
    {
        "elapsed": "01:33:00",
        "physical_index": 1,
        "utilization_percent": 65,
        "memory_used_mib": 11407,
    },
)

IMPLEMENTATION_SOURCES = {
    "source_execution_protocol": {
        "path": Path("scripts/issue53_gap_weight_dual_ar_screen_execution_protocol.py"),
        "sha256": "344b336980457118bbfa94f51d7b6c5e67b17d4b501e9803f977a7dbe096be2a",
    },
    "source_collector": {
        "path": Path("scripts/run_issue53_gap_weight_dual_ar_screen.py"),
        "sha256": SOURCE_RUNNER_SHA256,
    },
    "generic_recovery_engine": {
        "path": Path("scripts/recover_issue53_gap_weight_sqrt_screen.py"),
        "sha256": "4f9168887539074fefec380a22a99880914b2919e37f294a201fa6000db12f77",
    },
    "recovery_adapter": {
        "path": Path("scripts/recover_issue53_gap_weight_dual_ar_screen.py"),
        "sha256": "5b56588f27d41b34a69a4d17fbd6dc9b61204bef0922e565b1715cbb199dde5c",
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
        "stage": "dual_ar_postcollection_state_count_erratum",
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
            "limitation_code": "runner_samples_lost_before_shard_report_write",
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
        raise RuntimeError("A/R 恢复源执行协议身份漂移")
    source.assert_frozen_protocol_identity(repository)
    if file_sha256(repository / RECOVERY_DOC) != RECOVERY_DOC_SHA256:
        raise RuntimeError("A/R 恢复协议文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(repository / item["path"]) != item["sha256"]:
            raise RuntimeError(f"A/R 恢复实现源码漂移：{name}")
    observed = recovery_sha256()
    if observed != FROZEN_RECOVERY_SHA256:
        raise RuntimeError(
            "A/R 恢复清单漂移："
            f"expected={FROZEN_RECOVERY_SHA256}, observed={observed}"
        )
    return observed


def require_confirmation(value: str | None) -> None:
    if value != FROZEN_RECOVERY_SHA256:
        raise PermissionError("A/R 结果盲恢复需要确认完整恢复协议 SHA-256")
