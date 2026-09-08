"""R8 单种子筛查四 case 采集后勘误恢复协议。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import issue53_gap_weight_r8_screen_execution_protocol as source


RECOVERY_VERSION = "issue53-gap-weight-r8-postcollection-erratum-recovery-v1"
RECOVERY_DOC = Path(
    "docs/设计/Issue53_BC问题一R8筛查采集后勘误恢复协议.md"
)
RECOVERY_DOC_SHA256 = (
    "659c81076058fbd7515bff6326079e659560e7fbca20463ff2f5c260e5680f8f"
)
FROZEN_RECOVERY_SHA256 = (
    "252f7bb9dd018a258e4eafe40a9b84a0afb7846e1bb3c1a30e22138cbb8290d0"
)

SOURCE_PROTOCOL_SHA256 = (
    "8c662f6d0568525b62a6002f3be4913a3334b432bde91faaa70d5b58e6606a4a"
)
SOURCE_GENERATION_COMMIT = "3996442c564b4b4125130aeac0f85c5ca050d256"
SOURCE_RUNNER_SHA256 = (
    "3a3151a09043bbcaa0684a18b0abc6b41325c2e3d59e3328d18912fcce6bb756"
)
SOURCE_STAGING_BASENAME = ".local_rtx4090.staging-sbjg5gqk"
SOURCE_STAGING_MANIFEST_SHA256 = (
    "72bbfe6773708f69d34962016c94218a592b98a0cd8d7e3e693a9c9fcd63b64e"
)

ARTIFACT_INVENTORY = {
    "seed_9908__gap_legacy_s8__test_300x10": {
        "case_manifest.json": {
            "size": 5365,
            "sha256": "b9289a187405629bf166f987627ef397f838cb223b73507f8e0a42c0f15c30d9",
        },
        "terminal_current.csv": {
            "size": 18356,
            "sha256": "f0f3d7fbbc91553e50dd0e38f66127169c788609e237529da3322c100f972851",
        },
        "checkpoint_query_answers.json": {
            "size": 6088,
            "sha256": "aca7aca3eaf1bac669ceeb6427f48a0af8e896b88f487485e565341ec96cd21d",
        },
        "transition_audit.json": {
            "size": 1689158,
            "sha256": "4ebdd941d06e40a3301a2d930afc1e360938b2280dd46baee1a46a539909b079",
        },
    },
    "seed_9908__gap_bounded_r8_s8__test_300x10": {
        "case_manifest.json": {
            "size": 5411,
            "sha256": "1f34f50abb2337cbd0406c8d9c4b3bba8f77c2fbd6060bdbcac6ecb8e496a23f",
        },
        "terminal_current.csv": {
            "size": 18729,
            "sha256": "a076c4f8e3f760660018004b795890a5334c922d93bed9bf39cf5763a3937141",
        },
        "checkpoint_query_answers.json": {
            "size": 6110,
            "sha256": "98f98756525bca1dccdac3545b5f2883abc6c2998069d74502a395f92dfa5230",
        },
        "transition_audit.json": {
            "size": 1655643,
            "sha256": "94b4e00dc52b2b573d7fc5b69f985f47393fd0ca25f10d893e7f292e8293dab1",
        },
    },
    "seed_9908__gap_legacy_s8__nltcs": {
        "case_manifest.json": {
            "size": 5356,
            "sha256": "63fbef561ed65b5d28d5a9c7552c4b1d5a03ffde37d84ccc09feb8ec26e8cc33",
        },
        "terminal_current.csv": {
            "size": 517911,
            "sha256": "694d00970c62d23ae0fff5ecf13f11689ca27794277422cd744141ca1193d4d5",
        },
        "checkpoint_query_answers.json": {
            "size": 95618,
            "sha256": "b4135158a9b1aeeca88c7f0ff6468e570e4f67a5b46db338f4531aaa99757a34",
        },
        "transition_audit.json": {
            "size": 3076421,
            "sha256": "6ae6d43eb05f679cdbc717fde23111dbdb6a19ee049e6244e4933a837ce5e075",
        },
    },
    "seed_9908__gap_bounded_r8_s8__nltcs": {
        "case_manifest.json": {
            "size": 5413,
            "sha256": "1a032643e05a1bc398662fdb12d939491ed139d728f0d74746d97a860ffd8b81",
        },
        "terminal_current.csv": {
            "size": 517911,
            "sha256": "aa47309b0dbe92b212144d7b11e7c6265f3398411e72d2a735bd5c93673575d6",
        },
        "checkpoint_query_answers.json": {
            "size": 81841,
            "sha256": "de1e47667b63b7893baec4005b87508adfd73f4c7ba8ba64860e00ab3123406a",
        },
        "transition_audit.json": {
            "size": 1798192,
            "sha256": "57ee920e11ab91a57c0916a5360670d564a98bba25627ee2b9660719e7b43573",
        },
    },
}

SUPERVISING_GPU_SAMPLES = (
    {
        "elapsed": "00:10:45",
        "physical_index": 1,
        "utilization_percent": 100,
        "memory_used_mib": 21805,
        "temperature_c": 54,
    },
    {
        "elapsed": "00:18:57",
        "physical_index": 1,
        "utilization_percent": 87,
        "memory_used_mib": 21805,
        "temperature_c": 57,
    },
    {
        "elapsed": "00:25:47",
        "physical_index": 1,
        "utilization_percent": 98,
        "memory_used_mib": 21805,
        "temperature_c": 58,
    },
    {
        "elapsed": "00:36:01",
        "physical_index": 1,
        "utilization_percent": 96,
        "memory_used_mib": 21805,
        "temperature_c": 53,
    },
    {
        "elapsed": "00:42:52",
        "physical_index": 1,
        "utilization_percent": 96,
        "memory_used_mib": 21805,
        "temperature_c": 55,
    },
    {
        "elapsed": "00:49:56",
        "physical_index": 1,
        "utilization_percent": 42,
        "memory_used_mib": 21805,
        "temperature_c": 55,
    },
)

IMPLEMENTATION_SOURCES = {
    "source_execution_protocol": {
        "path": Path("scripts/issue53_gap_weight_r8_screen_execution_protocol.py"),
        "sha256": "d0cde17c11af422f437bc872ef25e51dc39e0882c8eb610ab15c771d55e59de1",
    },
    "source_collector": {
        "path": Path("scripts/run_issue53_gap_weight_r8_screen.py"),
        "sha256": SOURCE_RUNNER_SHA256,
    },
    "recovery_collector": {
        "path": Path("scripts/recover_issue53_gap_weight_r8_screen.py"),
        "sha256": "d11b41d3d7f3f60678ca51e71f632f4df4c58923c20e7b64a3c5fa2f0e50e1b8",
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
        "stage": "r8_postcollection_state_evaluation_count_erratum",
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
        raise RuntimeError("R8 恢复源执行协议身份漂移")
    source.assert_frozen_protocol_identity(repository)
    if file_sha256(repository / RECOVERY_DOC) != RECOVERY_DOC_SHA256:
        raise RuntimeError("R8 恢复协议文档漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(repository / item["path"]) != item["sha256"]:
            raise RuntimeError(f"R8 恢复实现源码漂移：{name}")
    observed = recovery_sha256()
    if observed != FROZEN_RECOVERY_SHA256:
        raise RuntimeError(
            "R8 恢复清单漂移："
            f"expected={FROZEN_RECOVERY_SHA256}, observed={observed}"
        )
    return observed


def require_confirmation(value: str | None) -> None:
    if value != FROZEN_RECOVERY_SHA256:
        raise PermissionError("R8 结果盲恢复需要确认完整恢复协议 SHA-256")
