from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_issue53_stage6d_v5_recovered_v2 as compat_auditor
from scripts import evaluate_issue53_stage6d_v5_recovered_v2 as compat_evaluator
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_evaluation_compat_protocol as compat_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6d_v5 as recovery_collection
from scripts import validate_issue53_stage6d_v5_recovered_collection as validator


def test_json_roundtrip_changes_only_the_two_frozen_order_count_key_sets():
    original = generation_protocol.frozen_protocol_manifest()
    normalized = validator.normalized_generation_protocol_manifest()

    assert original != normalized
    assert tuple(original["datasets"]["test_300x10"]["order_counts"]) == (
        2,
        3,
        4,
    )
    assert tuple(normalized["datasets"]["test_300x10"]["order_counts"]) == (
        "2",
        "3",
        "4",
    )
    assert tuple(original["datasets"]["nltcs"]["order_counts"]) == (2, 3)
    assert tuple(normalized["datasets"]["nltcs"]["order_counts"]) == (
        "2",
        "3",
    )
    assert recovery_protocol.canonical_sha256(normalized) == (
        generation_protocol.FROZEN_PROTOCOL_SHA256
    )


def test_temporary_manifest_adapter_restores_function_after_failure():
    original_function = generation_protocol.frozen_protocol_manifest

    with (
        pytest.raises(RuntimeError, match="intentional"),
        validator._temporary_normalized_generation_manifest() as normalized,
    ):
        assert generation_protocol.frozen_protocol_manifest() == normalized
        assert generation_protocol.frozen_protocol_manifest is not original_function
        raise RuntimeError("intentional")

    assert generation_protocol.frozen_protocol_manifest is original_function


def test_both_collection_paths_reuse_v1_with_only_temporary_json_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    root = Path("/unused")
    collection_sha = compat_protocol.CONFIRMED_COLLECTION_REPORT_SHA256
    compatibility_sha = "c" * 64
    calls: list[str] = []
    indexed = {(353, "nltcs", "gap_b_s8"): {"task_id": "example"}}

    monkeypatch.setattr(
        compat_protocol, "assert_frozen_compatibility_identity", lambda _root: None
    )
    monkeypatch.setattr(
        compat_protocol, "require_evaluation_confirmation", lambda *_args: None
    )

    def fake_validation(_root: Path, confirmed: str):
        assert confirmed == collection_sha
        embedded = generation_protocol.frozen_protocol_manifest()
        assert tuple(embedded["datasets"]["nltcs"]["order_counts"]) == ("2", "3")
        calls.append("collector")
        return {"source_generation_protocol": embedded}, indexed

    def fake_independent(_root: Path, confirmed: str):
        assert confirmed == collection_sha
        embedded = generation_protocol.frozen_protocol_manifest()
        assert tuple(embedded["datasets"]["test_300x10"]["order_counts"]) == (
            "2",
            "3",
            "4",
        )
        calls.append("auditor")
        return {"source_generation_protocol": embedded}, indexed

    monkeypatch.setattr(
        validator.recovery_collection_v1, "load_collection", fake_validation
    )
    monkeypatch.setattr(
        validator.recovery_auditor_v1,
        "_audit_collection_independently",
        fake_independent,
    )
    original_function = generation_protocol.frozen_protocol_manifest

    assert validator.load_collection(root, collection_sha, compatibility_sha)[1] == (
        indexed
    )
    assert generation_protocol.frozen_protocol_manifest is original_function
    assert (
        validator.audit_collection_independently(
            root, collection_sha, compatibility_sha
        )[1]
        == indexed
    )
    assert generation_protocol.frozen_protocol_manifest is original_function
    assert calls == ["collector", "auditor"]


def test_confirmation_requires_both_exact_frozen_hashes():
    with pytest.raises(PermissionError, match="采集报告"):
        compat_protocol.require_evaluation_confirmation(
            None,
            compat_protocol.FROZEN_COMPATIBILITY_PROTOCOL_SHA256,
        )
    with pytest.raises(PermissionError, match="兼容协议"):
        compat_protocol.require_evaluation_confirmation(
            compat_protocol.CONFIRMED_COLLECTION_REPORT_SHA256,
            None,
        )


def test_evaluator_validates_compatible_collection_before_reference_access(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        compat_protocol, "assert_frozen_compatibility_identity", lambda _root: "ok"
    )
    monkeypatch.setattr(
        compat_protocol, "require_evaluation_confirmation", lambda *_args: None
    )
    monkeypatch.setattr(recovery_collection, "_git_text", lambda *_args: "")

    def fake_load(_root: Path, collection_sha: str, compatibility_sha: str):
        calls.append(("collection", collection_sha, compatibility_sha))
        raise RuntimeError("stop-before-reference")

    monkeypatch.setattr(validator, "load_collection", fake_load)
    monkeypatch.setattr(
        compat_evaluator.recovery_evaluator_v1.legacy_evaluator,
        "_evaluate_cases",
        lambda *_args: pytest.fail("兼容 collection（采集）复核前不得读取参考数据"),
    )

    with pytest.raises(RuntimeError, match="stop-before-reference"):
        compat_evaluator.evaluate("a" * 64, "b" * 64)
    assert calls == [("collection", "a" * 64, "b" * 64)]


def test_auditor_validates_compatible_collection_before_reference_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(compat_auditor, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(compat_protocol, "OUTPUT_DIR", Path("out"))
    monkeypatch.setattr(
        compat_protocol, "assert_frozen_compatibility_identity", lambda _root: "ok"
    )
    monkeypatch.setattr(
        compat_protocol, "require_evaluation_confirmation", lambda *_args: None
    )
    monkeypatch.setattr(recovery_collection, "_git_text", lambda *_args: "")
    monkeypatch.setattr(
        validator,
        "audit_collection_independently",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("stop-before-reference")),
    )
    monkeypatch.setattr(
        compat_auditor.recovery_auditor_v1.legacy_auditor,
        "_recompute_cases",
        lambda *_args: pytest.fail("兼容 collection（采集）复核前不得读取参考数据"),
    )

    with pytest.raises(RuntimeError, match="stop-before-reference"):
        compat_auditor.audit("a" * 64, "b" * 64, "c" * 64)


def test_v2_evaluation_identity_adapter_always_restores_v1_contract():
    source_evaluator = compat_auditor.recovery_auditor_v1.recovery_evaluator
    source_binding = (
        compat_auditor.recovery_auditor_v1.recovery_protocol.IMPLEMENTATION_SOURCES[
            "recovery_evaluator"
        ]
    )
    original_version = source_evaluator.EVALUATION_VERSION
    original_sha256 = source_binding["sha256"]

    with (
        pytest.raises(RuntimeError, match="intentional"),
        compat_auditor._temporary_v2_evaluation_identity(),
    ):
        assert source_evaluator.EVALUATION_VERSION == (
            compat_evaluator.EVALUATION_VERSION
        )
        assert (
            source_binding["sha256"]
            == compat_protocol.IMPLEMENTATION_SOURCES["compatibility_evaluator"][
                "sha256"
            ]
        )
        raise RuntimeError("intentional")

    assert source_evaluator.EVALUATION_VERSION == original_version
    assert source_binding["sha256"] == original_sha256


def test_old_generation_source_fails_closed_and_compat_plans_are_read_only(
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="实现源码漂移：full_generator"):
        compat_protocol.assert_frozen_compatibility_identity(root)

    monkeypatch.setattr(
        compat_protocol,
        "assert_frozen_compatibility_identity",
        lambda _root: compat_protocol.FROZEN_COMPATIBILITY_PROTOCOL_SHA256,
    )
    assert validator.build_plan()["quality_metrics_interpreted"] is False
    assert compat_evaluator.build_plan()["evaluation_started"] is False
    assert compat_evaluator.build_plan()["new_generation_allowed"] is False
    assert compat_auditor.build_plan()["audit_started"] is False
    assert compat_auditor.build_plan()["rerun_generation"] is False
