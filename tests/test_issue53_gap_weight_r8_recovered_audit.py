from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import audit_issue53_gap_weight_r8_screen as source_auditor
from scripts import audit_issue53_gap_weight_r8_screen_recovered as adapter
from scripts import issue53_gap_weight_r8_recovered_audit_protocol as protocol
from scripts import issue53_gap_weight_r8_recovered_evaluation_protocol as evaluation
from scripts import issue53_gap_weight_r8_screen_execution_protocol as source_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_recovered_audit_protocol_binds_all_three_published_artifacts():
    for path, expected in (
        (evaluation.SOURCE_COLLECTION_PATH, protocol.SOURCE_COLLECTION_SHA256),
        (protocol.SOURCE_EVALUATION_PATH, protocol.SOURCE_EVALUATION_SHA256),
        (protocol.SOURCE_METRICS_CSV_PATH, protocol.SOURCE_METRICS_CSV_SHA256),
    ):
        assert protocol.file_sha256(ROOT / path) == expected
    assert (
        protocol.assert_frozen_audit_adapter_identity(ROOT)
        == protocol.FROZEN_AUDIT_ADAPTER_SHA256
    )


def test_audit_plan_does_not_load_collection_or_recompute_reference(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("独立审计 plan 不得读取 collection 或参考表")

    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter.source_auditor, "_recompute_cases", forbidden)
    plan = adapter.build_plan()

    assert plan["audit_started"] is False
    assert plan["raw_reference_data_accessed"] is False
    assert plan["rerun_generation"] is False
    assert plan["evaluation_artifacts_modified"] is False
    assert plan["imports_primary_r8_evaluator_arithmetic"] is False


def test_audit_confirmation_gate_fails_before_artifact_or_reference_access(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("错误确认不得继续访问")

    monkeypatch.setattr(
        protocol,
        "assert_frozen_audit_adapter_identity",
        forbidden,
    )
    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter.source_auditor, "_recompute_cases", forbidden)
    with pytest.raises(PermissionError, match="适配协议"):
        adapter.audit(
            "wrong",
            protocol.SOURCE_COLLECTION_SHA256,
            protocol.SOURCE_EVALUATION_SHA256,
        )
    with pytest.raises(PermissionError, match="collection"):
        adapter.audit(
            protocol.FROZEN_AUDIT_ADAPTER_SHA256,
            "wrong",
            protocol.SOURCE_EVALUATION_SHA256,
        )
    with pytest.raises(PermissionError, match="evaluation"):
        adapter.audit(
            protocol.FROZEN_AUDIT_ADAPTER_SHA256,
            protocol.SOURCE_COLLECTION_SHA256,
            "wrong",
        )


def test_adapter_imports_no_primary_evaluator_and_defines_no_metric_arithmetic():
    path = ROOT / "scripts/audit_issue53_gap_weight_r8_screen_recovered.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "evaluate_issue53_gap_weight_r8_screen" not in imported_modules
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(
        {
            "_independent_vector_metrics",
            "_checkpoint_independent",
            "_recompute_cases",
            "_independent_ratio",
            "_summary_independent",
            "_csv_rows_independent",
            "_assert_equal",
            "_audit_csv",
        }
    )
    assert adapter.source_auditor._recompute_cases is source_auditor._recompute_cases
    assert (
        adapter.source_auditor._summary_independent
        is source_auditor._summary_independent
    )
    assert (
        adapter.source_auditor._csv_rows_independent
        is source_auditor._csv_rows_independent
    )
    assert adapter.source_auditor._assert_equal is source_auditor._assert_equal
    assert adapter.source_auditor._audit_csv is source_auditor._audit_csv


def test_stub_audit_orchestration_uses_recovery_then_independent_auditor(
    monkeypatch,
):
    calls: list[str] = []
    collection = {"structural": True}
    indexed = {("stub",): {"row": True}}
    cases = [{"case": index} for index in range(4)]
    identities = {"identity": True}
    evaluation_report = {"evaluation_commit": "e" * 40}
    evidence = {"pass": True, "verified_frozen_classification": "stub"}

    monkeypatch.setattr(
        protocol,
        "assert_frozen_audit_adapter_identity",
        lambda _root: protocol.FROZEN_AUDIT_ADAPTER_SHA256,
    )

    def git_text(_root, *arguments):
        return "" if arguments[0] == "status" else "f" * 40

    monkeypatch.setattr(adapter.recovery_collection, "_git_text", git_text)

    def load_collection(_root, confirmed):
        assert confirmed == protocol.SOURCE_COLLECTION_SHA256
        calls.append("load_collection")
        return collection, indexed

    def recompute(_root, observed):
        assert observed is indexed
        calls.append("_recompute_cases")
        return cases, identities

    def audit_evaluation(*args):
        assert args[3] is collection
        assert args[4] is cases
        assert args[5] is identities
        calls.append("_audit_recovered_evaluation")
        return evaluation_report, evidence

    def publish(_root, payload):
        calls.append("_publish_audit")
        assert payload["evaluator_arithmetic_imported"] is False
        assert payload["generation_rerun"] is False
        assert payload["pass"] is True
        return ROOT / source_protocol.OUTPUT_DIR / source_protocol.AUDIT_REPORT

    monkeypatch.setattr(adapter.recovery_collection, "load_collection", load_collection)
    monkeypatch.setattr(adapter.source_auditor, "_recompute_cases", recompute)
    monkeypatch.setattr(adapter, "_audit_recovered_evaluation", audit_evaluation)
    monkeypatch.setattr(adapter, "_publish_audit", publish)

    result = adapter.audit(
        protocol.FROZEN_AUDIT_ADAPTER_SHA256,
        protocol.SOURCE_COLLECTION_SHA256,
        protocol.SOURCE_EVALUATION_SHA256,
    )

    assert result.name == source_protocol.AUDIT_REPORT
    assert calls == [
        "load_collection",
        "_recompute_cases",
        "_audit_recovered_evaluation",
        "_publish_audit",
    ]


def test_stub_evaluation_audit_compares_cases_summary_classification_and_csv(
    monkeypatch,
):
    collection = {
        "execution_commit": "a" * 40,
        "postcollection_recovery": {"recovery_commit": "b" * 40},
    }
    cases = [{"case": True}]
    identities = {"identity": True}
    summary = {"summary": True}
    precedence = source_protocol.scientific.frozen_protocol_manifest()[
        "screen_decision"
    ]["precedence"]
    report = {
        "contract_version": evaluation.ADAPTER_VERSION,
        "evaluation_adapter_protocol_sha256": evaluation.FROZEN_ADAPTER_SHA256,
        "source_evaluation_contract_version": evaluation.SOURCE_EVALUATOR_VERSION,
        "source_evaluator_sha256": evaluation.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": (
            evaluation.SOURCE_RECOVERY_PROTOCOL_SHA256
        ),
        "collection_report_sha256": protocol.SOURCE_COLLECTION_SHA256,
        "collection_generation_commit": "a" * 40,
        "collection_recovery_commit": "b" * 40,
        "collection_recovered_after_generation": True,
        "collection_execution_monitoring_evidence_complete": False,
        "collection_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "case_count": 4,
        "evaluation_started": True,
        "generation_started": False,
        "source_metric_arithmetic_modified": False,
        "source_classification_modified": False,
        "raw_reference_data_accessed": True,
        "all_four_cases_evaluated": True,
        "terminal_and_reached_checkpoints_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "automatic_followup_authorized": False,
        "recovered_evaluator_sha256": evaluation.IMPLEMENTATION_SOURCES[
            "recovered_evaluation_adapter"
        ]["sha256"],
        "delegated_source_functions": list(evaluation.DELEGATED_SOURCE_FUNCTIONS),
        "evaluation_commit": protocol.SOURCE_EVALUATION_COMMIT,
        "query_and_reference_identity_audit": identities,
        "cases": cases,
        "summary": summary,
        "execution_valid": True,
        "quality_interpretation_allowed": True,
        "frozen_classification": "stub_classification",
        "classification_precedence": precedence,
    }
    calls: list[str] = []
    monkeypatch.setattr(
        protocol,
        "file_sha256",
        lambda _path: protocol.SOURCE_EVALUATION_SHA256,
    )
    monkeypatch.setattr(adapter.source_auditor, "_load_json", lambda _path: report)
    monkeypatch.setattr(adapter, "_validate_commit_ancestry", lambda *_args: None)

    def assert_equal(left, right, path):
        calls.append(path)
        assert left == right

    monkeypatch.setattr(adapter.source_auditor, "_assert_equal", assert_equal)
    monkeypatch.setattr(
        adapter.source_auditor,
        "_summary_independent",
        lambda observed: (summary, True, "stub_classification"),
    )
    monkeypatch.setattr(
        adapter.source_auditor,
        "_csv_rows_independent",
        lambda observed: [{"csv": True}],
    )
    monkeypatch.setattr(
        adapter.source_auditor,
        "_audit_csv",
        lambda _root, observed, rows: {"rows_exact": observed is report and bool(rows)},
    )

    observed, evidence = adapter._audit_recovered_evaluation(
        ROOT,
        protocol.SOURCE_EVALUATION_SHA256,
        protocol.SOURCE_COLLECTION_SHA256,
        collection,
        cases,
        identities,
    )

    assert observed is report
    assert calls == ["cases", "summary"]
    assert evidence["pass"] is True
    assert evidence["classification_independently_reproduced"] is True
    assert evidence["verified_frozen_classification"] == "stub_classification"
    assert evidence["screen_metrics_csv"]["rows_exact"] is True


def test_real_independent_audit_output_absent_before_authorized_run():
    assert not (ROOT / protocol.AUDIT_OUTPUT_PATH).exists()
