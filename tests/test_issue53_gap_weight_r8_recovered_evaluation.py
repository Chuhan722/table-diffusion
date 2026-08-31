from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import evaluate_issue53_gap_weight_r8_screen_recovered as adapter
from scripts import issue53_gap_weight_r8_recovered_evaluation_protocol as protocol
from scripts import issue53_gap_weight_r8_screen_execution_protocol as source_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_recovered_evaluation_protocol_binds_collection_and_source_evaluator():
    assert protocol.file_sha256(
        ROOT / protocol.SOURCE_COLLECTION_PATH
    ) == protocol.SOURCE_COLLECTION_SHA256
    assert protocol.file_sha256(
        ROOT / protocol.IMPLEMENTATION_SOURCES["source_evaluator"]["path"]
    ) == protocol.SOURCE_EVALUATOR_SHA256
    assert (
        protocol.assert_frozen_adapter_identity(ROOT)
        == protocol.FROZEN_ADAPTER_SHA256
    )


def test_plan_is_result_blind_and_does_not_call_loader_or_evaluator(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("结果前 plan 不得读取 collection 或评价 case")

    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter.source_evaluator, "_evaluate_cases", forbidden)
    plan = adapter.build_plan()

    assert plan["evaluation_started"] is False
    assert plan["raw_reference_data_accessed"] is False
    assert plan["new_generation_allowed"] is False
    assert plan["source_metric_arithmetic_modified"] is False
    assert plan["source_classification_modified"] is False
    assert plan["collection_report_sha256"] == protocol.SOURCE_COLLECTION_SHA256


def test_confirmation_gate_fails_before_collection_or_reference_access(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("错误确认不得继续访问")

    monkeypatch.setattr(protocol, "assert_frozen_adapter_identity", forbidden)
    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter.source_evaluator, "_evaluate_cases", forbidden)
    with pytest.raises(PermissionError, match="适配协议"):
        adapter.evaluate("wrong", protocol.SOURCE_COLLECTION_SHA256)
    with pytest.raises(PermissionError, match="collection"):
        adapter.evaluate(protocol.FROZEN_ADAPTER_SHA256, "wrong")


def test_adapter_defines_no_metric_ratio_classification_or_csv_arithmetic():
    path = ROOT / "scripts/evaluate_issue53_gap_weight_r8_screen_recovered.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(
        {
            "_integer_vectors",
            "_vector_metrics",
            "_evaluate_cases",
            "_ratio_record",
            "_decision_ratio",
            "_comparison",
            "_summary_and_decision",
            "_csv_rows",
            "_write_csv",
            "_publish",
        }
    )
    assert adapter.source_evaluator._evaluate_cases is source_evaluator._evaluate_cases
    assert (
        adapter.source_evaluator._summary_and_decision
        is source_evaluator._summary_and_decision
    )
    assert adapter.source_evaluator._csv_rows is source_evaluator._csv_rows
    assert adapter.source_evaluator._publish is source_evaluator._publish


def test_result_blind_stub_wiring_delegates_to_all_frozen_source_functions(
    monkeypatch,
):
    calls: list[str] = []
    collection = {
        "execution_commit": "b" * 40,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "postcollection_recovery": {"recovery_commit": "c" * 40},
    }
    indexed = {("stub",): {"structural": True}}
    cases = [{"case": index} for index in range(4)]
    summary = {
        "datasets": {"stub": True},
        "execution_valid": True,
        "quality_interpretation_allowed": True,
        "frozen_classification": "stub_classification",
        "classification_precedence": ["stub"],
    }
    published: dict[str, object] = {}

    monkeypatch.setattr(
        protocol,
        "assert_frozen_adapter_identity",
        lambda _root: protocol.FROZEN_ADAPTER_SHA256,
    )

    def git_text(_root, *arguments):
        return "" if arguments[0] == "status" else "d" * 40

    monkeypatch.setattr(adapter.recovery_collection, "_git_text", git_text)

    def load_collection(_root, confirmed):
        assert confirmed == protocol.SOURCE_COLLECTION_SHA256
        calls.append("load_collection")
        return collection, indexed

    def evaluate_cases(_root, observed):
        assert observed is indexed
        calls.append("_evaluate_cases")
        return cases, {"stub_identity": True}

    def summarize(observed):
        assert observed is cases
        calls.append("_summary_and_decision")
        return summary

    def csv_rows(observed):
        assert observed is cases
        calls.append("_csv_rows")
        return [{"stub_csv": True}]

    def publish(destination, report, rows):
        calls.append("_publish")
        published.update(destination=destination, report=report, rows=rows)
        return destination / "evaluation_report.json", destination / "screen_metrics.csv"

    monkeypatch.setattr(adapter.recovery_collection, "load_collection", load_collection)
    monkeypatch.setattr(adapter.source_evaluator, "_evaluate_cases", evaluate_cases)
    monkeypatch.setattr(
        adapter.source_evaluator,
        "_summary_and_decision",
        summarize,
    )
    monkeypatch.setattr(adapter.source_evaluator, "_csv_rows", csv_rows)
    monkeypatch.setattr(adapter.source_evaluator, "_publish", publish)

    report_path, csv_path = adapter.evaluate(
        protocol.FROZEN_ADAPTER_SHA256,
        protocol.SOURCE_COLLECTION_SHA256,
    )

    assert calls == [
        "load_collection",
        "_evaluate_cases",
        "_summary_and_decision",
        "_csv_rows",
        "_publish",
    ]
    assert report_path.name == source_protocol.EVALUATION_REPORT
    assert csv_path.name == source_protocol.L1_RESULTS_CSV
    report = published["report"]
    assert report["source_metric_arithmetic_modified"] is False
    assert report["frozen_classification"] == "stub_classification"
    assert report["collection_execution_monitoring_evidence_complete"] is False
    assert report["new_generation_performed_by_evaluator"] is False
    assert published["rows"] == [{"stub_csv": True}]


def test_real_evaluation_outputs_do_not_exist_during_adapter_freeze():
    destination = ROOT / source_protocol.OUTPUT_DIR
    assert not (destination / source_protocol.EVALUATION_REPORT).exists()
    assert not (destination / source_protocol.L1_RESULTS_CSV).exists()
