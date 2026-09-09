from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import audit_issue53_gap_weight_r8_screen as source_auditor
from scripts import audit_issue53_gap_weight_sqrt_screen_recovered as adapter
from scripts import issue53_gap_weight_sqrt_recovered_audit_protocol as protocol
from scripts import issue53_gap_weight_sqrt_recovered_evaluation_protocol as evaluation
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific


ROOT = Path(__file__).resolve().parents[1]


def _case(dataset: str, arm: str, *, value: float) -> dict:
    target_bins = {
        item["name"]: {"absolute_count_error_mean": value}
        for item in scientific.TARGET_COUNT_BINS[dataset]
    }
    by_order = {
        str(order): {"absolute_count_error_mean": value}
        for order in scientific.DATASETS[dataset]["order_counts"]
    }
    group_names = (
        evaluation.TEST_GROUP_ORDER
        if dataset == "test_300x10"
        else tuple(evaluation.NLTCS_GROUP_COUNTS)
    )
    return {
        "task_id": f"seed_9908__{arm}__{dataset}",
        "dataset": dataset,
        "arm": arm,
        "seed": scientific.DEVELOPMENT_SEED,
        "metrics": {
            "validity": {"valid_row_rate": 1.0},
            "measured": {
                "overall": {"normalized_l1_mean": value},
                "proxy_objectives": {
                    "legacy_relative_gap": value,
                    "bounded_r8_gap": value,
                },
                "target_count_bins": target_bins,
                "by_order": by_order,
            },
            "offline_query_groups": {
                name: {"normalized_l1_mean": value}
                for name in group_names
            },
        },
        "trajectory": {"fixed_checkpoints": [], "terminal": {}},
        "kernel_audit": {"gap_weighting_guard_passed": True},
        "cost": {"elapsed_sec": value},
    }


def _synthetic_cases() -> tuple[list[dict], list[dict]]:
    candidate = [
        _case(dataset, evaluation.CANDIDATE_ARM, value=9.0)
        for dataset in scientific.DATASET_ORDER
    ]
    baseline = []
    for dataset in scientific.DATASET_ORDER:
        baseline.append(_case(dataset, adapter.LEGACY_ARM, value=10.0))
        baseline.append(_case(dataset, adapter.R8_ARM, value=8.0))
    candidate_nltcs = next(
        case for case in candidate if case["dataset"] == "nltcs"
    )
    candidate_nltcs["metrics"]["measured"]["target_count_bins"][
        scientific.NLTCS_RARE_BIN
    ]["absolute_count_error_mean"] = 7.0
    return candidate, baseline


def test_audit_protocol_binds_candidate_and_audited_baseline_artifacts():
    # 收束线：已发布产物与基线的死记录仍须自恰（缺产物则 skip）；
    # 适配身份链对活树的校验预期失败关闭。
    if not (ROOT / evaluation.SOURCE_COLLECTION_PATH).exists():
        pytest.skip("本机没有 sqrt 冻结 collection 产物")
    for path, expected in (
        (evaluation.SOURCE_COLLECTION_PATH, protocol.SOURCE_COLLECTION_SHA256),
        (protocol.SOURCE_EVALUATION_PATH, protocol.SOURCE_EVALUATION_SHA256),
        (protocol.SOURCE_METRICS_CSV_PATH, protocol.SOURCE_METRICS_CSV_SHA256),
    ):
        assert protocol.file_sha256(ROOT / path) == expected
    for item in evaluation.BASELINE_ARTIFACTS.values():
        assert protocol.file_sha256(ROOT / item["path"]) == item["sha256"]
    with pytest.raises(RuntimeError, match="漂移"):
        protocol.assert_frozen_audit_adapter_identity(ROOT)


def test_plan_does_not_load_collection_baseline_or_recompute_reference(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("独立审计 plan 不得读取结果、基线或参考表")

    # 收束线：绕过对活树的适配身份校验，只测 plan 的结果盲性。
    monkeypatch.setattr(
        protocol,
        "assert_frozen_audit_adapter_identity",
        lambda _root: protocol.FROZEN_AUDIT_ADAPTER_SHA256,
    )
    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter, "_load_audited_baseline", forbidden)
    monkeypatch.setattr(adapter.source_auditor, "_recompute_cases", forbidden)
    plan = adapter.build_plan()

    assert plan["audit_started"] is False
    assert plan["raw_reference_data_accessed"] is False
    assert plan["candidate_results_accessed"] is False
    assert plan["baseline_quality_values_accessed"] is False
    assert plan["rerun_generation"] is False
    assert plan["evaluation_artifacts_modified"] is False


def test_confirmation_gate_fails_before_worktree_artifact_or_reference_access(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("错误确认不得继续访问")

    monkeypatch.setattr(adapter, "_repo_root", forbidden)
    monkeypatch.setattr(adapter.recovery_collection, "load_collection", forbidden)
    monkeypatch.setattr(adapter, "_load_audited_baseline", forbidden)
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


def test_adapter_imports_no_primary_evaluator_and_no_terminal_metric_arithmetic():
    path = ROOT / "scripts/audit_issue53_gap_weight_sqrt_screen_recovered.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "scripts"
        for alias in node.names
    }
    assert "evaluate_issue53_gap_weight_sqrt_screen_recovered" not in imported_names
    assert "evaluate_issue53_gap_weight_r8_screen" not in imported_names
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
            "_csv_rows_independent",
            "_assert_equal",
            "_audit_csv",
        }
    )
    assert adapter.source_auditor._recompute_cases is source_auditor._recompute_cases
    assert (
        adapter.source_auditor._independent_ratio
        is source_auditor._independent_ratio
    )
    assert (
        adapter.source_auditor._csv_rows_independent
        is source_auditor._csv_rows_independent
    )
    assert adapter.source_auditor._audit_csv is source_auditor._audit_csv


def test_source_auditor_runtime_binding_is_scoped(monkeypatch):
    original = source_auditor.protocol

    def recompute(_root, _indexed):
        assert source_auditor.protocol is evaluation
        return [{"case": True}], {"identity": True}

    monkeypatch.setattr(source_auditor, "_recompute_cases", recompute)
    with adapter._source_auditor_runtime():
        cases, identities = source_auditor._recompute_cases(ROOT, {})
    assert cases == [{"case": True}]
    assert identities == {"identity": True}
    assert source_auditor.protocol is original


def test_independent_summary_rebuilds_six_gates_and_passes_only_full_design():
    candidate, baseline = _synthetic_cases()
    summary = adapter._summary_independent(
        candidate,
        baseline,
        {"execution_valid": True},
        {"pass": True},
    )

    assert summary["execution_valid"] is True
    assert summary["quality_interpretation_allowed"] is True
    assert summary["frozen_classification"] == (
        "advance_to_fresh_seed_confirmation"
    )
    assert summary["all_six_gate_families_passed"] is True
    gates = summary["screen_gate_evidence"]
    assert gates["nltcs_measured_l1_vs_legacy"]["operator"] == (
        "strictly_less_than"
    )
    assert gates["nltcs_rare_bin_vs_legacy"]["threshold"] == 1.25
    assert gates["nltcs_rare_bin_vs_r8"]["operator"] == (
        "strictly_less_than"
    )
    assert set(gates["one_way_safety_vs_legacy"]) == set(
        scientific.DATASET_ORDER
    )


def test_independent_summary_explicitly_preserves_frozen_precedence(
    monkeypatch,
):
    candidate, baseline = _synthetic_cases()
    candidate_nltcs = next(
        case for case in candidate if case["dataset"] == "nltcs"
    )
    candidate_nltcs["metrics"]["measured"]["overall"][
        "normalized_l1_mean"
    ] = 10.0
    candidate_nltcs["metrics"]["measured"]["target_count_bins"][
        scientific.NLTCS_RARE_BIN
    ]["absolute_count_error_mean"] = 100.0
    monkeypatch.setattr(
        scientific,
        "classify_screen",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("独立分类不得调用主科学分类函数")
        ),
    )
    summary = adapter._summary_independent(
        candidate,
        baseline,
        {"execution_valid": True},
        {"pass": True},
    )
    assert summary["frozen_classification"] == "common_mechanism_not_retained"
    assert summary["classification_precedence"] == (
        scientific.frozen_protocol_manifest()["screen_gates"][
            "classification_order"
        ]
    )


def test_stub_audit_orchestration_uses_recovery_baseline_then_independent_auditor(
    monkeypatch,
):
    calls: list[str] = []
    collection = {
        "execution_commit": "a" * 40,
        "postcollection_recovery": {"recovery_commit": "b" * 40},
    }
    indexed = {("stub",): {"row": True}}
    candidate_cases, baseline_cases = _synthetic_cases()
    identities = {"identity": True}
    baseline_evaluation = {
        "query_and_reference_identity_audit": identities,
    }
    baseline_audit = {"pass": True}
    evaluation_report = {"evaluation_commit": "c" * 40}
    evidence = {"pass": True, "verified_frozen_classification": "stub"}

    monkeypatch.setattr(
        protocol,
        "assert_frozen_audit_adapter_identity",
        lambda _root: protocol.FROZEN_AUDIT_ADAPTER_SHA256,
    )
    monkeypatch.setattr(adapter, "_assert_audit_output_absent", lambda _root: None)

    def git_text(_root, *arguments):
        return "" if arguments[0] == "status" else "d" * 40

    monkeypatch.setattr(adapter.recovery_collection, "_git_text", git_text)

    def load_collection(_root, confirmed):
        assert confirmed == protocol.SOURCE_COLLECTION_SHA256
        calls.append("load_collection")
        return collection, indexed

    def load_baseline(_root):
        calls.append("_load_audited_baseline")
        return baseline_evaluation, baseline_cases, baseline_audit

    def recompute(_root, observed):
        assert source_auditor.protocol is evaluation
        assert observed is indexed
        calls.append("_recompute_cases")
        return candidate_cases, identities

    def audit_evaluation(*args):
        assert args[3] is collection
        assert args[4] is candidate_cases
        assert args[5] is identities
        assert args[6] is baseline_evaluation
        assert args[7] is baseline_cases
        assert args[8] is baseline_audit
        calls.append("_audit_recovered_evaluation")
        return evaluation_report, evidence

    def publish(_root, payload):
        calls.append("_publish_audit")
        assert payload["evaluator_arithmetic_imported"] is False
        assert payload["generation_rerun"] is False
        assert payload["pass"] is True
        return ROOT / protocol.AUDIT_OUTPUT_PATH

    monkeypatch.setattr(adapter.recovery_collection, "load_collection", load_collection)
    monkeypatch.setattr(adapter, "_load_audited_baseline", load_baseline)
    monkeypatch.setattr(adapter.source_auditor, "_recompute_cases", recompute)
    monkeypatch.setattr(adapter, "_audit_recovered_evaluation", audit_evaluation)
    monkeypatch.setattr(adapter, "build_plan", lambda: {"plan": "stub"})
    monkeypatch.setattr(adapter, "_publish_audit", publish)

    result = adapter.audit(
        protocol.FROZEN_AUDIT_ADAPTER_SHA256,
        protocol.SOURCE_COLLECTION_SHA256,
        protocol.SOURCE_EVALUATION_SHA256,
    )

    assert result.name == protocol.AUDIT_OUTPUT_PATH.name
    assert calls == [
        "load_collection",
        "_load_audited_baseline",
        "_recompute_cases",
        "_audit_recovered_evaluation",
        "_publish_audit",
    ]


def test_real_independent_audit_output_matches_published_verdict_when_present():
    # 冻结期本断言为"报告不存在"；正式审计已完成后，报告在位时改为
    # 校验其记录与已发布判决一致。
    report_path = ROOT / protocol.AUDIT_OUTPUT_PATH
    if not report_path.exists():
        return
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["verified_frozen_classification"] == (
        "rare_query_protection_not_recovered"
    )
