from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import evaluate_issue53_gap_weight_sqrt_screen_recovered as adapter
from scripts import issue53_gap_weight_sqrt_recovered_evaluation_protocol as protocol
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
        protocol.TEST_GROUP_ORDER
        if dataset == "test_300x10"
        else tuple(protocol.NLTCS_GROUP_COUNTS)
    )
    return {
        "task_id": f"seed_9908__{arm}__{dataset}",
        "dataset": dataset,
        "arm": arm,
        "seed": 9908,
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
        _case(dataset, protocol.CANDIDATE_ARM, value=9.0)
        for dataset in scientific.DATASET_ORDER
    ]
    baseline = []
    for dataset in scientific.DATASET_ORDER:
        baseline.append(_case(dataset, adapter.LEGACY_ARM, value=10.0))
        baseline.append(_case(dataset, adapter.R8_ARM, value=8.0))
    rare = scientific.NLTCS_RARE_BIN
    candidate_nltcs = next(
        case for case in candidate if case["dataset"] == "nltcs"
    )
    candidate_nltcs["metrics"]["measured"]["target_count_bins"][rare][
        "absolute_count_error_mean"
    ] = 7.0
    return candidate, baseline


def test_adapter_identity_plan_and_confirmation_gate():
    assert (
        protocol.assert_frozen_adapter_identity(ROOT)
        == protocol.FROZEN_ADAPTER_SHA256
    )
    plan = adapter.build_plan()
    assert plan["raw_reference_data_accessed"] is False
    assert plan["candidate_quality_accessed"] is False
    assert plan["baseline_quality_values_accessed"] is False
    assert plan["evaluation_started"] is False
    assert plan["screen_gates"] == scientific.frozen_protocol_manifest()[
        "screen_gates"
    ]
    with pytest.raises(PermissionError, match="确认"):
        protocol.require_evaluation_confirmation(None, None)


def test_confirmation_gate_fails_before_collection_baseline_or_reference_access(
    monkeypatch,
):
    monkeypatch.setattr(
        adapter,
        "_repo_root",
        lambda: (_ for _ in ()).throw(AssertionError("提前访问工作树")),
    )
    monkeypatch.setattr(
        adapter,
        "_load_frozen_baseline",
        lambda _root: (_ for _ in ()).throw(AssertionError("提前读基线")),
    )
    monkeypatch.setattr(
        adapter,
        "_evaluate_candidate_cases",
        lambda _root, _indexed: (_ for _ in ()).throw(
            AssertionError("提前评价候选")
        ),
    )
    with pytest.raises(PermissionError, match="确认"):
        adapter.evaluate("wrong", protocol.SOURCE_COLLECTION_SHA256)


def test_source_runtime_binding_is_scoped_and_delegates_candidate_evaluation(
    monkeypatch,
):
    original = source_evaluator.protocol
    expected_cases = [
        {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "arm": task.arm,
            "seed": task.seed,
        }
        for task in protocol.task_plan().tasks
    ]

    def stub(_root, _indexed):
        assert source_evaluator.protocol is protocol
        return expected_cases, {"identity": "stub"}

    monkeypatch.setattr(source_evaluator, "_evaluate_cases", stub)
    cases, identities = adapter._evaluate_candidate_cases(ROOT, {})
    assert cases == expected_cases
    assert identities == {"identity": "stub"}
    assert source_evaluator.protocol is original


def test_summary_uses_frozen_six_gates_and_passes_only_complete_design():
    candidate, baseline = _synthetic_cases()
    summary = adapter._summary_and_decision(
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


def test_summary_preserves_frozen_classification_precedence():
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
    summary = adapter._summary_and_decision(
        candidate,
        baseline,
        {"execution_valid": True},
        {"pass": True},
    )
    assert summary["frozen_classification"] == (
        "common_mechanism_not_retained"
    )
    assert summary["classification_precedence"] == (
        scientific.frozen_protocol_manifest()["screen_gates"][
            "classification_order"
        ]
    )


def test_adapter_defines_no_terminal_checkpoint_or_reference_metric_arithmetic():
    source = (
        ROOT / "scripts/evaluate_issue53_gap_weight_sqrt_screen_recovered.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "def _vector_metrics",
        "def _integer_vectors",
        "evaluate_table(",
        "_load_references(",
        "pd.read_csv(",
        "np.asarray(",
    ):
        assert forbidden not in source


def test_result_blind_stub_wiring_publishes_only_after_all_frozen_loaders(
    monkeypatch,
):
    identities = {"identity": "same"}
    candidate_cases = [{"task_id": "candidate-1"}, {"task_id": "candidate-2"}]
    baseline_cases = [{"task_id": f"baseline-{index}"} for index in range(4)]
    collection_report = {
        "execution_commit": "a" * 40,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "postcollection_recovery": {"recovery_commit": "b" * 40},
    }
    baseline_evaluation = {
        "frozen_classification": "baseline_stub",
        "query_and_reference_identity_audit": identities,
    }
    summary = {
        "datasets": {"stub": True},
        "screen_gate_evidence": {"stub": True},
        "candidate_execution_valid": True,
        "baseline_execution_valid": True,
        "execution_valid": True,
        "quality_interpretation_allowed": True,
        "frozen_classification": "advance_to_fresh_seed_confirmation",
        "classification_precedence": ["stub"],
        "all_six_gate_families_passed": True,
    }
    monkeypatch.setattr(
        protocol,
        "assert_frozen_adapter_identity",
        lambda _root: protocol.FROZEN_ADAPTER_SHA256,
    )
    monkeypatch.setattr(adapter, "_repo_root", lambda: ROOT)
    monkeypatch.setattr(adapter, "_assert_outputs_absent", lambda _root: None)
    monkeypatch.setattr(
        adapter.recovery_collection,
        "_git_text",
        lambda _root, *args: "" if args[0] == "status" else "c" * 40,
    )
    monkeypatch.setattr(
        adapter.recovery_collection,
        "load_collection",
        lambda _root, _confirmed: (collection_report, {}),
    )
    monkeypatch.setattr(
        adapter,
        "_load_frozen_baseline",
        lambda _root: (baseline_evaluation, baseline_cases, {"pass": True}),
    )
    monkeypatch.setattr(
        adapter,
        "_evaluate_candidate_cases",
        lambda _root, _indexed: (candidate_cases, identities),
    )
    monkeypatch.setattr(
        adapter,
        "_summary_and_decision",
        lambda *_args: summary,
    )
    monkeypatch.setattr(adapter, "build_plan", lambda: {"plan": "stub"})
    monkeypatch.setattr(
        source_evaluator,
        "_csv_rows",
        lambda cases: [{"n": len(cases)}],
    )

    published = {}

    def publish(destination, report, rows):
        assert source_evaluator.protocol is protocol
        published.update({"destination": destination, "report": report, "rows": rows})
        return (
            destination / protocol.EVALUATION_REPORT,
            destination / protocol.L1_RESULTS_CSV,
        )

    monkeypatch.setattr(source_evaluator, "_publish", publish)
    report_path, csv_path = adapter.evaluate(
        protocol.FROZEN_ADAPTER_SHA256,
        protocol.SOURCE_COLLECTION_SHA256,
    )

    assert report_path.name == protocol.EVALUATION_REPORT
    assert csv_path.name == protocol.L1_RESULTS_CSV
    assert published["rows"] == [{"n": 2}]
    assert published["report"]["candidate_case_count"] == 2
    assert published["report"]["baseline_case_count"] == 4
    assert published["report"]["frozen_classification"] == (
        "advance_to_fresh_seed_confirmation"
    )
    assert published["report"]["independent_audit_automatically_run"] is False
