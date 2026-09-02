from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_issue53_gap_weight_dual_ar_screen_recovered as adapter
from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import issue53_gap_weight_dual_ar_recovered_evaluation_protocol as protocol
from scripts import issue53_gap_weight_dual_ar_screen_protocol as scientific


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
                name: {"normalized_l1_mean": value} for name in group_names
            },
        },
        "trajectory": {"fixed_checkpoints": [], "terminal": {}},
        "kernel_audit": {
            "gap_weighting": (
                protocol.DUAL_WEIGHTING
                if arm == protocol.CANDIDATE_ARM
                else "baseline"
            ),
            "gap_weighting_guard_passed": True,
        },
        "cost": {"elapsed_sec": value},
    }


def _synthetic_inputs():
    candidate = [
        _case(dataset, protocol.CANDIDATE_ARM, value=9.0)
        for dataset in scientific.DATASET_ORDER
    ]
    baseline = []
    for dataset in scientific.DATASET_ORDER:
        baseline.extend(
            [
                _case(dataset, adapter.LEGACY_ARM, value=10.0),
                _case(dataset, adapter.R8_ARM, value=8.0),
                _case(dataset, adapter.SQRT_ARM, value=8.0),
            ]
        )
    candidate_nltcs = next(
        case for case in candidate if case["dataset"] == "nltcs"
    )
    candidate_nltcs["metrics"]["measured"]["target_count_bins"][
        scientific.NLTCS_RARE_BIN
    ]["absolute_count_error_mean"] = 7.0
    provenance = {
        "r8_evaluation": {"execution_valid": True},
        "r8_audit": {"pass": True},
        "sqrt_evaluation": {"execution_valid": True},
        "sqrt_audit": {"pass": True},
    }
    return candidate, baseline, provenance


def test_adapter_identity_and_confirmation_gate():
    assert (
        protocol.assert_frozen_adapter_identity(ROOT)
        == protocol.FROZEN_ADAPTER_SHA256
    )
    with pytest.raises(PermissionError, match="确认"):
        protocol.require_evaluation_confirmation(None, None)


def test_dual_gap_objective_handles_positive_and_all_zero_targets():
    assert adapter._dual_gap_objective([10, 20], [11, 21], 100) == pytest.approx(
        0.01
    )
    assert adapter._dual_gap_objective([0, 0], [1, 3], 100) == pytest.approx(
        0.02
    )


def test_source_runtime_binding_is_scoped():
    original_protocol = source_evaluator.protocol
    original_checkpoint = source_evaluator._audit_checkpoint_artifact
    with adapter._source_evaluator_runtime():
        assert source_evaluator.protocol is protocol
        assert source_evaluator._audit_checkpoint_artifact is adapter._audit_checkpoint_artifact
    assert source_evaluator.protocol is original_protocol
    assert source_evaluator._audit_checkpoint_artifact is original_checkpoint


def test_summary_uses_frozen_six_gates_and_all_three_baselines():
    candidate, baseline, provenance = _synthetic_inputs()
    summary = adapter._summary_and_decision(candidate, baseline, provenance)
    assert summary["frozen_classification"] == "advance_to_fresh_seed_confirmation"
    assert summary["all_six_gate_families_passed"] is True
    measured = summary["datasets"]["nltcs"]["measured_normalized_l1"]
    assert set(measured) == {"vs_legacy", "vs_r8", "vs_sqrt"}
    assert summary["screen_gate_evidence"]["nltcs_rare_bin_vs_sqrt"][
        "operator"
    ] == "strictly_less_than"


def test_summary_preserves_classification_precedence():
    candidate, baseline, provenance = _synthetic_inputs()
    candidate_nltcs = next(
        case for case in candidate if case["dataset"] == "nltcs"
    )
    candidate_nltcs["metrics"]["measured"]["overall"][
        "normalized_l1_mean"
    ] = 10.0
    summary = adapter._summary_and_decision(candidate, baseline, provenance)
    assert summary["frozen_classification"] == "common_mechanism_not_retained"
    assert summary["classification_precedence"] == scientific.frozen_protocol_manifest()[
        "screen_gates"
    ]["classification_order"]


def test_candidate_proxy_does_not_mutate_source_rows():
    row = {
        "gap_relative_positive_weight_ratio_min": 8,
        "gap_relative_positive_weight_ratio_max": 9,
    }
    indexed = {(9908, "nltcs", protocol.CANDIDATE_ARM): row}
    proxies = adapter._candidate_proxy_rows(indexed)
    assert row == {
        "gap_relative_positive_weight_ratio_min": 8,
        "gap_relative_positive_weight_ratio_max": 9,
    }
    assert proxies[next(iter(proxies))]["gap_actual_weight_ratio_min"] == 8


def test_confirmation_fails_before_any_quality_access(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_repo_root",
        lambda: (_ for _ in ()).throw(AssertionError("提前访问工作树")),
    )
    with pytest.raises(PermissionError, match="确认"):
        adapter.evaluate("wrong", protocol.SOURCE_COLLECTION_SHA256)

