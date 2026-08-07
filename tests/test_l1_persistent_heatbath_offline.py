"""L1 持久热浴阶段 I 离线分类与统计边界测试。"""

import copy

import numpy as np
import pytest

import scripts.analyze_l1_persistent_heatbath_offline as analysis


def _row(value):
    return {
        metric: float(value)
        for metric in analysis.QUALITY_METRIC_DIRECTIONS
    }


def _metric(mean):
    return {"mean": float(mean)}


def _paired(improved, nonworse=None):
    return {
        "improved": int(improved),
        "nonworse": int(improved if nonworse is None else nonworse),
    }


def _classification_inputs(
    *,
    measured_candidate=0.1,
    measured_initial=0.2,
    measured_baseline=0.15,
    three_candidate=0.1,
    three_initial=0.1,
    three_baseline=0.12,
    measured_wins=14,
    three_wins=14,
    three_nonworse=11,
    entropy=0.8,
    uphill=0.1,
    semantic=True,
):
    by_variant = {
        "initial": {
            "training_normalized_l1": _metric(measured_initial),
            "unmeasured_3way_l1": _metric(three_initial),
        },
        "baseline": {
            "training_normalized_l1": _metric(measured_baseline),
            "unmeasured_3way_l1": _metric(three_baseline),
        },
        "candidate": {
            "training_normalized_l1": _metric(measured_candidate),
            "unmeasured_3way_l1": _metric(three_candidate),
        },
    }
    paired = {
        "candidate_minus_initial": {
            "training_normalized_l1": _paired(measured_wins),
            "unmeasured_3way_l1": _paired(
                three_nonworse, nonworse=three_nonworse
            ),
        },
        "candidate_minus_baseline": {
            "training_normalized_l1": _paired(measured_wins),
            "unmeasured_3way_l1": _paired(three_wins),
        },
    }
    payload = {
        "aggregate": {
            "all_diagnostic_gates_passed": semantic,
            "paired_metrics": {
                "conditional_normalized_entropy_mean": {
                    "candidate": {"mean": entropy}
                },
                "uphill_probability_mass_mean": {
                    "candidate": {"mean": uphill}
                },
            },
        },
        "independent_audit": {"passed": semantic},
    }
    return by_variant, paired, payload


def test_paired_summary_reports_median_interval_effect_and_direction():
    candidate = [_row(1.0), _row(3.0), _row(5.0)]
    reference = [_row(2.0), _row(2.0), _row(2.0)]

    lower = analysis._paired_summary(
        candidate, reference, "unmeasured_3way_l1"
    )
    higher = analysis._paired_summary(
        candidate, reference, "raw_unique_states"
    )

    assert lower["mean"] == 1.0
    assert lower["median"] == 1.0
    assert lower["improved"] == 1
    assert lower["worsened"] == 2
    assert lower["nonworse"] == 1
    assert lower["paired_cohens_dz"] == pytest.approx(0.5)
    assert higher["improved"] == 2


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "target_aligned_and_independent_quality_not_degraded"),
        ({"three_candidate": 0.11}, "training_target_only"),
        ({"measured_wins": 13}, "not_supported"),
        ({"entropy": 0.49}, "exploration_collapse_risk"),
        ({"uphill": 0.009}, "exploration_collapse_risk"),
        ({"semantic": False}, "not_supported"),
    ],
)
def test_classification_follows_preregistered_gates(overrides, expected):
    by_variant, paired, payload = _classification_inputs(**overrides)

    result = analysis._classification(by_variant, paired, payload)

    assert result["label"] == expected


def test_classification_threshold_boundaries_are_inclusive_where_declared():
    by_variant, paired, payload = _classification_inputs(
        three_candidate=0.1,
        three_initial=0.1,
        measured_wins=14,
        three_wins=14,
        three_nonworse=11,
        entropy=0.5,
        uphill=0.01,
    )

    result = analysis._classification(by_variant, paired, payload)

    assert result["label"] == (
        "target_aligned_and_independent_quality_not_degraded"
    )
    assert result["exploration_collapse_risk"] is False


def test_source_validation_requires_formal_generation_and_audit():
    experiment = analysis.experiment
    payload = {
        "experiment": "l1_persistent_workload_heatbath",
        "formal_protocol": True,
        "protocol": {
            "n_records": experiment.N_RECORDS,
            "seeds": list(experiment.FORMAL_SEEDS),
            "steps": experiment.FORMAL_STEPS,
            "tail_window": experiment.FORMAL_TAIL,
            "tau": experiment.FORMAL_TAU,
            "verify_every": experiment.FORMAL_VERIFY_EVERY,
            "device": experiment.FORMAL_DEVICE,
            "baseline_energy_mode": "squared",
            "candidate_energy_mode": "normalized_l1",
            "acceptance_or_checkpoint_selection": False,
        },
        "aggregate": {
            "classification": "generation_complete_pending_offline_quality",
            "all_diagnostic_gates_passed": True,
        },
        "independent_audit": {"passed": True},
    }
    analysis._validate_source_payload(payload)

    invalid = copy.deepcopy(payload)
    invalid["independent_audit"]["passed"] = False
    with pytest.raises(ValueError, match="正式输出"):
        analysis._validate_source_payload(invalid)


def test_summarize_rejects_nonfinite_and_reports_median():
    result = analysis._summarize([1.0, 4.0, 9.0])
    assert result["median"] == 4.0
    with pytest.raises(ValueError, match="非空有限"):
        analysis._summarize([np.nan])
