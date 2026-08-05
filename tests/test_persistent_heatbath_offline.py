"""持久化热浴生成后离线评价的汇总与边界测试。"""

import copy

import numpy as np
import pytest

import scripts.analyze_persistent_heatbath_offline as analysis


def _row(value):
    return {
        metric: float(value)
        for metric in analysis.QUALITY_METRIC_DIRECTIONS
    }


def test_summary_reports_seed_level_interval_and_rejects_nonfinite():
    result = analysis._summarize([1.0, 2.0, 3.0])

    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["mean_t_interval_95"][0] < 2.0
    assert result["mean_t_interval_95"][1] > 2.0
    with pytest.raises(ValueError, match="非空有限"):
        analysis._summarize([np.nan])


def test_paired_summary_respects_metric_direction():
    candidate = [_row(1.0), _row(3.0)]
    reference = [_row(2.0), _row(2.0)]

    lower = analysis._paired_summary(
        candidate, reference, "unmeasured_3way_l1"
    )
    higher = analysis._paired_summary(
        candidate, reference, "raw_unique_states"
    )

    assert lower["improved"] == 1
    assert lower["worsened"] == 1
    assert higher["improved"] == 1
    assert higher["worsened"] == 1


def test_validate_source_requires_formal_passed_generation():
    payload = {
        "experiment": "persistent_workload_heatbath",
        "formal_protocol": True,
        "protocol": {
            "n_records": analysis.experiment.N_RECORDS,
            "seeds": list(analysis.experiment.FORMAL_SEEDS),
            "steps": analysis.experiment.FORMAL_STEPS,
            "tail_window": analysis.experiment.FORMAL_TAIL,
            "tau": analysis.experiment.FORMAL_TAU,
            "verify_every": analysis.experiment.FORMAL_VERIFY_EVERY,
            "device": analysis.experiment.FORMAL_DEVICE,
            "acceptance_or_checkpoint_selection": False,
        },
        "aggregate": {
            "classification": "supports_persistent_heatbath_smoke",
            "all_diagnostic_gates_passed": True,
        },
        "independent_audit": {"passed": True},
    }
    analysis._validate_source_payload(payload)

    invalid = copy.deepcopy(payload)
    invalid["independent_audit"]["passed"] = False
    with pytest.raises(ValueError, match="正式输出"):
        analysis._validate_source_payload(invalid)


def test_relative_change_handles_zero_reference_explicitly():
    assert analysis._relative_change(0.0, 0.0) == 0.0
    assert analysis._relative_change(1.0, 0.0) is None
    assert analysis._relative_change(1.1, 1.0) == pytest.approx(0.1)
