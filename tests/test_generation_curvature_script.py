"""整代曲率 Gibbs 冻结脚本的配对、门禁与预注册判断测试。"""

import numpy as np
import pandas as pd
import pytest

import scripts.probe_generation_curvature_gibbs as probe
from table_diffevo.schema import AttributeBlock, Schema


def _schema_queries():
    schema = Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in ("a", "b", "c")
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
    ]
    return schema, queries


def _comparison(baseline, candidate):
    return {
        "baseline": {"mean": float(baseline)},
        "candidate": {"mean": float(candidate)},
        "difference": {"mean": float(candidate - baseline)},
    }


def _conditional_row(entropy=0.6, count=10):
    return {
        "conditional_probability_count": count,
        "conditional_entropy_mean": entropy if count else None,
        "conditional_probability_min": 0.1 if count else None,
        "conditional_probability_max": 0.9 if count else None,
        "all_conditionals_bidirectional": True,
    }


def _state(net_difference, positive_difference, entropy=0.6):
    return {
        "state_rounds": 500,
        "paired": {
            "net_gain": {"difference": {"mean": net_difference}},
            "positive_gain": {
                "difference": {"mean": positive_difference}
            },
        },
        "baseline_rows": [_conditional_row(0.6)],
        "candidate_rows": [_conditional_row(entropy)],
    }


@pytest.mark.parametrize(
    "net_differences,positive_differences,expected",
    [
        (
            [1.0, 2.0, 3.0],
            [0.01, 0.00, -0.01],
            "supports_late_curvature_kernel",
        ),
        (
            [1.0, 2.0, -0.5],
            [0.01, 0.01, 0.01],
            "late_curvature_inconclusive",
        ),
        (
            [1.0, -2.0, -0.5],
            [0.01, 0.01, 0.01],
            "late_curvature_not_supported",
        ),
        (
            [1.0, 2.0, 3.0],
            [-0.01, -0.01, -0.01],
            "late_curvature_inconclusive",
        ),
    ],
)
def test_decision_applies_preregistered_late_state_rule(
    net_differences, positive_differences, expected
):
    states = [
        _state(net, positive)
        for net, positive in zip(net_differences, positive_differences)
    ]
    late_paired = {
        "net_gain": _comparison(1.0, 1.0 + np.mean(net_differences)),
        "positive_gain": _comparison(
            0.4, 0.4 + np.mean(positive_differences)
        ),
    }
    initial_paired = {"net_gain": _comparison(10.0, 10.0)}

    result = probe._decision_from_state_results(
        states, late_paired, initial_paired
    )
    assert result["decision"] == expected


def test_decision_marks_preregistered_stage_and_entropy_risks():
    states = [
        _state(1.0, 0.01, entropy=0.5),
        _state(1.0, 0.01, entropy=0.5),
        _state(1.0, 0.01, entropy=0.5),
    ]
    result = probe._decision_from_state_results(
        states,
        {
            "net_gain": _comparison(1.0, 2.0),
            "positive_gain": _comparison(0.4, 0.41),
        },
        {"net_gain": _comparison(10.0, 9.0)},
    )

    assert result["decision"] == "supports_late_curvature_kernel"
    assert result["initial_deterioration_over_5pct_risk"] is True
    assert result["conditional_entropy_concentration_risk"] is True


def test_conditional_summary_handles_empty_and_weights_microsteps():
    assert probe._conditional_summary([_conditional_row(count=0)]) == {
        "n_microsteps": 0,
        "mean_entropy": None,
        "min_probability": None,
        "max_probability": None,
        "all_bidirectional": True,
    }
    result = probe._conditional_summary([
        _conditional_row(0.4, count=10),
        _conditional_row(0.6, count=30),
    ])
    assert result["n_microsteps"] == 40
    assert result["mean_entropy"] == pytest.approx(0.55)
    assert result["all_bidirectional"] is True


def test_probe_state_passes_exact_regression_and_energy_gates(monkeypatch):
    schema, queries = _schema_queries()
    state = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
        "c": [0, 1, 1, 0],
    })
    target = np.asarray([2.5, 2.5, 1.5])
    monkeypatch.setattr(probe, "N_RECORDS", len(state))
    monkeypatch.setattr(probe, "RHO", 1.0)

    result = probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=3,
        state_index=0,
        state_rounds=500,
        proposals=3,
        temperature=2.0,
        sweeps=2,
        max_factor_order=3,
        device="numpy",
        fixed_reference_scale=0.25,
    )

    assert result["n_proposals"] == 3
    assert result["gates"]["primary_rng_aligned"] is True
    assert result["gates"]["gibbs_rng_aligned"] is True
    assert result["gates"]["gamma_zero_frame_exact"] is True
    assert result["gates"]["gamma_zero_mask_exact"] is True
    assert result["gates"]["gamma_zero_diagnostics_exact"] is True
    assert result["gates"]["internal_initial_masks_aligned"] is True
    assert result["gates"]["initial_query_delta_max_error"] == 0.0
    assert result["gates"]["final_query_delta_max_error"] == 0.0
    assert result["gates"]["candidate_energy_identity_max_error"] <= 1e-10
    assert result["gates"]["row_delta_sum_max_error"] == 0.0
    assert result["gates"]["quadratic_identity_max_error"] == 0.0
    assert result["gates"]["gain_identity_max_error"] == 0.0
    assert result["gates"]["all_conditionals_bidirectional"] is True
    assert result["baseline_conditional"]["n_microsteps"] > 0
    assert result["candidate_conditional"]["n_microsteps"] > 0


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_probe_state_rejects_invalid_fixed_scale(scale, monkeypatch):
    schema, queries = _schema_queries()
    state = pd.DataFrame({
        "a": [0, 1],
        "b": [0, 1],
        "c": [0, 1],
    })
    monkeypatch.setattr(probe, "N_RECORDS", len(state))
    with pytest.raises(ValueError, match="fixed_reference_scale"):
        probe._probe_state(
            state,
            np.asarray([1.0, 1.0, 1.0]),
            queries,
            schema,
            seed=0,
            state_index=0,
            state_rounds=0,
            proposals=1,
            temperature=2.0,
            sweeps=2,
            max_factor_order=3,
            device="numpy",
            fixed_reference_scale=scale,
        )
