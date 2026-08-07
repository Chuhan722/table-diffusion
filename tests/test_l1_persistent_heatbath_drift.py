"""L1 持久热浴事后期望漂移诊断的数值与边界测试。"""

import numpy as np
import pytest

import scripts.analyze_l1_persistent_heatbath_drift as drift
import scripts.probe_l1_persistent_heatbath as probe
from table_diffevo.persistent_heatbath import (
    ENERGY_MODE_NORMALIZED_L1,
    initialize_persistent_heatbath_state,
)
from table_diffevo.schema import AttributeBlock, Schema


def _tiny_problem():
    schema = Schema([
        AttributeBlock(
            name="a", type="categorical", description="", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="", values=[0, 1]
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
    ]
    return schema, queries, np.asarray([1.0, 1.0, 1.0])


def test_random_scan_drift_is_monotonic_with_negative_variance_derivative():
    energy_sets = {
        "source_energy": 0.2,
        "candidate_energies": [
            np.asarray([0.2, 0.0, 1.0]),
            np.asarray([0.2, 0.1, 0.8]),
        ],
    }
    center = drift._random_scan_drift(energy_sets, 0.7)
    epsilon = 1e-5
    numerical = (
        drift._random_scan_drift(
            energy_sets, 0.7 + epsilon
        )["mean_drift"]
        - drift._random_scan_drift(
            energy_sets, 0.7 - epsilon
        )["mean_drift"]
    ) / (2.0 * epsilon)

    assert drift._random_scan_drift(
        energy_sets, 0.0
    )["mean_drift"] > center["mean_drift"]
    assert numerical == pytest.approx(
        center["derivative_wrt_inverse_scale"], rel=1e-8, abs=1e-10
    )
    assert center["minimum_conditional_probability"] > 0.0


def test_critical_tau_finds_unique_finite_drift_crossing():
    energy_sets = {
        "source_energy": 0.2,
        "candidate_energies": [
            np.asarray([0.2, 0.0, 1.0]),
            np.asarray([0.2, 0.1, 0.8]),
        ],
    }

    result = drift._critical_tau(energy_sets, scale=0.1)

    assert result["exists_finite"] is True
    assert result["critical_tau"] > 0.0
    assert abs(result["drift_at_critical"]) <= 1e-15
    below = drift._random_scan_drift(
        energy_sets, (result["critical_tau"] - 1e-6) / 0.1
    )
    above = drift._random_scan_drift(
        energy_sets, (result["critical_tau"] + 1e-6) / 0.1
    )
    assert below["mean_drift"] > 0.0
    assert above["mean_drift"] < 0.0


def test_strict_coordinate_local_minimum_has_no_finite_neutral_tau():
    energy_sets = {
        "source_energy": 0.0,
        "candidate_energies": [
            np.asarray([0.0, 0.1]),
            np.asarray([0.0, 0.2]),
        ],
    }

    result = drift._critical_tau(energy_sets, scale=0.1)

    assert result["exists_finite"] is False
    assert result["critical_tau"] is None
    assert drift._random_scan_drift(
        energy_sets, 100.0
    )["mean_drift"] > 0.0


def test_candidate_energy_sets_cover_every_coordinate_and_are_finite():
    schema, queries, target = _tiny_problem()
    table = probe._oracle_table((0, 0, 1, 0))
    state = initialize_persistent_heatbath_state(
        table, schema, queries, target
    )

    result = drift._candidate_energy_sets(
        state,
        schema,
        queries,
        target,
        ENERGY_MODE_NORMALIZED_L1,
    )

    assert result["coordinates"] == 4
    assert result["candidate_state_evaluations"] == 8
    assert len(result["candidate_energies"]) == 4
    assert all(
        np.all(np.isfinite(values))
        for values in result["candidate_energies"]
    )


def test_build_analysis_reuses_recorded_initial_state_and_trajectory(
    monkeypatch,
):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        65,
        schema,
        queries,
        target,
        None,
        steps=6,
        tail=2,
        tau=1.0,
        verify_every=3,
    )
    payload = {
        "protocol": {"n_records": 2},
        "runs": [run],
    }

    result = drift.build_analysis(
        payload, schema, queries, target, None
    )

    assert result["scope"].startswith("posthoc")
    for variant in ("baseline", "candidate"):
        row = result["rows"][variant][0]
        assert row["coordinates"] == 4
        assert row["formal_tau"] == 1.0
        assert np.isfinite(row["initial_random_scan"]["mean_drift"])
        assert result["by_variant"][variant][
            "initial_mean_drift"
        ]["n"] == 1
