"""平方根目标计数查询权重的最小差量测试。"""

import copy

import numpy as np
import pandas as pd
import pytest

from table_diffevo import gap_l1_diffusion as gap
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


def _schema() -> Schema:
    return Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=[0, 1],
        )
    ])


def _equals(value: int = 1) -> dict:
    return {"attribute": "a", "operator": "==", "value": value}


def test_sqrt_target_denominators_use_one_count_quantum():
    targets = np.array([0.0, 1.0, 4.0, 9.0, 100.0])
    denominators, spec = gap._build_gap_l1_denominators(
        targets,
        floor=8.0,
        weighting=gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        max_weight_ratio=None,
        n_records=None,
    )

    np.testing.assert_array_equal(denominators, [1.0, 1.0, 2.0, 3.0, 10.0])
    assert spec.mode == gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE
    assert spec.max_weight_ratio is None
    assert spec.smoothing_count is None
    assert spec.actual_weight_ratio == 10.0


def test_sqrt_target_objective_is_independent_of_legacy_floor_and_row_count():
    targets = np.array([0.0, 1.0, 4.0, 9.0, 100.0])
    counts = targets + 1.0
    expected = np.mean([1.0, 1.0, 0.5, 1.0 / 3.0, 0.1])

    first = gap.normalized_gap_l1_error(
        counts,
        targets,
        floor=8.0,
        weighting=gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
    )
    second = gap.normalized_gap_l1_error(
        counts,
        targets,
        floor=123.0,
        weighting=gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        n_records=10_000,
    )

    assert first == expected
    assert second == expected


def test_sqrt_target_rejects_bounded_ratio_parameter():
    with pytest.raises(ValueError, match="sqrt_target_relative.*不允许"):
        gap.normalized_gap_l1_error(
            np.array([1.0]),
            np.array([0.0]),
            weighting=gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
            max_weight_ratio=8.0,
        )


def test_sqrt_target_does_not_claim_legacy_integer_cancellation_rule():
    schema = _schema()
    queries = [{"conditions": [_equals()]}]
    current = pd.DataFrame({"a": [0]})
    donors = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="整数公共分母精确抵消"):
        gap.isolated_gap_l1_scores(
            current,
            donors,
            schema,
            queries,
            np.array([1.0]),
            np.array([0]),
            exact_target_numerators=np.array([1]),
            exact_target_denominator=1,
            weighting=gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        )


def test_sqrt_target_scan_replays_exactly_eight_k_and_records_identity():
    schema = _schema()
    queries = [
        {"conditions": [_equals(1)]},
        {"conditions": [_equals(0)]},
    ]
    current = pd.DataFrame({"a": [0, 0, 1, 0]})
    donors = pd.DataFrame({"a": [1, 1, 0, 1]})
    counts = evaluate_table(current, queries)
    kwargs = {
        "participate": np.ones(len(current), dtype=bool),
        "initial_mask": np.zeros((len(current), 1), dtype=bool),
        "reference_scale": 0.1,
        "n_sweeps": 8,
        "weighting": gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
    }
    first_rng = np.random.default_rng(20260831)
    second_rng = np.random.default_rng(20260831)
    initial_rng_state = copy.deepcopy(first_rng.bit_generator.state)

    first = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1.0, 3.0]),
        counts,
        rng=first_rng,
        **kwargs,
    )
    second = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1.0, 3.0]),
        counts,
        rng=second_rng,
        **kwargs,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    diagnostics = first[2]
    assert diagnostics["kernel"].endswith("sqrt_target_relative")
    assert diagnostics["gap_l1_weighting"] == (
        gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE
    )
    assert diagnostics["gap_l1_max_weight_ratio"] is None
    assert diagnostics["gap_l1_smoothing_count"] is None
    assert diagnostics["gap_l1_target_count_quantum"] == 1.0
    assert diagnostics["gap_l1_actual_weight_ratio"] == np.sqrt(3.0)
    assert diagnostics["gibbs_microsteps"] == (
        8 * diagnostics["active_switches_k"]
    )
    assert diagnostics["conditional_error_evaluations"] == (
        2 * diagnostics["gibbs_microsteps"]
    )
    assert diagnostics["clip_hit_count"] == 0
    assert diagnostics["nonfinite_condition_count"] == 0
    assert diagnostics["exact_zero_or_one_probability_count"] == 0
    assert diagnostics["minimum_binary_outcome_probability"] > 0.0
    assert diagnostics["microstep_trace_sha256"] == (
        second[2]["microstep_trace_sha256"]
    )
    assert first_rng.bit_generator.state != initial_rng_state
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
