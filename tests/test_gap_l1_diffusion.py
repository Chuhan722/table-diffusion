"""剩余缺口感知全局绝对误差 Gibbs（吉布斯）核测试。"""

import copy

import numpy as np
import pandas as pd
import pytest

from table_diffevo.gap_l1_diffusion import (
    GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
    GAP_L1_WEIGHTING_LEGACY_RELATIVE,
    compile_gap_l1_workload,
    evaluate_gap_l1_condition,
    evolve_step_gap_l1_global,
    gap_l1_conditional_probability,
    isolated_gap_l1_scores,
    normalized_gap_l1_error,
    stable_nonzero_rms,
)
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


def _schema(*names):
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in names
    ])


def _equals(attribute, value=1):
    return {"attribute": attribute, "operator": "==", "value": value}


def test_normalized_error_uses_target_floor_and_all_queries():
    actual = normalized_gap_l1_error(
        np.array([2, 12]), np.array([0, 10]), floor=8
    )
    assert actual == (2 / 8 + 2 / 10) / 2


def test_bounded_relative_uses_fractional_row_scaled_smoothing():
    n_records = 300
    rare = normalized_gap_l1_error(
        np.array([1]),
        np.array([0]),
        weighting=GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
        max_weight_ratio=8.0,
        n_records=n_records,
    )
    common = normalized_gap_l1_error(
        np.array([n_records - 1]),
        np.array([n_records]),
        weighting=GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
        max_weight_ratio=8.0,
        n_records=n_records,
    )

    smoothing = n_records / 7.0
    assert rare == 1.0 / smoothing
    assert common == 1.0 / (n_records + smoothing)
    assert rare / common == 8.0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"weighting": "unknown"}, "weighting"),
        (
            {"weighting": GAP_L1_WEIGHTING_BOUNDED_RELATIVE},
            "max_weight_ratio",
        ),
        (
            {
                "weighting": GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
                "max_weight_ratio": 1.0,
            },
            "大于 1",
        ),
        (
            {
                "weighting": GAP_L1_WEIGHTING_LEGACY_RELATIVE,
                "max_weight_ratio": 8.0,
            },
            "不允许",
        ),
    ],
)
def test_gap_l1_weighting_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        normalized_gap_l1_error(
            np.array([1]), np.array([1]), n_records=10, **kwargs
        )


def test_same_row_conjunction_is_not_an_isolated_linear_sum():
    schema = _schema("a", "b")
    queries = [{"conditions": [_equals("a"), _equals("b")]}]
    current = pd.DataFrame({"a": [0], "b": [0]})
    donors = pd.DataFrame({"a": [1], "b": [1]})
    participate = np.array([True])

    without_b = evaluate_gap_l1_condition(
        current,
        donors,
        schema,
        queries,
        np.array([1]),
        np.array([0]),
        participate=participate,
        mask=np.array([[False, False]]),
        row_index=0,
        attribute_index=0,
        reference_scale=0.125,
    )
    with_b = evaluate_gap_l1_condition(
        current,
        donors,
        schema,
        queries,
        np.array([1]),
        np.array([0]),
        participate=participate,
        mask=np.array([[False, True]]),
        row_index=0,
        attribute_index=0,
        reference_scale=0.125,
    )

    assert without_b["score"] == 0.0
    assert without_b["probability"] == 0.5
    assert with_b["e0"] == 0.125
    assert with_b["e1"] == 0.0
    assert with_b["score"] == 0.125
    assert with_b["probability"] > 0.5


def test_global_context_brakes_a_second_row_that_would_cross_target():
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]}]
    current = pd.DataFrame({"a": [0, 0]})
    donors = pd.DataFrame({"a": [1, 1]})
    condition = evaluate_gap_l1_condition(
        current,
        donors,
        schema,
        queries,
        np.array([1]),
        np.array([0]),
        participate=np.array([True, True]),
        mask=np.array([[True], [False]]),
        row_index=1,
        attribute_index=0,
        reference_scale=0.125,
    )

    assert condition["e0"] == 0.0
    assert condition["e1"] == 0.125
    assert condition["score"] == -0.125
    assert condition["probability"] < 0.5


def test_exact_target_is_not_frozen_but_departure_gets_negative_score():
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]}]
    current = pd.DataFrame({"a": [1, 0]})
    donors = pd.DataFrame({"a": [1, 1]})
    condition = evaluate_gap_l1_condition(
        current,
        donors,
        schema,
        queries,
        np.array([1]),
        np.array([1]),
        participate=np.array([False, True]),
        mask=np.array([[False], [False]]),
        row_index=1,
        attribute_index=0,
        reference_scale=0.125,
    )

    assert condition["e0"] == 0.0
    assert condition["e1"] == 0.125
    assert condition["score"] < 0.0
    assert 0.0 < condition["probability"] < 0.5


def test_isolated_scale_excludes_zero_scores_and_is_stable():
    schema = _schema("a", "b")
    queries = [
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("a")]},
    ]
    current = pd.DataFrame({"a": [0], "b": [0]})
    donors = pd.DataFrame({"a": [1], "b": [1]})
    isolated = isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        np.array([1, 1]),
        np.array([0, 0]),
    )
    scale, diagnostics = stable_nonzero_rms(isolated["scores"])

    np.testing.assert_array_equal(isolated["coordinates"], [[0, 0], [0, 1]])
    np.testing.assert_allclose(isolated["scores"], [1 / 16, 0])
    assert scale == 1 / 16
    assert diagnostics["nonzero_count"] == 1
    assert diagnostics["zero_count"] == 1
    assert diagnostics["absolute_max_over_rms"] == 1.0


def test_calibration_uses_exact_rational_zero_not_float_tail():
    schema = _schema("a")
    # 三个查询有同一个指示量，但目标不同。当前计数为 10，单独复制后为 11：
    # -1/8 + 1/12 + 1/24 = 0，数学上必须从定尺集合排除。
    queries = [{"conditions": [_equals("a")]} for _ in range(3)]
    current = pd.DataFrame({"a": [1] * 10 + [0] * 15})
    donors = current.copy()
    donors.at[10, "a"] = 1
    isolated = isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        np.array([8.0, 12.0, 24.0]),
        np.array([10, 10, 10]),
        exact_target_numerators=np.array([8, 12, 24]) * len(current),
        exact_target_denominator=len(current),
    )
    scale, diagnostics = stable_nonzero_rms(isolated["scores"])

    np.testing.assert_array_equal(isolated["coordinates"], [[10, 0]])
    assert isolated["scores"][0] == 0.0
    assert scale == 0.0
    assert diagnostics["zero_count"] == 1


def test_bounded_calibration_preserves_exact_rational_zero():
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]} for _ in range(3)]
    current = pd.DataFrame({"a": [1] * 10 + [0] * 60})
    donors = current.copy()
    donors.at[10, "a"] = 1
    targets = np.array([2.0, 14.0, 14.0])
    isolated = isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        np.array([10, 10, 10]),
        exact_target_numerators=targets.astype(np.int64),
        exact_target_denominator=1,
        weighting=GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
        max_weight_ratio=8.0,
    )
    scale, diagnostics = stable_nonzero_rms(isolated["scores"])

    np.testing.assert_array_equal(isolated["coordinates"], [[10, 0]])
    assert isolated["scores"][0] == 0.0
    assert scale == 0.0
    assert diagnostics["zero_count"] == 1


def test_random_scan_runs_exactly_eight_k_and_replays_bit_for_bit():
    schema = _schema("a", "b")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
    ]
    current = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
    donors = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
    current_counts = evaluate_table(current, queries)
    compiled = compile_gap_l1_workload(schema, queries)
    kwargs = dict(
        participate=np.array([True, True]),
        initial_mask=np.zeros((2, 2), dtype=bool),
        reference_scale=0.1,
        n_sweeps=8,
        compiled_workload=compiled,
    )
    first = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1, 1, 1]),
        current_counts,
        rng=np.random.default_rng(1234),
        **kwargs,
    )
    second = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1, 1, 1]),
        current_counts,
        rng=np.random.default_rng(1234),
        **kwargs,
    )
    explicit_legacy = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1, 1, 1]),
        current_counts,
        rng=np.random.default_rng(1234),
        weighting=GAP_L1_WEIGHTING_LEGACY_RELATIVE,
        **kwargs,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[0], explicit_legacy[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[1], explicit_legacy[1])
    assert first[2]["active_switches_k"] == 4
    assert first[2]["gibbs_microsteps"] == 32
    assert first[2]["conditional_error_evaluations"] == 64
    assert (
        first[2]["microstep_trace_sha256"]
        == second[2]["microstep_trace_sha256"]
    )
    assert (
        first[2]["microstep_trace_sha256"]
        == explicit_legacy[2]["microstep_trace_sha256"]
    )
    assert "gap_l1_weighting" not in first[2]
    assert first[2]["clip_hit_count"] == 0
    assert first[2]["exact_zero_or_one_probability_count"] == 0
    assert first[2]["minimum_binary_outcome_probability"] > 0.0
    assert first[2]["no_gate"] is True
    np.testing.assert_array_equal(
        evaluate_table(first[0], queries), first[2]["final_query_counts"]
    )


def test_bounded_scan_records_weight_cap_without_changing_scan_shape():
    schema = _schema("a")
    queries = [
        {"conditions": [_equals("a", 1)]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": np.zeros(70, dtype=int)})
    donors = pd.DataFrame({"a": np.ones(70, dtype=int)})
    counts = evaluate_table(current, queries)
    _, _, diagnostics = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([0, 70]),
        counts,
        participate=np.ones(70, dtype=bool),
        initial_mask=np.zeros((70, 1), dtype=bool),
        reference_scale=0.1,
        rng=np.random.default_rng(7),
        n_sweeps=0,
        weighting=GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
        max_weight_ratio=8.0,
    )

    assert diagnostics["kernel"].endswith("bounded_relative")
    assert diagnostics["gap_l1_weighting"] == (
        GAP_L1_WEIGHTING_BOUNDED_RELATIVE
    )
    assert diagnostics["gap_l1_max_weight_ratio"] == 8.0
    assert diagnostics["gap_l1_smoothing_count"] == 10.0
    assert diagnostics["gap_l1_actual_weight_ratio"] == 8.0
    assert diagnostics["gibbs_microsteps"] == 0


def test_k_zero_consumes_no_rng_and_returns_initial_table():
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]}]
    current = pd.DataFrame({"a": [0, 1]})
    donors = current.copy()
    rng = np.random.default_rng(77)
    before = copy.deepcopy(rng.bit_generator.state)
    result, mask, diagnostics = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        np.array([1]),
        np.array([1]),
        participate=np.array([True, True]),
        initial_mask=np.zeros((2, 1), dtype=bool),
        reference_scale=0.1,
        rng=rng,
    )

    pd.testing.assert_frame_equal(result, current)
    assert not np.any(mask)
    assert diagnostics["active_switches_k"] == 0
    assert diagnostics["gibbs_microsteps"] == 0
    assert rng.bit_generator.state == before


def test_logit_guard_preserves_both_outcomes_and_reports_real_clip():
    probability, raw, effective, clipped = gap_l1_conditional_probability(
        1e300, 1e300, strength=1e6, logit_clip=30
    )
    assert clipped is True
    assert raw > 30
    assert effective == 30
    assert 0.0 < probability < 1.0
