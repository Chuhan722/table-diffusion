"""A/R 双通道 ``max`` 查询权重的结果前契约测试。"""

import copy

import numpy as np
import pandas as pd
import pytest

from table_diffevo import gap_l1_diffusion as gap
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


MODE = gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX


def _schema(*names: str) -> Schema:
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in names
    ])


def _equals(attribute: str, value: int = 1) -> dict:
    return {"attribute": attribute, "operator": "==", "value": value}


def _channels(
    counts: np.ndarray, targets: np.ndarray, n_records: int
) -> tuple[float, float]:
    residuals = np.abs(targets - counts)
    absolute = float(np.mean(residuals) / n_records)
    positive = targets > 0
    if not np.any(positive):
        return absolute, 0.0
    inverse = 1.0 / targets[positive]
    relative = float(
        np.sum(residuals[positive] * inverse)
        / np.sum(inverse)
        / n_records
    )
    return absolute, relative


@pytest.mark.parametrize(
    "errors,dominant",
    [
        ([1.0, 1.0], "tie"),
        ([0.0, 10.0], "absolute"),
        ([1.0, 0.0], "relative"),
    ],
)
def test_dual_objective_calibrates_channels_and_uses_worse_one(
    errors, dominant
):
    targets = np.array([1.0, 100.0])
    counts = targets + np.asarray(errors)
    absolute, relative = _channels(counts, targets, n_records=100)
    observed = gap.normalized_gap_l1_error(
        counts,
        targets,
        n_records=100,
        weighting=MODE,
    )

    if dominant == "tie":
        assert absolute == relative == 0.01
    elif dominant == "absolute":
        assert absolute > relative
    else:
        assert relative > absolute
    assert observed == max(absolute, relative)


def test_dual_switch_has_no_hysteresis_or_channel_sum():
    targets = np.array([1.0, 3.0])
    expected = []
    for errors in (
        np.array([2.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 2.0]),
    ):
        counts = targets + errors
        absolute, relative = _channels(counts, targets, n_records=10)
        objective = gap.normalized_gap_l1_error(
            counts, targets, n_records=10, weighting=MODE
        )
        expected.append((absolute, relative, objective))

    assert expected[0][1] > expected[0][0]
    assert expected[1][0] == expected[1][1] == 0.1
    assert expected[2][0] > expected[2][1]
    assert [row[2] for row in expected] == [
        max(row[0], row[1]) for row in expected
    ]
    assert expected[1][2] != sum(expected[1][:2])


def test_zero_targets_enter_only_absolute_and_all_zero_falls_back_to_it():
    mixed_counts = np.array([2.0, 10.0])
    mixed_targets = np.array([0.0, 10.0])
    assert gap.normalized_gap_l1_error(
        mixed_counts,
        mixed_targets,
        n_records=100,
        weighting=MODE,
    ) == 0.01

    all_zero_counts = np.array([2.0, 4.0])
    all_zero_targets = np.array([0.0, 0.0])
    assert gap.normalized_gap_l1_error(
        all_zero_counts,
        all_zero_targets,
        n_records=100,
        weighting=MODE,
    ) == 0.03


def test_dual_has_no_floor_or_ratio_parameter():
    targets = np.array([0.0, 1.0, 100.0])
    counts = np.array([2.0, 2.0, 110.0])
    first = gap.normalized_gap_l1_error(
        counts, targets, n_records=300, floor=1.0, weighting=MODE
    )
    second = gap.normalized_gap_l1_error(
        counts, targets, n_records=300, floor=1e12, weighting=MODE
    )
    assert first == second

    with pytest.raises(ValueError, match="dual_abs_relative_max.*不允许"):
        gap.normalized_gap_l1_error(
            counts,
            targets,
            n_records=300,
            weighting=MODE,
            max_weight_ratio=8.0,
        )
    with pytest.raises(ValueError, match="n_records"):
        gap.normalized_gap_l1_error(counts, targets, weighting=MODE)


def test_dual_rejects_negative_targets_without_changing_old_modes():
    with pytest.raises(ValueError, match="非负计数"):
        gap.normalized_gap_l1_error(
            np.array([0.0]),
            np.array([-1.0]),
            n_records=10,
            weighting=MODE,
        )

    # 这是旧模式的历史输入边界；新模式的验证不得改它。
    assert gap.normalized_gap_l1_error(
        np.array([0.0]), np.array([-1.0])
    ) == 0.125


def test_dual_denominators_and_relative_weights_match_frozen_formula():
    targets = np.array([0.0, 1.0, 4.0, 20.0])
    denominators, spec = gap._build_gap_l1_denominators(
        targets,
        floor=8.0,
        weighting=MODE,
        max_weight_ratio=None,
        n_records=100,
    )
    weights = gap._build_dual_relative_weights(
        targets, n_records=100, weighting=spec
    )

    np.testing.assert_array_equal(denominators, np.full(4, 100.0))
    assert weights is not None
    assert weights[0] == 0.0
    np.testing.assert_allclose(
        weights[1:] * 100,
        (1.0 / targets[1:]) / np.sum(1.0 / targets[1:]),
        rtol=0,
        atol=1e-17,
    )
    assert np.sum(weights) == pytest.approx(0.01)
    assert spec.actual_weight_ratio == 1.0
    assert spec.positive_target_query_count == 3
    assert spec.relative_inverse_target_normalizer == 1.3
    assert spec.relative_positive_weight_ratio == 20.0


def test_dual_incremental_conditions_and_commits_match_full_recounts():
    schema = _schema("a", "b")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": [0, 1, 0], "b": [0, 0, 1]})
    donors = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 0]})
    targets = np.array([0.0, 1.5, 1.0, 2.0])
    plan = gap._prepare_plan(
        current,
        donors,
        schema,
        queries,
        targets,
        evaluate_table(current, queries),
        np.ones(len(current), dtype=bool),
        np.zeros((len(current), 2), dtype=bool),
        floor=8.0,
        compiled_workload=None,
        weighting=MODE,
    )

    for step, coordinate in enumerate(plan.active_coordinates):
        row_index, attribute_index = map(int, coordinate)
        e0, e1, failures0, failures1, indicators0, indicators1 = (
            gap._condition_pair(plan, row_index, attribute_index)
        )
        expected = []
        for selected in (False, True):
            candidate_mask = plan.mask.copy()
            candidate_mask[row_index, attribute_index] = selected
            frame = gap._materialize_copy_table(
                current,
                donors,
                plan.compiled.attribute_names,
                candidate_mask,
            )
            candidate_counts = evaluate_table(frame, queries)
            expected.append(gap.normalized_gap_l1_error(
                candidate_counts,
                targets,
                n_records=len(current),
                weighting=MODE,
            ))
        np.testing.assert_allclose([e0, e1], expected, rtol=0, atol=1e-16)

        selected = bool(step % 2 == 0)
        gap._set_coordinate(
            plan,
            row_index,
            attribute_index,
            selected,
            failures1 if selected else failures0,
            indicators1 if selected else indicators0,
        )
        frame = gap._materialize_copy_table(
            current, donors, plan.compiled.attribute_names, plan.mask
        )
        recounted = evaluate_table(frame, queries)
        np.testing.assert_array_equal(plan.plan_counts, recounted)
        assert gap._gap_l1_energy_numpy(plan) == pytest.approx(
            gap.normalized_gap_l1_error(
                recounted,
                targets,
                n_records=len(current),
                weighting=MODE,
            ),
            abs=1e-16,
        )


def test_dual_scan_identity_diagnostics_and_rng_replay():
    schema = _schema("a")
    queries = [
        {"conditions": [_equals("a", 1)]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": [0, 0, 1, 0]})
    donors = pd.DataFrame({"a": [1, 1, 0, 1]})
    targets = np.array([0.0, 3.0])
    counts = evaluate_table(current, queries)
    kwargs = {
        "participate": np.ones(len(current), dtype=bool),
        "initial_mask": np.zeros((len(current), 1), dtype=bool),
        "reference_scale": 0.1,
        "n_sweeps": 8,
        "weighting": MODE,
    }
    first_rng = np.random.default_rng(20260902)
    second_rng = np.random.default_rng(20260902)
    initial_state = copy.deepcopy(first_rng.bit_generator.state)

    first = gap.evolve_step_gap_l1_global(
        current, donors, schema, queries, targets, counts, rng=first_rng, **kwargs
    )
    second = gap.evolve_step_gap_l1_global(
        current, donors, schema, queries, targets, counts, rng=second_rng, **kwargs
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    diagnostics = first[2]
    assert diagnostics["kernel"].endswith(MODE)
    assert diagnostics["gap_l1_weighting"] == MODE
    assert diagnostics["gap_l1_max_weight_ratio"] is None
    assert diagnostics["gap_l1_smoothing_count"] is None
    assert diagnostics["gap_l1_actual_weight_ratio"] == 1.0
    assert diagnostics["gap_l1_channel_aggregation"] == "max"
    assert diagnostics["gap_l1_zero_target_policy"] == (
        "absolute_channel_only"
    )
    assert diagnostics["gap_l1_floor_applied"] is False
    assert diagnostics["gap_l1_positive_target_query_count"] == 1
    assert diagnostics["gap_l1_relative_inverse_target_normalizer"] == (
        1.0 / 3.0
    )
    assert diagnostics["gap_l1_relative_positive_weight_ratio"] == 1.0
    assert diagnostics["gibbs_microsteps"] == (
        8 * diagnostics["active_switches_k"]
    )
    assert diagnostics["microstep_trace_sha256"] == (
        second[2]["microstep_trace_sha256"]
    )
    assert first_rng.bit_generator.state != initial_state
    assert first_rng.bit_generator.state == second_rng.bit_generator.state


def test_dual_does_not_claim_linear_integer_cancellation_rule():
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]}]
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
            weighting=MODE,
        )
