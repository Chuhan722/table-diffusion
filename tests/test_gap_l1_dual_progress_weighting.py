"""A/R 相对初始进度 ``max`` 的公式、边界和扫描契约。"""

import numpy as np
import pandas as pd
import pytest

from table_diffevo import gap_l1_diffusion as gap
from table_diffevo.evolution import run_evolution
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


MODE = gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_PROGRESS_MAX


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


def test_progress_objective_divides_each_channel_by_its_initial_value():
    targets = np.array([0.0, 10.0, 50.0])
    initial_counts = np.array([10.0, 30.0, 20.0])
    reference = gap.build_gap_l1_channel_reference(
        initial_counts, targets, n_records=100
    )

    assert reference.absolute_initial == pytest.approx(0.2)
    assert reference.relative_initial == pytest.approx(13.0 / 60.0)
    assert reference.source == "initial_current_before_round_1"

    counts = np.array([4.0, 20.0, 40.0])
    observed = gap.normalized_gap_l1_error(
        counts,
        targets,
        n_records=100,
        weighting=MODE,
        channel_reference=reference,
    )
    absolute_progress = 0.08 / reference.absolute_initial
    relative_progress = 0.1 / reference.relative_initial
    assert relative_progress > absolute_progress
    assert observed == pytest.approx(relative_progress)


def test_progress_reference_is_required_and_rejected_by_old_modes():
    targets = np.array([1.0, 10.0])
    counts = np.array([2.0, 8.0])
    reference = gap.build_gap_l1_channel_reference(
        counts, targets, n_records=20
    )

    with pytest.raises(ValueError, match="要求显式提供"):
        gap.normalized_gap_l1_error(
            counts, targets, n_records=20, weighting=MODE
        )
    with pytest.raises(ValueError, match="只允许用于"):
        gap.normalized_gap_l1_error(
            counts,
            targets,
            n_records=20,
            weighting=gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
            channel_reference=reference,
        )


def test_progress_all_zero_targets_structurally_omit_relative_channel():
    targets = np.zeros(2)
    initial_counts = np.array([2.0, 4.0])
    reference = gap.build_gap_l1_channel_reference(
        initial_counts, targets, n_records=10
    )
    assert reference.absolute_initial == pytest.approx(0.3)
    assert reference.relative_initial is None

    observed = gap.normalized_gap_l1_error(
        np.array([1.0, 2.0]),
        targets,
        n_records=10,
        weighting=MODE,
        channel_reference=reference,
    )
    assert observed == 0.5


def test_progress_reference_rejects_zero_initial_absolute_or_relative():
    with pytest.raises(ValueError, match="A_init.*精确命中"):
        gap.build_gap_l1_channel_reference(
            np.array([0.0, 2.0]),
            np.array([0.0, 2.0]),
            n_records=10,
        )

    with pytest.raises(ValueError, match="R_init.*没有定义"):
        gap.build_gap_l1_channel_reference(
            np.array([3.0, 2.0]),
            np.array([0.0, 2.0]),
            n_records=10,
        )


def test_progress_incremental_condition_matches_full_recount():
    schema = _schema("a", "b")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": [0, 1, 0], "b": [0, 0, 1]})
    donors = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 0]})
    targets = np.array([0.5, 2.5, 1.5, 1.5])
    counts = evaluate_table(current, queries)
    reference = gap.build_gap_l1_channel_reference(
        counts, targets, n_records=len(current)
    )
    plan = gap._prepare_plan(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        np.ones(len(current), dtype=bool),
        np.zeros((len(current), 2), dtype=bool),
        floor=8.0,
        compiled_workload=None,
        weighting=MODE,
        channel_reference=reference,
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
            expected.append(gap.normalized_gap_l1_error(
                evaluate_table(frame, queries),
                targets,
                n_records=len(current),
                weighting=MODE,
                channel_reference=reference,
            ))
        np.testing.assert_allclose([e0, e1], expected, rtol=0, atol=1e-15)

        selected = bool(step % 2 == 0)
        gap._set_coordinate(
            plan,
            row_index,
            attribute_index,
            selected,
            failures1 if selected else failures0,
            indicators1 if selected else indicators0,
        )


def test_progress_scan_records_channel_dominance_and_final_channels():
    schema = _schema("a")
    queries = [
        {"conditions": [_equals("a", 1)]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": [0, 0, 1, 0]})
    donors = pd.DataFrame({"a": [1, 1, 0, 1]})
    targets = np.array([2.0, 3.0])
    counts = evaluate_table(current, queries)
    reference = gap.build_gap_l1_channel_reference(
        counts, targets, n_records=len(current)
    )
    _, _, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=np.ones(len(current), dtype=bool),
        initial_mask=np.zeros((len(current), 1), dtype=bool),
        reference_scale=0.1,
        rng=np.random.default_rng(20260903),
        n_sweeps=2,
        weighting=MODE,
        channel_reference=reference,
    )

    assert diagnostics["gap_l1_channel_aggregation"] == (
        "max_relative_to_initial"
    )
    assert diagnostics["gap_l1_channel_reference_source"] == (
        "initial_current_before_round_1"
    )
    assert diagnostics["gap_l1_absolute_channel_initial_reference"] == (
        reference.absolute_initial
    )
    assert diagnostics["gap_l1_relative_channel_initial_reference"] == (
        reference.relative_initial
    )
    dominance = diagnostics["gap_l1_channel_dominance_counts"]
    microsteps = diagnostics["gibbs_microsteps"]
    assert sum(dominance["candidate_0"].values()) == microsteps
    assert sum(dominance["candidate_1"].values()) == microsteps
    assert sum(dominance["pair"].values()) == microsteps
    final = diagnostics["gap_l1_final_channels"]
    assert final["dominant_channel"] in {"absolute", "relative", "tie"}
    assert final["absolute_relative_to_initial"] == pytest.approx(
        final["absolute_raw"] / reference.absolute_initial
    )
    assert final["relative_relative_to_initial"] == pytest.approx(
        final["relative_raw"] / reference.relative_initial
    )


def test_run_evolution_freezes_one_initial_reference_for_all_rounds():
    schema = _schema("a", "b", "c")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("a", 0), _equals("c")]},
    ]
    target = np.array([8.5, 10.5, 4.5, 2.5])
    _, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records=20,
        n_rounds=2,
        seed=17,
        rho=0.8,
        eta=0.5,
        mu=0.1,
        tol=float("inf"),
        device="numpy",
        distance_mode="geometric",
        residual_directed_diffusion=True,
        gap_l1_sweeps=8,
        gap_l1_weighting=MODE,
        log_every=100,
    )

    reference = diagnostics["gap_l1_channel_reference"]
    assert reference["source"] == "initial_current_before_round_1"
    attempts = [
        attempt
        for round_attempts in diagnostics[
            "gap_l1_attempt_diagnostics_history"
        ]
        for attempt in round_attempts
        if attempt["gap_l1_scan_applied"]
    ]
    assert attempts
    for attempt in attempts:
        assert attempt["gap_l1_absolute_channel_initial_reference"] == (
            reference["absolute_initial"]
        )
        assert attempt["gap_l1_relative_channel_initial_reference"] == (
            reference["relative_initial"]
        )


def test_exact_initial_state_stops_before_progress_reference_is_needed(
    monkeypatch,
):
    schema = _schema("a")
    queries = [{"conditions": [_equals("a")]}]
    initial = pd.DataFrame({"a": [0, 1, 0, 1]})
    monkeypatch.setattr(
        "table_diffevo.evolution.init_synthetic_table",
        lambda *args, **kwargs: initial.copy(),
    )
    result, diagnostics = run_evolution(
        np.array([2.0]),
        queries,
        schema,
        n_records=4,
        n_rounds=2,
        seed=9,
        gap_l1_sweeps=8,
        gap_l1_weighting=MODE,
        residual_directed_diffusion=True,
        tol=float("inf"),
        device="numpy",
        log_every=100,
    )

    pd.testing.assert_frame_equal(result, initial)
    assert diagnostics["stopped_early"] is True
    assert diagnostics["gap_l1_channel_reference"] is None
    assert diagnostics["gap_l1_microsteps"] == 0
