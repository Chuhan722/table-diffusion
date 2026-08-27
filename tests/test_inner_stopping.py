"""Artificial boundary tests for the exact-target best-loss stopper."""

import math

import pytest

from table_diffevo.inner_stopping import (
    BestLossInnerStopper,
    BestLossStoppingConfig,
)


def _started(
    *,
    initial_loss: float = 100.0,
    n_records: int = 4,
    block_windows: int = 3,
    cap: int = 20,
) -> BestLossInnerStopper:
    stopper = BestLossInnerStopper(
        BestLossStoppingConfig(
            n_records=n_records,
            stall_block_windows=block_windows,
            max_normalized_work=cap,
        )
    )
    decision = stopper.observe_initial(initial_loss)
    assert decision.termination_reason == "in_progress"
    return stopper


def test_exact_initial_state_stops_without_post_round_or_work() -> None:
    stopper = BestLossInnerStopper(BestLossStoppingConfig(n_records=10))

    decision = stopper.observe_initial(0.0)

    assert decision.should_stop is True
    assert decision.termination_reason == "exact_residual"
    assert decision.inner_complete is True
    assert decision.best_loss == 0.0
    assert decision.best_updated is True
    assert decision.cumulative_participating_rows == 0
    assert decision.completed_work_windows == 0


def test_temporary_ascent_is_allowed_and_later_new_best_resets_stall() -> None:
    stopper = _started()

    first = stopper.observe_post_round(
        current_loss=110.0,
        participating_rows=4,
    )
    second = stopper.observe_post_round(
        current_loss=105.0,
        participating_rows=4,
    )
    third = stopper.observe_post_round(
        current_loss=95.0,
        participating_rows=4,
    )

    assert first.best_loss == 100.0
    assert first.consecutive_no_progress_windows == 1
    assert second.best_loss == 100.0
    assert second.consecutive_no_progress_windows == 2
    assert third.current_loss == 95.0
    assert third.best_loss == 95.0
    assert third.best_updated is True
    assert third.completed_window_had_progress is True
    assert third.consecutive_no_progress_windows == 0
    assert third.termination_reason == "in_progress"


def test_falling_loss_above_best_needs_candidate_and_confirmation_blocks() -> None:
    stopper = _started()

    decisions = [
        stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        for loss in (110.0, 105.0, 101.0, 108.0, 104.0, 102.0)
    ]

    assert [row.best_updated for row in decisions] == [False] * 6
    assert [row.best_loss for row in decisions] == [100.0] * 6
    assert decisions[2].should_stop is False
    assert decisions[2].consecutive_no_progress_windows == 3
    assert decisions[-1].should_stop is True
    assert decisions[-1].termination_reason == "optimization_stalled"
    assert decisions[-1].inner_complete is True
    assert decisions[-1].consecutive_no_progress_windows == 6


def test_new_best_on_final_confirmation_boundary_cancels_stall() -> None:
    stopper = _started()
    for loss in (110.0, 105.0, 101.0, 108.0, 104.0):
        decision = stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        assert decision.should_stop is False

    decision = stopper.observe_post_round(
        current_loss=95.0,
        participating_rows=4,
    )

    assert decision.completed_work_windows == 6
    assert decision.best_updated is True
    assert decision.best_loss == 95.0
    assert decision.consecutive_no_progress_windows == 0
    assert decision.termination_reason == "in_progress"


def test_work_window_uses_accumulated_participating_rows() -> None:
    stopper = _started(n_records=10)

    first = stopper.observe_post_round(
        current_loss=110.0,
        participating_rows=3,
    )
    second = stopper.observe_post_round(
        current_loss=105.0,
        participating_rows=2,
    )
    third = stopper.observe_post_round(
        current_loss=99.5,
        participating_rows=5,
    )

    assert first.work_window_completed is False
    assert first.normalized_work == pytest.approx(0.3)
    assert second.work_window_completed is False
    assert second.normalized_work == pytest.approx(0.5)
    assert third.work_window_completed is True
    assert third.normalized_work == 1.0
    assert third.completed_work_windows == 1
    assert third.completed_window_had_progress is True
    assert third.consecutive_no_progress_windows == 0


def test_equal_loss_is_not_progress_but_half_unit_improvement_is() -> None:
    stopper = _started(initial_loss=10.0)

    equal = stopper.observe_post_round(
        current_loss=10.0,
        participating_rows=4,
    )
    improved = stopper.observe_post_round(
        current_loss=9.5,
        participating_rows=4,
    )

    assert equal.best_updated is False
    assert equal.consecutive_no_progress_windows == 1
    assert improved.best_updated is True
    assert improved.best_loss == 9.5
    assert improved.consecutive_no_progress_windows == 0


def test_exact_at_confirmation_boundary_beats_stall_and_resource_cap() -> None:
    stopper = _started(initial_loss=2.0, cap=6)
    for loss in (3.0, 2.5, 4.0, 3.5, 2.5):
        stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )

    decision = stopper.observe_post_round(
        current_loss=0.0,
        participating_rows=4,
    )

    assert decision.completed_work_windows == 6
    assert decision.best_updated is True
    assert decision.consecutive_no_progress_windows == 0
    assert decision.termination_reason == "exact_residual"
    assert decision.inner_complete is True


def test_stall_beats_resource_cap_at_the_same_boundary() -> None:
    stopper = _started(cap=6)
    decisions = [
        stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        for loss in (110.0, 105.0, 101.0, 108.0, 104.0, 102.0)
    ]
    decision = decisions[-1]

    assert decision.normalized_work == 6.0
    assert decision.termination_reason == "optimization_stalled"
    assert decision.inner_complete is True


def test_continued_progress_can_reach_fail_closed_resource_cap() -> None:
    stopper = _started(initial_loss=10.0, cap=6)

    decisions = [
        stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        for loss in (9.5, 9.0, 8.5, 8.0, 7.5, 7.0)
    ]

    assert all(row.best_updated for row in decisions)
    assert decisions[-1].normalized_work == 6.0
    assert decisions[-1].termination_reason == "resource_cap_reached"
    assert decisions[-1].inner_complete is False


@pytest.mark.parametrize(
    "loss",
    [math.nan, math.inf, -math.inf, -0.5],
)
def test_nonfinite_or_negative_initial_loss_fails_closed(loss) -> None:
    stopper = BestLossInnerStopper(BestLossStoppingConfig(n_records=4))

    with pytest.raises(ValueError, match="current_loss"):
        stopper.observe_initial(loss)


@pytest.mark.parametrize(
    "loss",
    [True, "1.0"],
)
def test_nonreal_initial_loss_fails_with_type_error(loss) -> None:
    stopper = BestLossInnerStopper(BestLossStoppingConfig(n_records=4))

    with pytest.raises(TypeError, match="current_loss"):
        stopper.observe_initial(loss)


@pytest.mark.parametrize("participating_rows", [-1, 5])
def test_out_of_range_participating_rows_fail_closed(
    participating_rows,
) -> None:
    stopper = _started()

    with pytest.raises(ValueError, match="participating_rows"):
        stopper.observe_post_round(
            current_loss=100.0,
            participating_rows=participating_rows,
        )


@pytest.mark.parametrize(
    "participating_rows",
    [1.5, True, "1"],
)
def test_noninteger_participating_rows_fail_with_type_error(
    participating_rows,
) -> None:
    stopper = _started()

    with pytest.raises(TypeError, match="participating_rows"):
        stopper.observe_post_round(
            current_loss=100.0,
            participating_rows=participating_rows,
        )


@pytest.mark.parametrize(
    "changes,error_type",
    [
        ({"n_records": 0}, ValueError),
        ({"n_records": True}, TypeError),
        ({"n_records": 4, "stall_block_windows": 0}, ValueError),
        ({"n_records": 4, "max_normalized_work": 0}, ValueError),
        (
            {
                "n_records": 4,
                "stall_block_windows": 4,
                "max_normalized_work": 7,
            },
            ValueError,
        ),
    ],
)
def test_invalid_configuration_fails_closed(changes, error_type) -> None:
    with pytest.raises(error_type):
        BestLossStoppingConfig(**changes)


def test_lifecycle_rejects_missing_duplicate_or_post_terminal_states() -> None:
    uninitialized = BestLossInnerStopper(BestLossStoppingConfig(n_records=4))
    with pytest.raises(RuntimeError, match="initial state"):
        uninitialized.observe_post_round(
            current_loss=1.0,
            participating_rows=4,
        )

    stopper = _started(initial_loss=0.5)
    with pytest.raises(RuntimeError, match="already been observed"):
        stopper.observe_initial(0.5)

    terminal = stopper.observe_post_round(
        current_loss=0.0,
        participating_rows=4,
    )
    assert terminal.should_stop is True
    with pytest.raises(RuntimeError, match="already been reached"):
        stopper.observe_post_round(
            current_loss=1.0,
            participating_rows=4,
        )


def test_stopper_api_does_not_accept_l1_or_reference_inputs() -> None:
    stopper = _started()

    with pytest.raises(TypeError):
        stopper.observe_post_round(
            current_loss=99.5,
            participating_rows=4,
            normalized_l1=0.01,
        )
