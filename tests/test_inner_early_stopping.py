"""Pure boundary contracts for terminal-current A/B/C early stopping."""

import math

import pytest

from table_diffevo.inner_early_stopping import (
    EarlyStoppingConfig,
    InnerEarlyStopper,
)


def _started(
    *,
    initial_loss: float = 10.0,
    n_records: int = 4,
    patience_ticks: int = 6,
) -> InnerEarlyStopper:
    stopper = InnerEarlyStopper(
        EarlyStoppingConfig(
            n_records=n_records,
            patience_ticks=patience_ticks,
        )
    )
    decision = stopper.observe_initial(initial_loss)
    assert decision.termination_reason == "in_progress"
    assert decision.terminal_output_state_index is None
    assert decision.terminal_output_loss is None
    return stopper


def test_A_stops_on_exact_initial_current_without_work() -> None:
    stopper = InnerEarlyStopper(EarlyStoppingConfig(n_records=10))

    decision = stopper.observe_initial(0.0)

    assert decision.should_stop is True
    assert decision.termination_reason == "fit_target_reached"
    assert decision.fit_target_reached is True
    assert decision.inner_complete is True
    assert decision.state_index == 0
    assert decision.terminal_output_state_index == 0
    assert decision.terminal_output_loss == 0.0
    assert decision.cumulative_participating_rows == 0


def test_A_preempts_external_C_on_the_initial_current_state() -> None:
    stopper = InnerEarlyStopper(EarlyStoppingConfig(n_records=10))

    decision = stopper.observe_initial(
        0.0,
        resource_cap_reached=True,
    )

    assert decision.external_resource_cap_reached is True
    assert decision.termination_reason == "fit_target_reached"
    assert decision.inner_complete is True
    assert decision.terminal_output_state_index == 0


def test_external_C_can_return_the_initial_current_state() -> None:
    stopper = InnerEarlyStopper(EarlyStoppingConfig(n_records=10))

    decision = stopper.observe_initial(
        5.0,
        resource_cap_reached=True,
    )

    assert decision.termination_reason == "resource_cap_reached"
    assert decision.inner_complete is False
    assert decision.terminal_output_state_index == 0
    assert decision.terminal_output_loss == 5.0


def test_B_uses_best_refresh_for_timing_but_outputs_terminal_current() -> None:
    stopper = _started()
    decisions = [
        stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        for loss in (8.0, 12.0, 11.0, 13.0, 9.0, 14.0, 15.0)
    ]

    decision = decisions[-1]
    assert decision.termination_reason == "early_stopped"
    assert decision.inner_complete is True
    assert decision.fit_target_reached is False
    assert decision.consecutive_no_progress_ticks == 6
    assert decision.current_loss == 15.0
    assert decision.best_loss_diagnostic_only == 8.0
    assert decision.best_state_index_diagnostic_only == 1
    assert decision.terminal_output_state_index == 7
    assert decision.terminal_output_loss == 15.0


def test_new_best_on_B_boundary_resets_patience_instead_of_stopping() -> None:
    stopper = _started()
    for loss in (8.0, 12.0, 11.0, 13.0, 9.0, 14.0):
        decision = stopper.observe_post_round(
            current_loss=loss,
            participating_rows=4,
        )
        assert decision.should_stop is False

    boundary = stopper.observe_post_round(
        current_loss=7.5,
        participating_rows=4,
    )

    assert boundary.completed_work_ticks == 7
    assert boundary.best_updated is True
    assert boundary.completed_tick_had_progress is True
    assert boundary.consecutive_no_progress_ticks == 0
    assert boundary.termination_reason == "in_progress"


def test_falling_current_above_best_is_not_progress() -> None:
    stopper = _started(initial_loss=8.0, patience_ticks=2)

    first = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=4,
    )
    second = stopper.observe_post_round(
        current_loss=9.0,
        participating_rows=4,
    )

    assert first.best_updated is False
    assert second.best_updated is False
    assert second.best_loss_diagnostic_only == 8.0
    assert second.termination_reason == "early_stopped"
    assert second.terminal_output_loss == 9.0


def test_natural_work_tick_accumulates_participating_rows() -> None:
    stopper = _started(n_records=10)

    first = stopper.observe_post_round(
        current_loss=11.0,
        participating_rows=3,
    )
    second = stopper.observe_post_round(
        current_loss=9.0,
        participating_rows=2,
    )
    third = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=5,
    )

    assert first.work_tick_completed is False
    assert first.normalized_work == pytest.approx(0.3)
    assert second.work_tick_completed is False
    assert second.normalized_work == pytest.approx(0.5)
    assert third.work_tick_completed is True
    assert third.normalized_work == 1.0
    assert third.completed_tick_had_progress is True
    assert third.consecutive_no_progress_ticks == 0


def test_B_only_fires_on_a_real_natural_work_boundary() -> None:
    stopper = _started(n_records=10, patience_ticks=1)

    partial = stopper.observe_post_round(
        current_loss=11.0,
        participating_rows=9,
    )
    boundary = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=1,
    )

    assert partial.should_stop is False
    assert partial.consecutive_no_progress_ticks == 0
    assert boundary.work_tick_completed is True
    assert boundary.termination_reason == "early_stopped"


def test_A_preempts_B_and_C_on_the_same_current_state() -> None:
    stopper = _started(initial_loss=2.0, patience_ticks=1)

    decision = stopper.observe_post_round(
        current_loss=0.0,
        participating_rows=4,
        resource_cap_reached=True,
    )

    assert decision.external_resource_cap_reached is True
    assert decision.termination_reason == "fit_target_reached"
    assert decision.fit_target_reached is True
    assert decision.terminal_output_loss == 0.0


def test_B_preempts_C_on_the_same_current_state() -> None:
    stopper = _started(patience_ticks=1)

    decision = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=4,
        resource_cap_reached=True,
    )

    assert decision.external_resource_cap_reached is True
    assert decision.termination_reason == "early_stopped"
    assert decision.inner_complete is True
    assert decision.terminal_output_loss == 12.0


def test_C_is_external_and_outputs_current_instead_of_historical_best() -> None:
    stopper = _started()
    improved = stopper.observe_post_round(
        current_loss=5.0,
        participating_rows=1,
    )
    assert improved.should_stop is False

    decision = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=0,
        resource_cap_reached=True,
    )

    assert decision.termination_reason == "resource_cap_reached"
    assert decision.inner_complete is False
    assert decision.fit_target_reached is False
    assert decision.current_loss == 12.0
    assert decision.best_loss_diagnostic_only == 5.0
    assert decision.terminal_output_state_index == 2
    assert decision.terminal_output_loss == 12.0


def test_patience_is_one_global_configurable_integer() -> None:
    stopper = _started(patience_ticks=2)

    first = stopper.observe_post_round(
        current_loss=11.0,
        participating_rows=4,
    )
    second = stopper.observe_post_round(
        current_loss=12.0,
        participating_rows=4,
    )

    assert first.should_stop is False
    assert second.termination_reason == "early_stopped"
    assert second.completed_work_ticks == 2


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf, -0.5])
def test_nonfinite_or_negative_loss_fails_closed(loss) -> None:
    stopper = InnerEarlyStopper(EarlyStoppingConfig(n_records=4))

    with pytest.raises(ValueError, match="current_loss"):
        stopper.observe_initial(loss)


@pytest.mark.parametrize("loss", [True, "1.0"])
def test_nonreal_loss_fails_with_type_error(loss) -> None:
    stopper = InnerEarlyStopper(EarlyStoppingConfig(n_records=4))

    with pytest.raises(TypeError, match="current_loss"):
        stopper.observe_initial(loss)


@pytest.mark.parametrize("participating_rows", [-1, 5])
def test_out_of_range_participating_rows_fail_closed(participating_rows) -> None:
    stopper = _started()

    with pytest.raises(ValueError, match="participating_rows"):
        stopper.observe_post_round(
            current_loss=10.0,
            participating_rows=participating_rows,
        )


@pytest.mark.parametrize("participating_rows", [1.5, True, "1"])
def test_noninteger_participating_rows_fail_with_type_error(
    participating_rows,
) -> None:
    stopper = _started()

    with pytest.raises(TypeError, match="participating_rows"):
        stopper.observe_post_round(
            current_loss=10.0,
            participating_rows=participating_rows,
        )


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"n_records": 0}, ValueError),
        ({"n_records": True}, TypeError),
        ({"n_records": 4, "patience_ticks": 0}, ValueError),
        ({"n_records": 4, "patience_ticks": 1.5}, TypeError),
    ],
)
def test_invalid_configuration_fails_closed(changes, error_type) -> None:
    with pytest.raises(error_type):
        EarlyStoppingConfig(**changes)


def test_resource_cap_flag_must_be_boolean() -> None:
    stopper = _started()

    with pytest.raises(TypeError, match="resource_cap_reached"):
        stopper.observe_post_round(
            current_loss=10.0,
            participating_rows=1,
            resource_cap_reached=1,
        )


def test_lifecycle_rejects_missing_duplicate_or_post_terminal_states() -> None:
    uninitialized = InnerEarlyStopper(EarlyStoppingConfig(n_records=4))
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
            current_loss=9.5,
            participating_rows=4,
            normalized_l1=0.01,
        )
