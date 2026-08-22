"""Pure contracts for the frozen Issue #53 two-level alpha controller."""

import json
from dataclasses import asdict

import pytest

from table_diffevo.inner_early_stopping import (
    EarlyStoppingConfig,
    InnerEarlyStopper,
)
from table_diffevo.stall_escape_alpha import (
    StallEscapeAlphaConfig,
    StallEscapeAlphaController,
)


def _started(*, n_records: int = 4, initial_loss: float = 10.0):
    stopper = InnerEarlyStopper(
        EarlyStoppingConfig(n_records=n_records, patience_ticks=6)
    )
    controller = StallEscapeAlphaController(StallEscapeAlphaConfig())
    initial = stopper.observe_initial(initial_loss)
    controller.observe_initial(initial)
    return stopper, controller


def _round(
    stopper: InnerEarlyStopper,
    controller: StallEscapeAlphaController,
    *,
    loss: float,
    participating_rows: int,
    resource_cap_reached: bool = False,
):
    alpha_used = controller.alpha_for_next_round
    stopping = stopper.observe_post_round(
        current_loss=loss,
        participating_rows=participating_rows,
        resource_cap_reached=resource_cap_reached,
    )
    observation = controller.observe_post_round(stopping)
    assert observation.alpha_used == alpha_used
    return alpha_used, stopping, observation


def test_two_stall_ticks_trigger_exactly_two_escape_ticks_then_restore() -> None:
    stopper, controller = _started()
    rounds = [
        _round(stopper, controller, loss=loss, participating_rows=4)
        for loss in (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    ]

    assert [row[0] for row in rounds] == [16.0, 16.0, 12.0, 12.0, 16.0, 16.0]
    assert rounds[0][2].events == ()
    assert rounds[1][2].events == ("escape_started",)
    assert rounds[1][2].alpha_for_next_round == 12.0
    assert rounds[2][2].events == ("escape_tick_completed",)
    assert rounds[2][2].escape_ticks_remaining_after == 1
    assert rounds[3][2].events == (
        "escape_tick_completed",
        "escape_completed",
    )
    assert rounds[3][2].alpha_for_next_round == 16.0
    assert rounds[-1][1].termination_reason == "early_stopped"
    assert rounds[-1][2].events == ("termination_observed",)
    assert controller.escape_count == 1
    assert controller.terminated is True


def test_new_best_during_escape_resets_patience_but_finishes_pulse() -> None:
    stopper, controller = _started()
    rounds = [
        _round(stopper, controller, loss=loss, participating_rows=4)
        for loss in (11.0, 12.0, 9.0, 11.0, 12.0)
    ]

    assert [row[0] for row in rounds] == [16.0, 16.0, 12.0, 12.0, 16.0]
    during_escape = rounds[2][2]
    assert during_escape.events == (
        "new_best",
        "escape_tick_completed",
    )
    assert during_escape.progress_epoch_after == 1
    assert during_escape.consecutive_no_progress_ticks == 0
    assert during_escape.escape_used_in_progress_epoch is False

    pulse_end = rounds[3][2]
    assert pulse_end.events == (
        "escape_tick_completed",
        "escape_completed",
    )
    assert pulse_end.consecutive_no_progress_ticks == 1
    assert pulse_end.alpha_for_next_round == 16.0

    next_epoch_trigger = rounds[4][2]
    assert next_epoch_trigger.events == ("escape_started",)
    assert next_epoch_trigger.escape_count == 2
    assert next_epoch_trigger.alpha_for_next_round == 12.0


def test_new_best_on_second_tick_prevents_trigger() -> None:
    stopper, controller = _started()
    _round(stopper, controller, loss=11.0, participating_rows=4)

    _, stopping, observation = _round(
        stopper,
        controller,
        loss=9.0,
        participating_rows=4,
    )

    assert stopping.best_updated is True
    assert stopping.completed_tick_had_progress is True
    assert stopping.consecutive_no_progress_ticks == 0
    assert observation.events == ("new_best",)
    assert observation.alpha_for_next_round == 16.0
    assert controller.escape_count == 0


def test_escape_duration_counts_work_not_rounds() -> None:
    stopper, controller = _started()
    rounds = [
        _round(stopper, controller, loss=11.0, participating_rows=2)
        for _ in range(8)
    ]

    assert [row[0] for row in rounds] == [
        16.0,
        16.0,
        16.0,
        16.0,
        12.0,
        12.0,
        12.0,
        12.0,
    ]
    assert rounds[3][2].events == ("escape_started",)
    assert rounds[4][2].events == ()
    assert rounds[5][2].events == ("escape_tick_completed",)
    assert rounds[6][2].events == ()
    assert rounds[7][2].events == (
        "escape_tick_completed",
        "escape_completed",
    )
    assert controller.alpha_for_next_round == 16.0


def test_terminal_A_does_not_schedule_or_advance_future_alpha() -> None:
    stopper, controller = _started()
    _round(stopper, controller, loss=11.0, participating_rows=4)
    _round(stopper, controller, loss=12.0, participating_rows=4)

    alpha_used, stopping, observation = _round(
        stopper,
        controller,
        loss=0.0,
        participating_rows=4,
    )

    assert alpha_used == 12.0
    assert stopping.termination_reason == "fit_target_reached"
    assert observation.events == ("termination_observed",)
    assert observation.alpha_for_next_round is None
    assert observation.escape_ticks_remaining_before == 2
    assert observation.escape_ticks_remaining_after == 2
    with pytest.raises(RuntimeError, match="terminated"):
        _ = controller.alpha_for_next_round


def test_observation_is_strict_json_and_contains_no_budget_field() -> None:
    stopper, controller = _started()
    _, _, observation = _round(
        stopper,
        controller,
        loss=11.0,
        participating_rows=4,
    )

    payload = asdict(observation)
    json.dumps(payload, allow_nan=False)
    assert "n_rounds" not in payload
    assert "candidate_budget" not in payload


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"normal_alpha": True}, TypeError),
        ({"normal_alpha": float("inf")}, ValueError),
        ({"escape_alpha": 16.0}, ValueError),
        ({"stall_trigger_ticks": 0}, ValueError),
        ({"escape_duration_ticks": 1.5}, TypeError),
    ],
)
def test_config_rejects_invalid_values(overrides, error) -> None:
    with pytest.raises(error):
        StallEscapeAlphaConfig(**overrides)
