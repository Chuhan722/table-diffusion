"""Pure contracts for the terminal-output early-stop development diagnostic."""

import pytest

from scripts.analyze_issue53_terminal_early_stop_development import (
    audit_online_wiring_case,
    locate_continuation_checkpoints,
    replay_fixed_early_stop,
)


def test_best_refresh_can_trigger_clock_but_output_state_is_terminal_current() -> None:
    losses = [10.0, 8.0, 12.0, 11.0, 13.0, 9.0, 14.0, 15.0]
    decision = replay_fixed_early_stop(
        losses,
        participating_rows=[4] * 7,
        n_records=4,
        patience_ticks=6,
    )

    assert decision.reason == "early_stopped"
    assert decision.state_index == 7
    assert losses[decision.state_index] == 15.0
    assert min(losses[: decision.state_index + 1]) == 8.0


def test_new_best_on_stop_boundary_resets_early_stop_clock() -> None:
    decision = replay_fixed_early_stop(
        [10.0, 8.0, 12.0, 12.0, 12.0, 12.0, 12.0, 7.0],
        participating_rows=[4] * 7,
        n_records=4,
        patience_ticks=6,
    )

    assert decision.reason == "reference_horizon_reached"
    assert decision.consecutive_no_progress_ticks == 0


def test_zero_noise_A_uses_current_state_and_preempts_B() -> None:
    decision = replay_fixed_early_stop(
        [10.0, 8.0, 12.0, 0.0, 12.0],
        participating_rows=[1] * 4,
        n_records=4,
        patience_ticks=1,
    )

    assert decision.reason == "fit_target_reached"
    assert decision.state_index == 3
    assert decision.normalized_work == pytest.approx(0.75)


def test_continuation_checkpoints_use_first_real_state_at_or_beyond_target() -> None:
    checkpoints = locate_continuation_checkpoints(
        participating_rows=[4, 1, 3, 4, 2, 4],
        stop_state_index=2,
        n_records=4,
        patience_ticks=1,
        patience_multiples=(1, 2, 4),
    )

    first, second, censored = checkpoints
    assert first.status == "observed"
    assert first.state_index == 4
    assert first.target_normalized_work == pytest.approx(2.25)
    assert first.actual_normalized_work == pytest.approx(3.0)
    assert first.actual_extra_work == pytest.approx(1.75)

    assert second.status == "observed"
    assert second.state_index == 5
    assert second.target_normalized_work == pytest.approx(3.25)
    assert second.actual_normalized_work == pytest.approx(3.5)
    assert second.actual_extra_work == pytest.approx(2.25)

    assert censored.status == "right_censored_by_known_trace_horizon"
    assert censored.state_index is None
    assert censored.actual_normalized_work is None
    assert censored.actual_extra_work is None


def test_continuation_checkpoint_does_not_substitute_last_state_when_censored() -> None:
    (checkpoint,) = locate_continuation_checkpoints(
        participating_rows=[4, 4, 1],
        stop_state_index=2,
        n_records=4,
        patience_ticks=1,
        patience_multiples=(1,),
    )

    assert checkpoint.status == "right_censored_by_known_trace_horizon"
    assert checkpoint.target_normalized_work == pytest.approx(3.0)
    assert checkpoint.state_index is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stop_state_index": -1}, "available state"),
        ({"stop_state_index": 4}, "available state"),
        ({"stop_state_index": 0, "patience_multiples": (2, 1)}, "increasing"),
        ({"stop_state_index": 0, "patience_multiples": (1, 1)}, "increasing"),
    ],
)
def test_continuation_checkpoint_contract_rejects_invalid_identity(
    kwargs: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        locate_continuation_checkpoints(
            participating_rows=[1, 1, 1],
            n_records=4,
            patience_ticks=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    (
        "seed",
        "rho",
        "n_rounds",
        "reason",
        "stop_state",
        "stop_work",
        "terminal_loss",
    ),
    [
        (20260816, 1.0, 40, "early_stopped", 12, 12.0, 12.0),
        (20260817, 1.0, 40, "early_stopped", 20, 20.0, 7.0),
        (20260818, 1.0, 40, "early_stopped", 8, 8.0, 7.5),
        (20260816, 0.25, 160, "early_stopped", 27, 7.0625, 4.0),
        (20260817, 0.25, 160, "early_stopped", 64, 16.0, 1.0),
        (20260818, 0.25, 160, "fit_target_reached", 11, 2.8125, 0.0),
    ],
)
def test_online_wiring_exactly_matches_known_offline_replay(
    seed: int,
    rho: float,
    n_rounds: int,
    reason: str,
    stop_state: int,
    stop_work: float,
    terminal_loss: float,
) -> None:
    audit = audit_online_wiring_case(
        seed=seed,
        rho=rho,
        n_rounds=n_rounds,
    )

    assert audit["offline_replay_reason"] == reason
    assert audit["online_reason"] == reason
    assert audit["stop_state"] == stop_state
    assert audit["stop_normalized_work"] == pytest.approx(stop_work)
    assert audit["terminal_current_loss"] == pytest.approx(terminal_loss)
    assert audit["main_output_equals_terminal_current"] is True
    assert audit["current_metrics_prefix_equal"] is True
    assert audit["transition_clocks_prefix_equal"] is True
    assert audit["accept_history_prefix_equal"] is True
    assert audit["primary_rng_prefix_equal"] is True
    assert audit["candidate_evaluations_prefix_equal"] is True
    assert audit["resource_C_preempted_A_or_B"] is False
    assert audit["terminal_current_above_best_diagnostic_only"] is (
        reason == "early_stopped"
    )
