"""Horizon-invariant two-level alpha controller for Issue #53.

The controller is deliberately small and deterministic.  It consumes only
completed observations from :mod:`table_diffevo.inner_early_stopping`, never
reads a round budget, and never owns a random-number generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

from table_diffevo.inner_early_stopping import EarlyStoppingDecision


STALL_ESCAPE_ALPHA_SCHEDULE_MODE = "stall_escape_16_12"
DEFAULT_NORMAL_ALPHA = 16.0
DEFAULT_ESCAPE_ALPHA = 12.0
DEFAULT_STALL_TRIGGER_TICKS = 2
DEFAULT_ESCAPE_DURATION_TICKS = 2
REQUIRED_EARLY_STOPPING_PATIENCE_TICKS = 6

AlphaPhase = Literal["normal", "escape"]
AlphaEvent = Literal[
    "new_best",
    "escape_started",
    "escape_tick_completed",
    "escape_completed",
    "termination_observed",
]


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


@dataclass(frozen=True)
class StallEscapeAlphaConfig:
    """Frozen two-level schedule parameters."""

    normal_alpha: float = DEFAULT_NORMAL_ALPHA
    escape_alpha: float = DEFAULT_ESCAPE_ALPHA
    stall_trigger_ticks: int = DEFAULT_STALL_TRIGGER_TICKS
    escape_duration_ticks: int = DEFAULT_ESCAPE_DURATION_TICKS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_alpha",
            _finite_nonnegative(self.normal_alpha, name="normal_alpha"),
        )
        object.__setattr__(
            self,
            "escape_alpha",
            _finite_nonnegative(self.escape_alpha, name="escape_alpha"),
        )
        object.__setattr__(
            self,
            "stall_trigger_ticks",
            _positive_integer(
                self.stall_trigger_ticks,
                name="stall_trigger_ticks",
            ),
        )
        object.__setattr__(
            self,
            "escape_duration_ticks",
            _positive_integer(
                self.escape_duration_ticks,
                name="escape_duration_ticks",
            ),
        )
        if self.escape_alpha >= self.normal_alpha:
            raise ValueError("escape_alpha must be lower than normal_alpha")


@dataclass(frozen=True)
class StallEscapeAlphaObservation:
    """One immutable post-round controller observation."""

    state_index: int
    alpha_used: float
    phase_before: AlphaPhase
    phase_after: AlphaPhase
    alpha_for_next_round: float | None
    progress_epoch_before: int
    progress_epoch_after: int
    escape_count: int
    escape_index_observed: int | None
    escape_ticks_completed_before: int
    escape_ticks_completed_after: int
    escape_ticks_remaining_before: int
    escape_ticks_remaining_after: int
    escape_used_in_progress_epoch: bool
    applied_participating_rows: int
    cumulative_participating_rows: int
    normalized_work: float
    work_tick_completed: bool
    completed_work_ticks: int
    completed_tick_had_progress: bool | None
    consecutive_no_progress_ticks: int
    best_updated: bool
    termination_reason: str
    events: tuple[AlphaEvent, ...]


class StallEscapeAlphaController:
    """Switch 16 -> 12 -> 16 using only completed natural-work history."""

    def __init__(self, config: StallEscapeAlphaConfig) -> None:
        if not isinstance(config, StallEscapeAlphaConfig):
            raise TypeError("config must be StallEscapeAlphaConfig")
        self._config = config
        self._initialized = False
        self._terminated = False
        self._phase: AlphaPhase = "normal"
        self._progress_epoch = 0
        self._escape_used_in_progress_epoch = False
        self._escape_count = 0
        self._active_escape_index: int | None = None
        self._escape_ticks_remaining = 0
        self._last_state_index = 0
        self._last_completed_work_ticks = 0

    @property
    def config(self) -> StallEscapeAlphaConfig:
        return self._config

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def escape_count(self) -> int:
        return self._escape_count

    @property
    def phase(self) -> AlphaPhase:
        return self._phase

    @property
    def alpha_for_next_round(self) -> float:
        if not self._initialized:
            raise RuntimeError("initial stopping state has not been observed")
        if self._terminated:
            raise RuntimeError("the alpha controller has already terminated")
        if self._phase == "escape":
            return self._config.escape_alpha
        return self._config.normal_alpha

    def observe_initial(self, decision: EarlyStoppingDecision) -> None:
        """Bind the controller to the same initial A/B/C observation."""

        if self._initialized:
            raise RuntimeError("initial stopping state has already been observed")
        if not isinstance(decision, EarlyStoppingDecision):
            raise TypeError("decision must be EarlyStoppingDecision")
        if decision.phase != "initial" or decision.state_index != 0:
            raise ValueError("controller requires the initial stopping decision")
        self._initialized = True
        self._terminated = decision.should_stop
        self._last_state_index = decision.state_index
        self._last_completed_work_ticks = decision.completed_work_ticks

    def observe_post_round(
        self,
        decision: EarlyStoppingDecision,
    ) -> StallEscapeAlphaObservation:
        """Observe one applied current state and schedule only the next round."""

        if not self._initialized:
            raise RuntimeError("initial stopping state must be observed first")
        if self._terminated:
            raise RuntimeError("the alpha controller has already terminated")
        if not isinstance(decision, EarlyStoppingDecision):
            raise TypeError("decision must be EarlyStoppingDecision")
        if decision.phase != "post_round":
            raise ValueError("controller requires a post-round stopping decision")
        if decision.state_index != self._last_state_index + 1:
            raise ValueError("stopping decisions must be observed in state order")
        newly_completed = (
            decision.completed_work_ticks - self._last_completed_work_ticks
        )
        if newly_completed not in (0, 1):
            raise ValueError("one observation must complete zero or one work tick")
        if decision.work_tick_completed != (newly_completed == 1):
            raise ValueError("work-tick fields are inconsistent")

        phase_before = self._phase
        alpha_used = (
            self._config.escape_alpha
            if phase_before == "escape"
            else self._config.normal_alpha
        )
        progress_epoch_before = self._progress_epoch
        active_escape_before = self._active_escape_index
        remaining_before = self._escape_ticks_remaining
        completed_before = (
            self._config.escape_duration_ticks - remaining_before
            if phase_before == "escape"
            else 0
        )
        events: list[AlphaEvent] = []
        escape_index_observed = active_escape_before

        if decision.should_stop:
            self._terminated = True
            events.append("termination_observed")
            remaining_after = self._escape_ticks_remaining
            completed_after = completed_before
            alpha_for_next_round = None
        else:
            if decision.best_updated:
                self._progress_epoch += 1
                self._escape_used_in_progress_epoch = False
                events.append("new_best")

            if phase_before == "escape":
                if decision.work_tick_completed:
                    self._escape_ticks_remaining -= 1
                    events.append("escape_tick_completed")
                    if self._escape_ticks_remaining == 0:
                        self._phase = "normal"
                        self._active_escape_index = None
                        events.append("escape_completed")
                remaining_after = self._escape_ticks_remaining
                completed_after = (
                    self._config.escape_duration_ticks - remaining_after
                )
            else:
                should_trigger = (
                    decision.work_tick_completed
                    and decision.completed_tick_had_progress is False
                    and decision.consecutive_no_progress_ticks
                    == self._config.stall_trigger_ticks
                    and not self._escape_used_in_progress_epoch
                )
                if should_trigger:
                    self._phase = "escape"
                    self._escape_count += 1
                    self._active_escape_index = self._escape_count
                    self._escape_ticks_remaining = (
                        self._config.escape_duration_ticks
                    )
                    self._escape_used_in_progress_epoch = True
                    escape_index_observed = self._active_escape_index
                    events.append("escape_started")
                remaining_after = self._escape_ticks_remaining
                completed_after = 0

            alpha_for_next_round = (
                self._config.escape_alpha
                if self._phase == "escape"
                else self._config.normal_alpha
            )

        self._last_state_index = decision.state_index
        self._last_completed_work_ticks = decision.completed_work_ticks
        return StallEscapeAlphaObservation(
            state_index=decision.state_index,
            alpha_used=alpha_used,
            phase_before=phase_before,
            phase_after=self._phase,
            alpha_for_next_round=alpha_for_next_round,
            progress_epoch_before=progress_epoch_before,
            progress_epoch_after=self._progress_epoch,
            escape_count=self._escape_count,
            escape_index_observed=escape_index_observed,
            escape_ticks_completed_before=completed_before,
            escape_ticks_completed_after=completed_after,
            escape_ticks_remaining_before=remaining_before,
            escape_ticks_remaining_after=remaining_after,
            escape_used_in_progress_epoch=(
                self._escape_used_in_progress_epoch
            ),
            applied_participating_rows=decision.applied_participating_rows,
            cumulative_participating_rows=(
                decision.cumulative_participating_rows
            ),
            normalized_work=decision.normalized_work,
            work_tick_completed=decision.work_tick_completed,
            completed_work_ticks=decision.completed_work_ticks,
            completed_tick_had_progress=(
                decision.completed_tick_had_progress
            ),
            consecutive_no_progress_ticks=(
                decision.consecutive_no_progress_ticks
            ),
            best_updated=decision.best_updated,
            termination_reason=decision.termination_reason,
            events=tuple(events),
        )
