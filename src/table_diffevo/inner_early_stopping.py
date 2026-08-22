"""Terminal-current early stopping for the gate-free generation inner loop.

This module is deliberately separate from ``inner_stopping.py``.  The latter
preserves the rejected 3+3 ``optimization_stalled`` prototype for historical
counterfactual tests.  The state machine here implements the current A/B/C
contract without controlling proposals, rejecting states, or selecting a
historical-best table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

DEFAULT_PATIENCE_TICKS = 6
WORK_TICK_NORMALIZED_WORK = 1

TerminationReason = Literal[
    "in_progress",
    "fit_target_reached",
    "early_stopped",
    "resource_cap_reached",
]
ObservationPhase = Literal["initial", "post_round"]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _finite_nonnegative_loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("current_loss must be a finite nonnegative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError("current_loss must be a finite nonnegative number")
    return normalized


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Natural-work patience for one inner generation run.

    C is intentionally absent from this configuration.  The caller owns the
    external resource limit and reports it through ``resource_cap_reached`` on
    the observed state.  This prevents a hidden quality-dependent C value.
    """

    n_records: int
    patience_ticks: int = DEFAULT_PATIENCE_TICKS
    work_tick_normalized_work: int = field(
        default=WORK_TICK_NORMALIZED_WORK,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_records",
            _positive_integer(self.n_records, name="n_records"),
        )
        object.__setattr__(
            self,
            "patience_ticks",
            _positive_integer(self.patience_ticks, name="patience_ticks"),
        )


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Immutable A/B/C decision for one observed current state."""

    phase: ObservationPhase
    state_index: int
    current_loss: float
    best_loss_diagnostic_only: float
    best_state_index_diagnostic_only: int
    best_updated: bool
    applied_participating_rows: int
    cumulative_participating_rows: int
    normalized_work: float
    work_tick_completed: bool
    completed_tick_had_progress: bool | None
    completed_work_ticks: int
    consecutive_no_progress_ticks: int
    external_resource_cap_reached: bool
    termination_reason: TerminationReason
    fit_target_reached: bool
    inner_complete: bool
    terminal_output_state_index: int | None
    terminal_output_loss: float | None

    @property
    def should_stop(self) -> bool:
        return self.termination_reason != "in_progress"


class InnerEarlyStopper:
    """Observe current states without controlling the gate-free transition."""

    def __init__(self, config: EarlyStoppingConfig) -> None:
        if not isinstance(config, EarlyStoppingConfig):
            raise TypeError("config must be EarlyStoppingConfig")
        self._config = config
        self._initialized = False
        self._terminated = False
        self._state_index = 0
        self._current_loss: float | None = None
        self._best_loss: float | None = None
        self._best_state_index = 0
        self._cumulative_participating_rows = 0
        self._completed_work_ticks = 0
        self._consecutive_no_progress_ticks = 0
        self._tick_had_best_improvement = False

    def observe_initial(
        self,
        current_loss: float,
        *,
        resource_cap_reached: bool = False,
    ) -> EarlyStoppingDecision:
        """Establish S0 and apply A before an optional external C."""

        if self._initialized:
            raise RuntimeError("initial state has already been observed")
        if not isinstance(resource_cap_reached, bool):
            raise TypeError("resource_cap_reached must be a boolean")
        loss = _finite_nonnegative_loss(current_loss)
        self._initialized = True
        self._current_loss = loss
        self._best_loss = loss
        reason: TerminationReason = "in_progress"
        if loss == 0.0:
            reason = "fit_target_reached"
        elif resource_cap_reached:
            reason = "resource_cap_reached"
        if reason != "in_progress":
            self._terminated = True
        return self._decision(
            phase="initial",
            best_updated=True,
            applied_participating_rows=0,
            work_tick_completed=False,
            completed_tick_had_progress=None,
            external_resource_cap_reached=resource_cap_reached,
            termination_reason=reason,
        )

    def observe_post_round(
        self,
        *,
        current_loss: float,
        participating_rows: int,
        resource_cap_reached: bool = False,
    ) -> EarlyStoppingDecision:
        """Observe one already-applied proposal and decide A, then B, then C."""

        if not self._initialized:
            raise RuntimeError("initial state must be observed first")
        if self._terminated:
            raise RuntimeError("stopping decision has already been reached")
        if not isinstance(resource_cap_reached, bool):
            raise TypeError("resource_cap_reached must be a boolean")

        loss = _finite_nonnegative_loss(current_loss)
        applied_rows = _nonnegative_integer(
            participating_rows,
            name="participating_rows",
        )
        if applied_rows > self._config.n_records:
            raise ValueError("participating_rows cannot exceed n_records")

        self._state_index += 1
        self._current_loss = loss
        assert self._best_loss is not None
        best_updated = loss < self._best_loss
        if best_updated:
            self._best_loss = loss
            self._best_state_index = self._state_index
            self._tick_had_best_improvement = True
            self._consecutive_no_progress_ticks = 0

        self._cumulative_participating_rows += applied_rows
        completed_now = self._cumulative_participating_rows // self._config.n_records
        newly_completed = completed_now - self._completed_work_ticks
        if newly_completed not in (0, 1):
            raise RuntimeError("one post-round crossed multiple natural work ticks")

        work_tick_completed = newly_completed == 1
        completed_tick_had_progress: bool | None = None
        if work_tick_completed:
            self._completed_work_ticks = completed_now
            completed_tick_had_progress = self._tick_had_best_improvement
            if completed_tick_had_progress:
                self._consecutive_no_progress_ticks = 0
            else:
                self._consecutive_no_progress_ticks += 1
            self._tick_had_best_improvement = False

        reason: TerminationReason = "in_progress"
        if loss == 0.0:
            reason = "fit_target_reached"
        elif (
            work_tick_completed
            and self._consecutive_no_progress_ticks >= self._config.patience_ticks
        ):
            reason = "early_stopped"
        elif resource_cap_reached:
            reason = "resource_cap_reached"

        if reason != "in_progress":
            self._terminated = True
        return self._decision(
            phase="post_round",
            best_updated=best_updated,
            applied_participating_rows=applied_rows,
            work_tick_completed=work_tick_completed,
            completed_tick_had_progress=completed_tick_had_progress,
            external_resource_cap_reached=resource_cap_reached,
            termination_reason=reason,
        )

    def _decision(
        self,
        *,
        phase: ObservationPhase,
        best_updated: bool,
        applied_participating_rows: int,
        work_tick_completed: bool,
        completed_tick_had_progress: bool | None,
        external_resource_cap_reached: bool,
        termination_reason: TerminationReason,
    ) -> EarlyStoppingDecision:
        assert self._current_loss is not None
        assert self._best_loss is not None
        should_stop = termination_reason != "in_progress"
        return EarlyStoppingDecision(
            phase=phase,
            state_index=self._state_index,
            current_loss=self._current_loss,
            best_loss_diagnostic_only=self._best_loss,
            best_state_index_diagnostic_only=self._best_state_index,
            best_updated=best_updated,
            applied_participating_rows=applied_participating_rows,
            cumulative_participating_rows=self._cumulative_participating_rows,
            normalized_work=(
                self._cumulative_participating_rows / self._config.n_records
            ),
            work_tick_completed=work_tick_completed,
            completed_tick_had_progress=completed_tick_had_progress,
            completed_work_ticks=self._completed_work_ticks,
            consecutive_no_progress_ticks=self._consecutive_no_progress_ticks,
            external_resource_cap_reached=external_resource_cap_reached,
            termination_reason=termination_reason,
            fit_target_reached=termination_reason == "fit_target_reached",
            inner_complete=termination_reason
            in {"fit_target_reached", "early_stopped"},
            terminal_output_state_index=(self._state_index if should_stop else None),
            terminal_output_loss=(self._current_loss if should_stop else None),
        )
