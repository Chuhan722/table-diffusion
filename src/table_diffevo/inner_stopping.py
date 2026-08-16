"""Best-loss stopping logic for the exact-target generation inner loop.

The state machine observes losses and applied participating-row counts only.
It does not evaluate queries, inspect a table, consume randomness, or use an
offline quality metric.  In particular, finite current-loss increases are
ordinary observations: only a strict improvement of the historical best loss
counts as progress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

WORK_WINDOW_NORMALIZED_WORK = 1
DEFAULT_STALL_BLOCK_WINDOWS = 3
DEFAULT_MAX_NORMALIZED_WORK = 20

TerminationReason = Literal[
    "in_progress",
    "exact_residual",
    "optimization_stalled",
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
class BestLossStoppingConfig:
    """Two-block stall confirmation and resource limits for one inner run."""

    n_records: int
    stall_block_windows: int = DEFAULT_STALL_BLOCK_WINDOWS
    max_normalized_work: int = DEFAULT_MAX_NORMALIZED_WORK
    window_normalized_work: int = field(
        default=WORK_WINDOW_NORMALIZED_WORK,
        init=False,
    )
    required_no_progress_windows: int = field(init=False)

    def __post_init__(self) -> None:
        n_records = _positive_integer(self.n_records, name="n_records")
        block_windows = _positive_integer(
            self.stall_block_windows,
            name="stall_block_windows",
        )
        required_windows = 2 * block_windows
        maximum = _positive_integer(
            self.max_normalized_work,
            name="max_normalized_work",
        )
        if maximum < required_windows:
            raise ValueError(
                "max_normalized_work must be at least 2 * stall_block_windows"
            )
        object.__setattr__(self, "n_records", n_records)
        object.__setattr__(self, "stall_block_windows", block_windows)
        object.__setattr__(self, "max_normalized_work", maximum)
        object.__setattr__(
            self,
            "required_no_progress_windows",
            required_windows,
        )


@dataclass(frozen=True)
class BestLossStoppingDecision:
    """Immutable snapshot returned after one observed generation state."""

    phase: ObservationPhase
    current_loss: float
    best_loss: float
    best_updated: bool
    applied_participating_rows: int
    cumulative_participating_rows: int
    normalized_work: float
    work_window_completed: bool
    completed_window_had_progress: bool | None
    completed_work_windows: int
    consecutive_no_progress_windows: int
    termination_reason: TerminationReason
    inner_complete: bool

    @property
    def should_stop(self) -> bool:
        return self.termination_reason != "in_progress"


class BestLossInnerStopper:
    """Track strict best-loss progress without controlling state transitions."""

    def __init__(self, config: BestLossStoppingConfig) -> None:
        if not isinstance(config, BestLossStoppingConfig):
            raise TypeError("config must be BestLossStoppingConfig")
        self._config = config
        self._initialized = False
        self._terminated = False
        self._current_loss: float | None = None
        self._best_loss: float | None = None
        self._cumulative_participating_rows = 0
        self._completed_work_windows = 0
        self._consecutive_no_progress_windows = 0
        self._window_had_best_improvement = False

    def observe_initial(self, current_loss: float) -> BestLossStoppingDecision:
        """Establish the initial checkpoint and detect an exact initial state."""

        if self._initialized:
            raise RuntimeError("initial state has already been observed")
        loss = _finite_nonnegative_loss(current_loss)
        self._initialized = True
        self._current_loss = loss
        self._best_loss = loss
        reason: TerminationReason = "exact_residual" if loss == 0.0 else "in_progress"
        if reason != "in_progress":
            self._terminated = True
        return self._decision(
            phase="initial",
            best_updated=True,
            applied_participating_rows=0,
            work_window_completed=False,
            completed_window_had_progress=None,
            termination_reason=reason,
        )

    def observe_post_round(
        self,
        *,
        current_loss: float,
        participating_rows: int,
    ) -> BestLossStoppingDecision:
        """Observe one applied post-round state without accepting or rejecting it."""

        if not self._initialized:
            raise RuntimeError("initial state must be observed first")
        if self._terminated:
            raise RuntimeError("stopping decision has already been reached")

        loss = _finite_nonnegative_loss(current_loss)
        applied_rows = _nonnegative_integer(
            participating_rows,
            name="participating_rows",
        )
        if applied_rows > self._config.n_records:
            raise ValueError("participating_rows cannot exceed n_records")

        self._current_loss = loss
        assert self._best_loss is not None
        best_updated = loss < self._best_loss
        if best_updated:
            self._best_loss = loss
            self._window_had_best_improvement = True
            # A new best cancels both the candidate and confirmation blocks.
            self._consecutive_no_progress_windows = 0

        self._cumulative_participating_rows += applied_rows
        completed_now = self._cumulative_participating_rows // self._config.n_records
        newly_completed = completed_now - self._completed_work_windows
        if newly_completed not in (0, 1):
            raise RuntimeError("one post-round crossed multiple work windows")

        work_window_completed = newly_completed == 1
        completed_window_had_progress: bool | None = None
        if work_window_completed:
            self._completed_work_windows = completed_now
            completed_window_had_progress = self._window_had_best_improvement
            if completed_window_had_progress:
                self._consecutive_no_progress_windows = 0
            else:
                self._consecutive_no_progress_windows += 1
            self._window_had_best_improvement = False

        # Normal completion wins over the engineering resource guard.
        reason: TerminationReason = "in_progress"
        if self._best_loss == 0.0:
            reason = "exact_residual"
        elif (
            work_window_completed
            and self._consecutive_no_progress_windows
            >= self._config.required_no_progress_windows
        ):
            reason = "optimization_stalled"
        elif self._cumulative_participating_rows >= (
            self._config.max_normalized_work * self._config.n_records
        ):
            reason = "resource_cap_reached"

        if reason != "in_progress":
            self._terminated = True
        return self._decision(
            phase="post_round",
            best_updated=best_updated,
            applied_participating_rows=applied_rows,
            work_window_completed=work_window_completed,
            completed_window_had_progress=completed_window_had_progress,
            termination_reason=reason,
        )

    def _decision(
        self,
        *,
        phase: ObservationPhase,
        best_updated: bool,
        applied_participating_rows: int,
        work_window_completed: bool,
        completed_window_had_progress: bool | None,
        termination_reason: TerminationReason,
    ) -> BestLossStoppingDecision:
        assert self._current_loss is not None
        assert self._best_loss is not None
        return BestLossStoppingDecision(
            phase=phase,
            current_loss=self._current_loss,
            best_loss=self._best_loss,
            best_updated=best_updated,
            applied_participating_rows=applied_participating_rows,
            cumulative_participating_rows=(self._cumulative_participating_rows),
            normalized_work=float(
                self._cumulative_participating_rows / self._config.n_records
            ),
            work_window_completed=work_window_completed,
            completed_window_had_progress=completed_window_had_progress,
            completed_work_windows=self._completed_work_windows,
            consecutive_no_progress_windows=(self._consecutive_no_progress_windows),
            termination_reason=termination_reason,
            inner_complete=termination_reason
            in {
                "exact_residual",
                "optimization_stalled",
            },
        )
