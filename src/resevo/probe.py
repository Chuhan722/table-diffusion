"""只读 beta 诊断的数值隔离；不可用点不替代主核，也不伪造零结果。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Callable, Literal, Sequence
import warnings

import numpy as np

from .stepsize import analytic_step, expected_improvement


class ProbeNumericalWarning(RuntimeWarning):
    """主核有效，但部分或全部诊断无法可靠计算。"""


@dataclass(frozen=True)
class ProbeUnavailablePoint:
    beta: float | None
    reason: str


@dataclass(frozen=True)
class ProbeDiagnostics:
    status: Literal["ok", "partial", "unavailable"]
    valid_points: int
    unavailable_points: tuple[ProbeUnavailablePoint, ...]


class ProbeAccumulator:
    """只收集可表示的正方向试探值；保留原零下降基准与严格大于的比较顺序。"""

    def __init__(self) -> None:
        self.best_beta = 0.0
        self.best_j = 0.0
        self.valid_points = 0
        self.unavailable_points: list[ProbeUnavailablePoint] = []

    def unavailable(self, beta: float, reason: str) -> None:
        value = float(beta) if np.isfinite(beta) else None
        self.unavailable_points.append(ProbeUnavailablePoint(value, reason))

    def grid(
        self, tilted: Callable[[float], tuple[object, float]], max_gain_sum: float, grid_size: int,
    ) -> list[float]:
        target = 0.99 * max_gain_sum
        beta_sat = 1.0 / max(max_gain_sum, 1e-300)
        try:
            # 初始评估加最多200次翻倍，与原网格定标相同。
            for attempt in range(201):
                if not np.isfinite(beta_sat):
                    raise FloatingPointError("numerical_indeterminate: 诊断网格上界不可表示")
                _, gain = tilted(beta_sat)
                if not np.isfinite(gain):
                    raise FloatingPointError("numerical_indeterminate: 诊断网格定标增益非有限")
                if gain >= target or attempt == 200:
                    break
                beta_sat *= 2.0
        except FloatingPointError as exc:
            if not str(exc).startswith("numerical_indeterminate:"):
                raise
            self.unavailable(beta_sat, str(exc))
            return []
        return [0.0] + [float(beta) for beta in np.geomspace(beta_sat / 2 ** 14, beta_sat, grid_size - 1)]

    def evaluate(
        self, beta: float, max_leave: float, direction_gain: float, interaction: float,
        damping: float, unit_rows: float, max_expected_rows: float | None,
    ) -> None:
        try:
            step = analytic_step(
                max_leave, direction_gain, interaction, damping,
                unit_expected_rows=unit_rows, max_expected_rows=max_expected_rows,
            )
            improvement = expected_improvement(step, direction_gain, interaction)
        except FloatingPointError as exc:
            if not str(exc).startswith("numerical_indeterminate:"):
                raise
            self.unavailable(beta, str(exc))
            return
        self.valid_points += 1
        if improvement > self.best_j:
            self.best_j, self.best_beta = improvement, float(beta)

    def finish(self) -> tuple[tuple[float, float] | None, ProbeDiagnostics]:
        status: Literal["ok", "partial", "unavailable"] = "ok"
        best: tuple[float, float] | None = (self.best_beta, self.best_j)
        if self.unavailable_points:
            status = "partial" if self.valid_points else "unavailable"
            if not self.valid_points:
                best = None
            warnings.warn(
                f"beta 网格诊断 {status}：{len(self.unavailable_points)} 个试探点/定标不可用，"
                f"{self.valid_points} 个正方向点可用；{self.unavailable_points[0].reason}",
                ProbeNumericalWarning, stacklevel=3,
            )
        return best, ProbeDiagnostics(status, self.valid_points, tuple(self.unavailable_points))


def probe_diagnostic_columns(
    diagnostics: Sequence[ProbeDiagnostics | None],
) -> tuple[list[str], list[list[object]]]:
    """仅存在受限诊断时追加列，正常探针的历史 CSV 格式保持。"""
    if not any(item is not None and item.status != "ok" for item in diagnostics):
        return [], [[] for _ in diagnostics]
    rows: list[list[object]] = []
    for item in diagnostics:
        if item is None:
            rows.append(["not_recorded", "", ""])
        else:
            rows.append([
                item.status, item.valid_points,
                json.dumps([asdict(point) for point in item.unavailable_points], ensure_ascii=False, allow_nan=False),
            ])
    return ["probe_status", "probe_valid_points", "probe_unavailable_points"], rows
