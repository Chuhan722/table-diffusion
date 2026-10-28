"""现有 NumPy 采样路径的有限精度合同，不提供任意微小概率的精确采样保证。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np
from numpy.typing import NDArray

SAMPLING_CONTRACT = "numpy-finite-precision-v1"


class SamplingPrecisionWarning(RuntimeWarning):
    """部分非保持质量的补数表示丢失；结果仍受有限精度近似限制。"""


@dataclass(frozen=True)
class SamplingPrecision:
    limited_segments: int
    max_limited_mass: float
    round_index: int | None = None


def resolve_sampling_mass(
    stay: NDArray[np.float64],
    mass: NDArray[np.float64],
    *,
    emit_warning: bool = True,
) -> tuple[NDArray[np.float64], SamplingPrecision | None]:
    """只恢复被补数回算吞掉的正质量，其余参数保留旧算术与数值。

    全部活跃段都无法用当前保持项区分时数值中断；部分受限则报告，不按收益符号选段。
    """
    if stay.ndim != 1 or mass.shape != stay.shape:
        raise ValueError("保持概率与非保持质量须为等长一维向量")
    if (
        not np.isfinite(stay).all() or not np.isfinite(mass).all()
        or np.any(stay < 0) or np.any(stay > 1) or np.any(mass < 0)
        or np.any(np.abs(stay + mass - 1.0) > 1e-12)
    ):
        raise FloatingPointError("numerical_indeterminate: 采样概率或分段质量非法")
    move = np.minimum(1.0, np.maximum(0.0, 1.0 - stay))
    active = mass > 0.0
    limited = active & (move == 0.0)
    if not np.any(limited):
        return move, None
    if not np.any(active & ~limited):
        raise FloatingPointError(
            "numerical_indeterminate: 全部非保持质量被保持项舍入掩盖，当前采样表示不可可靠区分"
        )
    precision = SamplingPrecision(int(limited.sum()), float(mass[limited].max()))
    move[limited] = mass[limited]
    if emit_warning:
        warnings.warn(
            "sampling_precision_limited: 部分正非保持质量被保持项舍入掩盖；"
            "继续使用有限精度近似，详见 sampling_precision",
            SamplingPrecisionWarning,
            stacklevel=2,
        )
    return move, precision


def flat_sampling_mass(
    probabilities: NDArray[np.float64],
    offsets: NDArray[np.int64],
    *,
    emit_warning: bool = True,
) -> tuple[NDArray[np.float64], SamplingPrecision | None]:
    """从已验证的最终核直接求非保持质量，不缓存可被调用方修改的概率数组。"""
    starts = offsets[:-1]
    nonstay = probabilities.copy()
    nonstay[starts] = 0.0
    mass = np.add.reduceat(nonstay, starts)
    return resolve_sampling_mass(probabilities[starts], mass, emit_warning=emit_warning)


def record_sampling_precision(
    precision: SamplingPrecision | None, round_index: int, records: list[SamplingPrecision],
) -> None:
    if precision is not None:
        records.append(replace(precision, round_index=round_index))


def sampling_summary(records: list[SamplingPrecision] | None) -> dict[str, int | float | str]:
    """计数按存储段跨轮累加，不将其解释成真实记录数或实际抽到的次数。"""
    return {
        "contract": SAMPLING_CONTRACT,
        "affected_rounds": len(records or []),
        "limited_segment_occurrences": sum(record.limited_segments for record in records or []),
        "max_limited_mass": max((record.max_limited_mass for record in records or []), default=0.0),
    }
