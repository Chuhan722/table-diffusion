"""分裂探针的有限均值；正常输入保持原规约顺序，溢出时才缩放重算。"""
from __future__ import annotations

import numpy as np


def finite_mean(values, *, xp=np) -> float:
    """用于 NumPy/CuPy 记账；非有限输入或结果须显式中断，不能当成正常诊断。"""
    if values.size == 0:
        raise ValueError("分裂探针均值输入不能为空")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(xp.mean(values))
    if np.isfinite(mean):
        return mean
    if not bool(xp.isfinite(values).all()):
        raise FloatingPointError("numerical_indeterminate: 分裂探针均值输入非有限")
    scale = float(xp.max(xp.abs(values)))
    mean = float(xp.mean(values / scale)) * scale if scale > 0.0 else 0.0
    if not np.isfinite(mean):
        raise FloatingPointError("numerical_indeterminate: 分裂探针均值不可表示")
    return mean
