"""固定独立高斯验证卷上的只读平方风险差监控，不参与生成或选择参考表。

条件：参考、候选轨迹与固定权重独立于验证卷；验证各题为未处理的独立高斯观测，
validation_sigma 是验证卷自身的已知计数标准差或可靠上界。数值校验不能认证这些统计条件。
预先限定观察次数并均分总误报容许率；不报警不表示候选未变差，也不保证最优停时。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, ndtri

from .gain import ExactGainOracle


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是非空、有限的一维向量")
    result.setflags(write=False)
    return result


def _counts(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = _vector(values, name)
    if np.any(result < 0) or np.any(result > 2.0**50) or np.any(result != np.rint(result)):
        raise ValueError(f"{name} 必须是 [0, 2**50] 内的整数计数答案")
    return result


def _standard_error_upper(variance: Fraction) -> float:
    if variance == 0:
        return 0.0
    with localcontext(Context(prec=80)):
        root = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
    value = float(root)
    if not math.isfinite(value) or value == 0.0:
        raise FloatingPointError("numerical_indeterminate: L2验证标准误无法以非零有限float64表示")
    upper = math.nextafter(value, math.inf)
    if not math.isfinite(upper) or Fraction.from_float(upper) ** 2 < variance:
        raise FloatingPointError("numerical_indeterminate: 无法建立有限的L2验证标准误上界")
    return upper


@dataclass(frozen=True)
class QuadraticRiskRecord:
    look: int
    risk_difference: float
    standard_error_upper: float
    critical_value: float
    lower_bound: float
    worse: bool
    zero_variance: bool
    error_probability_per_look: float


class IndependentGaussianQuadraticMonitor:
    """对一个固定参考作有限次单侧比较，输入的统计条件由调用方负责保证。

    D = L_B(candidate) - L_B(reference)，条件均值为真值平方风险差；
    Var(D) <= sum((w * (candidate-reference) * sigma_B)**2)，精确sigma时取等号。
    数值差复用实际目标/答案的精确增益计算，不能用两个总损失相减或反推答案。
    """

    def __init__(
        self, reference: ArrayLike, validation: ArrayLike, validation_sigma: ArrayLike,
        weights: ArrayLike, *, error_probability: float, max_looks: int,
    ) -> None:
        self._reference = _counts(reference, "参考答案")
        self._validation = _vector(validation, "验证目标")
        self._sigma = _vector(validation_sigma, "验证标准差")
        self._weights = _vector(weights, "监控权重")
        if any(value.shape != self._reference.shape for value in (self._validation, self._sigma, self._weights)):
            raise ValueError("参考、验证目标、标准差与权重必须等长")
        if np.any(self._sigma < 0) or np.any(self._weights <= 0):
            raise ValueError("验证标准差须非负，监控权重须严格为正")
        if not math.isfinite(error_probability) or not 0.0 < error_probability < 0.5:
            raise ValueError("总误报容许率须有限且位于 (0, 0.5)")
        if (
            isinstance(max_looks, (bool, np.bool_)) or not isinstance(max_looks, (int, np.integer))
            or not 1 <= int(max_looks) <= 2**63 - 1
        ):
            raise ValueError("最大观察次数须为正的int64范围整数")
        self._max_looks = int(max_looks)
        share = Fraction.from_float(float(error_probability)) / self._max_looks
        per_look = float(share)
        if Fraction.from_float(per_look) > share:
            per_look = math.nextafter(per_look, 0.0)
        if per_look == 0.0:
            raise ValueError("每次观察的误报容许率在float64中不可表示")
        self._per_look = per_look
        critical = math.nextafter(float(-ndtri(per_look)), math.inf)
        if not math.isfinite(critical) or critical <= 0.0:
            raise FloatingPointError("numerical_indeterminate: L2临界值不可表示")
        if float(log_ndtr(-critical)) > math.log(per_look):
            raise FloatingPointError("numerical_indeterminate: L2正态临界值未通过尾概率核对")
        self._critical = critical
        self._gain = ExactGainOracle.from_answers(self._validation, self._weights, self._reference)
        self._noise_coefficients = tuple(
            (Fraction.from_float(float(w)) * Fraction.from_float(float(sigma))) ** 2
            for w, sigma in zip(self._weights, self._sigma)
        )
        self._looks_used = 0

    @property
    def max_looks(self) -> int:
        return self._max_looks

    @property
    def looks_used(self) -> int:
        return self._looks_used

    @property
    def critical_value(self) -> float:
        return self._critical

    @property
    def error_probability_per_look(self) -> float:
        return self._per_look

    def observe(self, candidate: ArrayLike) -> QuadraticRiskRecord:
        """只读取计数并消耗一次观察；重复观察相同答案也计入预算。

        输入或数值错误不伪造记录，也不更新已成功观察次数。调用方不得利用验证值、
        报警或错误信息调整后续候选/权重/参考后继续引用本实例的统计保证。
        """
        if self._looks_used >= self._max_looks:
            raise ValueError("L2监控观察预算已耗尽")
        current = _counts(candidate, "当前答案")
        if current.shape != self._reference.shape:
            raise ValueError("当前答案须与参考答案等长")
        delta = current - self._reference
        live = np.flatnonzero(delta)
        difference = -self._gain.sparse_gain(delta[live], live)
        if difference == 0.0:
            difference = 0.0
        variance = sum(
            (self._noise_coefficients[int(j)] * int(delta[j]) ** 2 for j in live),
            Fraction(0),
        )
        standard_error = _standard_error_upper(variance)
        lower = math.nextafter(difference, -math.inf) if difference != 0.0 else 0.0
        if variance != 0:
            threshold = self._critical * standard_error
            if not math.isfinite(threshold) or threshold == 0.0:
                raise FloatingPointError("numerical_indeterminate: L2临界风险差无法以非零有限float64表示")
            threshold = math.nextafter(threshold, math.inf)
            lower = math.nextafter(lower - threshold, -math.inf)
        if not math.isfinite(lower):
            raise FloatingPointError("numerical_indeterminate: L2风险差下界不可表示")
        record = QuadraticRiskRecord(
            self._looks_used + 1, difference, standard_error, self._critical,
            lower, lower > 0.0, variance == 0, self._per_look,
        )
        self._looks_used += 1
        return record
