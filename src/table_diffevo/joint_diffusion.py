"""联合属性块扩散核的精确小规模 oracle。

本模块枚举一条 recipient 记录与其 donor 不同属性上的全部复制 mask，并在固定
残差场中计算完整 hybrid 的方向势能。有限温度 Gibbs 核只改变这些 mask 的概率，
不执行正收益门槛、argmax 方向选择或 proposal 接受检查。

精确枚举随活跃属性数指数增长，因此这里只用于小表辨识“独立单块分解”造成的
损失；默认护栏不允许把它误当成大表生产实现。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table_diffevo.schema import Schema
from table_diffevo.vectorized_eval import evaluate_directional_potential


@dataclass(frozen=True)
class JointMaskLandscape:
    """一条 recipient-donor 对的完整复制 mask 方向地形。"""

    row_index: int
    active_attribute_indices: np.ndarray
    active_attributes: Tuple[str, ...]
    masks: np.ndarray
    directions: np.ndarray


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} 必须是非负整数，得到 {value!r}")
    return int(value)


def enumerate_copy_masks(
    n_active_attributes: int,
    *,
    max_active_attributes: int = 16,
) -> np.ndarray:
    """按二进制顺序枚举 ``2**k`` 个复制 mask。

    第 0 行恒为空 mask；第 ``j`` 列对应第 ``j`` 个活跃属性。枚举规模由显式护栏
    限制，避免调用方在属性数较大时意外耗尽内存。
    """
    n_active = _validate_nonnegative_integer(
        n_active_attributes, "n_active_attributes"
    )
    maximum = _validate_nonnegative_integer(
        max_active_attributes, "max_active_attributes"
    )
    # 即使调用方主动调大软护栏，也保留绝对上限，避免 2**k 在进入后续评价前
    # 就分配不可控内存。20 个布尔属性的 mask 本身已约 20 MiB。
    if maximum > 20:
        raise ValueError("max_active_attributes 不得超过绝对护栏 20")
    if n_active > maximum:
        raise ValueError(
            "活跃属性数超过精确枚举护栏："
            f"{n_active} > {maximum}"
        )
    if n_active == 0:
        return np.zeros((1, 0), dtype=bool)

    states = np.arange(1 << n_active, dtype=np.uint64)[:, None]
    bits = np.arange(n_active, dtype=np.uint64)[None, :]
    return ((states >> bits) & 1).astype(bool)


def _validate_complete_masks(masks: np.ndarray) -> np.ndarray:
    raw = np.asarray(masks)
    if raw.ndim != 2:
        raise ValueError(f"masks 必须是二维数组，得到 shape {raw.shape}")
    if raw.dtype.kind not in "biuf":
        raise ValueError("masks 必须只包含 0/1 或布尔值")
    if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
        raise ValueError("masks 必须只包含 0/1 或布尔值")
    result = raw.astype(bool, copy=False)
    n_states, n_active = result.shape
    if n_active > 62 or n_states != (1 << n_active):
        raise ValueError("masks 必须完整枚举全部 2**k 个状态")
    if len(np.unique(result, axis=0)) != n_states:
        raise ValueError("masks 不得包含重复状态")
    return result


def _validate_open_probability(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{name} 必须是 (0, 1) 内的有限数值，得到 {value!r}")
    return float(value)


def _normalize_log_probabilities(log_weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(log_weights)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("log_weights 必须是非空一维数组")
    if raw.dtype.kind not in "iuf":
        raise ValueError("log_weights 必须是数值数组")
    values = raw.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("log_weights 必须全部为有限数值")
    maximum = float(np.max(values))
    shifted = values - maximum
    normalizer = maximum + float(np.log(np.exp(shifted).sum()))
    return values - normalizer


def baseline_mask_log_probabilities(
    masks: np.ndarray,
    eta: float,
) -> np.ndarray:
    """返回历史独立 ``Bernoulli(eta)`` 在完整 mask 空间上的对数概率。"""
    checked_masks = _validate_complete_masks(masks)
    probability = _validate_open_probability(eta, "eta")
    copied = checked_masks.sum(axis=1, dtype=float)
    retained = checked_masks.shape[1] - copied
    log_probabilities = (
        copied * np.log(probability)
        + retained * np.log1p(-probability)
    )
    return _normalize_log_probabilities(log_probabilities)


def independent_mask_log_probabilities(
    masks: np.ndarray,
    copy_probabilities: np.ndarray,
) -> np.ndarray:
    """把逐属性 Bernoulli 概率展开为完整 mask 乘积分布。"""
    checked_masks = _validate_complete_masks(masks)
    raw_probabilities = np.asarray(copy_probabilities)
    expected_shape = (checked_masks.shape[1],)
    if raw_probabilities.shape != expected_shape:
        raise ValueError(
            "copy_probabilities 必须与 mask 属性数一致，"
            f"得到 {raw_probabilities.shape}，期望 {expected_shape}"
        )
    if raw_probabilities.dtype.kind not in "iuf":
        raise ValueError("copy_probabilities 必须是 (0, 1) 内的有限数值")
    probabilities = raw_probabilities.astype(float, copy=False)
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("copy_probabilities 必须是 (0, 1) 内的有限数值")
    if probabilities.size == 0:
        return np.zeros(1, dtype=float)

    log_probabilities = (
        checked_masks.astype(float) @ np.log(probabilities)
        + (~checked_masks).astype(float) @ np.log1p(-probabilities)
    )
    return _normalize_log_probabilities(log_probabilities)


def gibbs_mask_log_probabilities(
    reference_log_probabilities: np.ndarray,
    directions: np.ndarray,
    strength: float,
) -> np.ndarray:
    """以方向势能连续倾斜一个完整支持集的参考 mask 分布。

    返回 ``log q(m) ∝ log q0(m) + strength * direction(m)``。计算始终在
    log 空间完成；有限强度下不设置方向阈值，也不把任何 mask 主动移出支持集。
    """
    reference = _normalize_log_probabilities(reference_log_probabilities)
    raw_directions = np.asarray(directions)
    if raw_directions.shape != reference.shape:
        raise ValueError(
            "directions 必须与 reference_log_probabilities 形状一致，"
            f"得到 {raw_directions.shape} 与 {reference.shape}"
        )
    if raw_directions.dtype.kind not in "iuf":
        raise ValueError("directions 必须是有限数值数组")
    energy = raw_directions.astype(float, copy=False)
    if not np.all(np.isfinite(energy)):
        raise ValueError("directions 必须是有限数值数组")
    if (
        isinstance(strength, (bool, np.bool_))
        or not isinstance(strength, (int, float, np.integer, np.floating))
        or not np.isfinite(strength)
        or strength < 0.0
    ):
        raise ValueError(f"strength 必须是非负有限数值，得到 {strength!r}")
    strength = float(strength)
    if strength == 0.0 or np.all(energy == energy[0]):
        return reference.copy()

    # 减去常数不改变 Gibbs 分布，同时尽量降低乘法溢出的风险。
    centered = energy - float(np.max(energy))
    with np.errstate(over="ignore", invalid="ignore"):
        tilt = strength * centered
    if not np.all(np.isfinite(tilt)):
        raise ValueError("strength * directions 超出 float64 可表示范围")
    return _normalize_log_probabilities(reference + tilt)


def additive_mask_directions(
    masks: np.ndarray,
    single_directions: np.ndarray,
) -> np.ndarray:
    """把单块方向相加，得到独立近似下每个完整 mask 的方向。"""
    checked_masks = _validate_complete_masks(masks)
    raw = np.asarray(single_directions)
    expected_shape = (checked_masks.shape[1],)
    if raw.shape != expected_shape:
        raise ValueError(
            "single_directions 必须与 mask 属性数一致，"
            f"得到 {raw.shape}，期望 {expected_shape}"
        )
    if raw.dtype.kind not in "iuf":
        raise ValueError("single_directions 必须是有限数值数组")
    values = raw.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("single_directions 必须是有限数值数组")
    return checked_masks.astype(float) @ values


def categorical_kl(
    log_probabilities: np.ndarray,
    reference_log_probabilities: np.ndarray,
) -> float:
    """返回两个完整支持集离散分布的 ``KL(q || q0)``。"""
    values = _normalize_log_probabilities(log_probabilities)
    reference = _normalize_log_probabilities(reference_log_probabilities)
    if values.shape != reference.shape:
        raise ValueError("两个对数概率数组必须形状一致")
    probabilities = np.exp(values)
    divergence = float(np.dot(probabilities, values - reference))
    return max(divergence, 0.0)


def categorical_entropy(log_probabilities: np.ndarray) -> float:
    """返回离散 mask 分布的 Shannon 熵（自然对数，单位 nat）。"""
    values = _normalize_log_probabilities(log_probabilities)
    probabilities = np.exp(values)
    entropy = float(-np.dot(probabilities, values))
    return max(entropy, 0.0)


def mask_distribution_diagnostics(
    log_probabilities: np.ndarray,
    reference_log_probabilities: np.ndarray,
    directions: np.ndarray,
) -> Dict[str, float]:
    """汇总一个 mask 核的 KL、熵、期望方向和正反向概率质量。"""
    values = _normalize_log_probabilities(log_probabilities)
    reference = _normalize_log_probabilities(reference_log_probabilities)
    raw_directions = np.asarray(directions)
    if raw_directions.shape != values.shape:
        raise ValueError("directions 必须与对数概率数组形状一致")
    if raw_directions.dtype.kind not in "iuf":
        raise ValueError("directions 必须是有限数值数组")
    energy = raw_directions.astype(float, copy=False)
    if not np.all(np.isfinite(energy)):
        raise ValueError("directions 必须是有限数值数组")
    if reference.shape != values.shape:
        raise ValueError("两个对数概率数组必须形状一致")

    probabilities = np.exp(values)
    scale = max(1.0, float(np.max(np.abs(energy))))
    tolerance = 1e-12 * scale
    negative = energy < -tolerance
    positive = energy > tolerance
    neutral = ~(negative | positive)
    return {
        "kl_to_baseline": categorical_kl(values, reference),
        "entropy": categorical_entropy(values),
        "expected_direction": float(np.dot(probabilities, energy)),
        "negative_direction_mass": float(probabilities[negative].sum()),
        "neutral_direction_mass": float(probabilities[neutral].sum()),
        "positive_direction_mass": float(probabilities[positive].sum()),
    }


def sample_mask_index(
    log_probabilities: np.ndarray,
    gumbels: np.ndarray,
) -> int:
    """用外部提供的 Gumbel 噪声精确抽一个离散 mask 索引。

    ``argmax(log q + G)`` 是 categorical sampling 的 Gumbel-max 实现；这里的
    argmax 只实现随机抽样，不按方向或 proposal 收益选择候选。外部显式传入
    Gumbel 向量，便于不同核使用共同随机量做配对比较。
    """
    values = _normalize_log_probabilities(log_probabilities)
    raw_gumbels = np.asarray(gumbels)
    if raw_gumbels.shape != values.shape:
        raise ValueError(
            "gumbels 必须与 log_probabilities 形状一致，"
            f"得到 {raw_gumbels.shape} 与 {values.shape}"
        )
    if raw_gumbels.dtype.kind not in "iuf":
        raise ValueError("gumbels 必须是有限数值数组")
    noise = raw_gumbels.astype(float, copy=False)
    if not np.all(np.isfinite(noise)):
        raise ValueError("gumbels 必须是有限数值数组")
    return int(np.argmax(values + noise))


def match_gibbs_strength_for_kl(
    reference_log_probabilities: np.ndarray,
    directions: np.ndarray,
    target_kl: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iterations: int = 100,
) -> float:
    """求使联合 Gibbs 核达到给定 KL 的非负强度。

    对有限 mask 空间，``KL(q_beta || q0)`` 沿非负 ``beta`` 单调不减。目标超过
    该方向地形的正温极限时显式报错，而不是悄悄使用更小 KL 冒充匹配。
    """
    reference = _normalize_log_probabilities(reference_log_probabilities)
    raw_directions = np.asarray(directions)
    if raw_directions.shape != reference.shape:
        raise ValueError("directions 必须与参考对数概率形状一致")
    if raw_directions.dtype.kind not in "iuf":
        raise ValueError("directions 必须是有限数值数组")
    energy = raw_directions.astype(float, copy=False)
    if not np.all(np.isfinite(energy)):
        raise ValueError("directions 必须是有限数值数组")
    if (
        isinstance(target_kl, (bool, np.bool_))
        or not isinstance(target_kl, (int, float, np.integer, np.floating))
        or not np.isfinite(target_kl)
        or target_kl < 0.0
    ):
        raise ValueError(f"target_kl 必须是非负有限数值，得到 {target_kl!r}")
    if (
        not np.isfinite(rtol) or rtol < 0.0
        or not np.isfinite(atol) or atol < 0.0
    ):
        raise ValueError("rtol 和 atol 必须是非负有限数值")
    iterations = _validate_nonnegative_integer(max_iterations, "max_iterations")
    if iterations == 0:
        raise ValueError("max_iterations 必须大于 0")

    target = float(target_kl)
    tolerance = float(atol + rtol * target)
    if target <= tolerance:
        return 0.0
    energy_range = float(np.max(energy) - np.min(energy))
    if energy_range == 0.0:
        raise ValueError("常数方向地形只能达到 KL=0，无法匹配 target_kl")

    maximum = float(np.max(energy))
    maximizers = energy == maximum
    max_reference_mass = float(np.exp(reference[maximizers]).sum())
    limiting_kl = float(-np.log(max_reference_mass))
    if target > limiting_kl + tolerance:
        raise ValueError(
            "target_kl 超过该方向地形的正温极限："
            f"{target:.6g} > {limiting_kl:.6g}"
        )

    def divergence(strength: float) -> float:
        probabilities = gibbs_mask_log_probabilities(
            reference, energy, strength
        )
        return categorical_kl(probabilities, reference)

    lower = 0.0
    upper = 1.0 / energy_range
    upper_kl = divergence(upper)
    for _ in range(max_iterations):
        if upper_kl >= target - tolerance:
            break
        upper *= 2.0
        upper_kl = divergence(upper)
    else:
        raise RuntimeError("未能在数值范围内括住 target_kl")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_kl = divergence(midpoint)
        if abs(midpoint_kl - target) <= tolerance:
            return midpoint
        if midpoint_kl < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def compute_joint_mask_landscapes(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    row_indices: Sequence[int],
    schema: Schema,
    queries: List[Dict[str, Any]],
    residual: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    batch_size: int = 256,
    device: str = "numpy",
    max_active_attributes: int = 16,
) -> List[JointMaskLandscape]:
    """批量计算若干 recipient-donor 对的完整 joint-mask 方向地形。"""
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 ({len(donors)}) 不一致"
        )
    attr_names = schema.attribute_names()
    missing_current = [name for name in attr_names if name not in current.columns]
    missing_donors = [name for name in attr_names if name not in donors.columns]
    if missing_current or missing_donors:
        raise ValueError(
            "current/donors 缺少 schema 属性列："
            f"current={missing_current}, donors={missing_donors}"
        )

    raw_indices = np.asarray(row_indices)
    if raw_indices.ndim != 1:
        raise ValueError("row_indices 必须是一维整数序列")
    if raw_indices.size == 0:
        indices = np.empty(0, dtype=np.intp)
    elif raw_indices.dtype.kind in "iu":
        indices = raw_indices.astype(np.intp, copy=False)
    else:
        raise ValueError("row_indices 必须是一维整数序列")
    if np.any(indices < 0) or np.any(indices >= len(current)):
        raise ValueError("row_indices 包含越界位置")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("row_indices 不得重复")

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    metadata = []
    hybrid_frames = []
    offset = 0
    for row_index in indices:
        current_values = current_reset.loc[row_index, attr_names].to_numpy()
        donor_values = donors_reset.loc[row_index, attr_names].to_numpy()
        active_indices = np.flatnonzero(current_values != donor_values)
        masks = enumerate_copy_masks(
            len(active_indices),
            max_active_attributes=max_active_attributes,
        )
        n_states = len(masks)
        repeated_positions = np.full(n_states, row_index, dtype=np.intp)
        hybrids = current_reset.iloc[repeated_positions].reset_index(drop=True)
        for local_index, attr_index in enumerate(active_indices):
            selected = masks[:, local_index]
            if np.any(selected):
                attr = attr_names[attr_index]
                hybrids.loc[selected, attr] = donors_reset.at[row_index, attr]
        hybrid_frames.append(hybrids)
        metadata.append((
            int(row_index),
            active_indices.astype(np.intp, copy=False),
            tuple(attr_names[index] for index in active_indices),
            masks,
            offset,
            offset + n_states,
        ))
        offset += n_states

    if hybrid_frames:
        all_hybrids = pd.concat(hybrid_frames, ignore_index=True)
    else:
        all_hybrids = current_reset.iloc[[]].copy()
    potentials = evaluate_directional_potential(
        all_hybrids,
        queries,
        schema,
        residual,
        weights=weights,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    landscapes = []
    for (
        row_index,
        active_indices,
        active_attributes,
        masks,
        start,
        end,
    ) in metadata:
        directions = potentials[start:end] - potentials[start]
        directions[0] = 0.0
        landscapes.append(JointMaskLandscape(
            row_index=row_index,
            active_attribute_indices=active_indices,
            active_attributes=active_attributes,
            masks=masks,
            directions=directions,
        ))
    return landscapes
