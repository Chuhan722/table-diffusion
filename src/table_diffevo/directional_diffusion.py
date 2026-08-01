"""残差驱动的局部扩散方向。

本模块只计算“把 recipient 的一个属性块改成 donor 对应值”在固定旧残差场中的
一阶方向量。它不判断方向是否为正、不筛掉候选，也不执行随机抽样。方向量随后
用于连续倾斜块复制概率，使扩散核本身产生偏向，同时保留反向转移支持。
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from table_diffevo.schema import Schema
from table_diffevo.vectorized_eval import evaluate_directional_potential


def direction_rms_scale(direction_scores: np.ndarray) -> float:
    """返回有限方向量的数值稳定 RMS，用于无量纲温度定标。

    空数组和全零数组返回 0。先除以最大绝对值再计算，避免有限大数平方溢出。
    本函数只给出尺度，不改变方向符号或执行筛选。
    """
    raw_scores = np.asarray(direction_scores)
    if raw_scores.dtype.kind not in "iuf":
        raise ValueError("direction_scores 必须是有限数值数组")
    scores = raw_scores.astype(float, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("direction_scores 必须是有限数值数组")
    if scores.size == 0:
        return 0.0

    max_abs = float(np.max(np.abs(scores)))
    if max_abs == 0.0:
        return 0.0
    return float(max_abs * np.sqrt(np.mean((scores / max_abs) ** 2)))


def tilted_copy_probabilities(
    eta: float,
    direction_scores: np.ndarray,
    strength: float,
) -> np.ndarray:
    """用方向量连续倾斜 Bernoulli 复制概率。

    采用 baseline odds 的指数倾斜：

    ``logit(p) = logit(eta) + strength * direction``。

    中性方向显式返回 eta；数值 logit 截断到 [-30, 30]，避免有限输入在浮点下
    变成精确 0/1，从而保留正反方向支持。eta 的 0/1 端点保持原语义。
    """
    if (
        isinstance(eta, (bool, np.bool_))
        or not isinstance(eta, (int, float, np.integer, np.floating))
        or not np.isfinite(eta)
        or not 0.0 <= eta <= 1.0
    ):
        raise ValueError(f"eta 必须是 [0, 1] 内的有限数值，得到 {eta!r}")
    if (
        isinstance(strength, (bool, np.bool_))
        or not isinstance(strength, (int, float, np.integer, np.floating))
        or not np.isfinite(strength)
        or strength < 0.0
    ):
        raise ValueError(
            f"strength 必须是非负有限数值，得到 {strength!r}"
        )

    raw_scores = np.asarray(direction_scores)
    if raw_scores.dtype.kind not in "iuf":
        raise ValueError("direction_scores 必须是有限数值数组")
    scores = raw_scores.astype(float, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("direction_scores 必须是有限数值数组")

    eta = float(eta)
    strength = float(strength)
    if eta <= 0.0:
        return np.zeros_like(scores, dtype=float)
    if eta >= 1.0:
        return np.ones_like(scores, dtype=float)
    if strength == 0.0:
        return np.full_like(scores, eta, dtype=float)

    base_logit = np.log(eta) - np.log1p(-eta)
    with np.errstate(over="ignore"):
        tilted_logits = base_logit + strength * scores
    logits = np.clip(tilted_logits, -30.0, 30.0)
    probs = np.empty_like(logits, dtype=float)
    positive = logits >= 0.0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    probs[scores == 0.0] = eta
    return probs


def compute_copy_direction_scores(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    residual: np.ndarray,
    weights: Optional[np.ndarray] = None,
    batch_size: int = 256,
    device: str = "numpy",
) -> np.ndarray:
    """计算每个 recipient-donor 属性块复制的局部方向量。

    第 ``i, g`` 个元素对应实际单块转移

    ``current.iloc[i] -> current.iloc[i] with block g from donors.iloc[i]``

    的 ``sum_j w_j * residual[j] * delta_a_j``。只评价包含属性块 ``g`` 的
    查询；不包含该块的查询贡献变化严格为零。donor 与 recipient 在该块相同时
    分数为零。

    所有分数都使用同一份调用时传入的旧残差，保持整代同步语义。这里评价的是
    实际单块 hybrid 编辑，而不是完整 donor 替换的适应度代理。
    """
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )

    attr_names = schema.attribute_names()
    missing_current = [name for name in attr_names if name not in current.columns]
    missing_donors = [name for name in attr_names if name not in donors.columns]
    if missing_current or missing_donors:
        raise ValueError(
            "current/donors 缺少 schema 属性列："
            f"current={missing_current}, donors={missing_donors}"
        )

    residual_array = np.asarray(residual)
    m = len(queries)
    if residual_array.shape != (m,):
        raise ValueError(
            f"residual 必须是长度与 queries 一致的一维数组，"
            f"得到 shape {residual_array.shape}，期望 ({m},)"
        )
    if (
        residual_array.dtype.kind not in "iuf"
        or not np.all(np.isfinite(residual_array))
    ):
        raise ValueError("residual 必须是有限数值数组")
    residual_array = residual_array.astype(float, copy=False)

    if weights is None:
        weights_array = None
    else:
        weights_array = np.asarray(weights)
        if weights_array.shape != (m,):
            raise ValueError(
                f"weights 必须是长度与 queries 一致的一维数组，"
                f"得到 shape {weights_array.shape}，期望 ({m},)"
            )
        if (
            weights_array.dtype.kind not in "iuf"
            or not np.all(np.isfinite(weights_array))
        ):
            raise ValueError("weights 必须是有限数值数组")
        weights_array = weights_array.astype(float, copy=False)

    if isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
        raise ValueError(f"batch_size 必须是正整数，得到 {batch_size!r}")
    if device not in ("numpy", "cuda", "cpu"):
        raise ValueError(
            f"device 必须是 'numpy'、'cuda' 或 'cpu'，得到 {device!r}"
        )

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    N = len(current_reset)
    scores = np.zeros((N, len(attr_names)), dtype=float)
    if N == 0 or m == 0:
        return scores

    query_indices_by_attr = {name: [] for name in attr_names}
    for query_idx, query in enumerate(queries):
        query_attrs = {
            condition["attribute"] for condition in query["conditions"]
        }
        for attr in query_attrs:
            if attr in query_indices_by_attr:
                query_indices_by_attr[attr].append(query_idx)

    for attr_idx, attr in enumerate(attr_names):
        relevant = query_indices_by_attr[attr]
        if not relevant:
            continue

        current_values = current_reset[attr].to_numpy()
        donor_values = donors_reset[attr].to_numpy()
        differs = current_values != donor_values
        differing_rows = np.flatnonzero(differs)
        if len(differing_rows) == 0:
            continue

        base_rows = current_reset.iloc[differing_rows].copy()
        candidate_rows = base_rows.copy()
        candidate_rows[attr] = donor_values[differing_rows]

        # 拼接后统一编码，保证字符串类别在 base/candidate 两半使用同一映射。
        combined = pd.concat(
            [base_rows, candidate_rows], ignore_index=True
        )
        relevant_queries = [queries[idx] for idx in relevant]
        relevant_residual = residual_array[relevant]
        relevant_weights = (
            None if weights_array is None else weights_array[relevant]
        )
        potential = evaluate_directional_potential(
            combined,
            relevant_queries,
            schema,
            relevant_residual,
            weights=relevant_weights,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        half = len(differing_rows)
        scores[differing_rows, attr_idx] = (
            potential[half:] - potential[:half]
        )

    return scores
