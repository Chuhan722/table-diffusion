"""
目标衡量：残差计算

残差是连接"目标"和"演化方向"的桥梁：

    带噪目标 y  ─┐
                 ├─→  残差 r = y − q(S)  ─→  驱动适应度  ─→  驱动演化
    当前答案 q(S)─┘

残差的物理含义（来自 temp.md 第七节）：
- r > 0：查询偏低，需要增加满足该查询的记录
- r < 0：查询偏高，需要减少满足该查询的记录
- r = 0：查询已达标，中性，不推动

本模块采用"比例残差"，即除以记录总数 N，使残差落在 [-1, 1] 区间，
让适应度公式的尺度与数据规模无关（300 行和 5 万行用同一套超参数）。

## 比例残差公式（temp.md 第七节）

    ε_j = sign(y_j - q_j(S)) · max(|y_j - q_j(S)| - κσ_j, 0) / N

其中：
- y_j - q_j(S)：原始残差（计数差）
- sign(...)：保留方向（+1 / -1 / 0）
- max(|...| - κσ_j, 0)：噪声容忍区，误差小于 κσ 时视为达标，残差归零
- / N：归一化成比例

## 无噪声玩具阶段

当前 σ_j = 0（无 DP 噪声），容忍区为 0，公式简化为：

    ε_j = (y_j - q_j(S)) / N

代码保留 σ/κ 接口（默认无噪声），为将来进入 DP 阶段铺路。

## 与其他模块的分工（temp.md 三层分离思想）

- objective.py（本模块）：目标衡量——当前状态离目标多远
- fitness.py：适应度——某种记录形态该不该繁殖
- 更新率/变异率 + 整代损失检查：控制一轮走多远、有没有过头
"""
from typing import Optional
import numpy as np


def compute_residual(
    target: np.ndarray,
    current: np.ndarray,
    n_records: int,
    sigma: Optional[np.ndarray] = None,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    计算比例残差 ε_j。

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        带噪目标向量 y（当前无噪声阶段即真实计数）
    current : np.ndarray, shape (m,)
        当前答案 q(S)，由 queries.evaluate_table 得到
    n_records : int
        记录总数 N，用于归一化。必须 > 0
    sigma : np.ndarray or None, shape (m,)
        各查询的噪声标准差。None 表示无噪声（σ=0，容忍区为 0）
    kappa : float
        噪声容忍系数。|残差| < κσ 时视为已达标，残差归零

    Returns
    -------
    np.ndarray, shape (m,)
        比例残差向量 ε，落在 [-1, 1] 区间

    Raises
    ------
    ValueError
        n_records <= 0，或 target 与 current 形状不一致

    Examples
    --------
    >>> target = np.array([180, 95, 42])
    >>> current = np.array([170, 100, 42])
    >>> compute_residual(target, current, n_records=300)
    array([ 0.03333333, -0.01666667,  0.        ])
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)

    if n_records <= 0:
        raise ValueError(f"n_records 必须为正数，收到: {n_records}")

    if target.shape != current.shape:
        raise ValueError(
            f"target 与 current 形状不一致: {target.shape} vs {current.shape}"
        )

    # 原始残差（计数差）
    raw = target - current

    # 噪声容忍区：误差小于 κσ 的部分归零，只保留超出容忍区的量
    if sigma is None:
        # 无噪声阶段：容忍区为 0，保留全部残差
        magnitude = np.abs(raw)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != target.shape:
            raise ValueError(
                f"sigma 与 target 形状不一致: {sigma.shape} vs {target.shape}"
            )
        magnitude = np.maximum(np.abs(raw) - kappa * sigma, 0.0)

    # 恢复方向并归一化为比例
    epsilon = np.sign(raw) * magnitude / n_records

    return epsilon
