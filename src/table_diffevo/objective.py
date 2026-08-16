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

## 残差几何（geometry 参数，Issue #57）

绝对几何（默认）把每单位计数残差视为等价，是 L2 计数损失的梯度口径；
它会把优化力气集中在大计数查询上，而评价指标（每查询平均归一化 L1）
对所有查询等权。相对几何把残差除以目标计数（带下限保护），近似
KL/最大熵拟合的梯度口径，稀有查询的推动力按其相对误差放大：

    absolute:  ε_j = sign(raw_j) · magnitude_j / N
    relative:  ε_j = sign(raw_j) · magnitude_j / max(y_j, floor) / N

其中 magnitude 是噪声容忍后的残差幅度（容忍先于相对化，σ/κ 语义不变）。
floor > 0 防止 y_j 极小或为 0 时分母爆炸。

## 与其他模块的分工（temp.md 三层分离思想）

- objective.py（本模块）：目标衡量——当前状态离目标多远
- fitness.py：适应度——某种记录形态该不该繁殖
- 更新率/变异率 + 整代损失检查：控制一轮走多远、有没有过头
"""
from typing import Optional
import numpy as np


RESIDUAL_GEOMETRIES = ("absolute", "relative")


def compute_residual(
    target: np.ndarray,
    current: np.ndarray,
    n_records: int,
    sigma: Optional[np.ndarray] = None,
    kappa: float = 1.0,
    geometry: str = "absolute",
    geometry_floor: float = 8.0,
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
    geometry : str, default "absolute"
        残差几何。"absolute"：现状口径 ε = sign·magnitude/N（L2 计数损失
        梯度）；"relative"：ε = sign·magnitude/max(y,floor)/N（近似 KL
        梯度，稀有查询推动力按相对误差放大）。相对化只使用公开 target
        计数做归一化，不引入新信息流。
    geometry_floor : float, default 8.0
        relative 几何的分母下限（计数单位），防止 y 极小或为 0 时分母
        爆炸。仅 geometry="relative" 时使用，必须 > 0。

    Returns
    -------
    np.ndarray, shape (m,)
        比例残差向量 ε。absolute 几何落在 [-1, 1]；relative 几何幅度
        上界为 1/floor（分子 magnitude ≤ N、分母 ≥ floor·N），整体
        常数尺度由下游选择核/方向场归一化吸收

    Raises
    ------
    ValueError
        n_records <= 0，target 与 current 形状不一致，geometry 非法，
        或 geometry_floor 非正有限数（inf/nan/bool 均拒绝——inf 会把
        全部残差压成 0 并错误触发精确达标早停）

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

    if geometry not in RESIDUAL_GEOMETRIES:
        raise ValueError(
            f"geometry 必须是 {RESIDUAL_GEOMETRIES} 之一，"
            f"得到 {geometry!r}"
        )
    if geometry == "relative":
        if (
            isinstance(geometry_floor, bool)
            or not isinstance(
                geometry_floor, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(geometry_floor)
            or not geometry_floor > 0
        ):
            raise ValueError(
                "geometry_floor 必须是正有限数（非布尔），"
                f"得到 {geometry_floor!r}"
            )

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

    # 恢复方向并归一化为比例；relative 几何先按目标计数相对化
    # （容忍先于相对化：magnitude 已是容忍后的幅度，σ/κ 语义不变）
    if geometry == "relative":
        magnitude = magnitude / np.maximum(target, geometry_floor)
    epsilon = np.sign(raw) * magnitude / n_records

    return epsilon


def compute_loss(
    target: np.ndarray,
    current: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    kappa: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    计算监控损失 E(S)（完整方案 8.2）。

        E(S) = ½ Σ_j w_j [max(|y_j - q_j| - κσ_j, 0)]²

    衡量合成表的查询答案离目标有多远。越小越好，E=0 表示所有查询达标。

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        目标向量 y（当前无噪声阶段即真实计数）
    current : np.ndarray, shape (m,)
        当前答案 q(S)，由 queries.evaluate_table 得到
    sigma : np.ndarray or None, shape (m,)
        各查询的噪声标准差。None 表示无噪声（σ=0，容忍区为 0）
    kappa : float
        噪声容忍系数。|残差| < κσ 的部分不计入损失
    weights : np.ndarray or None, shape (m,)
        查询权重 w_j，默认全 1

    Returns
    -------
    float
        损失值 E(S) ≥ 0

    Raises
    ------
    ValueError
        target 与 current 形状不一致，或 weights 长度不匹配

    Notes
    -----
    **口径：使用计数残差** e_j = y_j - q_j（不是比例残差）。
    与 compute_residual 的比例残差差一个 N 倍数，但用于比较大小时排序一致。

    **无噪声玩具阶段**（σ=None、w=1）简化为：

        E(S) = ½ Σ_j (y_j - q_j)²

    即计数残差平方和的一半。

    Examples
    --------
    >>> target = np.array([180, 95, 42])
    >>> current = np.array([170, 100, 42])
    >>> compute_loss(target, current)  # ½(10² + 5² + 0²) = ½·125
    62.5
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)

    if target.shape != current.shape:
        raise ValueError(
            f"target 与 current 形状不一致: {target.shape} vs {current.shape}"
        )

    raw = target - current

    # 噪声容忍区：只保留超出 κσ 的部分
    if sigma is None:
        magnitude = np.abs(raw)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != target.shape:
            raise ValueError(
                f"sigma 与 target 形状不一致: {sigma.shape} vs {target.shape}"
            )
        magnitude = np.maximum(np.abs(raw) - kappa * sigma, 0.0)

    if weights is None:
        weights = np.ones_like(target)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != target.shape:
            raise ValueError(
                f"weights 与 target 形状不一致: {weights.shape} vs {target.shape}"
            )

    return 0.5 * float(np.sum(weights * magnitude**2))
