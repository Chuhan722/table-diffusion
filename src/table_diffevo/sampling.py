"""
参考记录抽样

把"适应度高且相近"这两个维度合成抽样概率，为每条当前记录抽取一个参考记录。

## 抽样分数与概率（依据"抽样分数+抽样概率.pdf"）

对当前记录 x_i 和候选参考记录 z_k，未归一化分数（logit）：

    ℓ_ik = β_t · F_t(z_k) − d(x_i, z_k)² / (2h_t²)

其中：
- β_t：选择强度，控制适应度的影响（≥0）
- F_t(z_k)：候选记录的适应度
- d(x_i, z_k)：当前记录与候选记录的归一化距离
- h_t：邻域尺度，控制距离惩罚的强弱（>0）

按行 softmax 归一化：

    Pr(J_i = k) = exp(ℓ_ik) / Σ_l exp(ℓ_il)

然后每条记录按此概率分布抽一次：z_i* = z_{J_i}

## 两个因子的作用

**适应度项** β·F：统计目标方向上的收益
- β 越大，越偏向高适应度候选
- β = 0 时完全忽略适应度

**距离惩罚项** −d²/(2h²)：限制学习范围
- h 越小，越强烈偏向近邻
- h 越大，距离影响越弱
- 高斯核形式：similarity = exp(−d²/2h²)

## 职责边界

本模块只负责"给定 β、h，算概率、抽样"。
β、h 随轮次的调度（如 h 从 0.8 降到 0.15）由主循环负责。

## 玩具阶段的使用

候选集 = 全表（K=N=300），允许记录抽到自己（=本轮保持不变）。
大规模时才用共享参考池（K=512）。
"""
from typing import Optional
import numpy as np


def compute_sampling_probs(
    fitness: np.ndarray,
    distances: np.ndarray,
    beta: float = 1.0,
    h: float = 0.8,
) -> np.ndarray:
    """
    计算每条当前记录对所有候选记录的抽样概率（softmax）。

    Parameters
    ----------
    fitness : np.ndarray, shape (K,)
        候选参考记录的适应度，来自 compute_fitness
    distances : np.ndarray, shape (N, K)
        当前记录与候选记录的归一化距离矩阵，来自 pairwise_block_distance
    beta : float, default 1.0
        选择强度 β_t ≥ 0，控制适应度的影响
        - 越大越偏向高适应度候选
        - = 0 时完全忽略适应度
    h : float, default 0.8
        邻域尺度 h_t > 0，控制距离惩罚的强弱
        - 越小越偏向近邻
        - 越大距离影响越弱
        - 文档建议：前期 0.8 → 后期 0.15（线性衰减）

    Returns
    -------
    np.ndarray, shape (N, K), dtype float
        抽样概率矩阵，probs[i, k] = Pr(J_i = k)
        每行非负、和为 1

    Raises
    ------
    ValueError
        输入形状不一致、beta < 0、h <= 0

    Notes
    -----
    **边界情况（自然处理，不特判）：**
    - 所有候选适应度相同 → 退化为纯距离选择或均匀分布
    - 残差全为 0 → 主循环应在抽样前终止，不会走到这里

    **数值稳定：** logit 减去行最大值再 exp，不改变 softmax 结果

    Examples
    --------
    >>> from table_diffevo.fitness import compute_fitness
    >>> from table_diffevo.distance import pairwise_block_distance
    >>> from table_diffevo.schema import load_schema
    >>>
    >>> # 假设已有 df、queries、residual、current_answer、schema
    >>> fitness = compute_fitness(df, queries, residual, current_answer)
    >>> distances = pairwise_block_distance(df, df, schema)  # 玩具阶段全对全
    >>>
    >>> probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8)
    >>> probs.shape
    (300, 300)
    >>> np.allclose(probs.sum(axis=1), 1.0)  # 每行和为 1
    True
    """
    if beta < 0:
        raise ValueError(f"beta 必须 ≥ 0，得到 {beta}")
    if h <= 0:
        raise ValueError(f"h 必须 > 0，得到 {h}")

    fitness = np.asarray(fitness, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if fitness.ndim != 1:
        raise ValueError(f"fitness 必须是 1 维，得到 shape {fitness.shape}")
    if distances.ndim != 2:
        raise ValueError(f"distances 必须是 2 维，得到 shape {distances.shape}")

    N, K = distances.shape
    if len(fitness) != K:
        raise ValueError(
            f"fitness 长度 ({len(fitness)}) 与 distances 列数 ({K}) 不一致"
        )

    # 计算 logit（未归一化分数）
    # ℓ_ik = β·F(z_k) − d(x_i, z_k)² / (2h²)
    fitness_term = beta * fitness  # (K,) 广播到每行
    distance_penalty = distances**2 / (2 * h**2)  # (N, K)
    logits = fitness_term[None, :] - distance_penalty  # (N, K)

    # softmax（减去行最大值做数值稳定）
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return probs


def sample_donors(
    probs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对每条当前记录，按概率分布抽取一个参考记录索引。

    Parameters
    ----------
    probs : np.ndarray, shape (N, K)
        抽样概率矩阵，来自 compute_sampling_probs
        每行应非负、和为 1
    rng : np.random.Generator or None
        随机数生成器。None 时使用全局随机状态（不推荐，除非已 set_seed）
        推荐显式传入：rng = np.random.default_rng(seed)

    Returns
    -------
    np.ndarray, shape (N,), dtype int
        每条记录抽到的候选索引，值在 [0, K)
        donor_indices[i] = J_i，即第 i 条记录抽到的候选编号

    Raises
    ------
    ValueError
        probs 不是 2 维、或某行和不为 1（容差 1e-6）

    Notes
    -----
    **复现性（铁律 5）：** 使用固定种子的 rng 保证结果可复现。

    **允许抽到自己：** 玩具阶段 K=N，记录可能抽到自己（索引相同），
    等价于本轮保持不变，这是合法的演化步骤。

    Examples
    --------
    >>> probs = compute_sampling_probs(fitness, distances)
    >>> rng = np.random.default_rng(42)
    >>> indices = sample_donors(probs, rng)
    >>> indices.shape
    (300,)
    >>> (indices >= 0).all() and (indices < 300).all()
    True
    >>>
    >>> # 固定种子可复现
    >>> rng1 = np.random.default_rng(42)
    >>> rng2 = np.random.default_rng(42)
    >>> idx1 = sample_donors(probs, rng1)
    >>> idx2 = sample_donors(probs, rng2)
    >>> np.array_equal(idx1, idx2)
    True
    """
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 2:
        raise ValueError(f"probs 必须是 2 维，得到 shape {probs.shape}")

    N, K = probs.shape

    # 验证每行和为 1（容差 1e-6）
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(
            f"probs 某些行和不为 1（容差 1e-6）：行 {bad_rows[:5]}... "
            f"行和范围 [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    if rng is None:
        rng = np.random

    # 对每行按 Categorical 分布抽样
    # numpy 没有直接的多行 categorical，用累积概率 + searchsorted
    cumprobs = probs.cumsum(axis=1)
    u = rng.uniform(size=N)[:, None]  # (N, 1)
    indices = (u < cumprobs).argmax(axis=1)  # 找第一个 cumprob >= u 的位置

    return indices.astype(np.intp)
