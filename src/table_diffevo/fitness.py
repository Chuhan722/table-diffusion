"""
适应度计算

适应度回答："某种记录形态，现在值不值得被繁殖？"

它连接残差和演化：

    残差 ε  ─→  适应度 F(z)  ─→  转成选择概率  ─→  选参考记录  ─→  演化

## 纯方向适应度公式（temp.md 第七节）

已删除二次步幅项，只保留残差方向项：

    F_t(z) = Σ_j w_j ε_j (a_j(z) - p_j)

其中：
- ε_j：查询 j 的比例残差（已由 objective.py 计算）
- a_j(z)：记录 z 对查询 j 的贡献（满足=1，不满足=0）
- p_j：当前满足比例 = q_j(S) / N
- w_j：查询权重（默认全 1）

## 物理含义

对于记录 z：
- 如果查询 j 偏低（ε_j > 0），且 z 满足 j（a_j > p_j），z 加分
- 如果查询 j 偏高（ε_j < 0），且 z 满足 j（a_j > p_j），z 扣分
- 如果查询 j 已达标（ε_j = 0），z 在这个查询上中性（不加不扣）

## 实现策略（应对大规模查询）

**不建贡献矩阵**（5万×5万=2.5GB），而是**逐查询累加**（内存O(N)）：

    fitness = 0
    for j in queries:
        mask_j = eval_query_mask(df, query_j)  # 临时掩码，用完就丢
        fitness += ε_j * (mask_j - p_j)

每次只在内存里存一个查询的掩码（长度N），与查询数量无关。

## 与 temp.md 四状态例子的验证

temp.md 第七节验证过，当：
- q1: A=1 目标3 当前2（偏低，ε>0）
- q2: B=1 目标1 当前2（偏高，ε<0）
- q3: A∧B 目标1 当前1（达标，ε=0）

适应度结果：
- 10（满足q1不满足q2）：+1（最该增加）
- 01（满足q2不满足q1）：-1（最该减少）
- 00、11：0（中性）

这个例子已写成测试，锚定实现正确性。
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from table_diffevo.queries import eval_query_mask


def compute_fitness(
    df: pd.DataFrame,
    queries: List[Dict[str, Any]],
    residual: np.ndarray,
    current_answer: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算每条记录的适应度（纯方向适应度，temp.md）。

    Parameters
    ----------
    df : pd.DataFrame
        数据表（合成表或原始数据）
    queries : List[Dict]
        查询定义列表
    residual : np.ndarray, shape (m,)
        比例残差向量 ε，由 objective.compute_residual 得到
    current_answer : np.ndarray, shape (m,)
        当前查询答案 q(S)，由 queries.evaluate_table 得到
    weights : np.ndarray or None, shape (m,)
        查询权重 w_j，默认全 1

    Returns
    -------
    np.ndarray, shape (N,)
        每条记录的适应度

    Raises
    ------
    ValueError
        输入形状不一致

    Notes
    -----
    **内存优化**：不建 (N, m) 贡献矩阵，而是逐查询累加。
    每次只存一个查询的掩码（长度 N），内存 O(N) 与查询数无关。

    虽然掩码会被计算两次（evaluate_table 一次，此函数一次），
    但在查询量很大时，多花计算时间（几秒）换来内存从 GB 降到 KB 是值得的。

    Examples
    --------
    >>> from table_diffevo.queries import load_queries, evaluate_table
    >>> from table_diffevo.objective import compute_residual
    >>>
    >>> df = ...  # 合成表
    >>> queries = load_queries("configs/measured_50query.json")
    >>> target = np.array([q["result"] for q in queries])
    >>>
    >>> current = evaluate_table(df, queries)
    >>> residual = compute_residual(target, current, n_records=len(df))
    >>> fitness = compute_fitness(df, queries, residual, current)
    >>>
    >>> # fitness[i] > 0：记录i更值得被繁殖
    >>> # fitness[i] < 0：记录i应该被淘汰
    """
    N = len(df)
    m = len(queries)

    if len(residual) != m:
        raise ValueError(
            f"residual 长度 ({len(residual)}) 与查询数 ({m}) 不一致"
        )
    if len(current_answer) != m:
        raise ValueError(
            f"current_answer 长度 ({len(current_answer)}) 与查询数 ({m}) 不一致"
        )

    if weights is None:
        weights = np.ones(m)
    elif len(weights) != m:
        raise ValueError(
            f"weights 长度 ({len(weights)}) 与查询数 ({m}) 不一致"
        )

    # 当前各查询的满足比例 p_j = q_j(S) / N
    p = current_answer / N

    # 初始化：所有记录的适应度为 0
    fitness = np.zeros(N)

    # 逐查询累加（内存 O(N)，与查询数无关）
    for j in range(m):
        # 1. 获取这个查询的掩码（布尔数组，长度 N）
        mask = eval_query_mask(df, queries[j])

        # 2. 转成 0/1，计算贡献偏差 a_j(z) - p_j
        a_j = mask.astype(float)  # 满足=1.0, 不满足=0.0
        contribution_deviation = a_j - p[j]

        # 3. 这个查询对各记录适应度的增量
        delta = weights[j] * residual[j] * contribution_deviation

        # 4. 累加到总适应度
        fitness += delta

        # mask 用完后作用域结束，下次循环被新 mask 覆盖，内存自动释放

    return fitness
