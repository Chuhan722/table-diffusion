"""
扩散演化主循环

把所有零件串起来：固定 S_t → 算残差 → 算适应度 → 抽 donor → 靠近一步
→ 整代检查 → 重算残差，一轮轮迭代逼近目标。

## 一轮流程（完整方案第 8、9 节）

1. evaluate_table(S, queries) → 当前答案 q
2. compute_residual(target, q, N) → 残差 ε
3. 检查终止：残差全 0 → 在抽样前停止
4. compute_fitness(S, queries, ε, q) → 适应度 F
5. pairwise_block_distance(S, S, schema) → 距离矩阵（玩具阶段全对全）
6. compute_sampling_probs + sample_donors → donor 索引 → 对齐 donors
7. evolve_step(S, donors, schema, ρ, η, μ, rng) → 提案 proposal
8. 整代安全检查：loss(proposal) ≤ loss(S) + 容差 → 接受，否则保持原表
9. 更新 best_S，进下一轮

## 第一版的简化（已与设计确认）

- 整代检查失败 → 保持原表（不重试、不缩小步幅）
- 参数 β/h/ρ/η/μ 用固定值（不随轮次衰减）
- 终止条件 = 残差全 0 或达到最大轮数 T
- 只接收 target（目标计数），不接收源数据（守铁律 6）
- 诊断只记每轮 loss + 少量汇总

## 铁律遵守

- 运行期不读真实私有答案：只用 target（已发布的目标）和 schema（公开）
- 全表同步：一轮内所有记录基于同一份 S_t 和同一份残差生成下一状态
- 固定种子可复现：seed → np.random.default_rng
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from table_diffevo.schema import Schema
from table_diffevo.queries import evaluate_table
from table_diffevo.objective import compute_residual, compute_loss
from table_diffevo.fitness import compute_fitness
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.update import evolve_step
from table_diffevo.generator import init_synthetic_table


def run_evolution(
    target: np.ndarray,
    queries: List[Dict[str, Any]],
    schema: Schema,
    n_records: int,
    n_rounds: int = 100,
    seed: int = 0,
    beta: float = 1.0,
    h: float = 0.8,
    rho: float = 0.1,
    eta: float = 0.5,
    mu: float = 0.01,
    tol: float = 1e-9,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    运行扩散演化主循环，返回历史最优合成表和诊断信息。

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        目标计数向量 y（各查询的正确答案）。运行期唯一接触的"目标"信息
    queries : List[Dict]
        查询定义列表
    schema : Schema
        属性 schema 定义
    n_records : int
        合成表记录条数 N（与源数据一致，值本身为公开信息）
    n_rounds : int, default 100
        最大轮数 T
    seed : int, default 0
        随机种子（复现，铁律 5）
    beta, h : float
        抽样参数：选择强度、邻域尺度（固定值，不衰减）
    rho, eta, mu : float
        更新参数：记录参与率、块复制率、变异率（固定值，不衰减）
    tol : float, default 1e-9
        整代检查的数值容差：loss(proposal) ≤ loss(S) + tol 时接受

    Returns
    -------
    best_S : pd.DataFrame, shape (n_records, n_attributes)
        演化过程中见过的 loss 最小的合成表
    diagnostics : dict
        诊断信息：
        - loss_history: List[float]，每轮开始时当前表的 loss
        - best_loss: float，最优 loss
        - rounds_run: int，实际跑的轮数
        - stopped_early: bool，是否因残差全 0 提前停止
        - accept_history: List[bool]，每轮整代检查是否接受提案

    Raises
    ------
    ValueError
        target 长度与 queries 数量不一致

    Notes
    -----
    **终止条件：** 残差全 0（达标）或达到 n_rounds。

    **整代检查失败：** 保持原表（第一版不重试）。best_S 保底，
    即使某轮无进展，最终返回的仍是历史最优表。

    Examples
    --------
    >>> from table_diffevo.schema import load_schema
    >>> from table_diffevo.queries import load_queries
    >>> import numpy as np
    >>>
    >>> schema = load_schema("configs/schema.yaml")
    >>> queries = load_queries("configs/measured_50query.json")
    >>> target = np.array([q["result"] for q in queries])
    >>>
    >>> best_S, diag = run_evolution(target, queries, schema,
    ...                              n_records=300, n_rounds=100, seed=0)
    >>> diag["best_loss"] <= diag["loss_history"][0]  # 不会比初始更差
    True
    """
    target = np.asarray(target, dtype=float)
    m = len(queries)
    if len(target) != m:
        raise ValueError(
            f"target 长度 ({len(target)}) 与查询数 ({m}) 不一致"
        )

    rng = np.random.default_rng(seed)

    # 初始表 S_0（不读源数据，只用 schema 合法域）
    S = init_synthetic_table(n_records, schema, rng)

    best_S = S.copy()
    best_loss = compute_loss(target, evaluate_table(S, queries))

    loss_history: List[float] = []
    accept_history: List[bool] = []
    stopped_early = False
    rounds_run = 0

    for t in range(n_rounds):
        rounds_run = t + 1

        # 1-2. 当前答案与残差
        q = evaluate_table(S, queries)
        residual = compute_residual(target, q, n_records)
        loss = compute_loss(target, q)
        loss_history.append(loss)

        # 3. 终止检查：残差全 0（达标）→ 抽样前停止
        if np.all(residual == 0):
            stopped_early = True
            # 达标的当前表即最优
            if loss < best_loss:
                best_loss = loss
                best_S = S.copy()
            break

        # 4. 适应度
        fitness = compute_fitness(S, queries, residual, q)

        # 5. 距离矩阵（玩具阶段全对全）
        distances = pairwise_block_distance(S, S, schema)

        # 6. 抽 donor
        probs = compute_sampling_probs(fitness, distances, beta=beta, h=h)
        donor_idx = sample_donors(probs, rng)
        donors = S.iloc[donor_idx].reset_index(drop=True)

        # 7. 靠近一步 → 提案
        proposal = evolve_step(
            S, donors, schema, rho=rho, eta=eta, mu=mu, rng=rng
        )

        # 8. 整代安全检查
        proposal_q = evaluate_table(proposal, queries)
        proposal_loss = compute_loss(target, proposal_q)
        accepted = proposal_loss <= loss + tol
        accept_history.append(accepted)
        if accepted:
            S = proposal
        # 否则保持原表（第一版不重试）

        # 9. 更新历史最优
        current_loss = compute_loss(target, evaluate_table(S, queries))
        if current_loss < best_loss:
            best_loss = current_loss
            best_S = S.copy()

    diagnostics = {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "rounds_run": rounds_run,
        "stopped_early": stopped_early,
        "accept_history": accept_history,
    }

    return best_S.reset_index(drop=True), diagnostics
