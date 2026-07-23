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
from table_diffevo.vectorized_eval import evaluate_vectorized


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
    device: str = 'numpy',
    eval_method: str = 'vectorized',
    batch_size: int = 256,
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
    device : str, default 'numpy'
        计算设备（用于距离计算）：
        - 'cuda': PyTorch GPU 加速
        - 'cpu': PyTorch CPU
        - 'numpy': 原始 NumPy 实现
    eval_method : str, default 'vectorized'
        查询评价方式（性能开关，不改变结果，仅改变算法实现）：
        - 'vectorized'（默认，快）：向量化+分块评价，计数与 fitness 一次算完
          （vectorized_eval.evaluate_vectorized）。当前表 S 只评价一次同时得到
          计数和 fitness，消除了 legacy 路径 evaluate_table+compute_fitness 的重复。
        - 'legacy'（慢）：原始逐查询 pandas 路径（evaluate_table + compute_fitness），
          保留作正确性基准、对拍、应急。结果与 vectorized 一致（numpy 逐位相同）。
    batch_size : int, default 256
        向量化评价的分块大小（一次算多少个查询），仅 eval_method='vectorized' 生效。
        内存峰值 ∝ N × batch_size。

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

    **GPU 加速：** device='cuda' 时，距离计算在 GPU 上进行（20-50x 加速），
    所有随机操作仍在 CPU（NumPy），确保相同种子下完全可复现。

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
    >>> # NumPy CPU 实现
    >>> best_S, diag = run_evolution(target, queries, schema,
    ...                              n_records=300, n_rounds=100, seed=0,
    ...                              device='numpy')
    >>> diag["best_loss"] <= diag["loss_history"][0]  # 不会比初始更差
    True
    >>>
    >>> # GPU 加速
    >>> best_S_gpu, diag_gpu = run_evolution(target, queries, schema,
    ...                                      n_records=300, n_rounds=100, seed=0,
    ...                                      device='cuda')
    """
    target = np.asarray(target, dtype=float)
    m = len(queries)
    if len(target) != m:
        raise ValueError(
            f"target 长度 ({len(target)}) 与查询数 ({m}) 不一致"
        )

    if eval_method not in ('vectorized', 'legacy'):
        raise ValueError(
            f"eval_method 必须是 'vectorized' 或 'legacy'，得到 {eval_method!r}"
        )

    rng = np.random.default_rng(seed)

    # ---- 查询评价分派：按 eval_method 选择向量化快路径或旧逐查询路径 ----
    # 两条路径结果一致（numpy 逐位相同），仅实现与速度不同。旧路径保留作对拍/应急。
    def _eval_counts(df):
        """只算计数 q（用于 proposal、初始 best_loss）。"""
        if eval_method == 'vectorized':
            q_, _, _ = evaluate_vectorized(
                df, queries, schema, batch_size=batch_size, device=device,
                want_fitness=False, verbose=False,
            )
            return q_
        return evaluate_table(df, queries)

    def _eval_counts_resid_fitness(df):
        """
        一次同时算计数 q、残差、fitness（用于当前表 S，消除重复评价）。

        vectorized：一次掩码扫描三样都出（计数、残差、fitness）。
        legacy：原路径 evaluate_table → compute_residual → compute_fitness。
        两条路径结果一致（numpy 逐位相同）。
        """
        if eval_method == 'vectorized':
            return evaluate_vectorized(
                df, queries, schema, target=target, n_records=n_records,
                batch_size=batch_size, device=device, want_fitness=True,
                verbose=False,
            )
        q_ = evaluate_table(df, queries)
        r_ = compute_residual(target, q_, n_records)
        f_ = compute_fitness(df, queries, r_, q_)
        return q_, r_, f_

    # 初始表 S_0（不读源数据，只用 schema 合法域）
    S = init_synthetic_table(n_records, schema, rng)

    best_S = S.copy()
    best_loss = compute_loss(target, _eval_counts(S))

    loss_history: List[float] = []
    accept_history: List[bool] = []
    stopped_early = False
    rounds_run = 0

    for t in range(n_rounds):
        rounds_run = t + 1

        # 1-2-4. 当前答案、残差、适应度（vectorized 一次掩码扫描全出；消除重复评价）
        q, residual, fitness = _eval_counts_resid_fitness(S)
        loss = compute_loss(target, q)
        loss_history.append(loss)

        # 逐轮进度
        print(f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e}", end="", flush=True)

        # 3. 终止检查：残差全 0（达标）→ 抽样前停止
        if np.all(residual == 0):
            stopped_early = True
            # 达标的当前表即最优
            if loss < best_loss:
                best_loss = loss
                best_S = S.copy()
            break

        # 5-6. 距离 → 抽样概率 → 抽 donor
        # cuda/cpu：距离留在设备上（return_tensor），softmax 和抽样也在设备上接力，
        #   数据不下显存，只回传 N 个 donor 索引；随机数仍用 numpy rng（保可复现）。
        # numpy：原路径，全程 NumPy。
        use_torch = device in ('cuda', 'cpu')
        distances = pairwise_block_distance(
            S, S, schema, device=device, return_tensor=use_torch
        )
        probs = compute_sampling_probs(fitness, distances, beta=beta, h=h, device=device)
        donor_idx = sample_donors(probs, rng, device=device)
        donors = S.iloc[donor_idx].reset_index(drop=True)

        # 7. 靠近一步 → 提案
        proposal = evolve_step(
            S, donors, schema, rho=rho, eta=eta, mu=mu, rng=rng
        )

        # 8. 整代安全检查（提案只需计数算 loss）
        proposal_q = _eval_counts(proposal)
        proposal_loss = compute_loss(target, proposal_q)
        accepted = proposal_loss <= loss + tol
        accept_history.append(accepted)
        if accepted:
            S = proposal
        # 否则保持原表（第一版不重试）

        # 逐轮进度：显示接受状态
        print(f" | 接受: {'是' if accepted else '否'}")

        # 9. 更新历史最优（直接用已知 loss，不重复评价）
        current_loss = proposal_loss if accepted else loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_S = S.copy()

    # 计算最终质量指标：平均相对误差（仅用于报告，不影响训练）
    best_q = evaluate_table(best_S, queries)
    abs_errors = np.abs(target - best_q)
    # 避免除零：只对非零目标计算相对误差
    mask = target > 0
    relative_errors = abs_errors[mask] / target[mask]
    mean_relative_error = float(np.mean(relative_errors))

    diagnostics = {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "rounds_run": rounds_run,
        "stopped_early": stopped_early,
        "accept_history": accept_history,
        "mean_relative_error": mean_relative_error,
        "params": {
            "n_records": n_records,
            "n_rounds": n_rounds,
            "seed": seed,
            "beta": beta,
            "h": h,
            "rho": rho,
            "eta": eta,
            "mu": mu,
            "tol": tol,
            "device": device,
            "eval_method": eval_method,
            "batch_size": batch_size,
        },
    }

    return best_S.reset_index(drop=True), diagnostics
