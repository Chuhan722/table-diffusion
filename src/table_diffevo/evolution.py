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

## 当前简化与可选增强

- 整代检查失败 → 默认保持原表；可选缩小 rho 重试
- 参数 β/h/ρ/η/μ 用固定值（不随轮次衰减）
- 终止条件 = 残差全 0 或达到最大轮数 T
- 只接收 target（目标计数），不接收源数据（守铁律 6）
- 诊断只记每轮 loss + 少量汇总
- 提案被拒后 S 不变，复用当前答案/适应度/距离；动态抽样概率仍按每轮 alpha 重算

## 铁律遵守

- 运行期不读真实私有答案：只用 target（已发布的目标）和 schema（公开）
- 全表同步：一轮内所有记录基于同一份 S_t 和同一份残差生成下一状态
- 固定种子可复现：seed → np.random.default_rng
"""
import time
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
from table_diffevo.pairwise_init import init_from_pairwise_maxent
from table_diffevo.vectorized_eval import evaluate_vectorized


def _mean_selected_distance(distances, donor_idx, use_torch: bool = False) -> float:
    """只提取每行选中 donor 的距离并求均值，避免回传完整 GPU 距离矩阵。"""
    donor_idx = np.asarray(donor_idx, dtype=np.intp)
    n_rows = len(donor_idx)

    if use_torch:
        import torch

        if isinstance(distances, torch.Tensor):
            rows = torch.arange(n_rows, device=distances.device)
            donors = torch.as_tensor(
                donor_idx, dtype=torch.long, device=distances.device
            )
            return float(distances[rows, donors].mean().item())

    distances_array = np.asarray(distances)
    return float(distances_array[np.arange(n_rows), donor_idx].mean())


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
    init_method: str = 'random',
    marginals: Optional[Dict[str, Any]] = None,
    log_every: int = 0,
    distance_mode: str = 'geometric',
    p: float = 1.0,
    lambda_param: float = 0.5,
    alpha_min: float = 2.0,
    alpha_max: float = 10.0,
    delta: float = 0.05,
    winsorize_quantiles: tuple = (0.01, 0.99),
    exclude_self: bool = True,
    max_retries: int = 0,
    retry_rho_decay: float = 0.5,
    maxent_max_states: int = 1_000_000,
    maxent_max_sweeps: int = 200,
    maxent_tol: float = 1e-8,
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
    init_method : str, default 'random'
        初始化方法：
        - 'random'（默认）：纯随机初始化（每格从 schema 合法域均匀抽样）
        - 'marginal'：按 1-way 边缘确定性初始化（需同时提供 marginals 参数）
        - 'pairwise_maxent'：从完整二阶等值查询拟合最大熵分布后抽样；当前仅支持
          全类别、状态空间可枚举的数据集，不读取原始表
    marginals : Dict or None, default None
        1-way 边缘测量（marginals.load_marginals 的返回值）。
        仅当 init_method='marginal' 时生效；为 None 时忽略。
    log_every : int, default 0
        逐轮进度打印频率：
        - 0（默认）：每轮都打印（向后兼容旧行为）
        - >0：每 log_every 轮打印一次（首轮与末轮总会打印），长实验更清爽
    distance_mode : str, default 'linear'
        距离项的处理方式：
        - 'linear'（默认，推荐）：exp(-d/h)，拉普拉斯核。实验表明在大数据上
          比 squared 优 ~20%（nltcs 1500轮，p<0.00001）。
        - 'squared'：exp(-d²/2h²)，高斯核（原实现，保留用于对比实验）
        - 'none'：不考虑距离，只用适应度驱动（实验中表现最差，不推荐）
        - 'geometric'：归一化 + 几何均值 + 动态锐度调度，跨数据集参数稳定。
          nltcs 1500轮 3种子实验中精度最优（loss 比 linear 低 ~93%，p=0.0013）。
          用 lambda_param/alpha_min/alpha_max/delta/winsorize_quantiles 调参。
    lambda_param : float, default 0.5
        （geometric 模式）适应度-距离权衡：λ↑ 更偏适应度，λ↓ 更偏距离。
    alpha_min, alpha_max : float, default 2.0, 10.0
        （geometric 模式）动态锐度调度的起止值，α_t 从 α_min 线性升到 α_max
        （早期平缓探索、后期锐利收敛）。推荐 2→10（实验最优配置）。
    exclude_self : bool, default True
        是否禁止记录抽到自己（对角线屏蔽）。主循环候选池=全表（全对全），
        抽到自己=该行本轮不变、对演化零贡献。默认 True 屏蔽之；传 False
        可复现屏蔽前的旧行为（做对照实验用）。将来若改用共享参考池（K≠N），
        距离非方阵会使 exclude_self=True 报错——届时需改为 False（池里无自身）。
    max_retries : int, default 0
        整代提案被拒后的最大重试次数。0 保持原行为；大于 0 时复用
        当轮 donor，逐次缩小 rho 重新生成提案，避免已经计算的距离和抽样浪费。
    retry_rho_decay : float, default 0.5
        每次重试的参与率缩放因子，必须在 (0, 1) 内。第 a 次尝试使用
        ``rho * retry_rho_decay ** a``（a 从 0 开始）。
    maxent_max_states : int, default 1_000_000
        pairwise_maxent 可枚举的最大联合状态数，防止意外耗尽内存。
    maxent_max_sweeps : int, default 200
        pairwise_maxent 的 IPF 最大扫描轮数。
    maxent_tol : float, default 1e-8
        pairwise_maxent 的最大二阶单元概率误差收敛阈值。

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
        - proposal_attempts_history: List[int]，每轮评估的提案数
        - accepted_attempt_history: List[int]，接受的尝试序号（0=全拒绝）
        - state_evaluation_count: int，实际执行当前表查询/适应度评价的次数
        - distance_evaluation_count: int，实际构造全对全距离矩阵的次数

    Raises
    ------
    ValueError
        target 长度与 queries 数量不一致

    Notes
    -----
    **终止条件：** 残差全 0（达标）或达到 n_rounds。

    **整代检查失败：** max_retries=0 时保持原表；否则缩小 rho
    重试。best_S 保底，即使某轮无进展，最终仍返回历史最优表。

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

    if init_method not in ('random', 'marginal', 'pairwise_maxent'):
        raise ValueError(
            "init_method 必须是 'random'、'marginal' 或 "
            f"'pairwise_maxent'，得到 {init_method!r}"
        )

    if isinstance(max_retries, bool) or not isinstance(max_retries, (int, np.integer)):
        raise ValueError(f"max_retries 必须是非负整数，得到 {max_retries!r}")
    if max_retries < 0:
        raise ValueError(f"max_retries 必须是非负整数，得到 {max_retries}")
    if not (0.0 < retry_rho_decay < 1.0):
        raise ValueError(
            f"retry_rho_decay 必须在 (0, 1) 内，得到 {retry_rho_decay}"
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

    # 初始表 S_0（不读源数据，只用 schema、已测量 target 与可选 1-way 边缘）
    if init_method == 'pairwise_maxent':
        S, initialization_diagnostics = init_from_pairwise_maxent(
            n_records=n_records,
            schema=schema,
            queries=queries,
            target=target,
            rng=rng,
            max_states=maxent_max_states,
            max_sweeps=maxent_max_sweeps,
            tol=maxent_tol,
        )
    elif init_method == 'marginal':
        S = init_synthetic_table(n_records, schema, rng, marginals=marginals)
        initialization_diagnostics = {"method": "marginal"}
    else:
        S = init_synthetic_table(n_records, schema, rng)
        initialization_diagnostics = {"method": "random"}

    loss_history: List[float] = []
    accept_history: List[bool] = []
    donor_fitness_history: List[float] = []      # 每轮选中 donor 的平均适应度
    donor_distance_history: List[float] = []     # 每轮到 donor 的平均距离
    donor_self_rate_history: List[float] = []    # 每轮抽到自己的比例（donor_idx==i）
    alpha_history: List[float] = []              # 每轮的锐度 α_t（geometric 模式）
    proposal_attempts_history: List[int] = []    # 每轮实际评估的提案数（含首次）
    accepted_attempt_history: List[int] = []     # 接受的尝试序号（1-based）；0=全部拒绝
    accepted_rho_history: List[Optional[float]] = []  # 接受时使用的 rho；全拒绝为 None
    stopped_early = False
    rounds_run = 0
    state_evaluation_count = 0
    distance_evaluation_count = 0

    # 当前表 S 没变化时，答案/残差/适应度/loss/距离也完全不变。整代提案被拒后
    # 保留这些量，下一轮只按新的 alpha 重算抽样概率并重新抽 donor。提案被接受
    # 后统一失效，确保缓存永远和 S 对齐。
    state_eval_cache = None
    distance_cache = None

    # 计时：主循环墙钟时间（不含 init/最终指标），用于扫描时估时与"快且好"对比。
    # 用 perf_counter（单调、不受系统时钟调整影响）。
    loop_start = time.perf_counter()

    # 直接完成第一轮需要的完整评价，同时据此初始化 best，避免先做一次 counts-only
    # 又在第一轮重复扫描同一张 S_0。
    initial_q, initial_residual, initial_fitness = _eval_counts_resid_fitness(S)
    initial_loss = compute_loss(target, initial_q)
    state_eval_cache = (
        initial_q, initial_residual, initial_fitness, initial_loss
    )
    state_evaluation_count += 1
    best_S = S.copy()
    best_loss = initial_loss

    for t in range(n_rounds):
        rounds_run = t + 1

        # 计算当前轮的动态锐度 α_t（geometric 模式用）
        if n_rounds > 1:
            progress = t / (n_rounds - 1)
        else:
            progress = 1.0
        alpha_t = alpha_min + (alpha_max - alpha_min) * progress
        alpha_history.append(alpha_t)

        # 1-2-4. 当前答案、残差、适应度。只有接受提案、S 真正更新后才重算；
        # 拒绝后的下一轮复用上一轮结果。
        if state_eval_cache is None:
            q, residual, fitness = _eval_counts_resid_fitness(S)
            loss = compute_loss(target, q)
            state_eval_cache = (q, residual, fitness, loss)
            state_evaluation_count += 1
        else:
            q, residual, fitness, loss = state_eval_cache
        loss_history.append(loss)

        # 是否在本轮打印进度：log_every=0 每轮打印；否则每 log_every 轮，
        # 且首轮和末轮总打印（长实验也能看到起点和终点）。
        do_log = (
            log_every <= 0
            or t == 0
            or t == n_rounds - 1
            or (t + 1) % log_every == 0
        )

        # 3. 终止检查：残差全 0（达标）→ 抽样前停止
        if np.all(residual == 0):
            if do_log:
                print(f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e} | 达标提前停止")
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
        if distance_cache is None:
            distance_cache = pairwise_block_distance(
                S, S, schema, device=device, return_tensor=use_torch
            )
            distance_evaluation_count += 1
        distances = distance_cache
        probs = compute_sampling_probs(
            fitness, distances, beta=beta, h=h, device=device,
            distance_mode=distance_mode, p=p,
            lambda_param=lambda_param, alpha=alpha_t, delta=delta,
            winsorize_quantiles=winsorize_quantiles,
            # 候选池=全表（全对全），排除对角线=禁止记录抽到自己。
            # 抽到自己 = 该行本轮不变、对演化零贡献；小表高锐度下自身率可达 8%
            # （见 scripts/diagnose_self_sampling.py），屏蔽后消除该浪费。
            # 默认 True；实验脚本可传 False 复现屏蔽前的 baseline 做对照。
            exclude_self=exclude_self,
        )
        donor_idx = sample_donors(probs, rng, device=device)
        # donor 索引得到后不再需要 N×N 概率矩阵，尽早释放设备内存。
        del probs
        donors = S.iloc[donor_idx].reset_index(drop=True)

        # 诊断：记录选中 donor 的适应度和距离
        N = len(S)  # 记录数
        selected_fitness = fitness[donor_idx]  # (N,) 每条记录选中的 donor 适应度
        # GPU/torch 路径只 gather 被选中的 N 个元素并回传一个均值标量，不再为了
        # 诊断把完整 N×N 距离矩阵搬回 CPU。
        selected_distance_mean = _mean_selected_distance(
            distances, donor_idx, use_torch=use_torch
        )

        # 记录平均值
        donor_fitness_history.append(float(selected_fitness.mean()))
        donor_distance_history.append(selected_distance_mean)
        # 自身抽样率：抽到自己（donor_idx==i）的比例。exclude_self=True 时恒为 0；
        # 全对全候选池下这是"自我复制空转"的直接度量（见 scripts/diagnose_self_sampling.py）。
        donor_self_rate_history.append(float(np.mean(donor_idx == np.arange(N))))

        # 7-8. 靠近一步 → 整代安全检查。首次失败时可复用当轮
        # donor 并缩小 rho 重试；距离/适应度/抽样都不重算，额外成本只是
        # evolve_step + 一次提案查询评价。max_retries=0 时与原逻辑完全一致。
        accepted = False
        accepted_attempt = 0
        accepted_rho = None
        proposal_attempts = 0
        current_loss = loss
        for attempt in range(max_retries + 1):
            attempt_rho = rho * (retry_rho_decay ** attempt)
            proposal = evolve_step(
                S, donors, schema, rho=attempt_rho, eta=eta, mu=mu, rng=rng
            )
            proposal_q = _eval_counts(proposal)
            proposal_loss = compute_loss(target, proposal_q)
            proposal_attempts += 1

            if proposal_loss <= loss + tol:
                accepted = True
                accepted_attempt = attempt + 1
                accepted_rho = attempt_rho
                current_loss = proposal_loss
                S = proposal
                break

        accept_history.append(accepted)
        proposal_attempts_history.append(proposal_attempts)
        accepted_attempt_history.append(accepted_attempt)
        accepted_rho_history.append(accepted_rho)

        if accepted:
            # S 已替换为 proposal，旧表对应的所有缓存立即失效。显式删除本地距离
            # 引用，避免下一轮计算新矩阵时旧矩阵仍占一份设备内存。
            state_eval_cache = None
            distance_cache = None
            del distances

        # 逐轮进度：单行输出 loss + 接受状态（受 log_every 控制）
        if do_log:
            print(f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e}"
                  f" | 接受: {'是' if accepted else '否'}"
                  f" | 尝试: {proposal_attempts}")

        # 9. 更新历史最优（直接用已知 loss，不重复评价）
        if current_loss < best_loss:
            best_loss = current_loss
            best_S = S.copy()

    elapsed_sec = time.perf_counter() - loop_start
    sec_per_round = elapsed_sec / rounds_run if rounds_run else 0.0

    # 计算最终质量指标：平均归一化 L1 误差（仅用于报告，不影响训练）
    # 对齐 AIM 论文的 workload error（方案 A：每条合取查询作为一个单元，权重取 1）：
    #     Error = (1 / (k·|D|)) · Σ_i |target_i − pred_i|
    #           = mean(|target − pred|) / |D|
    # 分母是记录数 |D|（而非逐查询除以自身的 target），因此不会被小 target 查询
    # 的极端相对误差拉高，可跨数据规模比较。
    best_q = evaluate_table(best_S, queries)
    abs_errors = np.abs(target - best_q)
    normalized_l1_error = float(np.mean(abs_errors) / n_records)
    # 分布统计：逐查询归一化误差 |target−pred|/N 的中位/P90/最大。
    # 均值易被少数难查询拉高，分布能看清"典型查询"和"最差查询"的差距。
    per_query_nl1 = abs_errors / n_records
    normalized_l1_median = float(np.median(per_query_nl1))
    normalized_l1_p90 = float(np.percentile(per_query_nl1, 90))
    normalized_l1_max = float(np.max(per_query_nl1))

    diagnostics = {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "rounds_run": rounds_run,
        "stopped_early": stopped_early,
        "accept_history": accept_history,
        "donor_fitness_history": donor_fitness_history,
        "donor_distance_history": donor_distance_history,
        "donor_self_rate_history": donor_self_rate_history,
        "alpha_history": alpha_history,
        "proposal_attempts_history": proposal_attempts_history,
        "accepted_attempt_history": accepted_attempt_history,
        "accepted_rho_history": accepted_rho_history,
        "state_evaluation_count": state_evaluation_count,
        "distance_evaluation_count": distance_evaluation_count,
        "initialization": initialization_diagnostics,
        "normalized_l1_error": normalized_l1_error,
        "normalized_l1_median": normalized_l1_median,
        "normalized_l1_p90": normalized_l1_p90,
        "normalized_l1_max": normalized_l1_max,
        "elapsed_sec": elapsed_sec,
        "sec_per_round": sec_per_round,
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
            "init_method": init_method,
            "distance_mode": distance_mode,
            "p": p if distance_mode == 'multiplicative' else None,
            "lambda": lambda_param if distance_mode == 'geometric' else None,
            "alpha_min": alpha_min if distance_mode == 'geometric' else None,
            "alpha_max": alpha_max if distance_mode == 'geometric' else None,
            "delta": delta if distance_mode == 'geometric' else None,
            "winsorize_quantiles": winsorize_quantiles if distance_mode == 'geometric' else None,
            "exclude_self": exclude_self,
            "max_retries": int(max_retries),
            "retry_rho_decay": retry_rho_decay,
            "maxent_max_states": int(maxent_max_states),
            "maxent_max_sweeps": int(maxent_max_sweeps),
            "maxent_tol": maxent_tol,
        },
    }

    return best_S.reset_index(drop=True), diagnostics
