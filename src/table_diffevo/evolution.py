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
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from table_diffevo.schema import Schema
from table_diffevo.queries import evaluate_table
from table_diffevo.objective import compute_residual, compute_loss
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.fitness import compute_fitness
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.update import evolve_step
from table_diffevo.directional_diffusion import (
    additive_copy_drift_diagnostics,
    bernoulli_entropy,
    bernoulli_kl,
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.pairwise_init import init_from_pairwise_maxent
from table_diffevo.vectorized_eval import evaluate_vectorized
from table_diffevo.factorized_diffusion import (
    DEFAULT_LOGIT_CLIP,
    evolve_step_factorized_gibbs,
)


def _self_cooling_factor(
    residual_l1: float,
    initial_residual_l1: float,
    exponent: float,
) -> Tuple[float, float]:
    """返回 (残差比 r, 冷却因子 r**exponent)。

    r = min(1, residual_l1 / initial_residual_l1)；初始残差为 0 时定义 r = 0
    （初始即达标，动力学应完全冻结）。两个返回值都落在 [0, 1]。
    """
    if initial_residual_l1 > 0.0:
        ratio = min(1.0, residual_l1 / initial_residual_l1)
    else:
        ratio = 0.0
    return ratio, ratio ** exponent


def _rng_state_sha256(rng: np.random.Generator) -> str:
    """返回 RNG 状态的稳定摘要，只用于等价性诊断。"""
    serialized = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _table_sha256(frame: pd.DataFrame) -> str:
    """返回合成表 CSV 表示的摘要，只用于复现诊断。"""
    serialized = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _current_state_metrics(
    target: np.ndarray,
    current: np.ndarray,
    n_records: int,
    squared_loss: float,
    *,
    state_index: int,
    round_index: int,
    phase: str,
) -> Dict[str, Any]:
    """构造一个不消费 RNG/查询评价的 current-state 观测点。"""
    return {
        "state_index": int(state_index),
        "round": int(round_index),
        "phase": phase,
        "current_normalized_l1": compute_normalized_l1(
            target, current, n_records
        ),
        "current_squared_loss": float(squared_loss),
    }


def _factorized_gibbs_seed(seed: int) -> int:
    """从公开主 seed 派生与主随机流独立的 Gibbs seed。"""
    sequence = np.random.SeedSequence([int(seed), 0x4749424253])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


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
    residual_directed_diffusion: bool = False,
    diffusion_direction_strength: float = 1.0,
    diffusion_direction_normalization: str = "initial_rms",
    factorized_gibbs_sweeps: int = 0,
    factorized_gibbs_max_order: int = 3,
    factorized_gibbs_logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP,
    candidate_budget: Optional[int] = None,
    residual_self_cooling: Optional[float] = None,
    self_cooling_monotone: bool = False,
    self_cooling_stop_ratio: Optional[float] = None,
    rho_anneal_end: Optional[float] = None,
    rho_anneal_rounds: Optional[int] = None,
    selection_scale_invariant: bool = False,
    selection_scale_invariant_min_spread: float = 1e-3,
    return_final_table: bool = False,
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
    residual_directed_diffusion : bool, default False
        是否让实际单块 donor 复制的比例残差方向量连续倾斜复制概率。默认关闭，
        保持历史算法与随机轨迹。该机制不执行正负门控或逐候选 top-k。
    diffusion_direction_strength : float, default 1.0
        残差驱动扩散的非负有限强度。0 在启用机制时也精确退化到历史固定 eta；
        正值越大，复制概率对实际局部方向越敏感。
    diffusion_direction_normalization : str, default 'initial_rms'
        方向强度的尺度口径：
        - 'none'：直接使用原始比例残差方向量；
        - 'initial_rms'：用本次运行首个非零方向矩阵的 RMS 固定定标。此时
          diffusion_direction_strength 是无量纲初始温度，后续残差变小时倾斜自然
          冷却，不逐轮重新标准化。
    factorized_gibbs_sweeps : int, default 0
        每条参与记录在独立定向初始 mask 后执行的随机扫描 Gibbs sweep 数。0 完全
        保留既有独立单块更新；正数只允许与 residual_directed_diffusion 一起启用。
        附加 Gibbs 使用从公开 seed 派生的独立随机流，不消费主随机流。
    factorized_gibbs_max_order : int, default 3
        查询局部因子的最高允许属性阶数。仅在 factorized_gibbs_sweeps > 0 时使用，
        超出时明确报错，不静默截断交互。
    factorized_gibbs_logit_clip : float or None, default 30
        Gibbs 条件 logit 的对称数值护栏。正有限数值保留极端有限温度下的双向
        float64 支持；显式传入 None 可关闭。该参数不改变 sweep=0 路径。
    candidate_budget : int or None, default None
        可选的全局候选评估次数上限。若指定，演化会在达到此预算时提前停止，
        与 n_rounds 并存（先达到者停止）。

        候选评估 = 生成候选表 → 评估所有查询 → 计算误差（算1次）。
        初始表评估不计入；只统计主循环中的候选提案评估（包括重试）。

        用途：为长时间探索实验设置计算成本上限，确保可比性。

    residual_self_cooling : float or None, default None
        残差自冷却指数（Issue #44 研究机制，默认关闭）。设为正数 p 时，每轮
        计算残差比 ``r_t = min(1, ||target - q_t||_1 / ||target - q_0||_1)``，
        并把参与率 ``rho`` 与变异率 ``mu`` 乘以冷却因子 ``c_t = r_t ** p``。
        扰动幅度随收敛进度自动衰减：残差趋零时动力学自然冻结（零残差为吸收
        态），无需接受门的棘轮作用。``p=1`` 为线性冷却，``p=2`` 更陡，
        ``p=0.5`` 更缓。None 时完全关闭，主循环行为与历史逐轨迹一致。

        该机制只作用于扰动幅度（分布侧），不引入任何接受/拒绝判定；与
        ``tol=inf``（关闭整代接受门）组合即为无门控扩散演化研究配置。
    self_cooling_monotone : bool, default False
        机制消融选项。False（默认）：冷却跟随当前残差比，状态回漂时温度回升
        （复燃）并快速重新收敛；True：使用历史最低残差比，温度只降不升
        （分布侧棘轮）。dev 定标（test_300x10、seed 42..44、2000 轮）显示
        单调冷却反而更差（final 96.8 vs 89.2）：温度锁死后偶发劣化步的恢复
        极慢。保留该开关仅作机制消融对照。两种模式都不读取候选评价。
    self_cooling_stop_ratio : float or None, default None
        内在停止阈值（需同时启用 residual_self_cooling）。当残差比
        ``r_t <= self_cooling_stop_ratio`` 时提前停止，作为达标早停之外的
        内在收敛停机信号；取值须在 (0, 1)。None 时不启用。
    rho_anneal_end : float or None, default None
        时间驱动的几何 rho 退火终点（Issue #44 机制迭代，默认关闭）。设为
        (0, rho] 内的值时，第 t 轮（t 从 0 起）的参与率为
        ``rho_t = rho * (rho_anneal_end / rho) ** (t / (n_rounds - 1))``，
        即从 ``rho`` 几何插值到 ``rho_anneal_end``——扩散模型意义上的盲
        噪声时间表（noise schedule）：调度只依赖轮次进度，不读取残差或任何
        候选评价，因而不存在残差反馈的过早冻结死锁。``rho_anneal_end ==
        rho`` 时每轮值恒为 rho（浮点上与关闭一致）。与
        ``residual_self_cooling`` 可组合：冷却因子乘在退火后的 rho_t 上。
        None 时完全关闭，rho 恒定，行为与历史逐轨迹一致。
    rho_anneal_rounds : int or None, default None
        两段式调度的快降段轮数 K（需同时启用 rho_anneal_end）。指定时退火
        进度按 ``min(1, t / K)`` 计算：前 K 轮从 ``rho`` 几何降温到
        ``rho_anneal_end``，之后恒定在 ``rho_anneal_end`` 深潜。None 时
        退火进度铺满全程 ``n_rounds``。动机：无门恒定动力学的噪声地板随
        rho 近似线性抬升，而到达高温地板只需少量轮数——快降段之后把预算
        留给低温深潜。仍是纯时间驱动的盲调度，不读取残差或候选评价。
    selection_scale_invariant : bool, default False
        尺度不变选择（仅 distance_mode='geometric'；Issue #44 机制迭代）。
        True 时 donor 选择 logits 先做行内标准化再乘 alpha：选择压力的
        有效温度恒等于 alpha，与联合分数行内离散度的绝对尺度解耦，消除
        种群同质化导致的晚期选择退化（否则需要靠调大 alpha 补偿）。纯
        分布侧机制：不读取候选评价、不引入接受/拒绝。False 保持历史行为。
        开启且 exclude_self=True 时行统计只在非自身候选上计算（第三轮
        审查修正）；启用时诊断新增 ``donor_top_share_history``（每轮被选
        最多的 donor 占比，选择集中度监控）。
    selection_scale_invariant_min_spread : float, default 1e-3
        尺度不变选择的低信号保护下限（需 selection_scale_invariant=True）。
        行内标准差低于该值时按该值截断：放大倍数有界
        （alpha/min_spread），离散度趋零时选择平滑退化为均匀，避免把
        噪声级微小差异放大成极端选择偏好。必须为正有限数。
    return_final_table : bool, default False
        为 True 时在诊断中附加 ``final_table``（最后一轮结束时的当前表深
        拷贝）。无门控研究的主输出是最终状态而非 best 追踪表；该字段是
        DataFrame，不可直接 JSON 序列化，调用方保存诊断前必须自行弹出。
        默认 False 保持诊断字典可序列化，行为与历史一致。

    Returns
    -------
    best_S : pd.DataFrame, shape (n_records, n_attributes)
        演化过程中见过的 loss 最小的合成表
    diagnostics : dict
        诊断信息：
        - loss_history: List[float]，每轮开始时当前表的 loss
        - best_loss: float，最优 loss
        - current_state_metrics_history: List[dict]，初始 current state 与
          每个实际轮后 current state 的 normalized L1/平方 loss。每项
          显式记录 state_index、round 和 phase；在 proposal 前提前停止
          时不伪造重复状态
        - final_current_normalized_l1/final_current_squared_loss:
          最终 current table 的两个权威终态指标
        - rounds_run: int，实际跑的轮数
        - stopped_early: bool，是否因残差全 0 提前停止
        - accept_history: List[bool]，每轮整代检查是否接受提案
        - proposal_attempts_history: List[int]，每轮评估的提案数
        - accepted_attempt_history: List[int]，接受的尝试序号（0=全拒绝）
        - raw_proposal_gain_history: List[List[float]]，每轮各次接受检查前的
          原始 proposal 精确收益
        - copy_direction_*_history: 局部方向分布与正/反向实际复制概率
        - copy_probability_entropy_history: 每轮 active Bernoulli 复制核的平均熵
        - copy_probability_kl_history: 每轮复制核相对历史 Bernoulli(eta) 的平均 KL
        - additive_copy_drift_*_history: 独立单块一阶近似相对符号极限的改善与
          利用率；不等于原始 proposal 的精确收益
        - state_evaluation_count: int，实际执行当前表查询/适应度评价的次数
        - candidate_evaluation_count: int，实际执行候选提案评价的次数（不含初始表）
        - distance_evaluation_count: int，实际构造全对全距离矩阵的次数
        - direction_evaluation_count: int，实际计算局部方向矩阵的次数
        - candidate_budget_exhausted: bool，是否因达到 candidate_budget 提前停止
        - self_cooling_history: List[float]，每轮的冷却因子 c_t（关闭时恒 1.0；
          与 loss_history 逐轮对齐）
        - self_cooling_stopped: bool，是否因残差比达到 self_cooling_stop_ratio
          提前停止
        - factorized_gibbs_attempt_diagnostics_history: 每轮每次尝试的因子构造、
          Gibbs 微步和墙钟诊断；不参与接受或早停
        - primary_rng_state_sha256/factorized_gibbs_rng_state_sha256:
          主随机流与附加随机流最终状态摘要

    Raises
    ------
    ValueError
        target 长度与 queries 数量不一致

    Notes
    -----
    **终止条件：** 残差全 0（达标）、达到 n_rounds、达到 candidate_budget
    （若指定）、或残差比达到 self_cooling_stop_ratio（若指定）。

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
    if not isinstance(residual_directed_diffusion, (bool, np.bool_)):
        raise ValueError(
            "residual_directed_diffusion 必须是布尔值，"
            f"得到 {residual_directed_diffusion!r}"
        )
    if candidate_budget is not None:
        if isinstance(candidate_budget, bool) or not isinstance(candidate_budget, (int, np.integer)):
            raise ValueError(f"candidate_budget 必须是正整数或 None，得到 {candidate_budget!r}")
        if candidate_budget <= 0:
            raise ValueError(f"candidate_budget 必须 > 0，得到 {candidate_budget}")
    if residual_self_cooling is not None:
        if (
            isinstance(residual_self_cooling, (bool, np.bool_))
            or not isinstance(
                residual_self_cooling,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(residual_self_cooling)
            or residual_self_cooling <= 0.0
        ):
            raise ValueError(
                "residual_self_cooling 必须是正有限数值或 None，"
                f"得到 {residual_self_cooling!r}"
            )
        residual_self_cooling = float(residual_self_cooling)
    if residual_self_cooling is not None and not isinstance(
        self_cooling_monotone, (bool, np.bool_)
    ):
        raise ValueError(
            f"self_cooling_monotone 必须是布尔值，得到 {self_cooling_monotone!r}"
        )
    if self_cooling_stop_ratio is not None:
        if residual_self_cooling is None:
            raise ValueError(
                "self_cooling_stop_ratio 需要同时启用 residual_self_cooling"
            )
        if (
            isinstance(self_cooling_stop_ratio, (bool, np.bool_))
            or not isinstance(
                self_cooling_stop_ratio,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self_cooling_stop_ratio)
            or not 0.0 < self_cooling_stop_ratio < 1.0
        ):
            raise ValueError(
                "self_cooling_stop_ratio 必须位于 (0, 1) 或为 None，"
                f"得到 {self_cooling_stop_ratio!r}"
            )
        self_cooling_stop_ratio = float(self_cooling_stop_ratio)
    if rho_anneal_end is not None:
        if (
            isinstance(rho_anneal_end, (bool, np.bool_))
            or not isinstance(
                rho_anneal_end,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(rho_anneal_end)
            or not 0.0 < rho_anneal_end <= rho
        ):
            raise ValueError(
                "rho_anneal_end 必须位于 (0, rho] 或为 None，"
                f"得到 {rho_anneal_end!r}（rho={rho}）"
            )
        rho_anneal_end = float(rho_anneal_end)
    if rho_anneal_rounds is not None:
        if rho_anneal_end is None:
            raise ValueError(
                "rho_anneal_rounds 需要同时启用 rho_anneal_end"
            )
        if (
            isinstance(rho_anneal_rounds, (bool, np.bool_))
            or not isinstance(rho_anneal_rounds, (int, np.integer))
            or rho_anneal_rounds < 1
        ):
            raise ValueError(
                "rho_anneal_rounds 必须是正整数或 None，"
                f"得到 {rho_anneal_rounds!r}"
            )
        rho_anneal_rounds = int(rho_anneal_rounds)
    if not isinstance(selection_scale_invariant, (bool, np.bool_)):
        raise ValueError(
            "selection_scale_invariant 必须是布尔值，"
            f"得到 {selection_scale_invariant!r}"
        )
    selection_scale_invariant = bool(selection_scale_invariant)
    if selection_scale_invariant and distance_mode != "geometric":
        raise ValueError(
            "selection_scale_invariant 仅支持 distance_mode='geometric'，"
            f"得到 {distance_mode!r}"
        )
    if (
        isinstance(selection_scale_invariant_min_spread, (bool, np.bool_))
        or not isinstance(
            selection_scale_invariant_min_spread,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(selection_scale_invariant_min_spread)
        or selection_scale_invariant_min_spread <= 0
    ):
        raise ValueError(
            "selection_scale_invariant_min_spread 必须是正有限数，"
            f"得到 {selection_scale_invariant_min_spread!r}"
        )
    selection_scale_invariant_min_spread = float(
        selection_scale_invariant_min_spread
    )
    if (
        isinstance(diffusion_direction_strength, (bool, np.bool_))
        or not isinstance(
            diffusion_direction_strength,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(diffusion_direction_strength)
        or diffusion_direction_strength < 0.0
    ):
        raise ValueError(
            "diffusion_direction_strength 必须是非负有限数值，"
            f"得到 {diffusion_direction_strength!r}"
        )
    diffusion_direction_strength = float(diffusion_direction_strength)
    if diffusion_direction_normalization not in ("none", "initial_rms"):
        raise ValueError(
            "diffusion_direction_normalization 必须是 'none' 或 "
            f"'initial_rms'，得到 {diffusion_direction_normalization!r}"
        )
    for value, name in (
        (factorized_gibbs_sweeps, "factorized_gibbs_sweeps"),
        (factorized_gibbs_max_order, "factorized_gibbs_max_order"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{name} 必须是非负整数，得到 {value!r}")
    factorized_gibbs_sweeps = int(factorized_gibbs_sweeps)
    factorized_gibbs_max_order = int(factorized_gibbs_max_order)
    if factorized_gibbs_max_order > 8:
        raise ValueError("factorized_gibbs_max_order 不得超过绝对护栏 8")
    if factorized_gibbs_logit_clip is not None:
        if (
            isinstance(factorized_gibbs_logit_clip, (bool, np.bool_))
            or not isinstance(
                factorized_gibbs_logit_clip,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(factorized_gibbs_logit_clip)
            or factorized_gibbs_logit_clip <= 0.0
        ):
            raise ValueError(
                "factorized_gibbs_logit_clip 必须是正有限数值或 None，"
                f"得到 {factorized_gibbs_logit_clip!r}"
            )
        factorized_gibbs_logit_clip = float(
            factorized_gibbs_logit_clip
        )
    if factorized_gibbs_sweeps > 0:
        if not residual_directed_diffusion:
            raise ValueError(
                "factorized_gibbs_sweeps > 0 要求启用 "
                "residual_directed_diffusion"
            )
        if factorized_gibbs_max_order == 0:
            raise ValueError(
                "factorized_gibbs_sweeps > 0 时 "
                "factorized_gibbs_max_order 必须至少为 1"
            )
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        ):
            raise ValueError(
                "factorized Gibbs 要求 seed 是整数，"
                f"得到 {seed!r}"
            )
        if (
            isinstance(eta, (bool, np.bool_))
            or not isinstance(eta, (int, float, np.integer, np.floating))
            or not np.isfinite(eta)
            or not 0.0 < eta < 1.0
        ):
            raise ValueError(
                "factorized Gibbs 要求 eta 是 (0, 1) 内的有限数值，"
                f"得到 {eta!r}"
            )

    rng = np.random.default_rng(seed)
    factorized_gibbs_rng = (
        np.random.default_rng(_factorized_gibbs_seed(seed))
        if factorized_gibbs_sweeps > 0 else None
    )
    factorized_gibbs_initial_rng_state_sha256 = (
        _rng_state_sha256(factorized_gibbs_rng)
        if factorized_gibbs_rng is not None else None
    )

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
    initial_table_sha256 = (
        _table_sha256(S) if residual_directed_diffusion else None
    )
    primary_rng_post_initialization_state_sha256 = _rng_state_sha256(rng)

    loss_history: List[float] = []
    accept_history: List[bool] = []
    donor_fitness_history: List[float] = []      # 每轮选中 donor 的平均适应度
    donor_distance_history: List[float] = []     # 每轮到 donor 的平均距离
    donor_self_rate_history: List[float] = []    # 每轮抽到自己的比例（donor_idx==i）
    alpha_history: List[float] = []              # 每轮的锐度 α_t（geometric 模式）
    proposal_attempts_history: List[int] = []    # 每轮实际评估的提案数（含首次）
    accepted_attempt_history: List[int] = []     # 接受的尝试序号（1-based）；0=全部拒绝
    accepted_rho_history: List[Optional[float]] = []  # 接受时使用的 rho；全拒绝为 None
    rho_schedule_history: List[float] = []  # 每轮退火后的 rho_t（关闭时恒为 rho）
    donor_top_share_history: List[float] = []  # 尺度不变选择时的集中度监控
    row_max_prob_mean_history: List[float] = []  # 逐行最大概率均值（每轮）
    row_max_prob_max_history: List[float] = []  # 逐行最大概率最大值（每轮）
    effective_donors_mean_history: List[float] = []  # exp(行熵)均值（每轮）
    copy_direction_mean_history: List[Optional[float]] = []
    copy_direction_positive_rate_history: List[Optional[float]] = []
    copy_direction_negative_rate_history: List[Optional[float]] = []
    negative_direction_copy_probability_history: List[Optional[float]] = []
    positive_direction_copy_probability_history: List[Optional[float]] = []
    copy_probability_entropy_history: List[Optional[float]] = []
    copy_probability_kl_history: List[Optional[float]] = []
    additive_copy_drift_improvement_history: List[Optional[float]] = []
    available_additive_copy_drift_improvement_history: List[
        Optional[float]
    ] = []
    additive_copy_drift_utilization_history: List[Optional[float]] = []
    effective_direction_strength_history: List[Optional[float]] = []
    direction_reference_scale_history: List[Optional[float]] = []
    raw_proposal_gain_history: List[List[float]] = []
    raw_proposal_linear_gain_history: List[List[float]] = []
    raw_proposal_quadratic_penalty_history: List[List[float]] = []
    factorized_gibbs_attempt_diagnostics_history: List[
        List[Dict[str, Any]]
    ] = []
    stopped_early = False
    candidate_budget_exhausted = False
    rounds_run = 0
    state_evaluation_count = 0
    candidate_evaluation_count = 0
    distance_evaluation_count = 0
    direction_evaluation_count = 0
    direction_evaluation_elapsed_sec = 0.0
    direction_reference_scale: Optional[float] = None
    factorized_gibbs_factor_build_elapsed_sec = 0.0
    factorized_gibbs_sample_elapsed_sec = 0.0
    factorized_gibbs_active_rows = 0
    factorized_gibbs_active_blocks = 0
    factorized_gibbs_factor_count = 0
    factorized_gibbs_factor_table_entries = 0
    factorized_gibbs_microsteps = 0

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
    self_cooling_initial_l1 = float(np.abs(target - initial_q).sum())
    self_cooling_history: List[float] = []
    self_cooling_stopped = False
    self_cooling_factor = 1.0
    self_cooling_min_ratio = 1.0
    state_eval_cache = (
        initial_q, initial_residual, initial_fitness, initial_loss
    )
    state_evaluation_count += 1
    best_S = S.copy()
    best_loss = initial_loss
    current_state_metrics_history: List[Dict[str, Any]] = [
        _current_state_metrics(
            target,
            initial_q,
            n_records,
            initial_loss,
            state_index=0,
            round_index=0,
            phase="initial",
        )
    ]

    for t in range(n_rounds):
        rounds_run = t + 1

        # 计算当前轮的动态锐度 α_t（geometric 模式用）
        if n_rounds > 1:
            progress = t / (n_rounds - 1)
        else:
            progress = 1.0
        alpha_t = alpha_min + (alpha_max - alpha_min) * progress
        alpha_history.append(alpha_t)

        # 时间驱动几何 rho 退火（盲噪声时间表）：只依赖轮次进度，不读取残差
        # 或候选评价。关闭时 rho_t 恒等于 rho，逐轨迹等价于历史行为。
        # rho_anneal_rounds 指定时为两段式：前 K 轮快降，其后恒定深潜。
        if rho_anneal_end is not None:
            if rho_anneal_rounds is not None:
                anneal_progress = min(1.0, t / rho_anneal_rounds)
            else:
                anneal_progress = progress
            rho_t = rho * (rho_anneal_end / rho) ** anneal_progress
        else:
            rho_t = rho
        rho_schedule_history.append(rho_t)

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

        # 残差自冷却因子：与 loss_history 逐轮对齐记录；内在停止检查在达标
        # 检查之后执行。
        self_cooling_ratio = None
        if residual_self_cooling is not None:
            self_cooling_ratio, _ = _self_cooling_factor(
                float(np.abs(target - q).sum()),
                self_cooling_initial_l1,
                residual_self_cooling,
            )
            if self_cooling_monotone:
                self_cooling_min_ratio = min(
                    self_cooling_min_ratio, self_cooling_ratio
                )
                self_cooling_ratio = self_cooling_min_ratio
            self_cooling_factor = self_cooling_ratio ** residual_self_cooling
            self_cooling_history.append(self_cooling_factor)
        else:
            self_cooling_history.append(1.0)

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

        # 3b. 残差自冷却内在停止：残差比达到阈值即停机（分布侧停止条件）。
        if (
            self_cooling_stop_ratio is not None
            and self_cooling_ratio is not None
            and self_cooling_ratio <= self_cooling_stop_ratio
        ):
            if do_log:
                print(
                    f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e} | "
                    f"残差比 {self_cooling_ratio:.4f} 达到内在停止阈值"
                    f" {self_cooling_stop_ratio}，提前停止"
                )
            stopped_early = True
            self_cooling_stopped = True
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
            scale_invariant=selection_scale_invariant,
            scale_invariant_min_spread=selection_scale_invariant_min_spread,
        )
        donor_idx = sample_donors(probs, rng, device=device)
        # 逐行选择集中度诊断（第二轮审查意见 4）：全局 top share 不能
        # 判断单行 softmax 是否接近确定性——即使每行都以 99.9% 概率选
        # 各自不同的 donor，全局 top share 仍可能很低。补充逐行最大
        # 概率与概率熵（有效 donor 数 = exp(熵)），在设备上归约成标量
        # 后回传。只在尺度不变选择启用时记录。
        if selection_scale_invariant:
            if use_torch:
                import torch
                row_max = probs.max(dim=1).values
                safe = torch.clamp(probs, min=1e-30)
                row_entropy = -(probs * torch.log(safe)).sum(dim=1)
                row_max_prob_mean_history.append(float(row_max.mean()))
                row_max_prob_max_history.append(float(row_max.max()))
                effective_donors_mean_history.append(
                    float(torch.exp(row_entropy).mean())
                )
            else:
                row_max = probs.max(axis=1)
                safe = np.clip(probs, 1e-30, None)
                row_entropy = -(probs * np.log(safe)).sum(axis=1)
                row_max_prob_mean_history.append(float(row_max.mean()))
                row_max_prob_max_history.append(float(row_max.max()))
                effective_donors_mean_history.append(
                    float(np.exp(row_entropy).mean())
                )
        # donor 索引得到后不再需要 N×N 概率矩阵，尽早释放设备内存。
        del probs
        donors = S.iloc[donor_idx].reset_index(drop=True)
        # 全局集中度（第三轮审查）：被选最多的 donor 占比。
        if selection_scale_invariant:
            donor_top_share_history.append(
                float(np.bincount(donor_idx).max() / len(donor_idx))
            )

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

        if residual_directed_diffusion:
            direction_start = time.perf_counter()
            copy_direction_scores = compute_copy_direction_scores(
                S,
                donors,
                schema,
                queries,
                residual,
                batch_size=batch_size,
                device=device,
            )
            direction_evaluation_elapsed_sec += (
                time.perf_counter() - direction_start
            )
            direction_evaluation_count += 1
            differing = np.column_stack([
                S[attr].reset_index(drop=True).to_numpy()
                != donors[attr].to_numpy()
                for attr in schema.attribute_names()
            ])
            active_directions = copy_direction_scores[differing]
            if (
                diffusion_direction_normalization == "initial_rms"
                and direction_reference_scale is None
            ):
                candidate_scale = direction_rms_scale(active_directions)
                if candidate_scale > 0.0:
                    direction_reference_scale = candidate_scale
            if diffusion_direction_normalization == "initial_rms":
                effective_direction_strength = (
                    diffusion_direction_strength / direction_reference_scale
                    if direction_reference_scale is not None else 0.0
                )
            else:
                effective_direction_strength = diffusion_direction_strength
            effective_direction_strength_history.append(
                float(effective_direction_strength)
            )
            direction_reference_scale_history.append(
                direction_reference_scale
            )
            if len(active_directions):
                active_probabilities = tilted_copy_probabilities(
                    eta,
                    active_directions,
                    effective_direction_strength,
                )
                negative_mask = active_directions < 0.0
                positive_mask = active_directions > 0.0
                copy_direction_mean_history.append(
                    float(np.mean(active_directions))
                )
                copy_direction_positive_rate_history.append(
                    float(np.mean(active_directions > 0.0))
                )
                copy_direction_negative_rate_history.append(
                    float(np.mean(negative_mask))
                )
                negative_direction_copy_probability_history.append(
                    float(np.mean(active_probabilities[negative_mask]))
                    if np.any(negative_mask) else None
                )
                positive_direction_copy_probability_history.append(
                    float(np.mean(active_probabilities[positive_mask]))
                    if np.any(positive_mask) else None
                )
                if effective_direction_strength == 0.0:
                    copy_probability_entropy_history.append(
                        float(bernoulli_entropy(np.asarray([eta]))[0])
                    )
                else:
                    copy_probability_entropy_history.append(
                        float(np.mean(
                            bernoulli_entropy(active_probabilities)
                        ))
                    )
                copy_probability_kl_history.append(
                    float(np.mean(
                        bernoulli_kl(active_probabilities, eta)
                    ))
                )
                drift_diagnostics = additive_copy_drift_diagnostics(
                    active_directions,
                    active_probabilities,
                    eta,
                )
                additive_copy_drift_improvement_history.append(
                    drift_diagnostics["additive_drift_improvement"]
                )
                available_additive_copy_drift_improvement_history.append(
                    drift_diagnostics[
                        "available_additive_drift_improvement"
                    ]
                )
                additive_copy_drift_utilization_history.append(
                    drift_diagnostics["additive_drift_utilization"]
                )
            else:
                copy_direction_mean_history.append(0.0)
                copy_direction_positive_rate_history.append(0.0)
                copy_direction_negative_rate_history.append(0.0)
                negative_direction_copy_probability_history.append(None)
                positive_direction_copy_probability_history.append(None)
                copy_probability_entropy_history.append(None)
                copy_probability_kl_history.append(None)
                additive_copy_drift_improvement_history.append(None)
                available_additive_copy_drift_improvement_history.append(None)
                additive_copy_drift_utilization_history.append(None)
        else:
            copy_direction_scores = None
            effective_direction_strength = None
            copy_direction_mean_history.append(None)
            copy_direction_positive_rate_history.append(None)
            copy_direction_negative_rate_history.append(None)
            negative_direction_copy_probability_history.append(None)
            positive_direction_copy_probability_history.append(None)
            copy_probability_entropy_history.append(
                float(bernoulli_entropy(np.asarray([eta]))[0])
            )
            copy_probability_kl_history.append(0.0)
            additive_copy_drift_improvement_history.append(None)
            available_additive_copy_drift_improvement_history.append(None)
            additive_copy_drift_utilization_history.append(None)
            effective_direction_strength_history.append(None)
            direction_reference_scale_history.append(None)

        # 7-8. 靠近一步 → 整代安全检查。首次失败时可复用当轮
        # donor 并缩小 rho 重试；距离/适应度/抽样都不重算，额外成本只是
        # evolve_step + 一次提案查询评价。max_retries=0 时与原逻辑完全一致。
        accepted = False
        accepted_attempt = 0
        accepted_rho = None
        proposal_attempts = 0
        current_loss = loss
        attempt_gains: List[float] = []
        attempt_linear_gains: List[float] = []
        attempt_quadratic_penalties: List[float] = []
        attempt_factorized_gibbs_diagnostics: List[Dict[str, Any]] = []
        count_residual = target - q
        direction_kwargs = (
            {
                "copy_direction_scores": copy_direction_scores,
                "copy_direction_strength": effective_direction_strength,
            }
            if residual_directed_diffusion else {}
        )
        for attempt in range(max_retries + 1):
            attempt_rho = (
                rho_t * self_cooling_factor * (retry_rho_decay ** attempt)
            )
            if factorized_gibbs_sweeps > 0:
                proposal, factorized_diagnostics = (
                    evolve_step_factorized_gibbs(
                        S,
                        donors,
                        schema,
                        queries,
                        residual,
                        rho=attempt_rho,
                        eta=eta,
                        mu=mu * self_cooling_factor,
                        copy_direction_scores=copy_direction_scores,
                        copy_direction_strength=effective_direction_strength,
                        n_sweeps=factorized_gibbs_sweeps,
                        rng=rng,
                        gibbs_rng=factorized_gibbs_rng,
                        max_factor_order=factorized_gibbs_max_order,
                        gibbs_logit_clip=factorized_gibbs_logit_clip,
                    )
                )
                attempt_factorized_gibbs_diagnostics.append(
                    factorized_diagnostics
                )
                factorized_gibbs_factor_build_elapsed_sec += (
                    factorized_diagnostics["factor_build_elapsed_sec"]
                )
                factorized_gibbs_sample_elapsed_sec += (
                    factorized_diagnostics["gibbs_sample_elapsed_sec"]
                )
                factorized_gibbs_active_rows += factorized_diagnostics[
                    "active_gibbs_rows"
                ]
                factorized_gibbs_active_blocks += factorized_diagnostics[
                    "active_blocks"
                ]
                factorized_gibbs_factor_count += factorized_diagnostics[
                    "factor_count"
                ]
                factorized_gibbs_factor_table_entries += (
                    factorized_diagnostics["factor_table_entries"]
                )
                factorized_gibbs_microsteps += factorized_diagnostics[
                    "gibbs_microsteps"
                ]
            else:
                proposal = evolve_step(
                    S,
                    donors,
                    schema,
                    rho=attempt_rho,
                    eta=eta,
                    mu=mu * self_cooling_factor,
                    rng=rng,
                    **direction_kwargs,
                )
            proposal_q = _eval_counts(proposal)
            proposal_loss = compute_loss(target, proposal_q)
            proposal_attempts += 1
            candidate_evaluation_count += 1

            delta_q = proposal_q - q
            linear_gain = float(np.dot(count_residual, delta_q))
            quadratic_penalty = float(0.5 * np.dot(delta_q, delta_q))
            attempt_linear_gains.append(linear_gain)
            attempt_quadratic_penalties.append(quadratic_penalty)
            attempt_gains.append(float(loss - proposal_loss))

            # 候选预算是硬上限：本次候选已被评估并计入，若刚好触边，就先标记
            # 耗尽。这必须在接受/拒绝分支之前判定——否则接受路径的 break 会绕过
            # 检查，导致连续接受时评估次数无限超出预算（见 #36 复现）。
            if (candidate_budget is not None
                    and candidate_evaluation_count >= candidate_budget):
                candidate_budget_exhausted = True

            # 边界上这个已评估的候选仍可正常应用（接受即生效），但随后必须停止。
            if proposal_loss <= loss + tol:
                accepted = True
                accepted_attempt = attempt + 1
                accepted_rho = attempt_rho
                current_loss = proposal_loss
                S = proposal

            if accepted or candidate_budget_exhausted:
                break

        accept_history.append(accepted)
        proposal_attempts_history.append(proposal_attempts)
        accepted_attempt_history.append(accepted_attempt)
        accepted_rho_history.append(accepted_rho)
        raw_proposal_gain_history.append(attempt_gains)
        raw_proposal_linear_gain_history.append(attempt_linear_gains)
        raw_proposal_quadratic_penalty_history.append(
            attempt_quadratic_penalties
        )
        factorized_gibbs_attempt_diagnostics_history.append(
            attempt_factorized_gibbs_diagnostics
        )

        post_round_q = proposal_q if accepted else q
        current_state_metrics_history.append(
            _current_state_metrics(
                target,
                post_round_q,
                n_records,
                current_loss,
                state_index=len(current_state_metrics_history),
                round_index=t + 1,
                phase="post_round",
            )
        )

        if accepted:
            # S 已替换为 proposal，旧表对应的所有缓存立即失效。显式删除本地距离
            # 引用，避免下一轮计算新矩阵时旧矩阵仍占一份设备内存。
            state_eval_cache = None
            distance_cache = None
            del distances

        # 逐轮进度：单行输出 loss + 接受状态（受 log_every 控制）
        if do_log:
            budget_info = ""
            if candidate_budget is not None:
                budget_info = f" | 候选: {candidate_evaluation_count}/{candidate_budget}"
            print(f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e}"
                  f" | 接受: {'是' if accepted else '否'}"
                  f" | 尝试: {proposal_attempts}{budget_info}")

        # 9. 更新历史最优（直接用已知 loss，不重复评价）
        if current_loss < best_loss:
            best_loss = current_loss
            best_S = S.copy()

        # 检查候选预算：轮次结束后再次检查，如果已耗尽则停止
        if candidate_budget_exhausted:
            if do_log or log_every > 0:
                print(f"轮次 {t+1}/{n_rounds} | loss: {loss:.2e} | "
                      f"达到候选预算 {candidate_budget} 提前停止")
            break

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
    final_current_metrics = current_state_metrics_history[-1]

    diagnostics = {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "best_loss_diagnostic_only": best_loss,
        "normalized_l1_at_best_squared_loss_diagnostic_only": (
            normalized_l1_error
        ),
        "current_state_metrics_history": current_state_metrics_history,
        "current_state_transition_count": (
            len(current_state_metrics_history) - 1
        ),
        "final_current_normalized_l1": final_current_metrics[
            "current_normalized_l1"
        ],
        "final_current_squared_loss": final_current_metrics[
            "current_squared_loss"
        ],
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
        "rho_schedule_history": rho_schedule_history,
        "donor_top_share_history": donor_top_share_history,
        "row_max_prob_mean_history": row_max_prob_mean_history,
        "row_max_prob_max_history": row_max_prob_max_history,
        "effective_donors_mean_history": effective_donors_mean_history,
        "copy_direction_mean_history": copy_direction_mean_history,
        "copy_direction_positive_rate_history": (
            copy_direction_positive_rate_history
        ),
        "copy_direction_negative_rate_history": (
            copy_direction_negative_rate_history
        ),
        "negative_direction_copy_probability_history": (
            negative_direction_copy_probability_history
        ),
        "positive_direction_copy_probability_history": (
            positive_direction_copy_probability_history
        ),
        "copy_probability_entropy_history": (
            copy_probability_entropy_history
        ),
        "copy_probability_kl_history": copy_probability_kl_history,
        "additive_copy_drift_improvement_history": (
            additive_copy_drift_improvement_history
        ),
        "available_additive_copy_drift_improvement_history": (
            available_additive_copy_drift_improvement_history
        ),
        "additive_copy_drift_utilization_history": (
            additive_copy_drift_utilization_history
        ),
        "effective_direction_strength_history": (
            effective_direction_strength_history
        ),
        "direction_reference_scale_history": (
            direction_reference_scale_history
        ),
        "raw_proposal_gain_history": raw_proposal_gain_history,
        "raw_proposal_linear_gain_history": raw_proposal_linear_gain_history,
        "raw_proposal_quadratic_penalty_history": (
            raw_proposal_quadratic_penalty_history
        ),
        "factorized_gibbs_attempt_diagnostics_history": (
            factorized_gibbs_attempt_diagnostics_history
        ),
        "state_evaluation_count": state_evaluation_count,
        "candidate_evaluation_count": candidate_evaluation_count,
        "candidate_budget_exhausted": candidate_budget_exhausted,
        "self_cooling_history": self_cooling_history,
        "self_cooling_stopped": self_cooling_stopped,
        "distance_evaluation_count": distance_evaluation_count,
        "direction_evaluation_count": direction_evaluation_count,
        "direction_evaluation_elapsed_sec": direction_evaluation_elapsed_sec,
        "direction_reference_scale": direction_reference_scale,
        "factorized_gibbs_factor_build_elapsed_sec": (
            factorized_gibbs_factor_build_elapsed_sec
        ),
        "factorized_gibbs_sample_elapsed_sec": (
            factorized_gibbs_sample_elapsed_sec
        ),
        "factorized_gibbs_active_rows": factorized_gibbs_active_rows,
        "factorized_gibbs_active_blocks": factorized_gibbs_active_blocks,
        "factorized_gibbs_factor_count": factorized_gibbs_factor_count,
        "factorized_gibbs_factor_table_entries": (
            factorized_gibbs_factor_table_entries
        ),
        "factorized_gibbs_microsteps": factorized_gibbs_microsteps,
        "initial_table_sha256": initial_table_sha256,
        "primary_rng_post_initialization_state_sha256": (
            primary_rng_post_initialization_state_sha256
        ),
        "primary_rng_state_sha256": _rng_state_sha256(rng),
        "factorized_gibbs_initial_rng_state_sha256": (
            factorized_gibbs_initial_rng_state_sha256
        ),
        "factorized_gibbs_rng_state_sha256": (
            _rng_state_sha256(factorized_gibbs_rng)
            if factorized_gibbs_rng is not None else None
        ),
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
            "residual_directed_diffusion": bool(
                residual_directed_diffusion
            ),
            "diffusion_direction_strength": diffusion_direction_strength,
            "diffusion_direction_normalization": (
                diffusion_direction_normalization
            ),
            "factorized_gibbs_sweeps": factorized_gibbs_sweeps,
            "factorized_gibbs_max_order": factorized_gibbs_max_order,
            "factorized_gibbs_logit_clip": factorized_gibbs_logit_clip,
            "candidate_budget": (
                int(candidate_budget) if candidate_budget is not None else None
            ),
            "residual_self_cooling": (
                float(residual_self_cooling)
                if residual_self_cooling is not None else None
            ),
            "self_cooling_monotone": (
                bool(self_cooling_monotone)
                if residual_self_cooling is not None else None
            ),
            "self_cooling_stop_ratio": (
                float(self_cooling_stop_ratio)
                if self_cooling_stop_ratio is not None else None
            ),
            "rho_anneal_end": (
                float(rho_anneal_end) if rho_anneal_end is not None else None
            ),
            "rho_anneal_rounds": (
                int(rho_anneal_rounds)
                if rho_anneal_rounds is not None else None
            ),
            "selection_scale_invariant": bool(selection_scale_invariant),
            "selection_scale_invariant_min_spread": (
                float(selection_scale_invariant_min_spread)
                if selection_scale_invariant else None
            ),
        },
    }
    if return_final_table:
        diagnostics["final_table"] = S.copy(deep=True).reset_index(drop=True)

    return best_S.reset_index(drop=True), diagnostics
