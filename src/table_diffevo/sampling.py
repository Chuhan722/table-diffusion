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
from typing import Optional, Literal
import numpy as np


def _exclude_self_numpy(probs):
    """把对角线概率置 0 并按行重归一化（numpy）。

    等价于抽样前将对角 logit 设为 -inf：softmax 后归一化仅在非自身候选上进行。
    要求 probs 为方阵（N==K），调用方已校验。
    行内除自身外全为 0 的极端情形（不会在 δ>0 的 geometric/含距离模式发生）
    会导致除零，此处不额外兜底——若真发生应上抛而非静默。
    """
    probs = probs.copy()
    n = probs.shape[0]
    idx = np.arange(n)
    probs[idx, idx] = 0.0
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def compute_sampling_probs(
    fitness,
    distances,
    beta: float = 1.0,
    h: float = 0.8,
    device: Literal['cuda', 'cpu', 'numpy'] = 'numpy',
    distance_mode: Literal['squared', 'linear', 'none', 'multiplicative', 'geometric'] = 'geometric',
    p: float = 1.0,
    lambda_param: float = 0.5,
    alpha: float = 1.0,
    delta: float = 0.05,
    winsorize_quantiles: tuple = (0.01, 0.99),
    exclude_self: bool = False,
    scale_invariant: bool = False,
    scale_invariant_min_spread: float = 1e-3,
):
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
    device : {'cuda', 'cpu', 'numpy'}, default 'numpy'
        计算设备：
        - 'numpy'（默认）：原始 NumPy 实现，distances 为 np.ndarray，返回 np.ndarray
        - 'cuda'/'cpu'：PyTorch 实现，softmax 在设备上算。distances 可为
          留在设备上的 torch.Tensor（来自 distance 的 return_tensor=True），
          fitness 为 np.ndarray；返回留在设备上的 torch.Tensor，供 sample_donors 接力。
    distance_mode : {'squared', 'linear', 'none', 'multiplicative', 'geometric'}, default 'linear'
        距离项的处理方式：
        - 'linear'（默认，推荐）：exp(-d/h)，拉普拉斯核。实验表明在大数据上
          比 squared 优 ~20%（nltcs 1500轮，p<0.00001）。
        - 'squared'：exp(-d²/2h²)，高斯核（原实现，保留用于对比实验）
        - 'none'：距离项设为0，只用适应度驱动（实验中表现最差，不推荐）
        - 'multiplicative'：乘法解耦，适应度先过 softmax，再乘距离权重 (1-d)^p，
          最后按行归一化。不需要匹配 F 和 d 的量级，调参解耦（β 管锐度、p 管陡度）。
        - 'geometric'：几何平均联合抽样，稳健归一化适应度（winsorize + min-max）
          和距离到 [δ,1]，用几何平均 f^λ · s^(1-λ) 结合，softmax(α·log A) 抽样。
          解决量级匹配问题，λ 控制倾斜、α 控制锐度。
    p : float, default 1.0
        距离陡度参数（仅 multiplicative 模式使用）
        - 控制距离权重 w=(1-d)^p 的衰减速度
        - p=0：w 恒为 1，忽略距离
        - p=1（默认）：线性衰减，温和
        - p=2：二次衰减，偏近邻
        - p 越大越强烈偏向近邻，但远候选权重始终 >0（保留逃逸通道）
    lambda_param : float, default 0.5
        倾斜参数（仅 geometric 模式使用）
        - 控制适应度与相似度的相对权重
        - λ=0.5（默认）：对称，两者等权
        - λ<0.5：偏相似度，更看重近
        - λ>0.5：偏适应度，更看重好
    alpha : float, default 1.0
        锐度参数（仅 geometric 模式使用）
        - 当前轮的动态锐度 α_t（由外层传入，通常从 α_min 线性升到 α_max）
        - 值越大分布越尖锐（贪心），越小越平坦（探索）
    delta : float, default 0.05
        底值（仅 geometric 模式使用）
        - 防 log(0)，保留逃逸通道
        - f, s ∈ [δ, 1]，越小越接近硬排除
    winsorize_quantiles : tuple, default (0.01, 0.99)
        稳健归一化分位点（仅 geometric 模式使用）
        - 截掉适应度的极端值，(q_low, q_high)
        - 越收越稳健，越宽越保留极值差异
    exclude_self : bool, default False
        是否排除对角线（禁止记录抽到自己）。
        - False（默认）：保持原行为，候选池含自身（自身距离=0、相似度=1，
          可能被抽中=本轮该行不变）。所有独立调用/测试维持不变。
        - True：抽样前把对角线 probs[i,i] 置 0 并按行重归一化，等价于候选池
          排除自身。仅在候选池=全表（K=N，行 i 与列 i 是同一条记录）时有意义，
          故要求 distances 为方阵（N==K），否则报错。主循环全对全时开启。
    scale_invariant : bool, default False
        尺度不变选择（仅 geometric 模式；Issue #44 机制迭代）。True 时对
        每行的联合分数做标准化 ``logits = alpha * (log_A - mean_i) /
        max(std_i, scale_invariant_min_spread)`` 再 softmax：选择压力的
        有效温度恒等于 alpha，与 log_A 行内离散度的绝对尺度解耦。动机：
        随着种群同质化，行内分数离散度收缩，固定 alpha 的 softmax 区分度
        衰减、选择退化为均匀——标准化把"锐度"变成机制内生性质，不再依赖
        对 alpha 的调参补偿。``exclude_self=True`` 时行统计只在非自身
        候选上计算（第三轮审查修正：自身条目距离恒 0、分数极端，参与
        统计会扭曲其余合法 donor 之间的选择强度）。False（默认）保持
        历史行为完全不变。
    scale_invariant_min_spread : float, default 1e-3
        低信号保护（仅 scale_invariant=True 时生效）。行内标准差的下限：
        放大倍数被限制在 ``alpha / scale_invariant_min_spread`` 以内；
        当行内离散度低于该值时选择强度随离散度线性平滑衰减（离散度趋零
        时退化为均匀），避免把纯噪声级微小差异放大成极端选择偏好。必须
        为正有限数。

    Returns
    -------
    np.ndarray 或 torch.Tensor, shape (N, K)
        抽样概率矩阵，probs[i, k] = Pr(J_i = k)
        每行非负、和为 1。numpy 路径返回 array，torch 路径返回设备上的 tensor。

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
    if distance_mode not in ('squared', 'linear', 'none', 'multiplicative', 'geometric'):
        raise ValueError(f"distance_mode 必须是 'squared'/'linear'/'none'/'multiplicative'/'geometric'，得到 {distance_mode}")
    if p < 0:
        raise ValueError(f"p 必须 ≥ 0，得到 {p}")
    if not (0 <= lambda_param <= 1):
        raise ValueError(f"lambda_param 必须在 [0,1]，得到 {lambda_param}")
    if alpha < 0:
        raise ValueError(f"alpha 必须 ≥ 0，得到 {alpha}")
    if not (0 < delta < 1):
        raise ValueError(f"delta 必须在 (0,1)，得到 {delta}")
    if len(winsorize_quantiles) != 2 or not (0 <= winsorize_quantiles[0] < winsorize_quantiles[1] <= 1):
        raise ValueError(f"winsorize_quantiles 必须是 (q_low, q_high)，0 <= q_low < q_high <= 1，得到 {winsorize_quantiles}")
    if not (
        isinstance(scale_invariant_min_spread, (int, float, np.integer, np.floating))
        and not isinstance(scale_invariant_min_spread, (bool, np.bool_))
        and np.isfinite(scale_invariant_min_spread)
        and scale_invariant_min_spread > 0
    ):
        raise ValueError(
            "scale_invariant_min_spread 必须是正有限数，"
            f"得到 {scale_invariant_min_spread!r}"
        )

    # exclude_self 只在候选池=全表（方阵，行 i 与列 i 同一条记录）时有意义。
    # 共享参考池（K≠N）里没有"自己"，盲目屏蔽第 i 列会误伤真实候选，故此处拦截。
    if exclude_self:
        dshape = getattr(distances, 'shape', None)
        if dshape is None or len(dshape) != 2 or dshape[0] != dshape[1]:
            raise ValueError(
                f"exclude_self=True 要求 distances 为方阵（N==K，全对全候选池），"
                f"得到 shape {dshape}"
            )

    # torch 路径：softmax 在设备上算（distances 可为设备上的 tensor）
    if device in ('cuda', 'cpu'):
        return _compute_sampling_probs_torch(
            fitness, distances, beta, h, device, distance_mode, p,
            lambda_param, alpha, delta, winsorize_quantiles, exclude_self,
            scale_invariant, scale_invariant_min_spread,
        )
    elif device != 'numpy':
        raise ValueError(f"Unknown device: {device}. Choose from 'cuda', 'cpu', 'numpy'.")

    # numpy 路径（默认，原逻辑不变）
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
    # ℓ_ik = β·F(z_k) − distance_penalty
    fitness_term = beta * fitness  # (K,) 广播到每行

    # 根据 distance_mode 计算距离惩罚项
    if distance_mode == 'squared':
        # 标准高斯核：exp(-d²/2h²)
        distance_penalty = distances**2 / (2 * h**2)  # (N, K)
    elif distance_mode == 'linear':
        # 拉普拉斯核：exp(-d/h)
        distance_penalty = distances / h  # (N, K)
    elif distance_mode == 'none':
        # 不考虑距离，距离项设为0（即权重为1）
        distance_penalty = np.zeros_like(distances)  # (N, K)
    elif distance_mode == 'multiplicative':
        # 乘法解耦：F 先 softmax → 乘距离权重 → 按行归一化
        # 第1步：F 过 softmax（数值稳定：减最大值）
        logits_F = fitness_term  # (K,)
        logits_F_shifted = logits_F - logits_F.max()
        exp_F = np.exp(logits_F_shifted)
        # 归一成概率（和为1）。注：这个 sum 是常数，会被第4步按行归一化精确
        # 吸收掉，数值上删掉也不变结果；保留是为了让 p_F 名副其实是"适应度概率"，
        # 与"适应度概率 × 距离权重"的设计叙述一致。减最大值那步则不可删（防 exp 溢出）。
        p_F = exp_F / exp_F.sum()  # (K,) 和为1

        # 第2步：距离权重
        w = (1 - distances) ** p  # (N, K)

        # 第3步：相乘
        unnormalized = p_F[None, :] * w  # (1,K) × (N,K) → (N,K)

        # 第4步：按行归一化
        probs = unnormalized / unnormalized.sum(axis=1, keepdims=True)

        if exclude_self:
            probs = _exclude_self_numpy(probs)
        return probs  # 提前返回，跳过下面的 softmax
    elif distance_mode == 'geometric':
        # 几何平均联合抽样：稳健归一化 + 几何平均 + 动态锐度
        # 第1步：稳健归一化适应度（winsorize + min-max）
        q_low, q_high = winsorize_quantiles
        q_low_val = np.quantile(fitness, q_low)
        q_high_val = np.quantile(fitness, q_high)

        if q_high_val - q_low_val < 1e-8:
            # 所有适应度相同 → f 全为 1，只剩 s 起作用（纯相似度抽样）
            f_norm = np.ones_like(fitness)
        else:
            f_clip = np.clip(fitness, q_low_val, q_high_val)
            f_norm = (f_clip - q_low_val) / (q_high_val - q_low_val + 1e-8)

        # 第2步：构造正值项（加底值 δ）
        # 先 clip 距离到 [0,1]（防止超界输入导致 s 为负）
        distances_clipped = np.clip(distances, 0.0, 1.0)
        f = delta + (1 - delta) * f_norm                      # (K,) ∈ [δ, 1]
        s = delta + (1 - delta) * (1.0 - distances_clipped)   # (N, K) ∈ [δ, 1]

        # 第3步：几何平均（log-space 计算防溢出）
        # A = f^λ · s^(1-λ) → log A = λ·log f + (1-λ)·log s
        log_f = np.log(f)                          # (K,)
        log_s = np.log(s)                          # (N, K)
        log_A = lambda_param * log_f[None, :] + (1 - lambda_param) * log_s  # (N, K)

        # 第4步：应用锐度 α_t。尺度不变模式先做行内标准化：有效温度恒等于
        # alpha，与 log_A 行内离散度的绝对尺度解耦。两项保护（第三轮审查）：
        # (a) exclude_self 时行统计只在非自身候选上计算——自身条目距离恒 0、
        #     分数极端，参与统计会扭曲其余合法 donor 之间的选择强度；自身
        #     logit 随后在 softmax 前置 -inf 移出支撑，softmax 后的通用清零
        #     重归一化步骤仅作为一致性防护；
        # (b) 标准差下限 scale_invariant_min_spread：放大倍数有界
        #     （alpha/min_spread），离散度低于下限时选择强度随之线性衰减、
        #     趋零时退化为均匀——不把噪声级微小差异放大成极端选择偏好。
        # 减行均值本身被 softmax 平移不变性吸收，保留是为了标准分语义自解释。
        if scale_invariant:
            if exclude_self:
                idx = np.arange(K)
                diag = log_A[idx, idx]
                k_eff = K - 1
                row_sum = log_A.sum(axis=1) - diag
                row_mean = (row_sum / k_eff)[:, None]
                row_sq = (log_A ** 2).sum(axis=1) - diag ** 2
                row_var = row_sq / k_eff - (row_sum / k_eff) ** 2
                row_std = np.sqrt(np.maximum(row_var, 0.0))[:, None]
            else:
                row_mean = log_A.mean(axis=1, keepdims=True)
                row_std = log_A.std(axis=1, keepdims=True)
            denom = np.maximum(row_std, scale_invariant_min_spread)
            logits = alpha * (log_A - row_mean) / denom
            if exclude_self:
                # NaN 漏洞修复（第二轮审查意见 1）：自身条目必须在 softmax
                # 前以 -inf 移出支撑。否则自身分数占优时（标准化放大后
                # logit 差可超过 float 下溢阈），其余合法 donor 概率全部
                # 下溢为 0，事后清零自身再归一化就是 0/0 → NaN。前置
                # -inf 与"softmax 后清零重归一化"在不下溢时数学等价。
                logits[idx, idx] = -np.inf
        else:
            logits = alpha * log_A  # (N, K)

        # 第5步：softmax（减最大值防溢出）
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if exclude_self:
            probs = _exclude_self_numpy(probs)
        return probs  # 提前返回

    logits = fitness_term[None, :] - distance_penalty  # (N, K)

    # softmax（减去行最大值做数值稳定）
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    if exclude_self:
        probs = _exclude_self_numpy(probs)
    return probs


def _compute_sampling_probs_torch(fitness, distances, beta, h, device, distance_mode, p,
                                  lambda_param, alpha, delta, winsorize_quantiles,
                                  exclude_self=False, scale_invariant=False,
                                  scale_invariant_min_spread=1e-3):
    """
    PyTorch 实现：softmax 在设备上算。与 numpy 版数学公式逐行对应。

    - distances 可为留在设备上的 torch.Tensor（避免 GPU→CPU 搬运），
      也接受 np.ndarray（会搬到设备）。
    - fitness 为 np.ndarray（长度 K）。
    - 返回留在设备上的 torch.Tensor (N, K)，供 sample_donors 接力。

    注：GPU 用 float32，与 numpy 的 float64 会有极小数值差（精度问题，
    非可复现问题）；同设备同输入下结果确定，torch 路径自身可复现。
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Use device='numpy' or install PyTorch: "
            "pip install torch"
        )

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    dev = torch.device(device)

    # distances：若已是 tensor 则原地用（可能已在设备上），否则转过去
    if isinstance(distances, torch.Tensor):
        dist_t = distances.to(dev).float()
    else:
        dist_t = torch.as_tensor(np.asarray(distances), dtype=torch.float32, device=dev)
    if dist_t.ndim != 2:
        raise ValueError(f"distances 必须是 2 维，得到 shape {tuple(dist_t.shape)}")

    fitness_t = torch.as_tensor(
        np.asarray(fitness, dtype=float), dtype=torch.float32, device=dev
    )
    if fitness_t.ndim != 1:
        raise ValueError(f"fitness 必须是 1 维，得到 shape {tuple(fitness_t.shape)}")
    N, K = dist_t.shape
    if fitness_t.shape[0] != K:
        raise ValueError(
            f"fitness 长度 ({fitness_t.shape[0]}) 与 distances 列数 ({K}) 不一致"
        )

    # logit：ℓ_ik = β·F(z_k) − distance_penalty
    if distance_mode == 'squared':
        distance_penalty = dist_t ** 2 / (2 * h ** 2)          # (N, K)
    elif distance_mode == 'linear':
        distance_penalty = dist_t / h                          # (N, K)
    elif distance_mode == 'none':
        distance_penalty = torch.zeros_like(dist_t)            # (N, K)
    elif distance_mode == 'multiplicative':
        # 乘法解耦：F 先 softmax → 乘距离权重 → 按行归一化
        # 第1步：F 过 softmax（数值稳定：减最大值）
        fitness_term = beta * fitness_t  # (K,)
        logits_F_shifted = fitness_term - fitness_term.max()
        exp_F = torch.exp(logits_F_shifted)
        # 归一成概率（和为1）。注：这个 sum 是常数，会被第4步按行归一化精确
        # 吸收掉，数值上删掉也不变结果；保留是为了让 p_F 名副其实是"适应度概率"，
        # 与"适应度概率 × 距离权重"的设计叙述一致。减最大值那步则不可删（防 exp 溢出）。
        p_F = exp_F / exp_F.sum()  # (K,) 和为1

        # 第2步：距离权重
        w = (1 - dist_t) ** p  # (N, K)

        # 第3步：相乘
        unnormalized = p_F.unsqueeze(0) * w  # (1,K) × (N,K) → (N,K)

        # 第4步：按行归一化
        probs = unnormalized / unnormalized.sum(dim=1, keepdim=True)

        if exclude_self:
            probs = _exclude_self_torch(probs)
        return probs  # 提前返回，跳过下面的 softmax
    elif distance_mode == 'geometric':
        # 几何平均联合抽样：稳健归一化 + 几何平均 + 动态锐度
        # 第1步：稳健归一化适应度（winsorize + min-max）
        q_low, q_high = winsorize_quantiles
        q_low_val = torch.quantile(fitness_t, q_low)
        q_high_val = torch.quantile(fitness_t, q_high)

        if (q_high_val - q_low_val) < 1e-8:
            # 所有适应度相同 → f 全为 1，只剩 s 起作用（纯相似度抽样）
            f_norm = torch.ones_like(fitness_t)
        else:
            f_clip = torch.clamp(fitness_t, q_low_val, q_high_val)
            f_norm = (f_clip - q_low_val) / (q_high_val - q_low_val + 1e-8)

        # 第2步：构造正值项（加底值 δ）
        # 先 clamp 距离到 [0,1]（防止超界输入导致 s 为负）
        dist_t_clipped = torch.clamp(dist_t, 0.0, 1.0)
        f = delta + (1 - delta) * f_norm                    # (K,) ∈ [δ, 1]
        s = delta + (1 - delta) * (1.0 - dist_t_clipped)    # (N, K) ∈ [δ, 1]

        # 第3步：几何平均（log-space 计算防溢出）
        # A = f^λ · s^(1-λ) → log A = λ·log f + (1-λ)·log s
        log_f = torch.log(f)                       # (K,)
        log_s = torch.log(s)                       # (N, K)
        log_A = lambda_param * log_f.unsqueeze(0) + (1 - lambda_param) * log_s  # (N, K)

        # 第4步：应用锐度 α_t（scale_invariant 时先行内标准化，与 numpy
        # 路径同语义；std 用总体口径对齐 np.std 默认）。两项保护同 numpy：
        # exclude_self 时行统计只在非自身候选上计算；标准差下限
        # scale_invariant_min_spread 使放大倍数有界、低离散度平滑退化均匀。
        if scale_invariant:
            if exclude_self:
                diag = torch.diagonal(log_A)
                k_eff = K - 1
                row_sum = log_A.sum(dim=1) - diag
                mean_off = row_sum / k_eff
                row_sq = (log_A ** 2).sum(dim=1) - diag ** 2
                row_var = row_sq / k_eff - mean_off ** 2
                row_mean = mean_off.unsqueeze(1)
                row_std = torch.sqrt(
                    torch.clamp(row_var, min=0.0)
                ).unsqueeze(1)
            else:
                row_mean = log_A.mean(dim=1, keepdim=True)
                row_std = log_A.std(dim=1, keepdim=True, unbiased=False)
            denom = torch.clamp(row_std, min=scale_invariant_min_spread)
            logits = alpha * (log_A - row_mean) / denom
            if exclude_self:
                # NaN 漏洞修复（第二轮审查意见 1）：softmax 前置 -inf，
                # 语义与 numpy 路径一致（防自身占优时其余 donor 全下溢）。
                n_rows = logits.shape[0]
                eye_idx = torch.arange(n_rows, device=logits.device)
                logits[eye_idx, eye_idx] = float("-inf")
        else:
            logits = alpha * log_A  # (N, K)

        # 第5步：softmax（torch 内置，自动减最大值）
        probs = torch.softmax(logits, dim=1)

        if exclude_self:
            probs = _exclude_self_torch(probs)
        return probs  # 提前返回

    logits = fitness_term_broadcast(fitness_t, beta) - distance_penalty  # (N, K)

    # softmax（减去行最大值做数值稳定），沿列（dim=1）
    logits_shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    if exclude_self:
        probs = _exclude_self_torch(probs)
    return probs


def _exclude_self_torch(probs):
    """把对角线概率置 0 并按行重归一化（torch）。numpy 版 _exclude_self_numpy 的对应实现。"""
    import torch
    n = probs.shape[0]
    idx = torch.arange(n, device=probs.device)
    probs = probs.clone()
    probs[idx, idx] = 0.0
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def fitness_term_broadcast(fitness_t, beta):
    """β·F(z_k) 广播成行向量 (1, K)，供逐行相减。"""
    return (beta * fitness_t).unsqueeze(0)


def sample_donors(
    probs,
    rng: Optional[np.random.Generator] = None,
    device: Literal['cuda', 'cpu', 'numpy'] = 'numpy',
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
    device : {'cuda', 'cpu', 'numpy'}, default 'numpy'
        计算设备：
        - 'numpy'（默认）：原始 NumPy 实现，probs 为 np.ndarray
        - 'cuda'/'cpu'：PyTorch 实现，cumsum 在设备上算。probs 可为
          留在设备上的 torch.Tensor（来自 compute_sampling_probs 的 torch 路径）。
          **随机数仍用 numpy 的 rng.uniform 抽**（保证与 numpy 路径消耗相同的
          随机状态、同种子可复现），只把 N 个索引搬回 CPU 返回。

    Returns
    -------
    np.ndarray, shape (N,), dtype int
        每条记录抽到的候选索引，值在 [0, K)
        donor_indices[i] = J_i，即第 i 条记录抽到的候选编号
        （torch 路径也返回 CPU 上的 np.ndarray，接口一致）

    Raises
    ------
    ValueError
        probs 不是 2 维、或某行和不为 1（容差 1e-6）

    Notes
    -----
    **复现性（铁律 5）：** 使用固定种子的 rng 保证结果可复现。
    torch 路径的随机数仍由 numpy rng 提供，GPU 只做确定性的 cumsum+比较，
    因此同种子下 torch 路径自身可复现。

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
    # torch 路径：cumsum 在设备上算，随机数仍用 numpy rng（保可复现）
    if device in ('cuda', 'cpu'):
        return _sample_donors_torch(probs, rng, device)
    elif device != 'numpy':
        raise ValueError(f"Unknown device: {device}. Choose from 'cuda', 'cpu', 'numpy'.")

    # numpy 路径（默认，原逻辑不变）
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


def _sample_donors_torch(probs, rng, device):
    """
    PyTorch 实现：cumsum 在设备上算，抽样逻辑与 numpy 版逐行对应。

    **可复现关键：** 随机数仍用 numpy 的 rng.uniform(size=N) 抽——与 numpy
    路径消耗完全相同的随机状态，同种子 → 同随机数 → 同索引。GPU 只负责
    确定性的 cumsum 和 (u < cumprobs).argmax 比较，不掺和随机。

    只把最终 N 个索引搬回 CPU 返回（约 N×8 字节，极小），
    避免把 (N,K) 概率矩阵搬回 CPU。
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Use device='numpy' or install PyTorch: "
            "pip install torch"
        )

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    dev = torch.device(device)

    if isinstance(probs, torch.Tensor):
        probs_t = probs.to(dev).float()
    else:
        probs_t = torch.as_tensor(np.asarray(probs), dtype=torch.float32, device=dev)

    if probs_t.ndim != 2:
        raise ValueError(f"probs 必须是 2 维，得到 shape {tuple(probs_t.shape)}")

    N, K = probs_t.shape

    # 验证每行和为 1（容差放宽到 1e-4，float32 精度下 softmax 和会有微小偏差）
    row_sums = probs_t.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        raise ValueError(
            f"probs 某些行和不为 1（容差 1e-4，float32）："
            f"行和范围 [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}]"
        )

    if rng is None:
        rng = np.random.default_rng()

    # 随机数仍在 CPU 用 numpy 抽（保可复现），再搬到设备做比较
    u = rng.uniform(size=N)
    u_t = torch.as_tensor(u, dtype=torch.float32, device=dev).unsqueeze(1)  # (N, 1)

    # 与 numpy 版一致：cumsum → 第一个 (u < cumprob) 的位置
    cumprobs = probs_t.cumsum(dim=1)                       # (N, K)
    # (u < cumprobs) 是布尔矩阵；argmax 取第一个 True 的列（int → argmax 取首个最大）
    indices = (u_t < cumprobs).int().argmax(dim=1)        # (N,)

    return indices.cpu().numpy().astype(np.intp)
