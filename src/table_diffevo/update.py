"""
向参考记录靠近一步

对每条当前记录 x_i 和它抽到的参考记录 z_i*，本轮做三件事（完整方案第 7 节）：

1. 可能保持不变（记录参与概率 ρ_t）
2. 可能复制参考记录的一部分属性块（属性块复制概率 η_t）
3. 以很小概率发生随机变异（变异概率 μ_t）

## 记录参与（7.2，ρ_t）

先抽 U_i ~ Bernoulli(ρ_t)：
- U_i = 0：整条记录保持不变
- U_i = 1：进入属性块复制过程

ρ_t 控制一轮中大约多少比例的记录有机会变化。

## 属性块复制（7.3，η_t）

对每个可修改属性块 g：
- 与参考记录相同 → 直接保持
- 与参考记录不同 → 以概率 η_t 复制参考记录该块，否则保持原值

可选的残差驱动扩散会在 Bernoulli 对数几率上连续加入实际单块转移的方向量：

    logit(p_copy) = logit(η_t) + strength * direction

方向为零或 strength=0 时精确保持 η_t；负方向概率降低但在有限数值下不被硬置零。
它改变随机转移核，不使用 ``direction > 0`` 资格筛选。

逐块靠近，不是一步整行复制。

## 变异（7.4，μ_t）

每条参与更新的记录最多变异一个块：
1. 以概率 μ_t 决定是否变异
2. 随机选一个块
3. 从该块的合法先验分布抽一个值

**玩具阶段简化（已与设计确认）：**
- 合法先验用 schema 合法值上的均匀分布（类别块）/ 范围内均匀整数（数值块）
- 暂不做合法性检查与回退（7.5），单字段值域天然合法，跨字段约束留待后续

## 职责边界

本模块只负责"给定当前记录和已对齐的参考记录，靠近一步"。
- donors 已按行对齐：donors.iloc[i] 是 current.iloc[i] 的参考记录
  （从候选池按抽样索引取 donor 的逻辑在上游，见 sampling.sample_donors）
- ρ、η、μ 随轮次的衰减调度由主循环负责，本函数只接收当前轮的标量值
"""
from typing import Optional
import numpy as np
import pandas as pd
from table_diffevo.schema import Schema
from table_diffevo.directional_diffusion import tilted_copy_probabilities


def evolve_step(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    rho: float = 0.1,
    eta: float = 0.5,
    mu: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    copy_direction_scores: Optional[np.ndarray] = None,
    copy_direction_strength: float = 0.0,
) -> pd.DataFrame:
    """
    全表同步向参考记录靠近一步，生成下一代 S_{t+1}。

    Parameters
    ----------
    current : pd.DataFrame, shape (N, n_attributes)
        当前记录表 S_t
    donors : pd.DataFrame, shape (N, n_attributes)
        已按行对齐的参考记录：donors.iloc[i] 是 current.iloc[i] 的参考记录
    schema : Schema
        属性 schema 定义
    rho : float, default 0.1
        记录参与概率 ρ_t，一轮中大约多少比例的记录有机会变化
    eta : float, default 0.5
        属性块复制概率 η_t，不同的块以此概率复制参考记录
    mu : float, default 0.01
        变异概率 μ_t，参与更新的记录以此概率变异一个块
    rng : np.random.Generator or None
        随机数生成器。推荐显式传入 np.random.default_rng(seed) 保证复现
    copy_direction_scores : np.ndarray or None, shape (N, A), default None
        每条记录、每个属性块的实际单块复制方向量。None 表示使用历史固定 η。
        提供时只连续倾斜复制概率，不执行正负阈值筛选。
    copy_direction_strength : float, default 0.0
        非负有限方向强度。0 精确退化到历史固定 η 路径；正值越大，复制概率对
        方向量越敏感。有限强度下负方向仍保留非零复制概率。

    Returns
    -------
    pd.DataFrame, shape (N, n_attributes)
        下一代记录表 S_{t+1}（新对象，不修改输入）

    Raises
    ------
    ValueError
        current 与 donors 形状不一致、概率参数越界

    Notes
    -----
    **复现性（铁律 5）：** 使用固定种子的 rng 保证结果可复现。

    **全表同步（铁律）：** 所有记录基于同一份输入同步生成下一状态。

    Examples
    --------
    >>> from table_diffevo.sampling import compute_sampling_probs, sample_donors
    >>> from table_diffevo.distance import pairwise_block_distance
    >>> from table_diffevo.schema import load_schema
    >>>
    >>> schema = load_schema("configs/schema.yaml")
    >>> probs = compute_sampling_probs(fitness, distances)
    >>> rng = np.random.default_rng(42)
    >>> donor_idx = sample_donors(probs, rng)
    >>> donors = current.iloc[donor_idx].reset_index(drop=True)
    >>> next_table = evolve_step(current, donors, schema, rng=rng)
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho 必须在 [0, 1]，得到 {rho}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta 必须在 [0, 1]，得到 {eta}")
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu 必须在 [0, 1]，得到 {mu}")

    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 ({len(donors)}) 不一致"
        )

    if (
        isinstance(copy_direction_strength, (bool, np.bool_))
        or not isinstance(
            copy_direction_strength,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(copy_direction_strength)
        or copy_direction_strength < 0.0
    ):
        raise ValueError(
            "copy_direction_strength 必须是非负有限数值，"
            f"得到 {copy_direction_strength!r}"
        )
    copy_direction_strength = float(copy_direction_strength)

    if rng is None:
        rng = np.random.default_rng()

    N = len(current)
    attr_names = schema.attribute_names()

    if copy_direction_scores is None:
        if copy_direction_strength != 0.0:
            raise ValueError(
                "copy_direction_strength 非零时必须提供 copy_direction_scores"
            )
        direction_scores = None
    else:
        direction_scores = np.asarray(copy_direction_scores)
        expected_shape = (N, len(attr_names))
        if direction_scores.shape != expected_shape:
            raise ValueError(
                "copy_direction_scores 必须是 shape (N, A) 的二维数组，"
                f"得到 {direction_scores.shape}，期望 {expected_shape}"
            )
        if direction_scores.dtype.kind not in "iuf":
            raise ValueError("copy_direction_scores 必须是数值数组")
        direction_scores = direction_scores.astype(float, copy=False)
        if not np.all(np.isfinite(direction_scores)):
            raise ValueError("copy_direction_scores 必须全部为有限数值")

    # 以当前表为基础构造下一代（新对象，索引对齐 0..N-1）
    next_table = current.reset_index(drop=True).copy()
    donors = donors.reset_index(drop=True)

    # 7.2 记录参与：U_i ~ Bernoulli(rho)
    participate = rng.random(N) < rho  # (N,) 布尔

    # 7.3 属性块复制：对每个块，参与且与参考不同的记录以概率 eta 复制
    for attr_idx, attr in enumerate(attr_names):
        cur_col = current[attr].reset_index(drop=True).to_numpy()
        donor_col = donors[attr].to_numpy()
        differ = cur_col != donor_col  # (N,) 与参考记录不同的位置
        if direction_scores is None or copy_direction_strength == 0.0:
            # 默认和 strength=0 端点严格复用历史表达式与随机数消耗。
            copy_roll = rng.random(N) < eta
        else:
            copy_probability = tilted_copy_probabilities(
                eta,
                direction_scores[:, attr_idx],
                copy_direction_strength,
            )
            copy_roll = rng.random(N) < copy_probability
        copy_mask = participate & differ & copy_roll
        if copy_mask.any():
            new_col = next_table[attr].to_numpy().copy()
            new_col[copy_mask] = donor_col[copy_mask]
            next_table[attr] = new_col

    # 7.4 变异：参与更新的记录以概率 mu 变异一个块
    mutate_mask = participate & (rng.random(N) < mu)  # (N,)
    mutate_rows = np.nonzero(mutate_mask)[0]
    for i in mutate_rows:
        block = _sample_mutation_block(schema, rng)
        new_value = _sample_legal_value(schema.get_block(block), rng)
        next_table.at[i, block] = new_value

    return next_table


def _sample_mutation_block(schema: Schema, rng: np.random.Generator) -> str:
    """随机选择一个可修改属性块的名字。"""
    names = schema.attribute_names()
    idx = rng.integers(0, len(names))
    return names[idx]


def _sample_legal_value(block, rng: np.random.Generator):
    """
    从块的合法先验分布抽一个值（玩具阶段：均匀分布）。

    - 类别块：合法取值集合上的均匀抽样
    - 数值块：[min, max] 范围内的均匀整数（含端点）
    """
    if block.is_numeric():
        low, high = block.range
        # 范围内均匀整数，含端点
        return int(rng.integers(int(low), int(high) + 1))
    else:
        idx = rng.integers(0, len(block.values))
        return block.values[idx]
