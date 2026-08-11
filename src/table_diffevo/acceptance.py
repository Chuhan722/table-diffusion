"""
接受规则模块

实现 Issue #33 预注册的接受规则：
- A0：平方 loss 主判
- A1：归一化 L1 主判 + Q 平局判

所有规则在候选覆盖前计算差值，确保四象限统计正确。

**严格改善口径（Issue #33 预注册定义）**：
A0/A1 采用严格不等号——必须有超过 eps 的**实质改善**才接受，Q/L1 平局与
容差内的微小恶化一律拒绝。这是接受规则对照实验（阶段 A）的被测对象，两条臂
只应在"看 Q 还是看 L1"上有差异，故口径保持一致的严格 `<`。

注意：这与主循环 `evolution.py` 的判据 `proposal_loss <= loss + tol`（非严格，
接受平局与 tol 容差内微小恶化）在边界处理上不同。因此 A0 符合 Issue #33 冻结
公式、可作预注册实验臂，但**不是**与旧主循环逐轨迹等价的 baseline——这一差异
在阶段 A 分析时须显式披露，不得把边界差异误归因于 L1/Q 主判逻辑。最终选出的
A* 接入主循环时，再单独决定其容差口径。
"""
import numpy as np
from typing import Tuple
from .metrics import compute_normalized_l1, compute_squared_loss


def check_acceptance(
    rule: str,
    target: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
    n_records: int,
    eps_L1: float = 0.0,
    eps_Q: float = 0.0,
) -> Tuple[bool, float, float]:
    """
    检查候选是否接受（统一接口）。

    Parameters
    ----------
    rule : str
        接受规则，'A0' 或 'A1'
    target : np.ndarray
        目标计数向量
    current : np.ndarray
        当前状态的查询答案
    candidate : np.ndarray
        候选状态的查询答案
    n_records : int
        记录总数 N
    eps_L1 : float, default=0.0
        归一化 L1 容差（A1 使用）：|delta_L1| <= eps_L1 视为 L1 平局，转由 Q 裁决
    eps_Q : float, default=0.0
        平方 loss 严格改善阈值：仅当 delta_Q < -eps_Q（Q 改善超过 eps_Q）才接受。
        eps_Q = 0 时要求严格改善（delta_Q < 0），拒绝 Q 平局与任何恶化。

    Returns
    -------
    accept : bool
        是否接受候选
    delta_L1 : float
        L1_candidate - L1_current（负值表示改善）
    delta_Q : float
        Q_candidate - Q_current（负值表示改善）

    Raises
    ------
    ValueError
        如果 rule 不是 'A0' 或 'A1'
    """
    # 计算差值（在覆盖前）
    L1_current = compute_normalized_l1(target, current, n_records)
    L1_candidate = compute_normalized_l1(target, candidate, n_records)
    Q_current = compute_squared_loss(target, current)
    Q_candidate = compute_squared_loss(target, candidate)

    delta_L1 = L1_candidate - L1_current
    delta_Q = Q_candidate - Q_current

    # 根据规则判断
    if rule == "A0":
        accept = _check_A0(delta_Q, eps_Q)
    elif rule == "A1":
        accept = _check_A1(delta_L1, delta_Q, eps_L1, eps_Q)
    else:
        raise ValueError(f"未知接受规则: {rule}，仅支持 'A0' 或 'A1'")

    return accept, delta_L1, delta_Q


def _check_A0(delta_Q: float, eps_Q: float) -> bool:
    """
    A0 规则：平方 loss 主判（严格改善口径，Issue #33 预注册定义）。

    接受条件：delta_Q < -eps_Q，即 Q_candidate < Q_current - eps_Q。
    要求 Q 有超过 eps_Q 的实质改善才接受；Q 平局与任何恶化一律拒绝。

    Parameters
    ----------
    delta_Q : float
        Q_candidate - Q_current
    eps_Q : float
        Q 严格改善阈值（非负）；仅 delta_Q < -eps_Q 时接受

    Returns
    -------
    bool
        是否接受
    """
    return delta_Q < -eps_Q


def _check_A1(
    delta_L1: float,
    delta_Q: float,
    eps_L1: float,
    eps_Q: float
) -> bool:
    """
    A1 规则：归一化 L1 主判 + Q 平局判（严格改善口径，Issue #33 预注册定义）。

    接受条件：
    1. 若 delta_L1 < -eps_L1（L1 严格改善超过容差）→ 接受
    2. 若 |delta_L1| <= eps_L1（L1 落在容差带内，视为平局）→ 用 Q 裁决
       - delta_Q < -eps_Q（Q 严格改善）→ 接受
       - 否则 → 拒绝
    3. 若 delta_L1 > eps_L1（L1 恶化超过容差）→ 拒绝

    Parameters
    ----------
    delta_L1 : float
        L1_candidate - L1_current
    delta_Q : float
        Q_candidate - Q_current
    eps_L1 : float
        L1 平局容差（平局带半宽，非负）
    eps_Q : float
        Q 严格改善阈值（非负）；平局带内仅 delta_Q < -eps_Q 才接受

    Returns
    -------
    bool
        是否接受
    """
    if delta_L1 < -eps_L1:
        # L1 严格改善超过容差 → 接受
        return True
    elif abs(delta_L1) <= eps_L1:
        # L1 落在容差带内（|delta_L1| <= eps_L1），视为平局 → 用 Q 裁决
        return delta_Q < -eps_Q
    else:
        # L1 恶化超过容差 → 拒绝
        return False
