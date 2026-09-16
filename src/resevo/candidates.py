"""候选集枚举与参考分布构造。

支持集回答"每个块允许变化成什么"，与残差和目标完全无关，
合法性在这里一次性解决，抽样阶段绝不做合法性修复。
第一版决策，候选集直接用全集，每个单行块枚举全部合法状态，不做裁剪。
参考分布 R 把保持质量 0.9 放在源元组，剩余 0.1 按迁移率均分给其他合法后继。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlockSupport:
    """一个记录块的合法支持，rows 为行索引，outcomes 为合法后继元组。

    mobility 为可选迁移率权重，None 视为全 1。
    结构与参考实现保持一致，便于对拍。
    """

    rows: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]
    mobility: tuple[float, ...] | None = None


@dataclass(frozen=True)
class NormalizedBlock:
    """规范化后的块，源元组固定在索引 0，参考分布已构造完毕。"""

    rows: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]  # 索引 0 恒为源元组
    reference: NDArray[np.float64]  # R，与 outcomes 对齐，和为 1


def full_single_row_supports(
    num_rows: int,
    num_states: int,
    legal_states: Sequence[int] | None = None,
) -> list[BlockSupport]:
    """全集候选，每行一个单行块，枚举全部合法状态。

    legal_states 为 None 时全部状态合法，否则只枚举给定的合法状态编号，
    例如规则"若 A=1 则 B=1"下合法状态只有 00,01,11，支持里不出现 10。
    """
    if num_rows < 1 or num_states < 1:
        raise ValueError("行数与状态数必须为正")
    if legal_states is None:
        legal = tuple(range(num_states))
    else:
        legal = tuple(int(u) for u in legal_states)
        if len(set(legal)) != len(legal):
            raise ValueError("合法状态列表不能重复")
        if any(u < 0 or u >= num_states for u in legal):
            raise ValueError("合法状态编号越界")
        if not legal:
            raise ValueError("合法状态列表不能为空")
    outcomes = tuple((u,) for u in legal)
    return [BlockSupport((i,), outcomes) for i in range(num_rows)]


def validate_partition(supports: Sequence[BlockSupport], num_rows: int) -> None:
    """块必须构成全部行索引的不相交划分。"""
    flat = [i for spec in supports for i in spec.rows]
    if sorted(flat) != list(range(num_rows)) or any(not spec.rows for spec in supports):
        raise ValueError("块必须构成全部行索引的不相交划分")


def normalize_block(
    spec: BlockSupport,
    state_ids: NDArray[np.int64],
    num_states: int,
    stay_probability: float = 0.9,
) -> NormalizedBlock:
    """把一个支持规范化，源元组放到索引 0 并构造参考分布 R。

    保持状态必须出现恰好一次，这里无论输入含不含源元组，输出都恰好含一次。
    相同后继必须先合并参考质量再倾斜，这里对重复后继累加迁移率。
    非保持质量 1-stay 按迁移率比例分配，运算顺序与参考实现一致。
    """
    if not (0 < stay_probability < 1):
        raise ValueError("stay_probability 必须落在 (0,1)")
    source = tuple(int(state_ids[i]) for i in spec.rows)
    mob = spec.mobility if spec.mobility is not None else (1.0,) * len(spec.outcomes)
    if len(mob) != len(spec.outcomes):
        raise ValueError("mobility 与 outcomes 长度不一致")
    merged: dict[tuple[int, ...], float] = {}
    for outcome, c in zip(spec.outcomes, mob):
        if len(outcome) != len(source) or any(
            not isinstance(u, (int, np.integer)) or u < 0 or u >= num_states
            for u in outcome
        ):
            raise ValueError("块后继元组非法")
        if not np.isfinite(c) or c < 0:
            raise ValueError("迁移率必须有限且非负")
        key = tuple(int(u) for u in outcome)
        if key == source:
            continue
        if c > 0:
            # 相同后继先合并参考质量，保序累加
            merged[key] = merged.get(key, 0.0) + float(c)
    dest = list(merged.keys())
    cs = [merged[u] for u in dest]
    outcomes = (source,) + tuple(dest)
    if dest:
        reference = np.r_[stay_probability, (1 - stay_probability) * np.asarray(cs) / sum(cs)]
    else:
        # 该块没有其他合法后继，保持概率直接为 1，不参与更新
        reference = np.ones(1)
    return NormalizedBlock(spec.rows, outcomes, reference)
