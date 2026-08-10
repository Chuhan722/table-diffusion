"""
轻量级 Checkpoint 机制，用于探测式 α 调度的三岔路分支。

支持保存和恢复演化状态，使得三个探测分支可以从同一起点出发。
"""

import copy
import numpy as np
from typing import Any, Tuple


class Checkpoint:
    """
    演化状态的 checkpoint，支持保存和恢复。

    用于探测式 α 调度：在触发探测时保存当前状态，然后从这个 checkpoint
    分叉出 DOWN/HOLD/UP 三个分支，各自独立运行后比较结果。

    Attributes:
        syn_table: 合成表的深拷贝
        best_L1: 当前最优 normalized L1
        rng_state: numpy RandomState 的状态（支持可复现分叉）
        round_count: 当前轮数
        alpha: 当前 α 值
    """

    def __init__(
        self,
        syn_table: np.ndarray,
        best_L1: float,
        rng_state: Any,
        round_count: int,
        alpha: float,
    ):
        """
        初始化 checkpoint（通常通过 Checkpoint.capture() 创建）。

        Args:
            syn_table: 合成表数据
            best_L1: 当前最优 normalized L1
            rng_state: numpy RandomState.get_state() 返回的状态
            round_count: 当前轮数
            alpha: 当前 α 值
        """
        self.syn_table = copy.deepcopy(syn_table)
        self.best_L1 = best_L1
        self.rng_state = copy.deepcopy(rng_state)
        self.round_count = round_count
        self.alpha = alpha

    @classmethod
    def capture(
        cls,
        syn_table: np.ndarray,
        best_L1: float,
        rng: np.random.Generator,
        round_count: int,
        alpha: float,
    ) -> "Checkpoint":
        """
        保存当前演化状态为 checkpoint。

        Args:
            syn_table: 当前合成表
            best_L1: 当前最优 normalized L1
            rng: numpy Generator 实例
            round_count: 当前轮数
            alpha: 当前 α 值

        Returns:
            Checkpoint 实例
        """
        return cls(
            syn_table=syn_table,
            best_L1=best_L1,
            rng_state=rng.bit_generator.state,
            round_count=round_count,
            alpha=alpha,
        )

    def restore(self) -> Tuple[np.ndarray, float, Any, int, float]:
        """
        恢复 checkpoint 中保存的状态（返回深拷贝，避免污染原 checkpoint）。

        Returns:
            (syn_table_copy, best_L1, rng_state_copy, round_count, alpha)

            使用示例：
                syn_table, best_L1, rng_state, round_count, alpha = checkpoint.restore()
                rng.bit_generator.state = rng_state
        """
        return (
            copy.deepcopy(self.syn_table),
            self.best_L1,
            copy.deepcopy(self.rng_state),
            self.round_count,
            self.alpha,
        )
