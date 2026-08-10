"""
探测式 α 调度控制器

实现三岔路探测机制：
- 停滞检测：块降幅占当前 L1 的比例低于 stall_rel（量纲无关，换数据不重调）
- 探测触发：连续 P 个块停滞
- 三岔路探测：DOWN/HOLD/UP 各跑 H 个块（边界处两条 α 相同的岔路由主循环去重）
- Winner 选择：final_L1 最低者胜，平局偏 HOLD
- 冷却机制：选定后 C 个块内不再探测
- 经验平台：连续 patience 次探测未刷新历史最好 L1 → 判定收敛
"""

from typing import List, Tuple, Optional, Dict, Any


class ProbeController:
    """
    探测式 α 调度控制器

    负责管理探测的触发、分支创建、winner 选择、冷却和经验平台判定。

    Attributes:
        alpha_min: α 的最小值（对应 u=0）
        alpha_max: α 的最大值（对应 u=1）
        P: 停滞块数阈值（触发探测）
        H: 每个探测分支运行的块数
        s: 探测步长（归一化 u 的比例，如 0.10）
        C: 冷却块数
        stall_rel: 停滞相对阈值（块降幅 / 当前 L1 低于此值记一次停滞）
        patience: 耐心值（连续 patience 次探测未刷新历史最好 L1 → 收敛）
        stall_blocks: 当前停滞块数
        cooldown_remaining: 冷却剩余块数
        no_improve_probes: 连续未刷新历史最好 L1 的探测次数
        best_L1_ever: 历史最好 L1
        probe_history: 探测历史记录
    """

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        P: int,
        H: int,
        s: float,
        C: int,
        stall_rel: float = 0.02,
        patience: int = 3,
    ):
        """
        初始化探测控制器

        Args:
            alpha_min: α 的最小值
            alpha_max: α 的最大值
            P: 停滞块数阈值
            H: 每个探测分支运行的块数
            s: 探测步长（归一化比例）
            C: 冷却块数
            stall_rel: 停滞相对阈值（块降幅 / 当前 L1）
            patience: 耐心值（连续多少次探测未刷新历史最好 L1 判收敛）
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.P = P
        self.H = H
        self.s = s
        self.C = C
        self.stall_rel = stall_rel
        self.patience = patience

        self.stall_blocks = 0
        self.cooldown_remaining = 0
        self.no_improve_probes = 0
        self.best_L1_ever = float("inf")
        self.probe_history: List[Dict[str, Any]] = []

    def normalize_alpha(self, alpha: float) -> float:
        """
        将 α 归一化到 [0, 1]

        Args:
            alpha: α 值

        Returns:
            u ∈ [0, 1]
        """
        return (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)

    def denormalize_u(self, u: float) -> float:
        """
        将归一化的 u 转回 α

        Args:
            u: 归一化值 ∈ [0, 1]

        Returns:
            α 值
        """
        return self.alpha_min + u * (self.alpha_max - self.alpha_min)

    def update_stall_status(self, block_improvement: float, current_L1: float):
        """
        更新停滞状态

        用相对比例判停滞（块降幅占当前 L1 的比例），量纲无关，换数据集不需重调阈值。

        Args:
            block_improvement: 本块内 best L1 的改善量（正数表示改善）
            current_L1: 当前 best L1（作分母；兜底防除零）
        """
        # 冷却期递减（不能小于 0）
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)

        # 停滞判断：降幅占当前 L1 的比例低于 stall_rel 记一次停滞
        # （冷却期内也要更新，但不会触发探测）
        rel = block_improvement / max(current_L1, 1e-12)
        if rel > self.stall_rel:
            self.stall_blocks = 0
        else:
            self.stall_blocks += 1

    def should_trigger_probe(self) -> bool:
        """
        判断是否应触发探测

        Returns:
            True 如果应该触发探测
        """
        return self.stall_blocks >= self.P and self.cooldown_remaining == 0

    def create_probe_branches(self, current_alpha: float) -> List[Tuple[str, float]]:
        """
        创建三个探测分支配置

        Args:
            current_alpha: 当前 α 值

        Returns:
            [(direction, alpha), ...] 三个分支的配置
            direction ∈ {'DOWN', 'HOLD', 'UP'}
        """
        u = self.normalize_alpha(current_alpha)
        branches = [
            ("DOWN", self.denormalize_u(max(0.0, u - self.s))),
            ("HOLD", current_alpha),
            ("UP", self.denormalize_u(min(1.0, u + self.s))),
        ]
        return branches

    @staticmethod
    def unique_branch_alphas(
        branches: List[Tuple[str, float]]
    ) -> List[Tuple[float, List[str]]]:
        """
        对分支去重（毛病一）：α 触及边界时 DOWN/HOLD 或 UP/HOLD 会算出同一个 α，
        它们从同一 checkpoint、同一 RNG 出发会跑出逐位相同的结果。这里把 α 相同的
        方向归成一组，主循环对每个不同 α 只跑一次、结果由同组方向共享，省下重复预算。

        Args:
            branches: create_probe_branches 的输出 [(direction, alpha), ...]

        Returns:
            [(alpha, [directions...]), ...]，按首次出现顺序排列
        """
        groups: List[Tuple[float, List[str]]] = []
        for direction, alpha in branches:
            for i, (a, dirs) in enumerate(groups):
                if a == alpha:
                    dirs.append(direction)
                    break
            else:
                groups.append((alpha, [direction]))
        return groups

    def select_winner(
        self, branch_results: List[Tuple[str, float, float, float]]
    ) -> Tuple[str, float]:
        """
        选择获胜分支

        排序规则：
        1. final_L1 最低者胜
        2. 平局偏 HOLD

        收敛与否不在此判定（改由 register_probe_outcome + 耐心值决定），
        因此不再返回 all_failed。

        Args:
            branch_results: [(direction, L1_improvement, final_L1, alpha), ...]

        Returns:
            (winner_direction, winner_alpha)
        """
        def sort_key(item):
            direction, _improvement, final_L1, _alpha = item
            hold_penalty = 0 if direction == "HOLD" else 1
            return (final_L1, hold_penalty)

        sorted_branches = sorted(branch_results, key=sort_key)
        winner_direction, _, _, winner_alpha = sorted_branches[0]

        return winner_direction, winner_alpha

    def register_probe_outcome(self, winner_L1: float):
        """
        探测结束后结算：启动冷却、重置停滞计数，并按“是否刷新历史最好 L1”
        更新耐心计数。

        Args:
            winner_L1: 获胜分支的 final_L1
        """
        self.cooldown_remaining = self.C
        self.stall_blocks = 0

        if winner_L1 < self.best_L1_ever:
            self.best_L1_ever = winner_L1
            self.no_improve_probes = 0
        else:
            self.no_improve_probes += 1

    def is_empirical_plateau(self) -> bool:
        """
        判断是否达到经验平台（连续 patience 次探测未刷新历史最好 L1）

        Returns:
            True 如果达到经验平台
        """
        return self.no_improve_probes >= self.patience

    def record_probe(
        self,
        round_num: int,
        trigger_alpha: float,
        branches: List[Tuple[str, float]],
        branch_improvements: List[float],
        winner_direction: str,
        winner_alpha: float,
    ):
        """
        记录探测历史

        Args:
            round_num: 触发探测的轮数
            trigger_alpha: 触发时的 α 值
            branches: [(direction, alpha), ...]
            branch_improvements: 各分支的 L1 改善量
            winner_direction: 获胜方向
            winner_alpha: 获胜后的 α 值
        """
        probe_record = {
            "round": round_num,
            "trigger_alpha": trigger_alpha,
            "branches": [
                {"direction": d, "alpha": a, "improvement": imp}
                for (d, a), imp in zip(branches, branch_improvements)
            ],
            "winner": winner_direction,
            "winner_alpha": winner_alpha,
            "no_improve_probes": self.no_improve_probes,
        }
        self.probe_history.append(probe_record)
