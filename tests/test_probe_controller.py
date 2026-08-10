"""
测试探测式 α 调度控制器（相对停滞阈值 + 耐心值收敛判定版）
"""

import pytest
from table_diffevo.probe_controller import ProbeController


def _make(stall_rel=0.02, patience=3, P=3, C=2):
    return ProbeController(
        alpha_min=2.0, alpha_max=10.0, P=P, H=2, s=0.1, C=C,
        stall_rel=stall_rel, patience=patience,
    )


class TestProbeController:
    """测试 ProbeController 类"""

    def test_normalize_and_denormalize(self):
        """测试 α 归一化和反归一化"""
        c = _make()
        assert c.normalize_alpha(2.0) == 0.0
        assert c.normalize_alpha(10.0) == 1.0
        assert c.normalize_alpha(6.0) == 0.5
        assert c.denormalize_u(0.0) == 2.0
        assert c.denormalize_u(1.0) == 10.0
        assert c.denormalize_u(0.5) == 6.0
        alpha = 7.5
        assert abs(c.denormalize_u(c.normalize_alpha(alpha)) - alpha) < 1e-10

    def test_stall_detection_relative(self):
        """停滞检测按相对比例：降幅占当前 L1 的比例低于 stall_rel 记停滞"""
        c = _make(stall_rel=0.02)
        assert c.stall_blocks == 0

        # current_L1=1.0，改善 0.01 → 相对 1% < 2% → 停滞
        c.update_stall_status(0.01, 1.0)
        assert c.stall_blocks == 1

        # 改善 0.015 → 1.5% < 2% → 继续停滞
        c.update_stall_status(0.015, 1.0)
        assert c.stall_blocks == 2

        # 改善 0.05 → 5% > 2% → 重置
        c.update_stall_status(0.05, 1.0)
        assert c.stall_blocks == 0

    def test_stall_detection_scale_invariant(self):
        """同一相对比例在不同 L1 量级下判定一致（量纲无关）"""
        c = _make(stall_rel=0.02)
        # L1=0.005 的小表：改善 0.00005 → 1% → 停滞
        c.update_stall_status(0.00005, 0.005)
        assert c.stall_blocks == 1
        # L1=0.03 的大表：改善 0.0003 → 1% → 同样停滞
        c.update_stall_status(0.0003, 0.03)
        assert c.stall_blocks == 2
        # 大表改善 0.003 → 10% → 重置
        c.update_stall_status(0.003, 0.03)
        assert c.stall_blocks == 0

    def test_stall_zero_division_guard(self):
        """current_L1=0 时不崩（兜底分母）"""
        c = _make(stall_rel=0.02)
        c.update_stall_status(0.0, 0.0)  # 不应抛异常
        assert c.stall_blocks == 1

    def test_trigger_probe_basic(self):
        """探测触发条件：连续 P 块停滞"""
        c = _make(stall_rel=0.02, P=3)
        assert not c.should_trigger_probe()
        c.update_stall_status(0.01, 1.0)
        assert not c.should_trigger_probe()
        c.update_stall_status(0.01, 1.0)
        assert not c.should_trigger_probe()
        c.update_stall_status(0.01, 1.0)
        assert c.should_trigger_probe()

    def test_trigger_probe_with_cooldown(self):
        """冷却期内不触发探测（冷却由 register_probe_outcome 启动）"""
        c = _make(stall_rel=0.02, P=3, C=2)
        for _ in range(3):
            c.update_stall_status(0.01, 1.0)
        assert c.should_trigger_probe()

        # 探测结算：启动冷却 C=2，重置停滞
        c.register_probe_outcome(winner_L1=0.5)
        assert c.cooldown_remaining == 2
        assert c.stall_blocks == 0

        # 冷却期内即使停滞也不触发
        c.update_stall_status(0.01, 1.0)
        assert c.cooldown_remaining == 1
        assert c.stall_blocks == 1
        assert not c.should_trigger_probe()

        c.update_stall_status(0.01, 1.0)
        assert c.cooldown_remaining == 0
        assert c.stall_blocks == 2
        assert not c.should_trigger_probe()

        # 冷却结束后可再触发
        c.update_stall_status(0.01, 1.0)
        assert c.stall_blocks == 3
        assert c.should_trigger_probe()

    def test_create_probe_branches(self):
        """三岔路分支创建"""
        c = _make()
        branches = c.create_probe_branches(6.0)
        assert [d for d, _ in branches] == ["DOWN", "HOLD", "UP"]
        alphas = [a for _, a in branches]
        # u=0.5, s=0.1 → DOWN u=0.4→5.2, HOLD 6.0, UP u=0.6→6.8
        assert abs(alphas[0] - 5.2) < 1e-10
        assert abs(alphas[1] - 6.0) < 1e-10
        assert abs(alphas[2] - 6.8) < 1e-10

    def test_create_probe_branches_boundary(self):
        """边界分支不越界"""
        c = _make()
        alphas = [a for _, a in c.create_probe_branches(2.5)]
        assert alphas[0] >= 2.0
        alphas = [a for _, a in c.create_probe_branches(9.5)]
        assert alphas[2] <= 10.0

    def test_unique_branch_alphas_middle(self):
        """毛病一去重：中间 α 三条互不相同 → 三组"""
        c = _make()
        branches = c.create_probe_branches(6.0)
        groups = c.unique_branch_alphas(branches)
        assert len(groups) == 3
        # 每组一个方向
        assert all(len(dirs) == 1 for _, dirs in groups)

    def test_unique_branch_alphas_at_lower_bound(self):
        """毛病一去重：α 在下界时 DOWN 与 HOLD 同 α → 归为一组"""
        c = _make()
        branches = c.create_probe_branches(2.0)  # u=0, DOWN=HOLD=2.0
        groups = c.unique_branch_alphas(branches)
        assert len(groups) == 2  # {2.0: [DOWN, HOLD], up_alpha: [UP]}
        alpha0, dirs0 = groups[0]
        assert alpha0 == 2.0
        assert dirs0 == ["DOWN", "HOLD"]

    def test_unique_branch_alphas_at_upper_bound(self):
        """毛病一去重：α 在上界时 UP 与 HOLD 同 α → 归为一组"""
        c = _make()
        branches = c.create_probe_branches(10.0)  # u=1, UP=HOLD=10.0
        groups = c.unique_branch_alphas(branches)
        assert len(groups) == 2  # {down_alpha: [DOWN], 10.0: [HOLD, UP]}
        # HOLD 与 UP 同组
        for alpha, dirs in groups:
            if alpha == 10.0:
                assert set(dirs) == {"HOLD", "UP"}

    def test_select_winner_by_lowest_L1(self):
        """胜者选择：final_L1 最低者胜"""
        c = _make()
        # branch_results: (direction, improvement, final_L1, alpha)
        branch_results = [
            ("DOWN", 0.050, 100.0, 2.5),
            ("HOLD", 0.020, 90.0, 5.0),
            ("UP", 0.015, 80.0, 7.5),
        ]
        winner_dir, winner_alpha = c.select_winner(branch_results)
        assert winner_dir == "UP"      # final_L1=80 最低
        assert winner_alpha == 7.5

    def test_select_winner_hold_tiebreak(self):
        """胜者选择：final_L1 相同时偏 HOLD"""
        c = _make()
        branch_results = [
            ("DOWN", 0.008, 100.0, 2.5),
            ("HOLD", 0.008, 100.0, 5.0),
            ("UP", 0.008, 100.0, 7.5),
        ]
        winner_dir, winner_alpha = c.select_winner(branch_results)
        assert winner_dir == "HOLD"
        assert winner_alpha == 5.0

    def test_register_probe_outcome_improve_resets_patience(self):
        """刷新历史最好 L1 → 耐心计数清零"""
        c = _make(patience=3)
        c.register_probe_outcome(winner_L1=1.0)   # 首次，刷新
        assert c.no_improve_probes == 0
        assert c.best_L1_ever == 1.0
        c.register_probe_outcome(winner_L1=0.8)   # 更低，刷新
        assert c.no_improve_probes == 0
        assert c.best_L1_ever == 0.8

    def test_register_probe_outcome_no_improve_counts(self):
        """未刷新历史最好 L1 → 耐心计数累加"""
        c = _make(patience=3)
        c.register_probe_outcome(winner_L1=1.0)   # 刷新
        assert c.no_improve_probes == 0
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新（相等不算进步）
        assert c.no_improve_probes == 1
        c.register_probe_outcome(winner_L1=1.2)   # 更差
        assert c.no_improve_probes == 2

    def test_empirical_plateau_uses_patience(self):
        """经验平台：连续 patience 次未刷新才判收敛"""
        c = _make(patience=3)
        c.register_probe_outcome(winner_L1=1.0)   # 刷新
        assert not c.is_empirical_plateau()
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新 1
        assert not c.is_empirical_plateau()
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新 2
        assert not c.is_empirical_plateau()
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新 3 → 达到 patience
        assert c.is_empirical_plateau()

    def test_empirical_plateau_reset_by_improvement(self):
        """中途刷新会打断连续未刷新，不判平台"""
        c = _make(patience=3)
        c.register_probe_outcome(winner_L1=1.0)   # 刷新
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新 1
        c.register_probe_outcome(winner_L1=1.0)   # 未刷新 2
        c.register_probe_outcome(winner_L1=0.5)   # 刷新 → 清零
        assert c.no_improve_probes == 0
        assert not c.is_empirical_plateau()

    def test_record_probe(self):
        """探测历史记录（新字段 no_improve_probes，不再有 all_failed）"""
        c = _make()
        branches = [("DOWN", 5.2), ("HOLD", 6.0), ("UP", 6.8)]
        improvements = [0.050, 0.020, 0.015]
        c.record_probe(
            round_num=100,
            trigger_alpha=6.0,
            branches=branches,
            branch_improvements=improvements,
            winner_direction="DOWN",
            winner_alpha=5.2,
        )
        assert len(c.probe_history) == 1
        rec = c.probe_history[0]
        assert rec["round"] == 100
        assert rec["trigger_alpha"] == 6.0
        assert rec["winner"] == "DOWN"
        assert rec["winner_alpha"] == 5.2
        assert "no_improve_probes" in rec
        assert "all_failed" not in rec
        assert len(rec["branches"]) == 3
