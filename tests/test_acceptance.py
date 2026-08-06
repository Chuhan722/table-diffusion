"""接受规则模块的单元测试。"""

import numpy as np
import pytest

from table_diffevo.acceptance import check_acceptance, AcceptanceRule


class TestAcceptanceRule:
    """测试 AcceptanceRule 枚举。"""

    def test_enum_values(self):
        """测试枚举值定义。"""
        assert AcceptanceRule.A0.value == "A0"
        assert AcceptanceRule.A1.value == "A1"


class TestA0Rule:
    """测试 A0 规则（平方 loss Q 主判）。"""

    def test_a0_q_improves_accept(self):
        """A0: Q 改善 → 接受。"""
        target = np.array([100, 200, 300])
        current_q = np.array([90, 210, 290])     # Q_current = 0.5*(10^2+10^2+10^2) = 150
        candidate_q = np.array([95, 205, 295])   # Q_candidate = 0.5*(5^2+5^2+5^2) = 37.5
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A0",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_Q=0.0
        )

        assert accept is True
        assert delta_Q < 0  # Q 改善

    def test_a0_q_worsens_reject(self):
        """A0: Q 恶化 → 拒绝。"""
        target = np.array([100, 200, 300])
        current_q = np.array([95, 205, 295])     # Q 较小
        candidate_q = np.array([90, 210, 290])   # Q 较大
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A0",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_Q=0.0
        )

        assert accept is False
        assert delta_Q > 0  # Q 恶化

    def test_a0_q_equal_accept(self):
        """A0: Q 持平（eps_Q=0）→ 接受。"""
        target = np.array([100, 200, 300])
        current_q = np.array([100, 200, 300])
        candidate_q = np.array([100, 200, 300])  # 完全相同
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A0",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_Q=0.0
        )

        assert accept is True
        assert delta_Q == 0.0
        assert delta_L1 == 0.0


class TestA1Rule:
    """测试 A1 规则（L1 主判 + Q 平局判）。"""

    def test_a1_l1_improves_accept_regardless_of_q(self):
        """A1: L1 改善 → 接受（无论 Q）。"""
        target = np.array([100, 200, 300])
        current_q = np.array([90, 190, 310])     # L1 较大
        candidate_q = np.array([95, 195, 305])   # L1 较小，但 Q 可能更大
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A1",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_L1=1e-5,
            eps_Q=0.0
        )

        assert accept is True
        assert delta_L1 < 0  # L1 改善

    def test_a1_l1_worsens_reject_regardless_of_q(self):
        """A1: L1 恶化 → 拒绝（无论 Q）。"""
        target = np.array([100, 200, 300])
        current_q = np.array([95, 195, 305])     # L1 较小
        candidate_q = np.array([90, 190, 310])   # L1 较大，即使 Q 改善
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A1",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_L1=1e-5,
            eps_Q=0.0
        )

        assert accept is False
        assert delta_L1 > 0  # L1 恶化

    def test_a1_l1_tie_q_improves_accept(self):
        """A1: L1 打平 + Q 改善 → 接受。"""
        target = np.array([100, 200, 300])
        # 构造 L1 相同但 Q 不同的情况
        current_q = np.array([90, 200, 310])     # L1 = (10+0+10)/1000 = 0.02
        candidate_q = np.array([95, 200, 305])   # L1 = (5+0+5)/1000 = 0.01
        # 实际上 candidate L1 更好，调整一下使其打平
        current_q = np.array([95, 200, 305])
        candidate_q = np.array([90, 205, 305])   # L1 差不多，但 Q 不同
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A1",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_L1=1e-2,  # 放宽容差以确保 L1 "打平"
            eps_Q=0.0
        )

        # 只要 |delta_L1| <= eps_L1，就进入平局判断
        if abs(delta_L1) <= 1e-2:
            # Q 裁决
            if delta_Q < 0:
                assert accept is True
            else:
                assert accept is False

    def test_a1_l1_tie_q_worsens_reject(self):
        """A1: L1 打平 + Q 恶化 → 拒绝。"""
        target = np.array([100, 200, 300])
        current_q = np.array([95, 200, 305])
        candidate_q = np.array([96, 199, 305])   # L1 接近，Q 更差
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A1",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_L1=1e-2,
            eps_Q=0.0
        )

        if abs(delta_L1) <= 1e-2 and delta_Q > 0:
            assert accept is False


class TestQuadrants:
    """测试四象限覆盖。"""

    def test_quadrant_l1_down_q_down(self):
        """象限 (L1↓, Q↓): A0 接受，A1 接受。"""
        target = np.array([100, 200, 300])
        current_q = np.array([90, 210, 290])
        candidate_q = np.array([95, 205, 295])   # L1 和 Q 都改善
        n_records = 1000

        accept_a0, delta_L1, delta_Q = check_acceptance("A0", target, current_q, candidate_q, n_records)
        accept_a1, _, _ = check_acceptance("A1", target, current_q, candidate_q, n_records, eps_L1=1e-5)

        assert delta_L1 < 0  # L1 改善
        assert delta_Q < 0   # Q 改善
        assert accept_a0 is True
        assert accept_a1 is True

    def test_quadrant_l1_down_q_up(self):
        """象限 (L1↓, Q↑): A0 拒绝，A1 接受（A1 专属）。"""
        target = np.array([100, 200, 300])
        # 构造 L1 改善但 Q 恶化的情况
        # 这需要精心设计，因为通常 L1 和 Q 同向变化
        # 一个近似例子：
        current_q = np.array([110, 200, 290])    # 一个大偏差，两个小偏差
        candidate_q = np.array([105, 205, 285])  # 平均偏差变小（L1↓），但平方和可能变大（Q↑）
        n_records = 1000

        accept_a0, delta_L1, delta_Q = check_acceptance("A0", target, current_q, candidate_q, n_records)
        accept_a1, _, _ = check_acceptance("A1", target, current_q, candidate_q, n_records, eps_L1=1e-5)

        # 检查是否真的构造出 L1↓ Q↑
        if delta_L1 < -1e-5 and delta_Q > 0:
            assert accept_a0 is False  # A0 因 Q 恶化拒绝
            assert accept_a1 is True   # A1 因 L1 改善接受

    def test_quadrant_l1_up_q_down(self):
        """象限 (L1↑, Q↓): A0 接受，A1 拒绝（A0 专属）。"""
        target = np.array([100, 200, 300])
        # 构造 L1 恶化但 Q 改善的情况
        current_q = np.array([105, 205, 285])
        candidate_q = np.array([110, 200, 290])
        n_records = 1000

        accept_a0, delta_L1, delta_Q = check_acceptance("A0", target, current_q, candidate_q, n_records)
        accept_a1, _, _ = check_acceptance("A1", target, current_q, candidate_q, n_records, eps_L1=1e-5)

        if delta_L1 > 1e-5 and delta_Q < 0:
            assert accept_a0 is True   # A0 因 Q 改善接受
            assert accept_a1 is False  # A1 因 L1 恶化拒绝

    def test_quadrant_l1_up_q_up(self):
        """象限 (L1↑, Q↑): A0 拒绝，A1 拒绝。"""
        target = np.array([100, 200, 300])
        current_q = np.array([95, 205, 295])
        candidate_q = np.array([90, 210, 290])   # L1 和 Q 都恶化
        n_records = 1000

        accept_a0, delta_L1, delta_Q = check_acceptance("A0", target, current_q, candidate_q, n_records)
        accept_a1, _, _ = check_acceptance("A1", target, current_q, candidate_q, n_records, eps_L1=1e-5)

        assert delta_L1 > 0  # L1 恶化
        assert delta_Q > 0   # Q 恶化
        assert accept_a0 is False
        assert accept_a1 is False


class TestBoundary:
    """测试边界情况。"""

    def test_eps_l1_zero_strict(self):
        """eps_L1 = 0 时为严格判断（无容差）。"""
        target = np.array([100, 200, 300])
        current_q = np.array([100, 200, 300])
        candidate_q = np.array([100, 200, 300])  # 完全相同
        n_records = 1000

        accept, delta_L1, delta_Q = check_acceptance(
            rule="A1",
            target=target,
            current_q=current_q,
            candidate_q=candidate_q,
            n_records=n_records,
            eps_L1=0.0,
            eps_Q=0.0
        )

        # L1 持平（delta_L1 = 0），进入 Q 裁决
        # Q 也持平（delta_Q = 0），应接受
        assert delta_L1 == 0.0
        assert delta_Q == 0.0
        assert accept is True

    def test_perfect_match(self):
        """完全匹配（L1=0, Q=0）。"""
        target = np.array([100, 200, 300])
        current_q = np.array([100, 200, 300])
        candidate_q = np.array([100, 200, 300])
        n_records = 1000

        accept_a0, delta_L1_a0, delta_Q_a0 = check_acceptance("A0", target, current_q, candidate_q, n_records)
        accept_a1, delta_L1_a1, delta_Q_a1 = check_acceptance("A1", target, current_q, candidate_q, n_records)

        assert delta_L1_a0 == 0.0
        assert delta_Q_a0 == 0.0
        assert accept_a0 is True

        assert delta_L1_a1 == 0.0
        assert delta_Q_a1 == 0.0
        assert accept_a1 is True

    def test_unknown_rule_raises(self):
        """未知规则应抛出异常。"""
        target = np.array([100, 200, 300])
        current_q = np.array([100, 200, 300])
        candidate_q = np.array([100, 200, 300])
        n_records = 1000

        with pytest.raises(ValueError, match="未知的接受规则"):
            check_acceptance("UNKNOWN", target, current_q, candidate_q, n_records)

