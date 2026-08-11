"""
接受规则模块的单元测试
"""
import pytest
import numpy as np
from table_diffevo.acceptance import check_acceptance, _check_A0, _check_A1


class TestA0Rule:
    """测试 A0 规则（平方 loss 主判）"""

    def test_a0_accepts_improvement(self):
        """Q 改善 → 接受"""
        target = np.array([100.0, 50.0])
        current = np.array([90.0, 60.0])  # Q = 100 + 100 = 200
        candidate = np.array([95.0, 55.0])  # Q = 25 + 25 = 50

        accept, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )

        assert accept is True
        assert delta_Q < 0  # Q 改善

    def test_a0_rejects_worsening(self):
        """Q 恶化 → 拒绝"""
        target = np.array([100.0, 50.0])
        current = np.array([95.0, 55.0])  # Q = 50
        candidate = np.array([90.0, 60.0])  # Q = 200

        accept, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )

        assert accept is False
        assert delta_Q > 0  # Q 恶化

    def test_a0_accepts_tie(self):
        """Q 持平（delta_Q = 0）→ 拒绝（严格改善口径，不满足 < -eps_Q）"""
        target = np.array([100.0, 50.0])
        current = np.array([95.0, 55.0])
        candidate = np.array([95.0, 55.0])  # 完全相同

        accept, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )

        assert accept is False  # delta_Q=0 不满足 < 0 → 拒绝平局
        assert delta_Q == 0.0

    def test_a0_epsilon_boundary(self):
        """测试 eps_Q 严格改善阈值：仅 Q 改善超过 eps_Q 才接受"""
        target = np.array([100.0])
        current = np.array([90.0])   # Q = 0.5*100 = 50
        candidate = np.array([95.0])  # Q = 0.5*25 = 12.5，delta_Q = -37.5（改善）

        # eps_Q=30 → delta_Q=-37.5 < -30 → 改善超过阈值 → 接受
        accept_loose, _, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=30.0
        )
        assert delta_Q == pytest.approx(-37.5)
        assert accept_loose is True

        # eps_Q=40 → delta_Q=-37.5 < -40 为假 → 改善不足阈值 → 拒绝
        accept_tight = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=40.0
        )[0]
        assert accept_tight is False


class TestA1Rule:
    """测试 A1 规则（L1 主判 + Q 平局判）"""

    def test_a1_accepts_l1_improvement(self):
        """L1 改善（无论 Q）→ 接受"""
        target = np.array([100.0, 50.0])
        current = np.array([80.0, 60.0])  # L1 = (20+10)/2/200 = 0.075
        candidate = np.array([90.0, 55.0])  # L1 = (10+5)/2/200 = 0.0375, Q 更差

        accept, delta_L1, delta_Q = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=1e-5, eps_Q=0.0
        )

        assert accept is True
        assert delta_L1 < 0  # L1 改善
        # Q 可能恶化，但不影响接受

    def test_a1_rejects_l1_worsening(self):
        """L1 恶化（无论 Q）→ 拒绝"""
        target = np.array([100.0, 50.0])
        current = np.array([90.0, 55.0])  # L1 = 0.0375
        candidate = np.array([80.0, 60.0])  # L1 = 0.075, Q 可能更好

        accept, delta_L1, delta_Q = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=1e-5, eps_Q=0.0
        )

        assert accept is False
        assert delta_L1 > 0  # L1 恶化

    def test_a1_tie_uses_q(self):
        """L1 严格打平（delta_L1=0）→ 落入平局带 → 由 Q 严格改善裁决"""
        target = np.array([0.0, 0.0, 0.0])
        # current 与 candidate 的 |偏差| 之和相等（均为 6）→ L1 完全相同
        current = np.array([4.0, 1.0, 1.0])   # Q = 0.5*(16+1+1) = 9
        candidate = np.array([2.0, 2.0, 2.0])  # Q = 0.5*(4+4+4) = 6，delta_Q = -3

        # delta_L1 = 0 落入平局带 → Q 严格改善（delta_Q=-3 < 0）→ 接受
        accept_q_good, delta_L1, delta_Q = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=0.01, eps_Q=0.0
        )
        assert delta_L1 == pytest.approx(0.0)
        assert delta_Q == pytest.approx(-3.0)
        assert accept_q_good is True

        # 反向：L1 仍打平，但 Q 恶化（delta_Q=+3，不满足 < 0）→ 拒绝
        accept_q_bad, delta_L1_bad, delta_Q_bad = check_acceptance(
            "A1", target, candidate, current, n_records=200, eps_L1=0.01, eps_Q=0.0
        )
        assert delta_L1_bad == pytest.approx(0.0)
        assert delta_Q_bad == pytest.approx(3.0)
        assert accept_q_bad is False

    def test_a1_epsilon_l1_boundary(self):
        """测试 eps_L1 边界"""
        target = np.array([100.0])
        current = np.array([95.0])  # L1 = 5/100 = 0.05
        candidate = np.array([96.0])  # L1 = 4/100 = 0.04, delta_L1 = -0.01

        # eps_L1=0.005 → delta_L1=-0.01 < -0.005 → L1 严格改善 → 接受
        accept_tight = check_acceptance(
            "A1", target, current, candidate, n_records=100, eps_L1=0.005, eps_Q=0.0
        )[0]

        # eps_L1=0.015 → |delta_L1|=0.01 <= 0.015 → L1 打平 → 用 Q 裁决（Q 改善）→ 接受
        accept_loose = check_acceptance(
            "A1", target, current, candidate, n_records=100, eps_L1=0.015, eps_Q=0.0
        )[0]

        assert accept_tight is True  # L1 严格改善
        assert accept_loose is True  # L1 打平但 Q 改善


class TestQuadrants:
    """测试四象限分类"""

    def test_quadrant_1_l1_down_q_down(self):
        """象限 1：L1↓ Q↓（双赢）"""
        target = np.array([100.0, 50.0])
        current = np.array([80.0, 60.0])
        candidate = np.array([90.0, 55.0])

        accept_a0, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )
        accept_a1, _, _ = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=1e-5, eps_Q=0.0
        )

        assert delta_L1 < 0 and delta_Q < 0
        assert accept_a0 is True  # A0 接受
        assert accept_a1 is True  # A1 接受

    def test_quadrant_2_l1_down_q_up(self):
        """象限 2：L1↓ Q↑（A1 专属接受区，A0 应拒绝）

        构造真正冲突：L1 是绝对值和（对误差如何分布无所谓），Q 是平方
        （惩罚集中的大误差）。target=[0,0] 时，从集中误差 [6,1] 走向分散
        误差 [4,4]：绝对值和 7→8 使 L1↑，但反过来 [4,4]→[6,1] 时
        绝对值和 8→7 使 L1↓、平方和 32→37 使 Q↑，两者真正反向。
        """
        target = np.array([0.0, 0.0])
        current = np.array([4.0, 4.0])   # L1∝8,  Q=0.5*(16+16)=16
        candidate = np.array([6.0, 1.0])  # L1∝7,  Q=0.5*(36+1)=18.5

        accept_a0, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=0.0
        )
        accept_a1, delta_L1_a1, delta_Q_a1 = check_acceptance(
            "A1", target, current, candidate, n_records=100, eps_L1=1e-5, eps_Q=0.0
        )

        # 断言真正处在象限 2：L1 改善、Q 恶化
        assert delta_L1 < 0, f"期望 L1↓，实得 delta_L1={delta_L1}"
        assert delta_Q > 0, f"期望 Q↑，实得 delta_Q={delta_Q}"
        # 精确值：delta_L1 = (7-8)/2/100 = -0.005，delta_Q = 18.5-16 = +2.5
        assert delta_L1 == pytest.approx(-0.005)
        assert delta_Q == pytest.approx(2.5)
        # A0 看 Q → 拒绝；A1 看 L1 → 接受。这是两规则的分歧点。
        assert accept_a0 is False
        assert accept_a1 is True

    def test_quadrant_3_l1_up_q_down(self):
        """象限 3：L1↑ Q↓（A0 专属接受区，A1 应拒绝）

        与象限 2 完全对称：交换 current/candidate，从分散误差 [4,4]
        走向集中误差 [6,1]，L1 恶化但 Q 改善。
        """
        target = np.array([0.0, 0.0])
        current = np.array([6.0, 1.0])   # L1∝7,  Q=18.5
        candidate = np.array([4.0, 4.0])  # L1∝8,  Q=16

        accept_a0, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=0.0
        )
        accept_a1, _, _ = check_acceptance(
            "A1", target, current, candidate, n_records=100, eps_L1=1e-5, eps_Q=0.0
        )

        # 断言真正处在象限 3：L1 恶化、Q 改善
        assert delta_L1 > 0, f"期望 L1↑，实得 delta_L1={delta_L1}"
        assert delta_Q < 0, f"期望 Q↓，实得 delta_Q={delta_Q}"
        assert delta_L1 == pytest.approx(0.005)
        assert delta_Q == pytest.approx(-2.5)
        # A0 看 Q → 接受；A1 看 L1 → 拒绝。
        assert accept_a0 is True
        assert accept_a1 is False

    def test_quadrant_4_l1_up_q_up(self):
        """象限 4：L1↑ Q↑（双输）"""
        target = np.array([100.0, 50.0])
        current = np.array([95.0, 52.0])
        candidate = np.array([80.0, 60.0])

        accept_a0, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )
        accept_a1, _, _ = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=1e-5, eps_Q=0.0
        )

        assert delta_L1 > 0 and delta_Q > 0
        assert accept_a0 is False  # A0 拒绝
        assert accept_a1 is False  # A1 拒绝


class TestEdgeCases:
    """边界情况测试"""

    def test_perfect_match(self):
        """完美匹配（L1=0, Q=0）"""
        target = np.array([100.0, 50.0])
        current = target.copy()
        candidate = target.copy()

        accept_a0, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )
        accept_a1, _, _ = check_acceptance(
            "A1", target, current, candidate, n_records=200, eps_L1=1e-5, eps_Q=0.0
        )

        assert delta_L1 == 0.0
        assert delta_Q == 0.0
        assert accept_a0 is False  # delta_Q=0 不满足严格改善 → 拒绝
        assert accept_a1 is False  # L1 平局，delta_Q=0 不满足 Q 严格改善 → 拒绝

    def test_strict_epsilon_zero(self):
        """eps_Q=0 时的严格不等号：delta_Q=0 不满足 < 0 → 拒绝"""
        target = np.array([100.0])
        current = np.array([99.0])  # Q=0.5
        candidate = np.array([99.0])  # Q=0.5，完全相同

        accept = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=0.0
        )[0]

        assert accept is False  # delta_Q=0 不满足 < 0

    def test_unknown_rule_raises(self):
        """未知规则抛出异常"""
        target = np.array([100.0])
        current = np.array([95.0])
        candidate = np.array([90.0])

        with pytest.raises(ValueError, match="未知接受规则"):
            check_acceptance(
                "A2", target, current, candidate, n_records=100
            )

    def test_a0_delta_exactly_neg_epsilon_rejected(self):
        """A0 恰好 delta_Q == -eps_Q 时应拒绝（严格 < 而非 <=）"""
        target = np.array([100.0])
        current = np.array([90.0])   # Q = 0.5*100 = 50
        candidate = np.array([95.0])  # Q = 0.5*25 = 12.5，delta_Q = -37.5

        # eps_Q = 37.5 → delta_Q(-37.5) < -37.5 为假 → 恰在边界，拒绝
        accept_boundary, _, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=37.5
        )
        assert delta_Q == pytest.approx(-37.5)
        assert accept_boundary is False  # 恰好等于 -eps_Q，严格不等号拒绝

        # eps_Q 略小 → delta_Q < -eps_Q 成立 → 接受
        accept_inside, _, _ = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=37.4
        )
        assert accept_inside is True

    def test_a1_delta_l1_exactly_neg_epsilon_uses_q(self):
        """A1 恰好 delta_L1 == -eps_L1 时落入平局分支，由 Q 裁决

        _check_A1 分支：delta_L1 < -eps_L1 判负后（相等不满足严格 <），
        |delta_L1| <= eps_L1 为真（相等），进入 Q 平局判。验证边界归属平局区
        而非改善区。
        """
        target = np.array([100.0])
        current = np.array([90.0])   # L1 = 10/100 = 0.10
        candidate = np.array([95.0])  # L1 = 5/100 = 0.05，delta_L1 = -0.05

        # eps_L1 = 0.05 → delta_L1(-0.05) < -0.05 为假，|−0.05| <= 0.05 为真 → 用 Q 裁决
        # 此例 Q 改善（delta_Q<0）→ 平局分支接受
        accept_q_good, delta_L1, delta_Q = check_acceptance(
            "A1", target, current, candidate, n_records=100, eps_L1=0.05, eps_Q=0.0
        )
        assert delta_L1 == pytest.approx(-0.05)
        assert delta_Q < 0
        assert accept_q_good is True  # 边界归平局区，Q 严格改善 → 接受

        # 同一边界，但把 Q 也卡在恰好相等 → 平局区内 Q 不满足严格改善 → 拒绝
        # 构造 candidate 使 delta_Q == -eps_Q 恰好相等
        accept_q_boundary = check_acceptance(
            "A1", target, current, candidate, n_records=100,
            eps_L1=0.05, eps_Q=abs(delta_Q),
        )[0]
        assert accept_q_boundary is False  # delta_Q == -eps_Q，严格不等号拒绝


class TestDeltaCalculation:
    """测试差值计算正确性"""

    def test_delta_calculated_before_overwrite(self):
        """确保差值在覆盖前计算（通过返回值验证）"""
        target = np.array([100.0, 50.0])
        current = np.array([90.0, 60.0])
        candidate = np.array([95.0, 55.0])

        accept, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=200, eps_Q=0.0
        )

        # 如果在覆盖后计算，delta 会是 0
        assert delta_L1 != 0.0 or delta_Q != 0.0  # 至少一个非零

        # 手动验证计算正确
        from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
        expected_delta_L1 = (
            compute_normalized_l1(target, candidate, 200) -
            compute_normalized_l1(target, current, 200)
        )
        expected_delta_Q = (
            compute_squared_loss(target, candidate) -
            compute_squared_loss(target, current)
        )

        assert abs(delta_L1 - expected_delta_L1) < 1e-12
        assert abs(delta_Q - expected_delta_Q) < 1e-12

    def test_rejection_still_logs_delta(self):
        """拒绝的候选也要记录非零差值"""
        target = np.array([100.0])
        current = np.array([95.0])
        candidate = np.array([80.0])  # 明显更差

        accept, delta_L1, delta_Q = check_acceptance(
            "A0", target, current, candidate, n_records=100, eps_Q=0.0
        )

        assert accept is False
        assert delta_L1 > 0  # L1 恶化
        assert delta_Q > 0  # Q 恶化
