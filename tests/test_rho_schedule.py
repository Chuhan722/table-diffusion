"""
测试 rho 衰减调度（Issue #29）

四个测试组：
1. 等价门（equivalence gate）：rho_schedule=None 时结果与不传调度参数完全一致
2. _compute_rho_t 线性公式：边界值 + 中间值
3. _compute_rho_t 指数公式：边界值 + 中间值
4. rho_t_history 长度与取值范围
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.evolution import run_evolution, _compute_rho_t


# ---------------------------------------------------------------------------
# 公共 fixture：小型 schema / queries，专为快速单测设计（20 轮）
# ---------------------------------------------------------------------------

def make_schema():
    return Schema([
        AttributeBlock(name="edu", type="categorical", description="学历",
                       values=["low", "mid", "high"]),
        AttributeBlock(name="job", type="categorical", description="职业",
                       values=["a", "b"]),
    ])


def make_queries():
    return [
        {"conditions": [{"attribute": "edu", "operator": "==", "value": "high"}]},
        {"conditions": [{"attribute": "job", "operator": "==", "value": "a"}]},
    ]


def run_small(seed=0, **extra_kwargs):
    """封装：跑 20 轮 / 50 行的最小演化，用于等价和 history 测试。"""
    schema = make_schema()
    queries = make_queries()
    target = np.array([20, 25])
    return run_evolution(
        target, queries, schema,
        n_records=50,
        n_rounds=20,
        seed=seed,
        rho=0.1,
        distance_mode='geometric',
        **extra_kwargs,
    )


def _table_sha256(df: pd.DataFrame) -> str:
    """将 DataFrame 序列化为稳定字节流后取 SHA-256，用于位级等价比对。"""
    raw = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 1. 等价门（Equivalence gate）
# ---------------------------------------------------------------------------

class TestEquivalenceGate:
    """rho_schedule=None 时，新参数不改变任何结果——位级等价。"""

    def test_table_identical(self):
        """最优合成表的 SHA-256 与不传调度参数时完全一致。"""
        s_old, _ = run_small(seed=7)
        s_new, _ = run_small(seed=7,
                              rho_schedule=None,
                              rho_max=0.1,
                              rho_min=0.1)
        assert _table_sha256(s_old) == _table_sha256(s_new)

    def test_loss_history_identical(self):
        """loss 轨迹列表逐元素相同。"""
        _, d_old = run_small(seed=7)
        _, d_new = run_small(seed=7,
                              rho_schedule=None,
                              rho_max=0.1,
                              rho_min=0.1)
        assert d_old["loss_history"] == d_new["loss_history"]

    def test_frame_identical(self):
        """pd.testing.assert_frame_equal 级别的位级等价。"""
        s_old, _ = run_small(seed=13)
        s_new, _ = run_small(seed=13,
                              rho_schedule=None,
                              rho_max=0.05,
                              rho_min=0.05)
        pd.testing.assert_frame_equal(s_old, s_new)


# ---------------------------------------------------------------------------
# 2. _compute_rho_t — 线性公式
# ---------------------------------------------------------------------------

class TestComputeRhoTLinear:
    """线性调度：边界值 + 中间值正确。"""

    RHO_MAX = 0.10
    RHO_MIN = 0.02
    N = 11  # n_rounds

    def _call(self, t):
        return _compute_rho_t(t, self.N, rho=0.05,
                               rho_schedule='linear',
                               rho_max=self.RHO_MAX,
                               rho_min=self.RHO_MIN)

    def test_first_round_equals_rho_max(self):
        """t=0 → rho_t == rho_max。"""
        assert self._call(0) == pytest.approx(self.RHO_MAX)

    def test_last_round_equals_rho_min(self):
        """t=N-1 → rho_t == rho_min。"""
        assert self._call(self.N - 1) == pytest.approx(self.RHO_MIN)

    def test_midpoint(self):
        """t=(N-1)/2 → 线性中点 = (rho_max + rho_min) / 2。"""
        mid = (self.N - 1) // 2  # t=5 when N=11
        expected = (self.RHO_MAX + self.RHO_MIN) / 2
        assert self._call(mid) == pytest.approx(expected)

    def test_monotone_decreasing(self):
        """rho_max > rho_min 时全程单调不增。"""
        vals = [self._call(t) for t in range(self.N)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_none_schedule_ignores_max_min(self):
        """rho_schedule=None 时返回固定 rho，与 rho_max/rho_min 无关。"""
        fixed = 0.05
        v = _compute_rho_t(3, self.N, rho=fixed,
                           rho_schedule=None,
                           rho_max=self.RHO_MAX,
                           rho_min=self.RHO_MIN)
        assert v == pytest.approx(fixed)


# ---------------------------------------------------------------------------
# 3. _compute_rho_t — 指数公式
# ---------------------------------------------------------------------------

class TestComputeRhoTExponential:
    """指数调度：边界值 + 中间值正确。"""

    RHO_MAX = 0.20
    RHO_MIN = 0.01
    N = 21

    def _call(self, t):
        return _compute_rho_t(t, self.N, rho=0.05,
                               rho_schedule='exponential',
                               rho_max=self.RHO_MAX,
                               rho_min=self.RHO_MIN)

    def test_first_round_equals_rho_max(self):
        """t=0 → rho_t == rho_max（指数 ^ 0 = 1）。"""
        assert self._call(0) == pytest.approx(self.RHO_MAX)

    def test_last_round_equals_rho_min(self):
        """t=N-1 → rho_t == rho_min（指数 ^ 1 = rho_min/rho_max）。"""
        assert self._call(self.N - 1) == pytest.approx(self.RHO_MIN)

    def test_midpoint_geometric_mean(self):
        """中点 t=(N-1)/2 → 几何均值 sqrt(rho_max * rho_min)。"""
        mid = (self.N - 1) // 2  # t=10
        expected = (self.RHO_MAX * self.RHO_MIN) ** 0.5
        assert self._call(mid) == pytest.approx(expected, rel=1e-9)

    def test_monotone_decreasing(self):
        """rho_max > rho_min 时全程单调不增。"""
        vals = [self._call(t) for t in range(self.N)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_single_round_n1(self):
        """n_rounds=1 时 progress 强制为 1.0，返回 rho_min。"""
        v = _compute_rho_t(0, 1, rho=0.05,
                           rho_schedule='exponential',
                           rho_max=0.10,
                           rho_min=0.02)
        assert v == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 4. rho_t_history 字段
# ---------------------------------------------------------------------------

class TestRhoTHistory:
    """diagnostics['rho_t_history'] 长度和取值正确。"""

    def test_history_length_equals_rounds_run(self):
        """rho_t_history 长度 == rounds_run。"""
        _, diag = run_small(seed=0)
        assert len(diag["rho_t_history"]) == diag["rounds_run"]

    def test_fixed_schedule_all_equal_rho(self):
        """rho_schedule=None → 每轮 rho_t 均等于 rho。"""
        _, diag = run_small(seed=0, rho_schedule=None,
                            rho_max=0.1, rho_min=0.1)
        for v in diag["rho_t_history"]:
            assert v == pytest.approx(0.1)

    def test_linear_history_bounds(self):
        """linear 调度：所有 rho_t ∈ [rho_min, rho_max]。"""
        rho_max, rho_min = 0.15, 0.03
        _, diag = run_small(seed=0,
                            rho_schedule='linear',
                            rho_max=rho_max, rho_min=rho_min)
        for v in diag["rho_t_history"]:
            assert rho_min - 1e-12 <= v <= rho_max + 1e-12

    def test_linear_history_matches_formula(self):
        """linear 调度：每轮 rho_t_history[t] 与 _compute_rho_t 公式严格一致。"""
        rho_max, rho_min = 0.15, 0.03
        _, diag = run_small(seed=0,
                            rho_schedule='linear',
                            rho_max=rho_max, rho_min=rho_min)
        n_rounds = 20  # run_small 默认值
        for t, v in enumerate(diag["rho_t_history"]):
            expected = _compute_rho_t(t, n_rounds, rho=0.1,
                                      rho_schedule='linear',
                                      rho_max=rho_max, rho_min=rho_min)
            assert v == pytest.approx(expected), f"round {t}: {v} != {expected}"

    def test_exponential_history_bounds(self):
        """exponential 调度：所有 rho_t ∈ [rho_min, rho_max]。"""
        rho_max, rho_min = 0.20, 0.02
        _, diag = run_small(seed=0,
                            rho_schedule='exponential',
                            rho_max=rho_max, rho_min=rho_min)
        for v in diag["rho_t_history"]:
            assert rho_min - 1e-12 <= v <= rho_max + 1e-12

    def test_params_stored_in_diagnostics(self):
        """diagnostics['params'] 中记录了 rho_schedule / rho_max / rho_min。"""
        _, diag = run_small(seed=0,
                            rho_schedule='linear',
                            rho_max=0.12, rho_min=0.03)
        p = diag["params"]
        assert p["rho_schedule"] == 'linear'
        assert p["rho_max"] == pytest.approx(0.12)
        assert p["rho_min"] == pytest.approx(0.03)

    def test_none_schedule_params_are_none_in_diagnostics(self):
        """rho_schedule=None 时，diagnostics['params']['rho_max/min'] 为 None。"""
        _, diag = run_small(seed=0)
        p = diag["params"]
        assert p["rho_schedule"] is None
        assert p["rho_max"] is None
        assert p["rho_min"] is None


# ---------------------------------------------------------------------------
# 5. 参数验证
# ---------------------------------------------------------------------------

class TestValidation:
    """无效参数应抛出 ValueError。"""

    def _base_kwargs(self):
        return dict(
            schema=make_schema(),
            queries=make_queries(),
            target=np.array([20, 25]),
            n_records=50, n_rounds=5, seed=0,
        )

    def test_invalid_schedule_string(self):
        """未知调度名报错。"""
        with pytest.raises(ValueError, match="rho_schedule"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='cosine',
            )

    def test_rho_max_zero(self):
        """rho_max=0 不合法（必须为正数）。"""
        with pytest.raises(ValueError, match="rho_max"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='linear', rho_max=0.0, rho_min=0.01,
            )

    def test_rho_min_negative(self):
        """rho_min<0 不合法。"""
        with pytest.raises(ValueError, match="rho_min"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='linear', rho_max=0.1, rho_min=-0.01,
            )

    def test_exponential_equal_max_min(self):
        """exponential 调度要求 rho_max != rho_min。"""
        with pytest.raises(ValueError, match="exponential"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='exponential', rho_max=0.05, rho_min=0.05,
            )

    def test_rho_max_less_than_min(self):
        """rho_max < rho_min 违反 rho_min <= rho_max。"""
        with pytest.raises(ValueError, match="rho_min <= rho_max"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='linear', rho_max=0.02, rho_min=0.05,
            )

    def test_rho_max_exceeds_one(self):
        """rho_max > 1.0 违反上界（参与率不能超过 1）。"""
        with pytest.raises(ValueError, match="rho_max <= 1.0"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='linear', rho_max=1.5, rho_min=0.01,
            )

    def test_rho_min_not_positive(self):
        """rho_min=0 不合法（必须为正数）。"""
        with pytest.raises(ValueError, match="rho_min"):
            run_evolution(
                np.array([20, 25]), make_queries(), make_schema(),
                n_records=50, n_rounds=5, seed=0,
                rho_schedule='linear', rho_max=0.1, rho_min=0.0,
            )
