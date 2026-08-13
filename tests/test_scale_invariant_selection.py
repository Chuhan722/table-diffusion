"""尺度不变选择（Issue #44 机制迭代）的单元与集成测试。"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.sampling import compute_sampling_probs
from table_diffevo.schema import AttributeBlock, Schema

DELTA = 0.05
LAMBDA = 0.5


def _distances_for_log_s(log_s: np.ndarray) -> np.ndarray:
    """给定目标 log_s 矩阵反解距离，使 s = delta + (1-delta)(1-d) = exp(log_s)。"""
    s = np.exp(log_s)
    assert np.all(s >= DELTA - 1e-12) and np.all(s <= 1.0 + 1e-12)
    return 1.0 - (s - DELTA) / (1.0 - DELTA)


class TestScaleInvarianceProperty:
    def test_probs_invariant_to_row_spread_scaling(self):
        # 构造两组输入：fitness 全等（f 项恒 1，log_f=0），log_s 第二组
        # 是第一组围绕行均值收缩 0.4 倍。历史路径下 softmax 变平（概率改变），
        # 尺度不变路径下概率精确不变。
        rng = np.random.default_rng(7)
        base_log_s = rng.uniform(np.log(DELTA) * 0.8, -0.05, size=(5, 6))
        row_mean = base_log_s.mean(axis=1, keepdims=True)
        shrunk_log_s = row_mean + 0.4 * (base_log_s - row_mean)

        fitness = np.zeros(6)
        kwargs = dict(
            distance_mode="geometric", lambda_param=LAMBDA, alpha=8.0,
            delta=DELTA, device="numpy",
        )
        p_base_si = compute_sampling_probs(
            fitness, _distances_for_log_s(base_log_s),
            scale_invariant=True, **kwargs,
        )
        p_shrunk_si = compute_sampling_probs(
            fitness, _distances_for_log_s(shrunk_log_s),
            scale_invariant=True, **kwargs,
        )
        # eps 正则（std+1e-8）带来 ~1e-8 级偏差；对照的历史路径差异是
        # 1e-1 量级，1e-6 容差足以区分两种行为。
        assert p_base_si == pytest.approx(p_shrunk_si, abs=1e-6)

        p_base_legacy = compute_sampling_probs(
            fitness, _distances_for_log_s(base_log_s),
            scale_invariant=False, **kwargs,
        )
        p_shrunk_legacy = compute_sampling_probs(
            fitness, _distances_for_log_s(shrunk_log_s),
            scale_invariant=False, **kwargs,
        )
        assert not np.allclose(p_base_legacy, p_shrunk_legacy, atol=1e-6)
        # 收缩后历史路径概率更接近均匀（区分度衰减的正是这个模式）
        uniform = np.full_like(p_base_legacy, 1.0 / 6.0)
        assert (
            np.abs(p_shrunk_legacy - uniform).sum()
            < np.abs(p_base_legacy - uniform).sum()
        )

    def test_degenerate_row_falls_back_to_uniform(self):
        fitness = np.zeros(4)
        distances = np.full((3, 4), 0.5)
        probs = compute_sampling_probs(
            fitness, distances, distance_mode="geometric",
            lambda_param=LAMBDA, alpha=10.0, delta=DELTA,
            device="numpy", scale_invariant=True,
        )
        assert probs == pytest.approx(np.full((3, 4), 0.25))

    def test_default_false_matches_legacy(self):
        rng = np.random.default_rng(11)
        fitness = rng.normal(size=6)
        distances = rng.uniform(0.05, 0.95, size=(5, 6))
        kwargs = dict(
            distance_mode="geometric", lambda_param=LAMBDA, alpha=4.0,
            delta=DELTA, device="numpy",
        )
        p_default = compute_sampling_probs(fitness, distances, **kwargs)
        p_explicit = compute_sampling_probs(
            fitness, distances, scale_invariant=False, **kwargs,
        )
        assert p_default == pytest.approx(p_explicit)


def _tiny_problem():
    schema = Schema([
        AttributeBlock(name="a", type="categorical", description="a",
                       values=["0", "1"]),
        AttributeBlock(name="b", type="categorical", description="b",
                       values=["0", "1"]),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"}]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": "1"}]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "b", "operator": "==", "value": "1"}]},
    ]
    target = np.asarray([6.0, 2.0, 2.0])
    return schema, queries, target


def _table_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _run(**overrides):
    schema, queries, target = _tiny_problem()
    kwargs = dict(
        target=target,
        queries=queries,
        schema=schema,
        n_records=8,
        n_rounds=12,
        seed=3,
        log_every=-1,
    )
    kwargs.update(overrides)
    return run_evolution(**kwargs)


class TestEvolutionIntegration:
    def test_default_false_keeps_trajectory(self):
        best_default, diag_default = _run()
        best_explicit, diag_explicit = _run(selection_scale_invariant=False)
        assert diag_default["loss_history"] == diag_explicit["loss_history"]
        assert _table_hash(best_default) == _table_hash(best_explicit)
        assert diag_default["params"]["selection_scale_invariant"] is False

    def test_enabled_runs_and_records_params(self):
        _, diag = _run(selection_scale_invariant=True, tol=float("inf"))
        assert diag["params"]["selection_scale_invariant"] is True
        assert len(diag["loss_history"]) > 0

    def test_requires_geometric_mode(self):
        with pytest.raises(ValueError, match="selection_scale_invariant"):
            _run(selection_scale_invariant=True, distance_mode="linear")

    @pytest.mark.parametrize("bad", [1, "yes", 0.5])
    def test_rejects_non_bool(self, bad):
        with pytest.raises(ValueError, match="selection_scale_invariant"):
            _run(selection_scale_invariant=bad)


class TestThirdRoundReviewProtections:
    """第三轮审查要求的保护与不变式。"""

    def test_near_identical_scores_not_amplified(self):
        # 意见一.1：候选分数几乎相同（噪声级差异）时不得放大成极端偏好。
        fitness = np.zeros(5)
        rng = np.random.default_rng(3)
        base = np.full((4, 5), -0.7)
        noisy = base + rng.uniform(-1e-7, 1e-7, size=base.shape)
        probs = compute_sampling_probs(
            np.zeros(5), _distances_for_log_s(noisy),
            distance_mode="geometric", lambda_param=LAMBDA, alpha=16.0,
            delta=DELTA, device="numpy", scale_invariant=True,
            scale_invariant_min_spread=1e-3,
        )
        uniform = np.full_like(probs, 1.0 / 5.0)
        assert np.abs(probs - uniform).max() < 1e-2

    def test_amplification_bounded_by_min_spread(self):
        # 放大倍数有界：同一模式、离散度缩小 100 倍且低于下限时，
        # 概率必须比"无下限的完全尺度不变"更接近均匀。
        rng = np.random.default_rng(11)
        base = rng.uniform(np.log(DELTA) * 0.8, -0.05, size=(4, 6))
        row_mean = base.mean(axis=1, keepdims=True)
        tiny = row_mean + 1e-5 * (base - row_mean)  # 离散度 ~1e-5 << 1e-3
        kwargs = dict(
            distance_mode="geometric", lambda_param=LAMBDA, alpha=8.0,
            delta=DELTA, device="numpy", scale_invariant=True,
            scale_invariant_min_spread=1e-3,
        )
        p_base = compute_sampling_probs(
            np.zeros(6), _distances_for_log_s(base), **kwargs)
        p_tiny = compute_sampling_probs(
            np.zeros(6), _distances_for_log_s(tiny), **kwargs)
        uniform = np.full_like(p_tiny, 1.0 / 6.0)
        assert np.abs(p_tiny - uniform).sum() < 0.05 * np.abs(
            p_base - uniform).sum()

    def test_excluded_self_does_not_affect_other_donors(self):
        # 意见二.5：当前行不得通过标准化统计影响其余 donor 的相对概率。
        rng = np.random.default_rng(7)
        n = 6
        log_s = rng.uniform(np.log(DELTA) * 0.8, -0.05, size=(n, n))
        distances = _distances_for_log_s(log_s)
        # 极端化对角线（自身距离本应为 0：分数最大端点）
        distances_extreme = distances.copy()
        np.fill_diagonal(distances_extreme, 0.0)
        distances_mild = distances.copy()
        np.fill_diagonal(distances_mild, 0.9)
        kwargs = dict(
            distance_mode="geometric", lambda_param=LAMBDA, alpha=8.0,
            delta=DELTA, device="numpy", scale_invariant=True,
            exclude_self=True,
        )
        p_extreme = compute_sampling_probs(np.zeros(n), distances_extreme, **kwargs)
        p_mild = compute_sampling_probs(np.zeros(n), distances_mild, **kwargs)
        # 对角线取值不同不得改变其余 donor 的归一化概率
        assert p_extreme == pytest.approx(p_mild, abs=1e-12)
        assert np.diagonal(p_extreme) == pytest.approx(np.zeros(n))

    def test_numpy_torch_paths_agree(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(13)
        n = 8
        fitness = rng.normal(size=n)
        distances = rng.uniform(0.05, 0.95, size=(n, n))
        np.fill_diagonal(distances, 0.0)
        kwargs = dict(
            distance_mode="geometric", lambda_param=LAMBDA, alpha=16.0,
            delta=DELTA, scale_invariant=True,
            scale_invariant_min_spread=1e-3, exclude_self=True,
        )
        p_np = compute_sampling_probs(
            fitness, distances, device="numpy", **kwargs)
        p_torch = compute_sampling_probs(
            fitness, distances, device="cpu", **kwargs)
        assert p_np == pytest.approx(
            np.asarray(p_torch), abs=5e-6)

    def test_min_spread_validation(self):
        for bad in (0.0, -1e-3, float("nan"), float("inf"), True):
            with pytest.raises((ValueError, TypeError)):
                compute_sampling_probs(
                    np.zeros(4), np.full((3, 4), 0.5),
                    distance_mode="geometric", device="numpy",
                    scale_invariant=True, scale_invariant_min_spread=bad,
                )

    def test_evolution_min_spread_validation_and_params(self):
        _, diag = _run(
            selection_scale_invariant=True,
            selection_scale_invariant_min_spread=5e-3,
            tol=float("inf"),
        )
        assert diag["params"][
            "selection_scale_invariant_min_spread"] == pytest.approx(5e-3)
        assert diag["params"]["selection_scale_invariant"] is True
        for bad in (0.0, -1.0, float("nan"), True):
            with pytest.raises(ValueError):
                _run(
                    selection_scale_invariant=True,
                    selection_scale_invariant_min_spread=bad,
                )

    def test_donor_top_share_recorded_when_enabled(self):
        _, diag = _run(selection_scale_invariant=True, tol=float("inf"))
        shares = diag["donor_top_share_history"]
        assert len(shares) == len(diag["loss_history"])
        assert all(0.0 < v <= 1.0 for v in shares)
        _, diag_off = _run()
        assert diag_off["donor_top_share_history"] == []

    def test_best_loss_not_greater_than_final_state_loss(self):
        # 意见二.4：best 必须包含末轮接受后的最终状态。
        from table_diffevo.objective import compute_loss
        from table_diffevo.queries import evaluate_table
        schema, queries, target = _tiny_problem()
        _, diag = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=8, n_rounds=12, seed=3, log_every=-1,
            tol=float("inf"), return_final_table=True,
        )
        final_q = evaluate_table(diag["final_table"], queries)
        final_loss = float(compute_loss(target, final_q))
        assert diag["best_loss"] <= final_loss + 1e-9

    def test_synthetic_table_matches_source_shape(self):
        # 意见二.7：合成表行数与列名必须与源数据规格一致。
        schema, queries, target = _tiny_problem()
        _, diag = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=8, n_rounds=5, seed=3, log_every=-1,
            return_final_table=True,
        )
        table = diag["final_table"]
        assert len(table) == 8
        assert list(table.columns) == schema.attribute_names()
