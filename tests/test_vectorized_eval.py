"""
测试向量化+分块查询评价（vectorized_eval）的正确性与一致性

核心验证：新向量化路径与旧逐查询路径（evaluate_table + compute_fitness）
结果一致，作为可复现铁律的锚点。

要点：
1. 计数与旧 evaluate_table 逐元素相同（nltcs 全==、toy 含 >=/between/字符串列）
2. fitness 与旧 compute_fitness 数值一致（numpy 逐位，cuda float32 极小差）
3. 非全 1 权重用例：权重不丢、不算错
4. σ≠0 噪声残差用例：噪声接口正确传递
5. 不同 batch_size 结果一致（分块不改变结果）
6. 回退组：未向量化算子走慢路径、结果对、有提醒
7. run_evolution 端到端：vectorized 与 legacy 逐位一致（numpy）
"""
import numpy as np
import pandas as pd
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries, load_data, evaluate_table
from table_diffevo.objective import compute_residual
from table_diffevo.fitness import compute_fitness
from table_diffevo.vectorized_eval import evaluate_vectorized


def _devices():
    devs = ["numpy"]
    if CUDA_AVAILABLE:
        devs.append("cuda")
    elif TORCH_AVAILABLE:
        devs.append("cpu")
    return devs


@pytest.fixture
def toy():
    schema = load_schema("configs/test_300x10/schema.yaml")
    queries = load_queries("configs/test_300x10/measured_50query.json")
    df = load_data("data/test_300x10/test_300x10.csv")
    target = np.array([q["result"] for q in queries], dtype=float)
    return schema, queries, df, target


class TestCountsMatchLegacy:
    """计数与旧 evaluate_table 逐元素相同"""

    @pytest.mark.parametrize("device", _devices())
    def test_toy_counts_match(self, toy, device):
        """toy 数据（含 >=/between/字符串类别列）计数逐元素相同"""
        schema, queries, df, target = toy
        q_old = evaluate_table(df, queries)
        q_new, _, _ = evaluate_vectorized(
            df, queries, schema, batch_size=16, device=device,
            want_fitness=False, verbose=False,
        )
        np.testing.assert_array_equal(q_old, q_new)

    @pytest.mark.parametrize("device", _devices())
    def test_toy_counts_with_fitness_path(self, toy, device):
        """want_fitness=True 时计数也与旧路径相同"""
        schema, queries, df, target = toy
        q_old = evaluate_table(df, queries)
        q_new, _, _ = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            batch_size=16, device=device, want_fitness=True, verbose=False,
        )
        np.testing.assert_array_equal(q_old, q_new)


class TestFitnessMatchLegacy:
    """fitness 与旧 compute_fitness 数值一致"""

    @pytest.mark.parametrize("device", _devices())
    def test_toy_fitness_match(self, toy, device):
        """toy fitness 与旧路径一致（numpy 逐位，torch float32 极小差）"""
        schema, queries, df, target = toy
        q_old = evaluate_table(df, queries)
        resid = compute_residual(target, q_old, len(df))
        fit_old = compute_fitness(df, queries, resid, q_old)

        _, _, fit_new = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            batch_size=16, device=device, want_fitness=True, verbose=False,
        )
        atol = 1e-9 if device == "numpy" else 1e-3
        np.testing.assert_allclose(fit_old, fit_new, atol=atol)

    def test_residual_returned_matches(self, toy):
        """返回的残差与 compute_residual 一致"""
        schema, queries, df, target = toy
        q_old = evaluate_table(df, queries)
        resid_old = compute_residual(target, q_old, len(df))
        _, resid_new, _ = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            device="numpy", want_fitness=True, verbose=False,
        )
        np.testing.assert_allclose(resid_old, resid_new, atol=1e-12)


class TestWeights:
    """非全 1 权重：权重不丢、不算错（锁住权重接口）"""

    def test_nonuniform_weights_match_legacy(self, toy):
        """自定义权重下 fitness 与旧 compute_fitness(weights=...) 一致"""
        schema, queries, df, target = toy
        m = len(queries)
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.5, 2.0, size=m)  # 非全 1 权重

        q_old = evaluate_table(df, queries)
        resid = compute_residual(target, q_old, len(df))
        fit_old = compute_fitness(df, queries, resid, q_old, weights=weights)

        _, _, fit_new = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            weights=weights, device="numpy", want_fitness=True, verbose=False,
        )
        np.testing.assert_allclose(fit_old, fit_new, atol=1e-9)


class TestNoise:
    """σ≠0 噪声残差：噪声接口正确传递（为 DP 阶段铺路）"""

    def test_sigma_kappa_match_legacy(self, toy):
        """σ≠0 时 fitness 与旧路径（compute_residual 带 σ/κ）一致"""
        schema, queries, df, target = toy
        m = len(queries)
        sigma = np.full(m, 5.0)  # 非零噪声
        kappa = 1.0

        q_old = evaluate_table(df, queries)
        resid = compute_residual(target, q_old, len(df), sigma=sigma, kappa=kappa)
        fit_old = compute_fitness(df, queries, resid, q_old)

        _, resid_new, fit_new = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            sigma=sigma, kappa=kappa, device="numpy", want_fitness=True,
            verbose=False,
        )
        np.testing.assert_allclose(resid, resid_new, atol=1e-12)
        np.testing.assert_allclose(fit_old, fit_new, atol=1e-9)


class TestBatchSizeInvariance:
    """不同 batch_size 结果一致（分块不改变结果）"""

    @pytest.mark.parametrize("bs", [1, 7, 50, 1000])
    def test_counts_and_fitness_invariant(self, toy, bs):
        schema, queries, df, target = toy
        q_ref, r_ref, f_ref = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            batch_size=1, device="numpy", want_fitness=True, verbose=False,
        )
        q, r, f = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            batch_size=bs, device="numpy", want_fitness=True, verbose=False,
        )
        np.testing.assert_array_equal(q_ref, q)
        np.testing.assert_allclose(f_ref, f, atol=1e-12)


class TestFallback:
    """回退组：未向量化算子走慢路径、结果对、有提醒

    构造手法：用 monkeypatch 临时把 '>=' 移出白名单，模拟"出现了快路径不支持
    的算子"。含 '>=' 的查询因此进回退组走旧 evaluate_table，结果应仍与全旧路径
    一致，并打印提醒。这样无需真的引入 eval_condition 不支持的新算子。
    """

    def test_fallback_matches_and_warns(self, toy, monkeypatch, capsys):
        """
        临时把 '>=' 移出白名单，含 '>=' 的查询应走回退（旧 evaluate_table），
        结果仍与全旧路径一致，并打印提醒。
        """
        import table_diffevo.vectorized_eval as ve
        schema, queries, df, target = toy

        # 找到含 '>=' 的查询，确保确实有回退组
        has_ge = any(
            c["operator"] == ">=" for q in queries for c in q["conditions"]
        )
        assert has_ge, "toy 查询里应有 >= 算子"

        # 缩小白名单：'>=' 变成未向量化 → 走回退
        monkeypatch.setattr(ve, "VECTORIZED_OPS", {"==", "between"})

        q_old = evaluate_table(df, queries)
        q_new, _, _ = evaluate_vectorized(
            df, queries, schema, batch_size=16, device="numpy",
            want_fitness=False, verbose=True,
        )
        np.testing.assert_array_equal(q_old, q_new)

        out = capsys.readouterr().out
        assert "未向量化算子" in out and ">=" in out

    def test_fallback_fitness_correct(self, toy, monkeypatch):
        """回退组的 fitness 贡献也正确（与全旧路径一致）"""
        import table_diffevo.vectorized_eval as ve
        schema, queries, df, target = toy

        q_old = evaluate_table(df, queries)
        resid = compute_residual(target, q_old, len(df))
        fit_old = compute_fitness(df, queries, resid, q_old)

        monkeypatch.setattr(ve, "VECTORIZED_OPS", {"==", "between"})
        _, _, fit_new = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            device="numpy", want_fitness=True, verbose=False,
        )
        np.testing.assert_allclose(fit_old, fit_new, atol=1e-9)


class TestEndToEnd:
    """run_evolution 端到端：vectorized 与 legacy 逐位一致"""

    def test_vectorized_equals_legacy(self, toy):
        from table_diffevo.evolution import run_evolution
        schema, queries, df, target = toy

        _, d_leg = run_evolution(
            target, queries, schema, n_records=300, n_rounds=15, seed=0,
            device="numpy", eval_method="legacy",
        )
        _, d_vec = run_evolution(
            target, queries, schema, n_records=300, n_rounds=15, seed=0,
            device="numpy", eval_method="vectorized",
        )
        assert d_leg["loss_history"] == d_vec["loss_history"]
        assert d_leg["accept_history"] == d_vec["accept_history"]
        assert d_leg["best_loss"] == d_vec["best_loss"]

    @pytest.mark.parametrize("bs", [1, 10, 500])
    def test_batch_size_end_to_end_invariant(self, toy, bs):
        from table_diffevo.evolution import run_evolution
        schema, queries, df, target = toy
        _, d = run_evolution(
            target, queries, schema, n_records=300, n_rounds=10, seed=0,
            device="numpy", eval_method="vectorized", batch_size=bs,
        )
        _, d_ref = run_evolution(
            target, queries, schema, n_records=300, n_rounds=10, seed=0,
            device="numpy", eval_method="vectorized", batch_size=256,
        )
        assert d["loss_history"] == d_ref["loss_history"]
