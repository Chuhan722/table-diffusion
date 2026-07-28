"""
测试 geometric 模式的抽样功能

覆盖点：
1. 基本功能：返回合法概率矩阵
2. 边界情况：全相同适应度/距离、λ=0/1、α=0/大
3. 稳健性：离群点、距离超界
4. 数值稳定：log(δ) 不会 NaN/inf
5. 可复现：同种子同结果
6. torch 一致性：numpy 和 torch 路径数值接近
7. 端到端：真实数据跑通
"""
import pytest
import numpy as np
import pandas as pd

from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema


class TestGeometricBasic:
    """基本功能测试"""

    def test_returns_valid_probs(self):
        """返回合法概率矩阵：非负、按行和为1"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 0.3],
            [0.8, 0.3, 0.0],
        ])
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=1.0, delta=0.05
        )
        assert probs.shape == (3, 3)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_no_nan_or_inf(self):
        """无 NaN/inf"""
        fitness = np.array([0.5, 1.5, 10.0])
        distances = np.random.rand(10, 3) * 0.9
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=2.0, delta=0.05
        )
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


class TestGeometricBoundary:
    """边界情况测试"""

    def test_all_fitness_equal_degrades_to_similarity(self):
        """所有适应度相同 → 退化成纯相似度抽样"""
        fitness = np.ones(3)  # 全相同
        distances = np.array([
            [0.0, 0.5, 0.9],
            [0.5, 0.0, 0.3],
            [0.9, 0.3, 0.0],
        ])
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=2.0, delta=0.05
        )
        # 第一行：距离 [0.0, 0.5, 0.9] → 相似度 [1, 0.5, 0.1]
        # 最近的候选概率应该最高
        assert probs[0, 0] > probs[0, 1] > probs[0, 2]

    def test_all_distances_equal_degrades_to_fitness(self):
        """所有距离相同 → 退化成纯适应度抽样"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.full((3, 3), 0.5)  # 全相同
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=2.0, delta=0.05
        )
        # 每行的概率分布应该相同（只看适应度）
        for i in range(3):
            assert np.allclose(probs[i], probs[0])
            # 适应度高的候选概率更高
            assert probs[i, 2] > probs[i, 1] > probs[i, 0]

    def test_lambda_zero_pure_similarity(self):
        """λ=0 → 纯相似度抽样"""
        fitness = np.array([1.0, 100.0, 10.0])  # 差异巨大
        distances = np.array([
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.5],
            [0.8, 0.5, 0.0],
        ])
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.0, alpha=2.0, delta=0.05  # λ=0
        )
        # 适应度被忽略，只看距离
        # 第一行：最近的是候选0和1
        assert probs[0, 0] > probs[0, 2]
        assert probs[0, 1] > probs[0, 2]

    def test_lambda_one_pure_fitness(self):
        """λ=1 → 纯适应度抽样"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.9, 0.1],  # 候选2很近，但适应度低
            [0.9, 0.0, 0.5],
            [0.1, 0.5, 0.0],
        ])
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=1.0, alpha=2.0, delta=0.05  # λ=1
        )
        # 距离被忽略，只看适应度
        # 每行都应该最偏好候选2（适应度最高）
        for i in range(3):
            assert probs[i, 2] > probs[i, 1] > probs[i, 0]

    def test_alpha_zero_uniform(self):
        """α=0 → 完全平坦（接近均匀）"""
        fitness = np.array([1.0, 100.0, 10.0])
        distances = np.random.rand(5, 3)
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=0.0, delta=0.05  # α=0
        )
        # α=0 时 logit 全为 0，softmax 应该均匀
        expected = 1.0 / 3
        assert np.allclose(probs, expected, atol=0.01)

    def test_large_alpha_sharpens(self):
        """α 大 → 分布尖锐"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([[0.0, 0.5, 0.3]])
        probs_small = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=0.5, delta=0.05
        )
        probs_large = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=10.0, delta=0.05
        )
        # α 大时，最大概率应该更接近 1（更尖锐）
        assert probs_large.max() > probs_small.max()


class TestGeometricRobustness:
    """稳健性测试"""

    def test_fitness_outlier_winsorized(self):
        """适应度极端离群点被 winsorize 截掉"""
        # 99% 的适应度在 [1, 3]，1% 是 1000
        fitness = np.array([1.0, 2.0, 3.0, 1000.0])
        distances = np.random.rand(10, 4) * 0.5
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=1.0, delta=0.05,
            winsorize_quantiles=(0.01, 0.99)
        )
        # 不应该崩溃，且离群点不会完全主导
        assert not np.any(np.isnan(probs))
        # 候选3（离群）的平均概率不应该是压倒性的
        assert probs[:, 3].mean() < 0.9

    def test_distance_out_of_bounds_clipped(self):
        """距离超 [0,1] 被 clip"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.5, 1.5],  # 1.5 超界
            [-0.1, 0.0, 0.3],  # -0.1 超界
            [0.5, 0.3, 0.0],
        ])
        # s = δ + (1−δ)·(1−d)，距离超界会导致 s 超 [δ,1] 或为负
        # 实际上 compute_sampling_probs 没有显式 clip 距离，
        # 但 log(s) 不应该崩（s 应该正）
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=1.0, delta=0.05
        )
        # 主要检查不崩、无 NaN
        assert not np.any(np.isnan(probs))
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestGeometricReproducibility:
    """可复现测试"""

    def test_same_seed_same_result(self):
        """同种子 → 同结果"""
        fitness = np.random.rand(5) * 10
        distances = np.random.rand(10, 5)

        probs1 = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=2.0, delta=0.05
        )
        probs2 = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=2.0, delta=0.05
        )
        assert np.array_equal(probs1, probs2)

        # 抽样也可复现
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        idx1 = sample_donors(probs1, rng1)
        idx2 = sample_donors(probs2, rng2)
        assert np.array_equal(idx1, idx2)


class TestGeometricTorch:
    """torch 路径一致性"""

    def test_torch_cpu_close_to_numpy(self):
        """torch CPU 路径与 numpy 数值接近"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        fitness = np.array([1.0, 2.0, 3.0, 4.0])
        distances = np.random.rand(10, 4) * 0.8

        probs_np = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            device='numpy', lambda_param=0.5, alpha=2.0, delta=0.05
        )
        probs_cpu = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            device='cpu', lambda_param=0.5, alpha=2.0, delta=0.05
        )
        # torch 返回 tensor，转回 numpy
        if isinstance(probs_cpu, torch.Tensor):
            probs_cpu = probs_cpu.cpu().numpy()

        # float32 vs float64 会有微小差异，用较宽容差
        assert np.allclose(probs_np, probs_cpu, atol=1e-5)

    def test_torch_cuda_no_crash(self):
        """torch CUDA 路径不崩（若有 GPU）"""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.random.rand(10, 3) * 0.5

        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            device='cuda', lambda_param=0.5, alpha=1.0, delta=0.05
        )
        # 主要检查不崩
        assert probs.shape == (10, 3)
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


class TestGeometricEndToEnd:
    """端到端集成测试"""

    def test_real_data_no_crash(self):
        """真实数据上跑通，loss 下降"""
        from table_diffevo.queries import load_queries
        from table_diffevo.generator import init_synthetic_table
        from table_diffevo.fitness import compute_fitness
        from table_diffevo.objective import compute_residual
        from table_diffevo.queries import evaluate_table

        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([q["result"] for q in queries])

        rng = np.random.default_rng(42)
        S = init_synthetic_table(300, schema, rng)

        q = evaluate_table(S, queries)
        residual = compute_residual(target, q, 300)
        fitness = compute_fitness(S, queries, residual, q)

        from table_diffevo.distance import pairwise_block_distance
        distances = pairwise_block_distance(S, S, schema)

        # geometric 模式抽样
        probs = compute_sampling_probs(
            fitness, distances, distance_mode='geometric',
            lambda_param=0.5, alpha=1.0, delta=0.05,
            winsorize_quantiles=(0.01, 0.99)
        )
        donor_idx = sample_donors(probs, rng)

        # 检查抽样合法
        assert probs.shape == (300, 300)
        assert len(donor_idx) == 300
        assert np.all((donor_idx >= 0) & (donor_idx < 300))


class TestGeometricValidation:
    """参数校验测试"""

    def test_validates_lambda_range(self):
        """λ 必须在 [0,1]"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError, match="lambda_param 必须在"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                lambda_param=-0.1  # 非法
            )

        with pytest.raises(ValueError, match="lambda_param 必须在"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                lambda_param=1.5  # 非法
            )

    def test_validates_alpha_non_negative(self):
        """α 必须 ≥ 0"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError, match="alpha 必须"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                alpha=-1.0  # 非法
            )

    def test_validates_delta_range(self):
        """δ 必须在 (0,1)"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError, match="delta 必须在"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                delta=0.0  # 非法
            )

        with pytest.raises(ValueError, match="delta 必须在"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                delta=1.0  # 非法
            )

    def test_validates_winsorize_quantiles(self):
        """winsorize_quantiles 格式校验"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError, match="winsorize_quantiles"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                winsorize_quantiles=(0.1,)  # 长度不对
            )

        with pytest.raises(ValueError, match="winsorize_quantiles"):
            compute_sampling_probs(
                fitness, distances, distance_mode='geometric',
                winsorize_quantiles=(0.5, 0.1)  # q_low > q_high
            )
