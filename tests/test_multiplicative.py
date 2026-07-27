"""
测试 multiplicative 距离模式
"""
import numpy as np
import pytest

from table_diffevo.sampling import compute_sampling_probs


def _cuda_available():
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestMultiplicativeMode:
    """测试 multiplicative 模式的基本功能"""

    def test_basic_functionality(self):
        """基本功能：能跑、不崩、概率和为1"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.3, 0.6],
            [0.3, 0.0, 0.4],
            [0.6, 0.4, 0.0]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=1.0, p=1.0,
            distance_mode='multiplicative'
        )

        assert probs.shape == (3, 3)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_p_zero_ignores_distance(self):
        """p=0 时应该退化成纯适应度抽样（距离权重恒为1）"""
        fitness = np.array([1.0, 5.0, 2.0])
        distances = np.array([
            [0.0, 0.9, 0.1],  # 第一行：距离差异很大
            [0.9, 0.0, 0.8],
            [0.1, 0.8, 0.0]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=1.0, p=0.0,  # p=0 应该忽略距离
            distance_mode='multiplicative'
        )

        # p=0 时，每行概率应该只和适应度有关，每行都一样
        assert np.allclose(probs[0], probs[1])
        assert np.allclose(probs[0], probs[2])

    def test_beta_zero_uniform(self):
        """β=0 时应该退化成纯距离权重（适应度概率均匀）"""
        fitness = np.array([1.0, 100.0, 2.0])  # 适应度差异很大
        distances = np.array([
            [0.0, 0.5, 0.9],
            [0.5, 0.0, 0.3],
            [0.9, 0.3, 0.0]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=0.0, p=1.0,  # β=0 应该忽略适应度
            distance_mode='multiplicative'
        )

        # β=0 时 softmax(0*F) = 均匀，每行只受距离影响
        # 近的候选（小 d → 大 w）应该概率更高
        assert probs[0, 0] > probs[0, 1]  # d=0.0 vs d=0.5
        assert probs[0, 1] > probs[0, 2]  # d=0.5 vs d=0.9

    def test_high_fitness_near_candidate_dominates(self):
        """高适应度+近距离的候选应该概率最高"""
        fitness = np.array([1.0, 10.0, 2.0])  # 第1个候选适应度最高
        distances = np.array([
            [0.5, 0.1, 0.8],  # 第0行：第1个候选距离最近且适应度最高
            [0.3, 0.6, 0.2],
            [0.7, 0.4, 0.5]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=2.0, p=2.0,  # 都偏锐利
            distance_mode='multiplicative'
        )

        # 第0行：候选1 适应度最高(10) + 距离最近(0.1) → 概率应该最高
        assert probs[0, 1] > probs[0, 0]
        assert probs[0, 1] > probs[0, 2]

    def test_validates_p_non_negative(self):
        """p 必须 ≥ 0"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError, match="p 必须 ≥ 0"):
            compute_sampling_probs(
                fitness, distances,
                beta=1.0, p=-1.0,
                distance_mode='multiplicative'
            )

    def test_no_nan_or_inf(self):
        """数值稳定性：不应该产生 NaN 或 inf"""
        fitness = np.array([100.0, 200.0, 150.0])  # 大数
        distances = np.array([
            [0.0, 0.99, 0.01],  # 极端距离
            [0.99, 0.0, 0.5],
            [0.01, 0.5, 0.0]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=10.0, p=5.0,  # 极端参数
            distance_mode='multiplicative'
        )

        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestMultiplicativeWithTorch:
    """测试 torch 路径的 multiplicative 模式"""

    def test_torch_cpu_consistent_with_numpy(self):
        """torch CPU 路径应该和 numpy 数值接近"""
        fitness = np.array([1.0, 2.0, 3.0, 1.5])
        distances = np.array([
            [0.0, 0.3, 0.6, 0.2],
            [0.3, 0.0, 0.4, 0.5],
            [0.6, 0.4, 0.0, 0.3],
            [0.2, 0.5, 0.3, 0.0]
        ])

        probs_numpy = compute_sampling_probs(
            fitness, distances,
            beta=1.0, p=1.0,
            device='numpy',
            distance_mode='multiplicative'
        )

        probs_torch = compute_sampling_probs(
            fitness, distances,
            beta=1.0, p=1.0,
            device='cpu',
            distance_mode='multiplicative'
        )

        # torch 用 float32, numpy 用 float64, 允许小误差
        probs_torch_np = probs_torch.cpu().numpy()
        assert np.allclose(probs_numpy, probs_torch_np, atol=1e-6)

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_torch_cuda_no_crash(self):
        """CUDA 路径不崩"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.3, 0.6],
            [0.3, 0.0, 0.4],
            [0.6, 0.4, 0.0]
        ])

        probs = compute_sampling_probs(
            fitness, distances,
            beta=1.0, p=1.0,
            device='cuda',
            distance_mode='multiplicative'
        )

        # 能返回、不崩、形状对
        assert probs.shape == (3, 3)
        probs_np = probs.cpu().numpy()
        assert np.allclose(probs_np.sum(axis=1), 1.0, atol=1e-6)
