"""
测试 GPU 距离计算的正确性和一致性

验证 PyTorch GPU 实现与 NumPy 实现数值上的一致性。
"""
import pytest
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from table_diffevo.distance import pairwise_block_distance
from table_diffevo.schema import load_schema


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchCPU:
    """测试 PyTorch CPU 实现"""

    def test_torch_cpu_matches_numpy_small(self):
        """小数据集：torch CPU 与 numpy 结果一致"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        # 构造小测试表
        data = {
            'age': [25, 30, 25],
            'education': ['high_school', 'bachelor', 'high_school'],
            'employment': ['full_time', 'full_time', 'part_time'],
            'income': ['medium', 'high', 'low'],
            'marital': ['single', 'married', 'single'],
            'children': [0, 2, 0],
            'housing': ['rent', 'own', 'rent'],
            'vehicle': ['car', 'car', 'no_vehicle'],
            'health': ['good', 'excellent', 'fair'],
            'region': ['urban', 'suburban', 'urban'],
        }
        df = pd.DataFrame(data)

        dist_numpy = pairwise_block_distance(df, df, schema, device='numpy')
        dist_torch = pairwise_block_distance(df, df, schema, device='cpu')

        # 数值应该非常接近（允许浮点误差）
        np.testing.assert_allclose(dist_numpy, dist_torch, atol=1e-6)

    def test_torch_cpu_self_distance_zero(self):
        """torch CPU 实现：自己与自己距离为 0"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        data = {
            'age': [30],
            'education': ['bachelor'],
            'employment': ['full_time'],
            'income': ['medium'],
            'marital': ['single'],
            'children': [0],
            'housing': ['rent'],
            'vehicle': ['car'],
            'health': ['good'],
            'region': ['urban'],
        }
        df = pd.DataFrame(data)

        dist = pairwise_block_distance(df, df, schema, device='cpu')

        assert dist.shape == (1, 1)
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestTorchGPU:
    """测试 PyTorch GPU 实现（需要 CUDA）"""

    def test_cuda_matches_numpy_small(self):
        """小数据集：CUDA 与 numpy 结果一致"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        data = {
            'age': [25, 30, 35, 40],
            'education': ['high_school', 'bachelor', 'master', 'bachelor'],
            'employment': ['full_time', 'full_time', 'part_time', 'unemployed'],
            'income': ['medium', 'high', 'high', 'low'],
            'marital': ['single', 'married', 'married', 'single'],
            'children': [0, 2, 1, 0],
            'housing': ['rent', 'own', 'own', 'rent'],
            'vehicle': ['car', 'car', 'car', 'no_vehicle'],
            'health': ['good', 'excellent', 'good', 'fair'],
            'region': ['urban', 'suburban', 'rural', 'urban'],
        }
        df = pd.DataFrame(data)

        dist_numpy = pairwise_block_distance(df, df, schema, device='numpy')
        dist_cuda = pairwise_block_distance(df, df, schema, device='cuda')

        # GPU 使用 float32，精度略低，但应该很接近
        np.testing.assert_allclose(dist_numpy, dist_cuda, atol=1e-5)

    def test_cuda_self_distance_zero(self):
        """CUDA 实现：自己与自己距离为 0"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        data = {
            'age': [30, 40],
            'education': ['bachelor', 'master'],
            'employment': ['full_time', 'full_time'],
            'income': ['medium', 'high'],
            'marital': ['single', 'married'],
            'children': [0, 2],
            'housing': ['rent', 'own'],
            'vehicle': ['car', 'car'],
            'health': ['good', 'excellent'],
            'region': ['urban', 'suburban'],
        }
        df = pd.DataFrame(data)

        dist = pairwise_block_distance(df, df, schema, device='cuda')

        assert dist.shape == (2, 2)
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert dist[1, 1] == pytest.approx(0.0, abs=1e-5)

    def test_cuda_symmetry(self):
        """CUDA 实现：距离矩阵对称"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        data = {
            'age': [25, 30, 35],
            'education': ['high_school', 'bachelor', 'master'],
            'employment': ['full_time', 'full_time', 'part_time'],
            'income': ['medium', 'high', 'high'],
            'marital': ['single', 'married', 'married'],
            'children': [0, 2, 1],
            'housing': ['rent', 'own', 'own'],
            'vehicle': ['car', 'car', 'car'],
            'health': ['good', 'excellent', 'good'],
            'region': ['urban', 'suburban', 'rural'],
        }
        df = pd.DataFrame(data)

        dist = pairwise_block_distance(df, df, schema, device='cuda')

        # 对称性：dist[i, j] == dist[j, i]
        np.testing.assert_allclose(dist, dist.T, atol=1e-5)

    def test_cuda_range(self):
        """CUDA 实现：距离在 [0, 1] 范围内"""
        schema = load_schema("configs/test_300x10/schema.yaml")

        data = {
            'age': [18, 50, 100],  # 跨度大
            'education': ['high_school', 'bachelor', 'phd'],
            'employment': ['full_time', 'part_time', 'unemployed'],
            'income': ['low', 'medium', 'high'],
            'marital': ['single', 'married', 'divorced'],
            'children': [0, 2, 5],
            'housing': ['rent', 'own', 'rent'],
            'vehicle': ['no_vehicle', 'car', 'car'],
            'health': ['poor', 'good', 'excellent'],
            'region': ['rural', 'suburban', 'urban'],
        }
        df = pd.DataFrame(data)

        dist = pairwise_block_distance(df, df, schema, device='cuda')

        assert np.all(dist >= 0.0)
        assert np.all(dist <= 1.0)


def test_device_parameter_validation():
    """测试 device 参数验证"""
    schema = load_schema("configs/test_300x10/schema.yaml")

    data = {
        'age': [30],
        'education': ['bachelor'],
        'employment': ['full_time'],
        'income': ['medium'],
        'marital': ['single'],
        'children': [0],
        'housing': ['rent'],
        'vehicle': ['car'],
        'health': ['good'],
        'region': ['urban'],
    }
    df = pd.DataFrame(data)

    # 无效的 device 参数
    with pytest.raises(ValueError, match="Unknown device"):
        pairwise_block_distance(df, df, schema, device='invalid')
