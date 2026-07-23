"""
测试 GPU 采样（softmax + donor 抽样）的正确性、一致性与可复现性

验证要点：
1. torch softmax 与 numpy softmax 数值接近（float32 精度）
2. torch donor 抽样与 numpy 抽样在同种子下逻辑一致（索引对齐）
3. torch 路径自身可复现（同种子两次相同）
4. run_evolution device='cuda' 端到端可复现
5. numpy 路径结果不受本次改动影响
"""
import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from table_diffevo.sampling import compute_sampling_probs, sample_donors


# ---------- 在两种设备上都跑：cpu 一定可用，cuda 视情况 ----------
def _devices():
    devs = []
    if TORCH_AVAILABLE:
        devs.append('cpu')
    if CUDA_AVAILABLE:
        devs.append('cuda')
    return devs


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSamplingProbsTorch:
    """torch softmax 与 numpy 数值接近"""

    @pytest.mark.parametrize("device", _devices())
    def test_probs_match_numpy_small(self, device):
        """小数据：torch 概率与 numpy 概率接近（float32 容差）"""
        rng = np.random.default_rng(0)
        fitness = rng.standard_normal(8)
        distances = rng.random((12, 8))

        p_np = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8, device='numpy')
        p_t = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8, device=device)
        p_t = np.asarray(p_t.cpu())  # tensor → numpy 便于比较

        np.testing.assert_allclose(p_np, p_t, atol=1e-5)

    @pytest.mark.parametrize("device", _devices())
    def test_probs_rows_sum_to_one(self, device):
        """torch 概率每行和为 1"""
        rng = np.random.default_rng(1)
        fitness = rng.standard_normal(10)
        distances = rng.random((15, 10))
        p_t = compute_sampling_probs(fitness, distances, beta=1.0, h=0.5, device=device)
        row_sums = np.asarray(p_t.sum(dim=1).cpu())
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    @pytest.mark.parametrize("device", _devices())
    def test_probs_accept_distance_tensor(self, device):
        """distances 作为设备上的 tensor 传入也能正确处理"""
        rng = np.random.default_rng(2)
        fitness = rng.standard_normal(6)
        distances = rng.random((9, 6))
        dist_t = torch.as_tensor(distances, dtype=torch.float32, device=device)

        p_from_tensor = compute_sampling_probs(fitness, dist_t, beta=1.0, h=0.8, device=device)
        p_from_array = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8, device=device)
        np.testing.assert_allclose(
            np.asarray(p_from_tensor.cpu()), np.asarray(p_from_array.cpu()), atol=1e-6
        )

    def test_invalid_device_raises(self):
        """非法 device 报错"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5]])
        with pytest.raises(ValueError, match="Unknown device"):
            compute_sampling_probs(fitness, distances, device='invalid')


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSampleDonorsTorch:
    """torch donor 抽样：与 numpy 逻辑一致 + 可复现"""

    @pytest.mark.parametrize("device", _devices())
    def test_indices_match_numpy_same_seed(self, device):
        """
        关键测试：同一概率矩阵 + 同种子，torch 抽样与 numpy 抽样索引一致。
        验证 (u < cumprobs).argmax 的 torch 实现与 numpy 语义相同。
        用规整的概率矩阵（非贴近的平局），确保 float32 不会导致边界翻转。
        """
        rng_seed = 42
        # 构造清晰的概率矩阵（每行明显区分，避免 float32 边界平局）
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.33, 0.34, 0.33],
            [0.9, 0.05, 0.05],
        ])

        idx_np = sample_donors(probs, np.random.default_rng(rng_seed), device='numpy')
        idx_t = sample_donors(probs, np.random.default_rng(rng_seed), device=device)

        np.testing.assert_array_equal(idx_np, idx_t)

    @pytest.mark.parametrize("device", _devices())
    def test_reproducible_same_seed(self, device):
        """torch 路径自身可复现：同种子两次结果相同"""
        rng = np.random.default_rng(3)
        probs = rng.random((20, 10))
        probs = probs / probs.sum(axis=1, keepdims=True)

        idx1 = sample_donors(probs, np.random.default_rng(7), device=device)
        idx2 = sample_donors(probs, np.random.default_rng(7), device=device)
        np.testing.assert_array_equal(idx1, idx2)

    @pytest.mark.parametrize("device", _devices())
    def test_returns_cpu_numpy(self, device):
        """torch 路径返回 CPU 上的 numpy 数组（接口一致）"""
        rng = np.random.default_rng(4)
        probs = rng.random((10, 5))
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx = sample_donors(probs, np.random.default_rng(1), device=device)
        assert isinstance(idx, np.ndarray)
        assert idx.shape == (10,)
        assert (idx >= 0).all() and (idx < 5).all()

    @pytest.mark.parametrize("device", _devices())
    def test_accept_probs_tensor(self, device):
        """probs 作为设备上的 tensor 传入也能抽样"""
        rng = np.random.default_rng(5)
        probs = rng.random((8, 4))
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs_t = torch.as_tensor(probs, dtype=torch.float32, device=device)
        idx = sample_donors(probs_t, np.random.default_rng(2), device=device)
        assert idx.shape == (8,)

    def test_invalid_device_raises(self):
        """非法 device 报错"""
        probs = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="Unknown device"):
            sample_donors(probs, np.random.default_rng(0), device='invalid')


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEvolutionGpuSampling:
    """run_evolution 端到端：GPU 采样链路可复现"""

    def _setup(self):
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_queries
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([q["result"] for q in queries], dtype=float)
        return schema, queries, target

    @pytest.mark.parametrize("device", _devices())
    def test_end_to_end_reproducible(self, device):
        """同种子 + 同 device 两次跑，best_loss 与 loss 历史完全一致"""
        from table_diffevo.evolution import run_evolution
        schema, queries, target = self._setup()

        _, d1 = run_evolution(target, queries, schema, n_records=300,
                              n_rounds=10, seed=0, device=device)
        _, d2 = run_evolution(target, queries, schema, n_records=300,
                              n_rounds=10, seed=0, device=device)

        assert d1["loss_history"] == d2["loss_history"]
        assert d1["best_loss"] == d2["best_loss"]
        assert d1["accept_history"] == d2["accept_history"]

    @pytest.mark.parametrize("device", _devices())
    def test_gpu_sampling_runs_and_reduces_loss(self, device):
        """GPU 采样链路能跑通且不比初始更差"""
        from table_diffevo.evolution import run_evolution
        schema, queries, target = self._setup()
        _, diag = run_evolution(target, queries, schema, n_records=300,
                                n_rounds=10, seed=0, device=device)
        assert diag["best_loss"] <= diag["loss_history"][0]


class TestNumpyPathUnchanged:
    """回归：本次改动不影响 numpy 路径的既有行为"""

    def test_numpy_probs_still_array(self):
        """numpy 路径仍返回 np.ndarray"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.5]])
        probs = compute_sampling_probs(fitness, distances, device='numpy')
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (2, 3)

    def test_numpy_donors_reproducible(self):
        """numpy 路径抽样仍可复现（默认 device）"""
        rng = np.random.default_rng(9)
        probs = rng.random((15, 6))
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx1 = sample_donors(probs, np.random.default_rng(3))
        idx2 = sample_donors(probs, np.random.default_rng(3))
        np.testing.assert_array_equal(idx1, idx2)
