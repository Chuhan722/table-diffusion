"""
exclude_self（对角线屏蔽）测试

验证 compute_sampling_probs(..., exclude_self=True) 在全对全候选池上禁止记录
抽到自己：对角概率恒为 0、每行仍和为 1、非方阵报错、numpy↔torch 一致，
且默认 False 时行为不变（回归）。

背景：候选池=全表时自身距离=0、相似度=1，高锐度 geometric 下小表自身抽样率
可达 8%（见 scripts/diagnose_self_sampling.py）。抽到自己 = 该行本轮不变、
对演化零贡献，故主循环开启 exclude_self=True 屏蔽之。
"""
import pytest
import numpy as np

from table_diffevo.sampling import compute_sampling_probs, sample_donors

ALL_MODES = ['squared', 'linear', 'none', 'multiplicative', 'geometric']


def _square_inputs(n=8, seed=0):
    """造一个方阵输入：fitness (n,)、对称距离 (n,n) 且对角为 0。"""
    rng = np.random.default_rng(seed)
    fitness = rng.uniform(0.0, 5.0, size=n)
    d = rng.uniform(0.0, 1.0, size=(n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return fitness, d


class TestExcludeSelfNumpy:
    """numpy 路径：对角屏蔽的核心性质，覆盖全部 distance_mode。"""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_diagonal_is_zero(self, mode):
        """exclude_self=True 时对角概率恒为 0。"""
        fitness, d = _square_inputs()
        probs = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='numpy',
            exclude_self=True,
        )
        diag = np.diag(probs)
        assert np.allclose(diag, 0.0), f"{mode}: 对角未清零 {diag}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_rows_still_sum_to_one(self, mode):
        """屏蔽后每行仍归一化（和为 1）。"""
        fitness, d = _square_inputs()
        probs = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='numpy',
            exclude_self=True,
        )
        assert np.allclose(probs.sum(axis=1), 1.0)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_off_diagonal_ratios_preserved(self, mode):
        """屏蔽=对该行非自身候选重归一化：非对角元素的相对比例不变。"""
        fitness, d = _square_inputs()
        base = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='numpy', exclude_self=False,
        )
        excl = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='numpy', exclude_self=True,
        )
        n = len(fitness)
        for i in range(n):
            mask = np.arange(n) != i
            renorm = base[i, mask] / base[i, mask].sum()
            assert np.allclose(renorm, excl[i, mask], atol=1e-10), f"{mode} 行 {i}"

    def test_default_false_unchanged(self):
        """默认 exclude_self=False 与不传参完全一致（回归）。"""
        fitness, d = _square_inputs()
        for mode in ALL_MODES:
            a = compute_sampling_probs(fitness, d, distance_mode=mode, device='numpy')
            b = compute_sampling_probs(fitness, d, distance_mode=mode, device='numpy',
                                       exclude_self=False)
            assert np.array_equal(a, b), mode

    def test_non_square_raises(self):
        """非方阵（K≠N）开启 exclude_self 报错。"""
        fitness = np.array([1.0, 2.0, 3.0])
        d = np.random.rand(5, 3)  # N=5, K=3
        with pytest.raises(ValueError, match="方阵"):
            compute_sampling_probs(fitness, d, distance_mode='geometric',
                                   device='numpy', exclude_self=True)

    def test_self_sampling_rate_is_zero(self):
        """端到端：屏蔽后实际抽样永不落在自身索引。"""
        fitness, d = _square_inputs(n=12, seed=3)
        probs = compute_sampling_probs(
            fitness, d, distance_mode='geometric', device='numpy',
            alpha=10.0, exclude_self=True,
        )
        n = len(fitness)
        for k in range(20):
            rng = np.random.default_rng(100 + k)
            donor = sample_donors(probs, rng, device='numpy')
            assert not np.any(donor == np.arange(n))


class TestExcludeSelfTorch:
    """torch 路径：与 numpy 一致 + 自身性质。"""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_torch_cpu_close_to_numpy(self, mode):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        fitness, d = _square_inputs(n=10, seed=1)
        probs_np = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='numpy', exclude_self=True,
        )
        probs_cpu = compute_sampling_probs(
            fitness, d, distance_mode=mode, device='cpu', exclude_self=True,
        )
        if isinstance(probs_cpu, torch.Tensor):
            probs_cpu = probs_cpu.cpu().numpy()
        assert np.allclose(probs_np, probs_cpu, atol=1e-5)
        # 对角也应为 0（float32 容差）
        assert np.allclose(np.diag(probs_cpu), 0.0, atol=1e-6)

    def test_torch_non_square_raises(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")
        fitness = np.array([1.0, 2.0, 3.0])
        d = np.random.rand(5, 3)
        with pytest.raises(ValueError, match="方阵"):
            compute_sampling_probs(fitness, d, distance_mode='geometric',
                                   device='cpu', exclude_self=True)


