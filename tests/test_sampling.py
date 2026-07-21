"""
测试参考记录抽样

锚定抽样概率计算和随机抽样的正确性。
"""
import numpy as np
import pytest
from table_diffevo.sampling import compute_sampling_probs, sample_donors


class TestComputeSamplingProbs:
    """测试抽样概率计算（确定性部分）"""

    def test_output_shape(self):
        """输出 shape 正确"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
        ])
        probs = compute_sampling_probs(fitness, distances)
        assert probs.shape == (2, 3)

    def test_each_row_sums_to_one(self):
        """每行和为 1"""
        fitness = np.random.rand(10)
        distances = np.random.rand(20, 10)
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.5)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_all_probs_non_negative(self):
        """所有概率非负"""
        fitness = np.array([-1.0, 0.0, 1.0])  # 适应度可以为负
        distances = np.random.rand(5, 3)
        probs = compute_sampling_probs(fitness, distances)
        assert (probs >= 0).all()

    def test_beta_zero_ignores_fitness(self):
        """β=0 时完全忽略适应度，只看距离"""
        fitness = np.array([100.0, 0.0, -100.0])  # 差距很大
        distances = np.array([
            [0.0, 0.0, 0.0],  # 第 0 行：所有候选距离相同
        ])
        probs = compute_sampling_probs(fitness, distances, beta=0.0, h=0.8)
        # 距离相同 + 忽略适应度 → 均匀分布
        assert np.allclose(probs[0], [1/3, 1/3, 1/3], atol=1e-6)

    def test_large_h_weakens_distance(self):
        """h 很大时，距离惩罚几乎消失，主要看适应度"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.5, 0.9],  # 第 0 行：距离差异较大
        ])
        # h=100 时距离项 d²/(2h²) ≈ 0
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=100.0)
        # 应该主要由适应度决定，fitness[2]=3 最大，概率应最高
        assert probs[0, 2] > probs[0, 1] > probs[0, 0]

    def test_small_h_emphasizes_distance(self):
        """h 很小时，距离惩罚占主导"""
        fitness = np.array([10.0, 5.0, 1.0])  # 适应度递减
        distances = np.array([
            [0.3, 0.5, 0.1],  # 第 0 行：候选 2 距离最近，候选 0 适应度最高但距离中等
        ])
        # h=0.01 时距离项很大，候选 2（距离最近 0.1）应比候选 1（距离远 0.5）概率高
        # 即使候选 2 适应度最低
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.01)
        assert probs[0, 2] > probs[0, 1]  # 距离近的候选 2 > 距离远的候选 1

    def test_uniform_fitness_falls_back_to_distance(self):
        """所有适应度相同时，退化为纯距离选择"""
        fitness = np.array([1.0, 1.0, 1.0])
        distances = np.array([
            [0.0, 0.5, 1.0],  # 候选 0 最近
        ])
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.5)
        # 适应度相同，距离越近概率越高
        assert probs[0, 0] > probs[0, 1] > probs[0, 2]

    def test_uniform_distance_falls_back_to_fitness(self):
        """所有距离相同时，退化为纯适应度选择"""
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.5, 0.5, 0.5],  # 距离全相同
        ])
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8)
        # 距离相同，适应度越高概率越高
        assert probs[0, 2] > probs[0, 1] > probs[0, 0]

    def test_both_uniform_gives_uniform_probs(self):
        """适应度和距离都相同 → 均匀分布"""
        fitness = np.array([1.0, 1.0, 1.0])
        distances = np.array([
            [0.5, 0.5, 0.5],
        ])
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8)
        assert np.allclose(probs[0], [1/3, 1/3, 1/3], atol=1e-6)

    def test_validates_beta_non_negative(self):
        """β < 0 报错"""
        fitness = np.array([1.0])
        distances = np.array([[0.0]])
        with pytest.raises(ValueError, match="beta 必须 ≥ 0"):
            compute_sampling_probs(fitness, distances, beta=-0.1, h=0.8)

    def test_validates_h_positive(self):
        """h <= 0 报错"""
        fitness = np.array([1.0])
        distances = np.array([[0.0]])
        with pytest.raises(ValueError, match="h 必须 > 0"):
            compute_sampling_probs(fitness, distances, beta=1.0, h=0.0)
        with pytest.raises(ValueError, match="h 必须 > 0"):
            compute_sampling_probs(fitness, distances, beta=1.0, h=-0.5)

    def test_validates_shape_mismatch(self):
        """fitness 长度与 distances 列数不一致报错"""
        fitness = np.array([1.0, 2.0])
        distances = np.array([[0.0, 0.5, 1.0]])  # 3 列
        with pytest.raises(ValueError, match="fitness 长度.*与 distances 列数.*不一致"):
            compute_sampling_probs(fitness, distances)

    def test_validates_fitness_not_1d(self):
        """fitness 不是 1 维报错"""
        fitness = np.array([[1.0, 2.0]])  # 2 维
        distances = np.array([[0.0, 0.5]])
        with pytest.raises(ValueError, match="fitness 必须是 1 维"):
            compute_sampling_probs(fitness, distances)

    def test_validates_distances_not_2d(self):
        """distances 不是 2 维报错"""
        fitness = np.array([1.0])
        distances = np.array([0.0])  # 1 维
        with pytest.raises(ValueError, match="distances 必须是 2 维"):
            compute_sampling_probs(fitness, distances)


class TestSampleDonors:
    """测试随机抽样（固定种子保证复现）"""

    def test_output_shape_and_range(self):
        """输出 shape 正确，索引在 [0, K) 内"""
        probs = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
        ])
        rng = np.random.default_rng(42)
        indices = sample_donors(probs, rng)
        assert indices.shape == (2,)
        assert (indices >= 0).all() and (indices < 3).all()

    def test_reproducible_with_same_seed(self):
        """固定种子可复现"""
        probs = np.random.rand(100, 10)
        probs /= probs.sum(axis=1, keepdims=True)

        rng1 = np.random.default_rng(123)
        indices1 = sample_donors(probs, rng1)

        rng2 = np.random.default_rng(123)
        indices2 = sample_donors(probs, rng2)

        assert np.array_equal(indices1, indices2)

    def test_different_seeds_give_different_results(self):
        """不同种子结果不同"""
        probs = np.random.rand(100, 10)
        probs /= probs.sum(axis=1, keepdims=True)

        rng1 = np.random.default_rng(111)
        indices1 = sample_donors(probs, rng1)

        rng2 = np.random.default_rng(222)
        indices2 = sample_donors(probs, rng2)

        # 100 行 × 10 候选，几乎不可能完全相同
        assert not np.array_equal(indices1, indices2)

    def test_deterministic_probs_always_pick_max(self):
        """确定性概率（某列=1）必定抽到那一列"""
        probs = np.array([
            [1.0, 0.0, 0.0],  # 必定抽候选 0
            [0.0, 1.0, 0.0],  # 必定抽候选 1
            [0.0, 0.0, 1.0],  # 必定抽候选 2
        ])
        rng = np.random.default_rng(999)
        indices = sample_donors(probs, rng)
        assert np.array_equal(indices, [0, 1, 2])

    def test_uniform_probs_cover_all_candidates(self):
        """均匀分布抽样足够多次后，所有候选都应被选中"""
        K = 5
        probs = np.full((1000, K), 1.0 / K)  # 1000 次抽样，均匀分布
        rng = np.random.default_rng(777)
        indices = sample_donors(probs, rng)
        # 所有候选都应至少被选中一次
        unique_indices = np.unique(indices)
        assert len(unique_indices) == K

    def test_validates_probs_not_2d(self):
        """probs 不是 2 维报错"""
        probs = np.array([0.5, 0.5])  # 1 维
        with pytest.raises(ValueError, match="probs 必须是 2 维"):
            sample_donors(probs)

    def test_validates_row_sums_not_one(self):
        """某行和不为 1 报错"""
        probs = np.array([
            [0.5, 0.3, 0.2],  # 和为 1，正常
            [0.4, 0.4, 0.4],  # 和为 1.2，异常
        ])
        with pytest.raises(ValueError, match="probs 某些行和不为 1"):
            sample_donors(probs)


class TestIntegration:
    """端到端集成测试：概率计算 + 抽样"""

    def test_full_pipeline(self):
        """完整流程：适应度 + 距离 → 概率 → 抽样索引"""
        # 构造 5 条当前记录，3 个候选
        fitness = np.array([1.0, 2.0, 3.0])
        distances = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.0],
            [0.3, 0.3, 0.3],
            [0.8, 0.2, 0.9],
        ])

        # 1. 计算概率
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.5)
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

        # 2. 抽样
        rng = np.random.default_rng(42)
        indices = sample_donors(probs, rng)
        assert indices.shape == (5,)
        assert (indices >= 0).all() and (indices < 3).all()

    def test_can_sample_self_in_full_table(self):
        """玩具阶段全对全(N=K)，允许抽到自己"""
        N = 10
        fitness = np.random.rand(N)
        distances = np.random.rand(N, N)
        np.fill_diagonal(distances, 0.0)  # 自己到自己距离为 0

        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=0.8)
        rng = np.random.default_rng(123)
        indices = sample_donors(probs, rng)

        # 统计抽到自己的次数（索引 i 抽到 i）
        self_selected = np.sum(indices == np.arange(N))
        # 抽到自己是低概率但合法事件，10 次里可能有几次
        # 这里只验证"允许"(不报错)，不验证具体次数
        assert 0 <= self_selected <= N
