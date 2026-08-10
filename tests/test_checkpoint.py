"""
Checkpoint 机制的单元测试

验证保存和恢复演化状态的正确性，支持探测式 α 调度的三岔路分支。
"""

import numpy as np
import pandas as pd
import pytest
from table_diffevo.checkpoint import Checkpoint


class TestCheckpoint:
    def test_basic_save_restore(self):
        """测试基本的保存和恢复功能"""
        syn_table = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        best_L1 = 0.05
        rng = np.random.default_rng(42)
        round_count = 100
        alpha = 5.0

        checkpoint = Checkpoint.capture(syn_table, best_L1, rng, round_count, alpha)
        restored_table, restored_L1, rng_state, restored_round, restored_alpha = (
            checkpoint.restore()
        )

        # 验证恢复的值
        assert restored_table.equals(syn_table)
        assert restored_L1 == best_L1
        assert restored_round == round_count
        assert restored_alpha == alpha

    def test_deep_copy(self):
        """测试 checkpoint 是深拷贝，修改原表不影响 checkpoint"""
        syn_table = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]})
        best_L1 = 0.1
        rng = np.random.default_rng(42)
        round_count = 50
        alpha = 3.0

        checkpoint = Checkpoint.capture(syn_table, best_L1, rng, round_count, alpha)

        # 修改原表
        original_value = syn_table.iloc[0, 0]
        syn_table.iloc[0, 0] = 999.0

        # 恢复 checkpoint，应该不受影响
        restored_table, _, _, _, _ = checkpoint.restore()
        assert restored_table.iloc[0, 0] == original_value
        assert syn_table.iloc[0, 0] == 999.0  # 原表已被修改

    def test_three_branches_from_same_checkpoint(self):
        """测试三个分支从同一 checkpoint 出发，初始状态完全相同"""
        syn_table = pd.DataFrame({"A": [10, 30], "B": [20, 40]})
        best_L1 = 0.25
        rng = np.random.default_rng(123)
        round_count = 200
        alpha = 7.0

        checkpoint = Checkpoint.capture(syn_table, best_L1, rng, round_count, alpha)

        # 模拟三个分支恢复
        branches = []
        for _ in range(3):
            restored = checkpoint.restore()
            branches.append(restored)

        # 验证三个分支的初始状态完全相同
        for i in range(1, 3):
            assert branches[i][0].equals(branches[0][0])  # syn_table
            assert branches[i][1] == branches[0][1]  # best_L1
            assert branches[i][3] == branches[0][3]  # round_count
            assert branches[i][4] == branches[0][4]  # alpha

    def test_rng_state_reproducibility(self):
        """测试 rng 状态的可复现性"""
        syn_table = pd.DataFrame({"A": [1], "B": [2]})
        best_L1 = 0.1
        rng = np.random.default_rng(999)

        # 生成一些随机数，改变 rng 状态
        rng.standard_normal(5)

        round_count = 10
        alpha = 3.0

        # 保存 checkpoint
        checkpoint = Checkpoint.capture(syn_table, best_L1, rng, round_count, alpha)

        # 恢复两次，验证生成的随机数相同
        _, _, rng_state_1, _, _ = checkpoint.restore()
        _, _, rng_state_2, _, _ = checkpoint.restore()

        rng1 = np.random.default_rng()
        rng1.bit_generator.state = rng_state_1
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = rng_state_2

        # 生成随机数并验证相同
        samples1 = rng1.random(10)
        samples2 = rng2.random(10)
        np.testing.assert_array_equal(samples1, samples2)

    def test_multiple_restore_does_not_mutate_checkpoint(self):
        """测试多次 restore 不会改变 checkpoint 本身"""
        syn_table = pd.DataFrame({"X": [100, 200]})
        best_L1 = 0.5
        rng = np.random.default_rng(777)
        round_count = 300
        alpha = 9.0

        checkpoint = Checkpoint.capture(syn_table, best_L1, rng, round_count, alpha)

        # 恢复并修改
        restored1, _, _, _, _ = checkpoint.restore()
        restored1.iloc[0, 0] = 999

        # 再次恢复，应该还是原值
        restored2, _, _, _, _ = checkpoint.restore()
        assert restored2.iloc[0, 0] == 100
