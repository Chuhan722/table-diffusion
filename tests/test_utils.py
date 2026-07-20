"""
测试随机种子工具的正确性

验证：
1. 相同种子 → 相同的随机数序列（可复现）
2. 不同种子 → 不同的随机数序列
3. 重复设置相同种子，能重置随机状态
"""
import numpy as np
from table_diffevo.utils import set_seed, get_rng


def test_same_seed_produces_same_sequence():
    """
    测试：相同种子产生相同的随机数序列

    这是可复现性的核心要求。
    """
    # 第一次运行
    set_seed(42)
    sequence1 = np.random.randn(10)

    # 第二次运行，用同样的种子
    set_seed(42)
    sequence2 = np.random.randn(10)

    # 两次结果必须完全一样
    np.testing.assert_array_equal(sequence1, sequence2)


def test_different_seeds_produce_different_sequences():
    """
    测试：不同种子产生不同的随机数序列

    确保种子确实起作用，不是总返回固定值。
    """
    set_seed(42)
    sequence1 = np.random.randn(10)

    set_seed(99)
    sequence2 = np.random.randn(10)

    # 两个序列应该不同（概率上几乎不可能完全相同）
    assert not np.array_equal(sequence1, sequence2)


def test_seed_resets_state():
    """
    测试：重新设置种子能重置随机状态

    即使中间生成过其他随机数，重设种子后从头开始。
    """
    set_seed(42)
    first_value = np.random.randn()

    # 中间生成一些随机数
    _ = np.random.randn(100)

    # 重新设置相同的种子
    set_seed(42)
    second_value = np.random.randn()

    # 应该得到和最开始一样的第一个值
    assert first_value == second_value


def test_seed_affects_multiple_operations():
    """
    测试：种子影响所有 np.random 操作

    不只是 randn，choice、shuffle 等都受影响。
    """
    set_seed(42)
    arr1 = np.arange(10)
    np.random.shuffle(arr1)
    choice1 = np.random.choice(100, size=5)

    set_seed(42)
    arr2 = np.arange(10)
    np.random.shuffle(arr2)
    choice2 = np.random.choice(100, size=5)

    np.testing.assert_array_equal(arr1, arr2)
    np.testing.assert_array_equal(choice1, choice2)


def test_get_rng_produces_independent_generators():
    """
    测试：get_rng 创建的生成器互相独立

    验证新方式（未来可能使用）的正确性。
    """
    rng1 = get_rng(42)
    rng2 = get_rng(42)

    # 相同种子的独立生成器，产生相同序列
    seq1 = rng1.standard_normal(10)
    seq2 = rng2.standard_normal(10)
    np.testing.assert_array_equal(seq1, seq2)

    # 但它们不影响全局状态
    set_seed(99)
    global_seq = np.random.randn(10)

    # rng1 继续生成，不受全局 seed 影响
    rng1_next = rng1.standard_normal(10)
    assert not np.array_equal(rng1_next, global_seq)


def test_get_rng_different_seeds():
    """
    测试：不同种子的 rng 产生不同序列
    """
    rng1 = get_rng(42)
    rng2 = get_rng(99)

    seq1 = rng1.standard_normal(10)
    seq2 = rng2.standard_normal(10)

    assert not np.array_equal(seq1, seq2)
