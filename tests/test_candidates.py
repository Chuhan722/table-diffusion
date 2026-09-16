"""候选集枚举与参考分布的锚点测试。"""
import numpy as np
import pytest

from resevo.candidates import (
    BlockSupport,
    full_single_row_supports,
    normalize_block,
    validate_partition,
)


def test_full_supports_partition_and_size():
    supports = full_single_row_supports(4, 4)
    validate_partition(supports, 4)
    assert len(supports) == 4
    for i, spec in enumerate(supports):
        assert spec.rows == (i,)
        assert spec.outcomes == ((0,), (1,), (2,), (3,))


def test_reference_distribution_anchor():
    # 四条记录例子，保持 0.9，其余三个后继各 1/30
    supports = full_single_row_supports(4, 4)
    ids = np.array([0, 1, 2, 3], dtype=np.int64)
    block = normalize_block(supports[1], ids, 4)
    assert block.outcomes[0] == (1,)
    assert len(block.outcomes) == 4
    assert np.allclose(block.reference, [0.9, 1 / 30, 1 / 30, 1 / 30], atol=1e-15)
    assert abs(block.reference.sum() - 1.0) < 1e-15


def test_stay_appears_exactly_once_even_if_missing_from_input():
    # 输入支持里没写源元组，规范化后保持状态也必须出现恰好一次
    spec = BlockSupport((0,), ((2,), (3,)))
    ids = np.array([0], dtype=np.int64)
    block = normalize_block(spec, ids, 4)
    assert block.outcomes == ((0,), (2,), (3,))
    assert block.outcomes.count((0,)) == 1
    assert np.allclose(block.reference, [0.9, 0.05, 0.05], atol=1e-15)


def test_duplicate_outcomes_merge_mass():
    # 相同后继必须先合并参考质量再倾斜，总质量守恒
    spec = BlockSupport((0,), ((1,), (1,), (2,)), mobility=(1.0, 1.0, 2.0))
    ids = np.array([0], dtype=np.int64)
    block = normalize_block(spec, ids, 4)
    assert block.outcomes == ((0,), (1,), (2,))
    assert np.allclose(block.reference, [0.9, 0.05, 0.05], atol=1e-15)
    assert abs(block.reference.sum() - 1.0) < 1e-15


def test_no_other_successor_keeps_probability_one():
    supports = full_single_row_supports(1, 1)
    ids = np.array([0], dtype=np.int64)
    block = normalize_block(supports[0], ids, 1)
    assert block.outcomes == ((0,),)
    assert np.array_equal(block.reference, [1.0])


def test_legal_states_exclude_forbidden():
    # 规则"若 A=1 则 B=1"，合法状态 00,01,11 编号 0,1,3，支持不出现 10 编号 2
    supports = full_single_row_supports(2, 4, legal_states=[0, 1, 3])
    for spec in supports:
        assert (2,) not in spec.outcomes
        assert spec.outcomes == ((0,), (1,), (3,))


def test_support_does_not_depend_on_target():
    # 支持与 R 的构造完全不接收目标 y，两次构造逐位一致
    a = full_single_row_supports(3, 4)
    b = full_single_row_supports(3, 4)
    assert a == b


def test_bad_inputs_rejected():
    ids = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError):
        normalize_block(BlockSupport((0,), ((4,),)), ids, 4)
    with pytest.raises(ValueError):
        normalize_block(BlockSupport((0,), ((1,),), mobility=(-1.0,)), ids, 4)
    with pytest.raises(ValueError):
        normalize_block(BlockSupport((0,), ((1, 2),)), ids, 4)
    with pytest.raises(ValueError):
        validate_partition([BlockSupport((0,), ((0,),)), BlockSupport((0,), ((0,),))], 2)
    with pytest.raises(ValueError):
        full_single_row_supports(2, 4, legal_states=[0, 0, 1])
