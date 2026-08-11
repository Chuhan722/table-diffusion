"""
测试度量计算模块
"""
import numpy as np
import pytest
from table_diffevo.metrics import (
    compute_normalized_l1,
    compute_squared_loss,
    compute_all_metrics
)
from table_diffevo.objective import compute_loss


def test_normalized_l1_basic():
    """测试 normalized_l1 的基本计算"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])
    n_records = 300

    result = compute_normalized_l1(target, current, n_records)
    # (10 + 5 + 0) / 3 / 300 = 15 / 900 = 0.01666...
    expected = (10 + 5 + 0) / 3 / 300

    assert abs(result - expected) < 1e-10


def test_normalized_l1_perfect_match():
    """完美匹配时 normalized_l1 应为 0"""
    target = np.array([100, 200, 50])
    current = target.copy()

    result = compute_normalized_l1(target, current, 300)
    assert result == 0.0


def test_normalized_l1_shape_mismatch():
    """测试形状不匹配的错误处理"""
    target = np.array([100, 200])
    current = np.array([90, 210, 50])

    with pytest.raises(ValueError, match="形状不一致"):
        compute_normalized_l1(target, current, 300)


def test_normalized_l1_invalid_n_records():
    """测试非法 n_records"""
    target = np.array([100, 200])
    current = np.array([90, 210])

    with pytest.raises(ValueError, match="必须为正数"):
        compute_normalized_l1(target, current, -1)

    with pytest.raises(ValueError, match="必须为正数"):
        compute_normalized_l1(target, current, 0)


def test_squared_loss_basic():
    """测试 squared_loss 的基本计算"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])

    result = compute_squared_loss(target, current)
    # ½(10² + 5² + 0²) = ½(100 + 25) = 62.5
    expected = 0.5 * (10**2 + 5**2 + 0**2)

    assert abs(result - expected) < 1e-10


def test_squared_loss_consistency_with_objective():
    """squared_loss 应与 objective.compute_loss 完全一致"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])

    result = compute_squared_loss(target, current)
    expected = compute_loss(target, current)

    assert result == expected


def test_squared_loss_perfect_match():
    """完美匹配时 squared_loss 应为 0"""
    target = np.array([100, 200, 50])
    current = target.copy()

    result = compute_squared_loss(target, current)
    assert result == 0.0


def test_compute_all_metrics():
    """测试一次性计算所有度量"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])
    n_records = 300

    l1, q, residual = compute_all_metrics(target, current, n_records)

    # 验证每个度量
    assert abs(l1 - compute_normalized_l1(target, current, n_records)) < 1e-10
    assert abs(q - compute_squared_loss(target, current)) < 1e-10
    assert np.allclose(residual, (target - current) / n_records)


def test_compute_all_metrics_perfect_match():
    """完美匹配时所有度量应为 0"""
    target = np.array([100, 200, 50])
    current = target.copy()
    n_records = 300

    l1, q, residual = compute_all_metrics(target, current, n_records)

    assert l1 == 0.0
    assert q == 0.0
    assert np.all(residual == 0.0)


def test_compute_all_metrics_residual_sign():
    """测试残差的符号正确性"""
    target = np.array([180, 95, 42])  # 目标
    current = np.array([170, 100, 42])  # 当前（第一个偏低，第二个偏高）
    n_records = 300

    l1, q, residual = compute_all_metrics(target, current, n_records)

    # 残差 = (target - current) / n_records
    # 第一个：(180-170)/300 = 10/300 > 0（需要增加）
    # 第二个：(95-100)/300 = -5/300 < 0（需要减少）
    # 第三个：(42-42)/300 = 0（已达标）
    assert residual[0] > 0
    assert residual[1] < 0
    assert residual[2] == 0


def test_all_metrics_handle_float_arrays():
    """测试处理浮点数组"""
    target = np.array([180.5, 95.3, 42.0])
    current = np.array([170.2, 100.1, 42.0])
    n_records = 300

    l1, q, residual = compute_all_metrics(target, current, n_records)

    # 应该正常计算，不抛出错误
    assert isinstance(l1, float)
    assert isinstance(q, float)
    assert isinstance(residual, np.ndarray)
