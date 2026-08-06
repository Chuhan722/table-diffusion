#!/usr/bin/env python3
"""验证 metrics.py 与 evolution.py 的计算一致性。

此脚本确保新实现的 metrics 模块与现有演化代码产生完全相同的结果。
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.objective import compute_loss


def verify_normalized_l1():
    """验证归一化 L1 计算与 evolution.py:904 一致。"""
    print("验证归一化 L1 计算...")

    # 测试用例 1：简单整数
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])
    n_records = 300

    # evolution.py:904 的逻辑
    abs_errors = np.abs(target - current)
    expected = float(np.mean(abs_errors) / n_records)

    # metrics.py 的实现
    actual = compute_normalized_l1(target, current, n_records)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 1 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 1: {actual:.10f}")

    # 测试用例 2：浮点数
    target = np.array([123.456, 789.012, 345.678])
    current = np.array([120.0, 800.0, 340.0])
    n_records = 16181

    abs_errors = np.abs(target - current)
    expected = float(np.mean(abs_errors) / n_records)
    actual = compute_normalized_l1(target, current, n_records)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 2 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 2: {actual:.10f}")

    # 测试用例 3：大规模随机
    np.random.seed(42)
    target = np.random.rand(1000) * 1000
    current = np.random.rand(1000) * 1000
    n_records = 16181

    abs_errors = np.abs(target - current)
    expected = float(np.mean(abs_errors) / n_records)
    actual = compute_normalized_l1(target, current, n_records)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 3 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 3: {actual:.10f}")

    # 测试用例 4：完全匹配
    target = np.array([100, 200, 300])
    current = np.array([100, 200, 300])
    n_records = 1000

    abs_errors = np.abs(target - current)
    expected = float(np.mean(abs_errors) / n_records)
    actual = compute_normalized_l1(target, current, n_records)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 4 失败: expected={expected}, actual={actual}"
    assert actual == 0.0, "完全匹配应返回 0"
    print(f"  ✓ 测试用例 4: {actual:.10f} (完全匹配)")

    print("✅ 归一化 L1 计算验证通过\n")


def verify_squared_loss():
    """验证平方 loss 计算与 objective.compute_loss 一致。"""
    print("验证平方 loss 计算...")

    # 测试用例 1：简单整数
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])

    # objective.compute_loss 的直接调用
    expected = compute_loss(target, current, sigma=None, kappa=1.0, weights=None)

    # metrics.py 的实现
    actual = compute_squared_loss(target, current)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 1 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 1: {actual:.2f}")

    # 测试用例 2：浮点数
    target = np.array([123.456, 789.012, 345.678])
    current = np.array([120.0, 800.0, 340.0])

    expected = compute_loss(target, current, sigma=None, kappa=1.0, weights=None)
    actual = compute_squared_loss(target, current)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 2 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 2: {actual:.2f}")

    # 测试用例 3：大规模随机
    np.random.seed(42)
    target = np.random.rand(1000) * 1000
    current = np.random.rand(1000) * 1000

    expected = compute_loss(target, current, sigma=None, kappa=1.0, weights=None)
    actual = compute_squared_loss(target, current)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 3 失败: expected={expected}, actual={actual}"
    print(f"  ✓ 测试用例 3: {actual:.2f}")

    # 测试用例 4：完全匹配
    target = np.array([100, 200, 300])
    current = np.array([100, 200, 300])

    expected = compute_loss(target, current, sigma=None, kappa=1.0, weights=None)
    actual = compute_squared_loss(target, current)

    assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), \
        f"测试用例 4 失败: expected={expected}, actual={actual}"
    assert actual == 0.0, "完全匹配应返回 0"
    print(f"  ✓ 测试用例 4: {actual:.2f} (完全匹配)")

    print("✅ 平方 loss 计算验证通过\n")


def main():
    print("=" * 60)
    print("度量计算一致性验证")
    print("=" * 60)
    print()

    try:
        verify_normalized_l1()
        verify_squared_loss()

        print("=" * 60)
        print("✅ 所有验证通过")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ 验证失败: {e}")
        print("=" * 60)
        return 1

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 意外错误: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
