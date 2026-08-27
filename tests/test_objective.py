"""
测试残差计算

验证：
1. 无噪声阶段的比例残差 (y - q) / N
2. 残差方向正确（偏低为正、偏高为负、达标为零）
3. 归一化正确（除以 N）
4. 噪声容忍区正确（|残差| < κσ 时归零）
5. 输入校验（形状不一致、N 非正）
"""
import numpy as np
import pytest
from table_diffevo.objective import compute_residual, compute_loss


def test_basic_proportional_residual():
    """测试无噪声阶段的比例残差 (y - q) / N"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])

    residual = compute_residual(target, current, n_records=300)

    expected = np.array([10 / 300, -5 / 300, 0 / 300])
    np.testing.assert_allclose(residual, expected)


def test_residual_direction():
    """测试残差方向：偏低为正、偏高为负、达标为零"""
    target = np.array([100, 50, 30])
    current = np.array([80, 70, 30])  # q1 偏低, q2 偏高, q3 达标

    residual = compute_residual(target, current, n_records=200)

    assert residual[0] > 0  # 偏低 → 正（需要增加）
    assert residual[1] < 0  # 偏高 → 负（需要减少）
    assert residual[2] == 0  # 达标 → 零（中性）


def test_normalization_by_n():
    """测试归一化：不同 N 下相同计数差得到不同比例"""
    target = np.array([50])
    current = np.array([40])  # 计数差固定为 10

    r_small = compute_residual(target, current, n_records=100)
    r_large = compute_residual(target, current, n_records=1000)

    # 大数据集下同样的计数差，比例残差更小
    assert r_small[0] == 10 / 100
    assert r_large[0] == 10 / 1000
    assert r_small[0] > r_large[0]


def test_residual_range():
    """测试残差落在 [-1, 1] 区间"""
    # 极端情况：目标全满足 vs 全不满足
    target = np.array([300, 0])
    current = np.array([0, 300])

    residual = compute_residual(target, current, n_records=300)

    assert np.all(residual >= -1)
    assert np.all(residual <= 1)
    assert residual[0] == 1.0   # (300-0)/300
    assert residual[1] == -1.0  # (0-300)/300


def test_zero_residual_when_exact():
    """测试完全达标时残差为零向量"""
    target = np.array([180, 95, 42, 100])
    current = np.array([180, 95, 42, 100])

    residual = compute_residual(target, current, n_records=300)

    np.testing.assert_array_equal(residual, np.zeros(4))


def test_noise_tolerance_zeros_small_residual():
    """测试噪声容忍区：误差小于 κσ 时归零"""
    target = np.array([100, 100])
    current = np.array([95, 80])  # 误差分别为 5, 20
    sigma = np.array([10.0, 10.0])  # 噪声标准差 10

    # kappa=1.0，容忍区 = 10
    residual = compute_residual(target, current, n_records=200, sigma=sigma, kappa=1.0)

    # q1: |5| < 10 → 归零
    assert residual[0] == 0
    # q2: |20| - 10 = 10，方向为正 → 10/200
    assert residual[1] == pytest.approx(10 / 200)


def test_noise_tolerance_preserves_direction():
    """测试噪声容忍区保留方向（偏高时超出容忍区为负）"""
    target = np.array([50])
    current = np.array([80])  # 偏高，误差 30
    sigma = np.array([10.0])

    residual = compute_residual(target, current, n_records=100, sigma=sigma, kappa=2.0)

    # 容忍区 = 2*10 = 20，超出量 = 30 - 20 = 10，方向为负
    assert residual[0] == pytest.approx(-10 / 100)


def test_sigma_none_equals_no_tolerance():
    """测试 sigma=None 时等价于无容忍区（保留全部残差）"""
    target = np.array([100, 50])
    current = np.array([98, 55])

    r_none = compute_residual(target, current, n_records=200, sigma=None)
    r_zero = compute_residual(target, current, n_records=200,
                              sigma=np.array([0.0, 0.0]), kappa=1.0)

    np.testing.assert_allclose(r_none, r_zero)


def test_invalid_n_records():
    """测试 n_records 非正时报错"""
    target = np.array([100])
    current = np.array([90])

    with pytest.raises(ValueError, match="n_records 必须为正数"):
        compute_residual(target, current, n_records=0)

    with pytest.raises(ValueError, match="n_records 必须为正数"):
        compute_residual(target, current, n_records=-5)


def test_shape_mismatch():
    """测试 target 与 current 形状不一致时报错"""
    target = np.array([100, 50, 30])
    current = np.array([90, 45])

    with pytest.raises(ValueError, match="形状不一致"):
        compute_residual(target, current, n_records=300)


def test_integration_with_queries():
    """
    集成测试：残差计算与查询评价器配合

    模拟主循环场景：目标达标时残差应为零。
    """
    from table_diffevo.queries import load_queries, load_data, evaluate_table

    df = load_data("data/test_300x10/test_300x10.csv")
    queries = load_queries("configs/test_300x10/measured_50query.json")

    # 当前答案 = 在原数据上评价（等于目标，因为目标就是原数据的真实计数）
    current = evaluate_table(df, queries)
    target = np.array([q["result"] for q in queries], dtype=int)

    residual = compute_residual(target, current, n_records=len(df))

    # 原数据上，当前答案 = 目标，残差应全为零
    np.testing.assert_array_equal(residual, np.zeros(len(queries)))


# ---- compute_loss 测试 ----

def test_loss_basic():
    """无噪声：E = ½·Σ(y-q)²"""
    target = np.array([180, 95, 42])
    current = np.array([170, 100, 42])
    # ½(10² + 5² + 0²) = ½·125 = 62.5
    assert compute_loss(target, current) == 62.5


def test_loss_zero_when_exact():
    """完全达标 → loss = 0"""
    target = np.array([10, 20, 30])
    assert compute_loss(target, target) == 0.0


def test_loss_non_negative():
    """loss 恒非负"""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 300, size=50)
    current = rng.integers(0, 300, size=50)
    assert compute_loss(target, current) >= 0


def test_loss_symmetric_in_sign():
    """偏高偏低对称：残差 +d 与 -d 的 loss 相同"""
    target = np.array([100])
    assert compute_loss(target, np.array([90])) == compute_loss(target, np.array([110]))


def test_loss_weights():
    """权重放大对应查询的贡献"""
    target = np.array([100, 100])
    current = np.array([90, 90])  # 各偏 10
    # 无权重：½(100+100)=100
    assert compute_loss(target, current) == 100.0
    # 第一个查询权重 2：½(2·100 + 1·100)=150
    loss_w = compute_loss(target, current, weights=np.array([2.0, 1.0]))
    assert loss_w == 150.0


def test_loss_noise_tolerance():
    """噪声容忍区内的残差不计入 loss"""
    target = np.array([100])
    current = np.array([95])  # 残差 5
    sigma = np.array([10.0])
    # κσ = 1·10 = 10 > |5| → 残差全被容忍 → loss = 0
    assert compute_loss(target, current, sigma=sigma, kappa=1.0) == 0.0


def test_loss_noise_partial():
    """超出容忍区的部分才计入"""
    target = np.array([100])
    current = np.array([80])  # 残差 20
    sigma = np.array([10.0])
    # max(20 - 10, 0) = 10 → ½·10² = 50
    assert compute_loss(target, current, sigma=sigma, kappa=1.0) == 50.0


def test_loss_shape_mismatch():
    """形状不一致报错"""
    with pytest.raises(ValueError, match="形状不一致"):
        compute_loss(np.array([1, 2]), np.array([1, 2, 3]))


def test_loss_ordering_matches_proportional():
    """loss 大小排序与比例残差平方和一致（差常数因子 N²）"""
    target = np.array([100, 200, 50])
    q1 = np.array([90, 190, 45])   # 偏差较小
    q2 = np.array([70, 160, 20])   # 偏差较大
    assert compute_loss(target, q1) < compute_loss(target, q2)


# ---- 残差几何（Issue #57）----

def test_geometry_absolute_explicit_matches_default():
    """显式 geometry='absolute' 与默认调用逐位一致（向后兼容）"""
    target = np.array([180.0, 95.0, 42.0])
    current = np.array([170.0, 100.0, 42.0])
    default = compute_residual(target, current, n_records=300)
    explicit = compute_residual(
        target, current, n_records=300, geometry="absolute"
    )
    np.testing.assert_array_equal(default, explicit)


def test_geometry_relative_hand_computed():
    """relative 几何手算锚定：ε = raw / max(y, floor) / N"""
    target = np.array([100.0, 4.0, 0.0])
    current = np.array([90.0, 8.0, 3.0])
    # raw = [10, -4, -3]; 分母 max(y, 8) = [100, 8, 8]
    expected = np.array([10.0 / 100, -4.0 / 8, -3.0 / 8]) / 300
    result = compute_residual(
        target, current, n_records=300,
        geometry="relative", geometry_floor=8.0,
    )
    np.testing.assert_allclose(result, expected)


def test_geometry_relative_amplifies_rare_queries():
    """同样的计数残差，稀有查询的相对残差幅度更大"""
    target = np.array([5000.0, 50.0])
    current = np.array([4990.0, 40.0])  # 都差 10 个计数
    r = compute_residual(
        target, current, n_records=16181, geometry="relative"
    )
    assert abs(r[1]) > abs(r[0]) * 50  # 5000/50 = 100 倍分母差


def test_geometry_sqrt_relative_has_fixed_middle_strength():
    """目标相差100倍时，sqrt_relative放大10倍而非relative的100倍。"""
    target = np.array([900.0, 9.0])
    current = np.array([897.0, 6.0])  # 都少3个计数，且均高于floor=8
    absolute = compute_residual(
        target, current, n_records=1000, geometry="absolute"
    )
    sqrt_relative = compute_residual(
        target, current, n_records=1000, geometry="sqrt_relative"
    )
    relative = compute_residual(
        target, current, n_records=1000, geometry="relative"
    )

    np.testing.assert_allclose(sqrt_relative, [0.1 / 1000, 1.0 / 1000])
    np.testing.assert_allclose(
        abs(sqrt_relative[1] / sqrt_relative[0]), 10.0
    )
    np.testing.assert_allclose(abs(absolute[1] / absolute[0]), 1.0)
    np.testing.assert_allclose(abs(relative[1] / relative[0]), 100.0)


def test_geometry_sqrt_relative_floor_protects_small_targets():
    """sqrt_relative 对 target=0 使用 sqrt(floor)，保持有限且不改零点。"""
    target = np.array([0.0, 2.0])
    current = np.array([5.0, 2.0])
    r = compute_residual(
        target, current, n_records=100,
        geometry="sqrt_relative", geometry_floor=8.0,
    )
    assert np.all(np.isfinite(r))
    np.testing.assert_allclose(r[0], -5.0 / np.sqrt(8.0) / 100)
    assert r[1] == 0.0


def test_geometry_relative_floor_protects_small_targets():
    """target < floor 时分母用 floor，target=0 不产生除零"""
    target = np.array([0.0, 2.0])
    current = np.array([5.0, 2.0])
    r = compute_residual(
        target, current, n_records=100,
        geometry="relative", geometry_floor=8.0,
    )
    assert np.all(np.isfinite(r))
    np.testing.assert_allclose(r[0], -5.0 / 8.0 / 100)


def test_geometry_relative_tolerance_before_relativization():
    """σ/κ 容忍先于相对化：容忍后为零的分量保持零"""
    target = np.array([100.0, 100.0])
    current = np.array([95.0, 80.0])  # 残差 5、20
    sigma = np.array([10.0, 10.0])
    r = compute_residual(
        target, current, n_records=300, sigma=sigma, kappa=1.0,
        geometry="relative",
    )
    assert r[0] == 0.0  # |5| < κσ=10 → 容忍归零
    # 第二个：raw=+20，max(20-10,0)=10 → +10/max(100,8)/300
    np.testing.assert_allclose(r[1], 10.0 / 100.0 / 300)


def test_geometry_relative_zero_when_exact():
    """精确匹配时 relative 残差为零（零点与 absolute 相同）"""
    target = np.array([100.0, 50.0])
    r = compute_residual(target, target.copy(), n_records=300,
                         geometry="relative")
    np.testing.assert_array_equal(r, np.zeros(2))


def test_geometry_invalid_name_raises():
    """非法 geometry 报错"""
    with pytest.raises(ValueError, match="geometry 必须是"):
        compute_residual(
            np.array([1.0]), np.array([1.0]), n_records=10,
            geometry="chi2",
        )


def test_geometry_floor_nonpositive_raises():
    """relative 下 floor <= 0 报错"""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="geometry_floor 必须是正有限数"):
            compute_residual(
                np.array([1.0]), np.array([1.0]), n_records=10,
                geometry="relative", geometry_floor=bad,
            )


def test_geometry_floor_ignored_for_absolute():
    """absolute 几何不使用 floor（非法 floor 也不报错、结果不变）"""
    target = np.array([100.0])
    current = np.array([90.0])
    r1 = compute_residual(target, current, n_records=300)
    r2 = compute_residual(
        target, current, n_records=300,
        geometry="absolute", geometry_floor=-5.0,
    )
    np.testing.assert_array_equal(r1, r2)


def test_geometry_floor_rejects_inf_nan_bool():
    """floor=inf 会把全部残差压成 0 并错误触发精确达标早停（PR #59
    审查复现的漏洞），inf/nan/bool 必须在入口拒绝"""
    target = np.array([100.0])
    current = np.array([90.0])
    for bad in (np.inf, float("inf"), np.nan, True):
        with pytest.raises(ValueError, match="geometry_floor 必须是正有限数"):
            compute_residual(
                target, current, n_records=300,
                geometry="relative", geometry_floor=bad,
            )
