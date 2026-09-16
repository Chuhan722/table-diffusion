"""熵校准的锚点测试，含 D(beta) 数值表、beta 根与微小增益反例。"""
import numpy as np
import pytest

from conftest import build_four_record_blocks
from resevo.tilt import calibrate_beta, tilted_distributions

BETA_STAR = 3.981402641214968


def four_gains_refs():
    _, _, blocks = build_four_record_blocks()
    gains = [g for _, _, g in blocks]
    refs = [nb.reference for nb, _, _ in blocks]
    return gains, refs


def test_max_gain_sum_and_requirement_anchor():
    gains, refs = four_gains_refs()
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    assert abs(result.max_gain_sum - 1.5) < 1e-12
    assert abs(result.required_gain - 0.75) < 1e-12


def test_direction_gain_table_anchor():
    # 文档 D(beta) 数值表，六位小数锚点
    gains, refs = four_gains_refs()
    table = {
        0.0: -0.366667,
        1.0: 0.065895,
        2.0: 0.261787,
        3.0: 0.493206,
        4.0: 0.754670,
        8.0: 1.316503,
    }
    for beta, expected in table.items():
        _, D = tilted_distributions(gains, refs, beta)
        assert abs(D - expected) < 5e-7, (beta, D)
    _, D_root = tilted_distributions(gains, refs, 3.9814026412)
    assert abs(D_root - 0.75) < 1e-9


def test_beta_root_anchor():
    gains, refs = four_gains_refs()
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    assert not result.frozen
    assert abs(result.beta - BETA_STAR) < 1e-9
    assert abs(result.direction_gain - 0.75) < 1e-12
    # 二分取上端点，约束必须被满足而不是近似差一点
    assert result.direction_gain >= result.required_gain


def test_monotone_direction_gain():
    gains, refs = four_gains_refs()
    values = [tilted_distributions(gains, refs, b)[1] for b in np.linspace(0, 8, 30)]
    assert all(x <= y + 1e-12 for x, y in zip(values, values[1:]))


def test_beta_zero_branch():
    # 参考分布已满足约束时直接取 beta=0
    gains = [np.array([0.0, 1.0])]
    refs = [np.array([0.5, 0.5])]
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    assert result.beta == 0.0
    assert not result.frozen
    assert abs(result.direction_gain - 0.5) < 1e-15


def test_all_zero_gain_freezes():
    gains = [np.zeros(3), np.zeros(2)]
    refs = [np.array([0.9, 0.05, 0.05]), np.array([0.9, 0.1])]
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    assert result.frozen
    assert result.beta == 0.0
    assert result.direction_gain == 0.0
    for P in result.probabilities:
        assert P[0] == 1.0
        assert P[1:].sum() == 0.0


def test_negative_gain_keeps_positive_probability():
    # 全支撑软概率，负增益后继不被删除，仍有严格正概率
    gains, refs = four_gains_refs()
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    for P in result.probabilities:
        assert np.all(P > 0.0)


def test_log_ratio_is_constant_within_block():
    # 直接检验程序真在计算指数倾斜，log(P/R)-beta*G 块内为同一常数
    gains, refs = four_gains_refs()
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    for P, R, g in zip(result.probabilities, refs, gains):
        c = np.log(P / R) - result.beta * g
        assert c.max() - c.min() < 1e-9


def test_common_scale_equivalence():
    # 所有块共同除以正尺度 s 后求 beta 再除回，概率不变
    gains, refs = four_gains_refs()
    base = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    s = 7.3
    scaled = calibrate_beta([g / s for g in gains], refs, old_loss=1.0, alpha=0.5)
    assert abs(scaled.beta / s - base.beta) < 1e-9
    for P_s, P_b in zip(scaled.probabilities, base.probabilities):
        assert np.allclose(P_s, P_b, atol=1e-12)


@pytest.mark.parametrize(
    "eps,beta_anchor",
    [(1e-1, 21.9722), (1e-3, 2197.2246), (1e-6, 2197224.5773)],
)
def test_tiny_gain_counterexample(eps, beta_anchor):
    # 微小增益反例，唯一非保持动作增益 eps，保持 0.9 与 alpha=0.5 逼出巨大 beta
    gains = [np.array([0.0, eps])]
    refs = [np.array([0.9, 0.1])]
    result = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    assert abs(result.beta - np.log(9.0) / eps) / (np.log(9.0) / eps) < 1e-9
    assert abs(result.beta - beta_anchor) / beta_anchor < 1e-5
    assert abs(result.probabilities[0][1] - 0.5) < 1e-9


def test_bad_alpha_rejected():
    with pytest.raises(ValueError):
        calibrate_beta([np.array([0.0, 1.0])], [np.array([0.9, 0.1])], 1.0, alpha=1.5)


def test_calibrate_beta_flat_matches_blockwise_random():
    """分段矢量化熵校准与逐块版在 30 个随机例上一致，含重数加权。"""
    from resevo.tilt import calibrate_beta_flat

    rng = np.random.default_rng(0)
    for trial in range(30):
        num_groups = int(rng.integers(1, 8))
        gains, refs, offsets = [], [], [0]
        for _ in range(num_groups):
            n = int(rng.integers(1, 6)) + 1
            g = rng.normal(0, 2, n)
            g[0] = 0.0
            R = rng.uniform(0.1, 1, n)
            R /= R.sum()
            gains.append(g)
            refs.append(R)
            offsets.append(offsets[-1] + n)
        mult = rng.integers(1, 5, num_groups).astype(float)
        old_loss = float(rng.uniform(0.5, 100))
        a = calibrate_beta(gains, refs, old_loss, multiplicities=mult)
        b = calibrate_beta_flat(
            np.concatenate(gains), np.concatenate(refs), np.array(offsets),
            old_loss, multiplicities=mult,
        )
        assert a.frozen == b.frozen
        assert b.beta == pytest.approx(a.beta, rel=1e-9, abs=1e-12)
        assert b.direction_gain == pytest.approx(a.direction_gain, rel=1e-9)
        assert b.max_gain_sum == pytest.approx(a.max_gain_sum, rel=1e-12)
        np.testing.assert_allclose(
            np.concatenate(a.probabilities), b.probabilities, rtol=1e-9, atol=1e-12
        )


def test_calibrate_beta_flat_frozen_puts_mass_on_sources():
    """全负增益时冻结，质量全部落在各段源项。"""
    from resevo.tilt import calibrate_beta_flat

    gains = np.array([0.0, -1.0, -2.0, 0.0, -3.0])
    refs = np.array([0.9, 0.05, 0.05, 0.9, 0.1])
    offsets = np.array([0, 3, 5])
    res = calibrate_beta_flat(gains, refs, offsets, 10.0)
    assert res.frozen
    np.testing.assert_allclose(res.probabilities, [1.0, 0.0, 0.0, 1.0, 0.0])
