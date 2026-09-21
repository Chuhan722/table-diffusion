"""alpha 透传与 beta 网格探针测试，探针默认关闭零改变。"""
import numpy as np

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs, query_field_sets

from test_batchkernel import _scene


def _provider(registry, weights, menu_seed=5):
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    return make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(menu_seed), joint_field_sets=fs,
    )


def _evolve(registry, ids, weights, rounds=30, **kwargs):
    provider = _provider(registry, weights)
    return evolve_batch(
        ids, rounds, np.random.default_rng(9), provider, registry,
        max_frozen_retries=3, **kwargs,
    )


def test_alpha_passthrough_changes_calibration():
    """alpha 越大要求越高首轮 beta 越大，多轮轨迹随之分叉。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    lo = _evolve(r1, ids1, w1, rounds=1, alpha=0.3)
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    hi = _evolve(r2, ids2, w2, rounds=1, alpha=0.7)
    # 首轮两跑的表与菜单相同，增益相同，要求更高解出的 beta 更大
    assert hi.records[0].beta > lo.records[0].beta
    assert lo.records[0].max_gain_sum == hi.records[0].max_gain_sum
    assert lo.records[0].max_gain_sum > 0.0


def test_probe_default_off_none():
    """不开探针时旁路日志为空，记录含 max_gain_sum。"""
    registry, ids, weights, _ = _scene(3, num_rows=30)
    out = _evolve(registry, ids, weights, rounds=5)
    assert out.probes == []
    assert all(r.max_gain_sum >= 0.0 for r in out.records)


def test_probe_readonly_bitwise_identical():
    """开关探针的演化轨迹与终表逐位一致，探针纯只读。"""
    r1, ids1, w1, _ = _scene(4, num_rows=30)
    base = _evolve(r1, ids1, w1, rounds=20)
    r2, ids2, w2, _ = _scene(4, num_rows=30)
    probed = _evolve(r2, ids2, w2, rounds=20, probe_interval=3)
    assert [r.old_loss for r in base.records] == [r.old_loss for r in probed.records]
    assert [r.beta for r in base.records] == [r.beta for r in probed.records]
    assert np.array_equal(base.state_ids, probed.state_ids)
    assert len(probed.probes) > 0


def test_probe_j_used_matches_records():
    """探针的 J_used 等于该轮账面期望下降，J_best 非负且探针轮号对齐间隔。"""
    registry, ids, weights, _ = _scene(5, num_rows=30)
    out = _evolve(registry, ids, weights, rounds=20, probe_interval=4)
    by_round = {r.round_index: r for r in out.records}
    for p in out.probes:
        rec = by_round[p.round_index]
        assert p.beta_used == rec.beta
        assert np.isclose(p.j_used, rec.old_loss - rec.expected_loss)
        assert p.j_best >= 0.0
        assert p.round_index % 4 == 0
