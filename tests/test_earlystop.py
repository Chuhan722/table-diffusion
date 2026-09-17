"""平台早停测试，默认关闭零改变，前缀逐位一致，平台触发与参数校验。"""
import numpy as np
import pytest

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs, query_field_sets

from test_batchkernel import _scene


def _provider(registry, weights, menu_seed=5):
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    return make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(menu_seed), joint_field_sets=fs,
    )


def _evolve(registry, ids, weights, rounds=40, **kwargs):
    provider = _provider(registry, weights)
    return evolve_batch(
        ids, rounds, np.random.default_rng(9), provider, registry,
        max_frozen_retries=3, **kwargs,
    )


def test_earlystop_default_off_bitwise_identical():
    """阈值取零与不传参数的轨迹逐位一致，旧行为零改变。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    base = _evolve(r1, ids1, w1)
    off = _evolve(r2, ids2, w2, stop_threshold=0.0, stop_lag=5)
    assert [r.old_loss for r in base.records] == [r.old_loss for r in off.records]
    assert np.array_equal(base.state_ids, off.state_ids)


def test_earlystop_triggers_on_plateau():
    """阈值取一必在第二个窗口边界停，停止原因为 loss_plateau。"""
    registry, ids, weights, _ = _scene(3, num_rows=30)
    out = _evolve(registry, ids, weights, rounds=40, stop_threshold=1.0, stop_lag=5)
    assert out.stop_reason == "loss_plateau"
    assert len(out.records) == 10


def test_earlystop_prefix_bitwise_identical():
    """早停轨迹是全量跑的前缀，返回表等于触发轮起点的全量表。"""
    r1, ids1, w1, _ = _scene(4, num_rows=30)
    stopped = _evolve(r1, ids1, w1, rounds=60, stop_threshold=1.0, stop_lag=5)
    assert stopped.stop_reason == "loss_plateau"
    k = len(stopped.records)  # 触发轮编号为 k-1，其移动未应用
    r2, ids2, w2, _ = _scene(4, num_rows=30)
    full = _evolve(r2, ids2, w2, rounds=60)
    assert [r.old_loss for r in stopped.records] == [
        r.old_loss for r in full.records[:k]
    ]
    r3, ids3, w3, _ = _scene(4, num_rows=30)
    prefix = _evolve(r3, ids3, w3, rounds=k - 1)
    assert np.array_equal(stopped.state_ids, prefix.state_ids)


def test_earlystop_validation():
    """负阈值与非正窗口轮数报错。"""
    registry, ids, weights, _ = _scene(5, num_rows=20)
    with pytest.raises(ValueError):
        _evolve(registry, ids, weights, rounds=5, stop_threshold=-0.1)
    with pytest.raises(ValueError):
        _evolve(registry, ids, weights, rounds=5, stop_threshold=0.5, stop_lag=0)
