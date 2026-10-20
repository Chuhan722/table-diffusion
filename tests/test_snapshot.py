"""损失快照测试（第六十一步）：默认关闭零改变，触发时刻与起点表正确，停止轮也触发，参数校验。"""
import numpy as np
import pytest

from test_batchkernel import _scene
from test_earlystop import _evolve


def _run_with_snapshots(scene_seed, thresholds, rounds=40, **kwargs):
    registry, ids, weights, _ = _scene(scene_seed, num_rows=30)
    fired: list[tuple] = []

    def hook(index, threshold, round_index, old_loss, state_ids):
        fired.append((index, threshold, round_index, old_loss, state_ids))

    out = _evolve(
        registry, ids, weights, rounds=rounds,
        snapshot_losses=thresholds, snapshot_hook=hook, **kwargs,
    )
    return out, fired


def test_snapshot_default_off_bitwise_identical():
    """开快照与不开的轨迹和终表逐位一致，钩子只读。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    base = _evolve(r1, ids1, w1)
    losses = [r.old_loss for r in base.records]
    thr = [losses[0] * 0.9, losses[-1] * 1.05]
    out, fired = _run_with_snapshots(2, thr)
    assert [r.old_loss for r in out.records] == losses
    assert np.array_equal(base.state_ids, out.state_ids)
    assert fired


def test_snapshot_fires_once_at_first_crossing_with_start_table():
    """每个阈值只触发一次，轮号是首次不高于阈值的轮，表等于该轮起点表。"""
    r0, ids0, w0, _ = _scene(3, num_rows=30)
    full = _evolve(r0, ids0, w0, rounds=40)
    losses = [r.old_loss for r in full.records]
    thr = [losses[0] * 0.95, losses[-1] * 1.02]
    out, fired = _run_with_snapshots(3, thr, rounds=40)
    assert sorted(f[0] for f in fired) == [0, 1]
    for index, threshold, k, old_loss, state_ids in fired:
        assert threshold == thr[index]
        assert old_loss == losses[k]
        assert old_loss <= threshold
        assert all(l > threshold for l in losses[:k])
        # 起点表 = 跑 k 轮后的表（k=0 为初始表）
        rk, idsk, wk, _ = _scene(3, num_rows=30)
        prefix = _evolve(rk, idsk, wk, rounds=k) if k > 0 else None
        expect = prefix.state_ids if prefix is not None else np.asarray(idsk, dtype=np.int64)
        assert np.array_equal(state_ids, expect)


def test_snapshot_never_fires_above_reach():
    """阈值低于全程最低损失则不触发。"""
    r0, ids0, w0, _ = _scene(4, num_rows=30)
    full = _evolve(r0, ids0, w0, rounds=30)
    floor = min(r.old_loss for r in full.records)
    _, fired = _run_with_snapshots(4, [floor * 0.5], rounds=30)
    assert fired == []


def test_snapshot_fires_on_stop_round():
    """平台停触发轮的起点损失若首次过线，快照仍触发且表等于返回表。"""
    thr_probe, _ = _run_with_snapshots(5, [1.0], rounds=40, stop_threshold=1.0, stop_lag=5)
    assert thr_probe.stop_reason == "loss_plateau"
    last = thr_probe.records[-1].old_loss
    out, fired = _run_with_snapshots(5, [last], rounds=40, stop_threshold=1.0, stop_lag=5)
    assert fired and fired[0][2] <= len(out.records) - 1
    if fired[0][2] == len(out.records) - 1:
        assert np.array_equal(fired[0][4], out.state_ids)


def test_snapshot_validation():
    """阈值与钩子须同时给出，阈值必须为正有限数。"""
    registry, ids, weights, _ = _scene(6, num_rows=20)
    with pytest.raises(ValueError):
        _evolve(registry, ids, weights, rounds=5, snapshot_losses=[1.0])
    with pytest.raises(ValueError):
        _evolve(registry, ids, weights, rounds=5, snapshot_hook=lambda *a: None)
    with pytest.raises(ValueError):
        _evolve(
            registry, ids, weights, rounds=5,
            snapshot_losses=[0.0], snapshot_hook=lambda *a: None,
        )


def test_snapshot_equals_stop_floor_table():
    """阈值 T 的快照表与 --stop-floor T 硬地板停返回的表逐位相同、轮号相同（第六十二步：×1.0 快照即地板停产物）。"""
    r0, ids0, w0, _ = _scene(7, num_rows=30)
    full = _evolve(r0, ids0, w0, rounds=40)
    losses = [r.old_loss for r in full.records]
    thr = (losses[0] + losses[-1]) / 2.0
    out, fired = _run_with_snapshots(7, [thr], rounds=40)
    assert len(fired) == 1 and fired[0][2] > 0
    rf, idsf, wf, _ = _scene(7, num_rows=30)
    floored = _evolve(rf, idsf, wf, rounds=40, stop_floor=thr)
    assert floored.stop_reason == "floor_reached"
    assert len(floored.records) - 1 == fired[0][2]
    assert np.array_equal(floored.state_ids, fired[0][4])
    assert not np.array_equal(floored.state_ids, out.state_ids)
