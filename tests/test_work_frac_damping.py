"""步长层开关测试，work-frac 比例折算与 damping 透传，默认零改变。"""
import importlib.util
from pathlib import Path

import numpy as np

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs, query_field_sets

from test_batchkernel import _scene


def _load_run_plants():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_plants", root / "scripts" / "run_plants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _provider(registry, weights, menu_seed=5, **kwargs):
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    return make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(menu_seed), joint_field_sets=fs, **kwargs,
    )


def _evolve(registry, ids, weights, rounds=30, provider_kwargs=None, **kwargs):
    provider = _provider(registry, weights, **(provider_kwargs or {}))
    return evolve_batch(
        ids, rounds, np.random.default_rng(9), provider, registry,
        max_frozen_retries=3, **kwargs,
    )


def test_resolve_work_rows_rounding():
    """比例折算四舍五入且至少一行，零比例沿用绝对值。"""
    rp = _load_run_plants()
    assert rp.resolve_work_rows(16181, 0, 0.20) == 3236
    assert rp.resolve_work_rows(17412, 0, 0.20) == 3482
    assert rp.resolve_work_rows(32561, 0, 0.20) == 6512
    assert rp.resolve_work_rows(17412, 0, 0.15) == 2612
    assert rp.resolve_work_rows(10, 0, 0.04) == 1
    assert rp.resolve_work_rows(16181, 4096, 0.0) == 4096
    assert rp.resolve_work_rows(16181, 0, 0.0) == 0


def test_work_frac_matches_equal_work_rows():
    """比例折算后透传与直接给等值绝对行数轨迹逐位一致。"""
    rp = _load_run_plants()
    k = rp.resolve_work_rows(30, 0, 0.27)
    assert k == 8
    r1, ids1, w1, _ = _scene(4, num_rows=30)
    r2, ids2, w2, _ = _scene(4, num_rows=30)
    a = _evolve(
        r1, ids1, w1, rounds=40,
        provider_kwargs=dict(work_rows=8, select_rng=np.random.default_rng(11)),
    )
    b = _evolve(
        r2, ids2, w2, rounds=40,
        provider_kwargs=dict(work_rows=k, select_rng=np.random.default_rng(11)),
    )
    assert [r.old_loss for r in a.records] == [r.old_loss for r in b.records]
    assert np.array_equal(a.state_ids, b.state_ids)


def test_damping_passthrough_scales_step():
    """damping 乘在步长上，0.5 首轮步长恰为 1.0 的一半，轨迹随之分叉。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    full = _evolve(r1, ids1, w1, rounds=1, damping=1.0)
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    half = _evolve(r2, ids2, w2, rounds=1, damping=0.5)
    assert full.records[0].step > 0
    assert np.isclose(half.records[0].step, 0.5 * full.records[0].step)
