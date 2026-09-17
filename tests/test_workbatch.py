"""工作批挑组测试，覆盖开关等价，可复现，批外冻结，端到端下降与跨后端。"""
import numpy as np
import pytest

from resevo.batchkernel import (
    build_batch_kernel,
    evolve_batch,
    make_batch_provider,
    sample_batch_next,
)
from resevo.dataset import target_from_specs, query_field_sets

from test_batchkernel import _scene


def _provider(registry, weights, menu_seed=5, **kwargs):
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    return make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(menu_seed), joint_field_sets=fs, **kwargs,
    )


def _evolve(registry, ids, weights, rounds=40, **kwargs):
    provider = _provider(registry, weights, **kwargs)
    out = evolve_batch(
        ids, rounds, np.random.default_rng(9), provider, registry,
        max_frozen_retries=3,
    )
    return out


def test_work_rows_zero_bitwise_identical():
    """开关关闭与不传参数的轨迹逐位一致，旧行为零改变。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    base = _evolve(r1, ids1, w1)
    off = _evolve(r2, ids2, w2, work_rows=0, select_rng=np.random.default_rng(3))
    assert [r.old_loss for r in base.records] == [r.old_loss for r in off.records]
    assert np.array_equal(base.state_ids, off.state_ids)


def test_work_selection_reproducible():
    """同挑组种子两遍轨迹逐位相同。"""
    runs = []
    for _ in range(2):
        registry, ids, weights, _ = _scene(4, num_rows=30)
        out = _evolve(
            registry, ids, weights, rounds=60,
            work_rows=8, select_rng=np.random.default_rng(11),
        )
        runs.append(out)
    assert [r.old_loss for r in runs[0].records] == [r.old_loss for r in runs[1].records]
    assert np.array_equal(runs[0].state_ids, runs[1].state_ids)


def test_first_round_full_and_outside_groups_frozen():
    """首轮无分数走全量，第二轮起未选组空段且行原地不动。"""
    registry, ids, weights, _ = _scene(3, num_rows=30)
    provider = _provider(
        registry, weights,
        work_rows=6, work_random_frac=0.0, select_rng=np.random.default_rng(7),
    )
    codes = provider.codebook.sync
    plan0 = provider(ids, 0)
    assert not np.any(np.diff(plan0.menu.offsets) == 0), "首轮应全量出菜单"
    res0 = build_batch_kernel(
        plan0.workload, plan0.grouped, plan0.menu,
        provider.structure[0], codes(), want_scores=True,
    )
    assert res0.group_scores is not None
    provider.feed_scores(res0.grouped.unique_ids, res0.group_scores)
    cur = sample_batch_next(res0, ids, np.random.default_rng(1), registry)

    plan1 = provider(cur, 1)
    empty = np.diff(plan1.menu.offsets) == 0
    assert empty.any(), "工作批应留下未选组"
    assert (~empty).any()
    res1 = build_batch_kernel(
        plan1.workload, plan1.grouped, plan1.menu, provider.structure[0], codes()
    )
    out = sample_batch_next(res1, cur, np.random.default_rng(2), registry)
    for g in np.flatnonzero(empty):
        for i in plan1.grouped.row_lists[int(g)]:
            assert out[i] == cur[i], "批外组的行不许动"
    # 选中组的行数不超过预算加最后一组的溢出
    chosen_rows = int(plan1.grouped.counts[~empty].sum())
    assert chosen_rows >= 6
    assert chosen_rows <= 6 + int(plan1.grouped.counts.max())


def test_work_mode_descends():
    """工作批模式端到端仍显著下降。"""
    registry, ids, weights, _ = _scene(1, num_rows=30)
    out = _evolve(
        registry, ids, weights, rounds=300,
        work_rows=10, select_rng=np.random.default_rng(13),
    )
    losses = [r.old_loss for r in out.records]
    assert losses[0] > 0
    assert min(losses) < 0.3 * losses[0]


def test_frozen_retry_falls_back_to_full():
    """冻结重试轮回退全量菜单，不受工作批限制。"""
    registry, ids, weights, _ = _scene(6, num_rows=24)
    provider = _provider(
        registry, weights,
        work_rows=5, work_random_frac=0.0, select_rng=np.random.default_rng(7),
    )
    plan0 = provider(ids, 0)
    res0 = build_batch_kernel(
        plan0.workload, plan0.grouped, plan0.menu,
        provider.structure[0], provider.codebook.sync(), want_scores=True,
    )
    provider.feed_scores(res0.grouped.unique_ids, res0.group_scores)
    working = provider(ids, 1, frozen_streak=0)
    assert np.any(np.diff(working.menu.offsets) == 0)
    retry = provider(ids, 2, frozen_streak=1)
    assert retry.mode == "batch"
    assert not np.any(np.diff(retry.menu.offsets) == 0), "重试轮应全量搜索"


def test_work_below_step_gate():
    """步长闸门，低于阈值前全量，触发后启用工作批且不回退。"""
    registry, ids, weights, _ = _scene(3, num_rows=30)
    provider = _provider(
        registry, weights,
        work_rows=6, work_random_frac=0.0,
        select_rng=np.random.default_rng(7), work_below_step=0.5,
    )
    plan0 = provider(ids, 0)
    res0 = build_batch_kernel(
        plan0.workload, plan0.grouped, plan0.menu,
        provider.structure[0], provider.codebook.sync(), want_scores=True,
    )
    provider.feed_scores(res0.grouped.unique_ids, res0.group_scores)
    provider.note_step(0.9)  # 步长还大，闸门不开
    full = provider(ids, 1)
    assert not np.any(np.diff(full.menu.offsets) == 0)
    provider.note_step(0.1)  # 低于阈值，开闸
    working = provider(ids, 2)
    assert np.any(np.diff(working.menu.offsets) == 0)
    provider.note_step(0.9)  # 单向切换，不回退
    still = provider(ids, 3)
    assert np.any(np.diff(still.menu.offsets) == 0)


def test_defer_workload_never_pairs():
    """惰性负载模式冻结重试不走配对回退，避免物化历史特征矩阵。"""
    registry, ids, weights, _ = _scene(5, num_rows=24)
    provider = _provider(registry, weights, defer_workload=True, pairing=True)
    plan = provider(ids, 1, frozen_streak=1)
    assert plan.mode == "batch"
    assert plan.workload.features.shape[0] == 0


def _gpu_ready() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ready(), reason="无可用 GPU")
def test_gpu_scores_match_cpu():
    """组错位分跨后端在容差内一致。"""
    from resevo.gpukernel import GpuBatchContext

    registry, ids, weights, _ = _scene(8, num_rows=30)
    provider = _provider(registry, weights)
    plan = provider(ids, 0)
    codes = provider.codebook.sync()
    cpu = build_batch_kernel(
        plan.workload, plan.grouped, plan.menu,
        provider.structure[0], codes, want_scores=True,
    )
    ctx = GpuBatchContext(
        provider.structure[0], plan.workload.target, plan.workload.weights
    )
    gpu = ctx.build(plan.grouped, plan.menu, codes, 0.9, 0.5, 1.0, None)
    assert gpu.group_scores is not None
    np.testing.assert_allclose(gpu.group_scores, cpu.group_scores, rtol=1e-9)


@pytest.mark.skipif(not _gpu_ready(), reason="无可用 GPU")
def test_gpu_work_mode_descends():
    """GPU 后端配工作批端到端下降。"""
    registry, ids, weights, _ = _scene(1, num_rows=30)
    provider = _provider(
        registry, weights,
        work_rows=10, select_rng=np.random.default_rng(13),
    )
    out = evolve_batch(
        ids, 200, np.random.default_rng(9), provider, registry,
        max_frozen_retries=3, backend="gpu",
    )
    losses = [r.old_loss for r in out.records]
    assert min(losses) < 0.3 * losses[0]
