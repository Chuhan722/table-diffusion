"""GPU 稀疏整表项/漂移项（第五十九步 乙）测试：与稠密通路对拍，答案残差逐位同、其余 1e-12 内，同卡可复现。"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("无可用 GPU", allow_module_level=True)
    _ = cp.zeros(2).sum()
except Exception:
    pytest.skip("GPU 运行时不可用", allow_module_level=True)

from test_batchkernel import _scene  # noqa: E402
from test_gpu_direct import _cell_scene, _inputs, _menu_gpu  # noqa: E402

from resevo.batchkernel import evolve_batch, make_batch_provider  # noqa: E402
from resevo.batchmenu import CodeBook, build_query_structure  # noqa: E402
from resevo.dataset import target_from_specs  # noqa: E402
from resevo.gpukernel import GpuBatchContext  # noqa: E402

_KWS = [{}, {"drop": 7}, {"shuffle": True, "drop": 3}, {"nofield": True}]


def _pair(registry, weights, **kw):
    target = target_from_specs(registry.specs)
    qs = build_query_structure(registry, CodeBook(registry))
    dense = GpuBatchContext(qs, target, weights, sparse=False)
    sparse = GpuBatchContext(qs, target, weights, **kw)
    assert dense.direct and not dense.sparse
    return dense, sparse


def test_sparse_resolution():
    """全格无半空间自动开稀疏；有半空间或区间题自动关、强制开报错；False 恒关。"""
    registry, _, weights, _ = _cell_scene(0)
    target = target_from_specs(registry.specs)
    qs = build_query_structure(registry, CodeBook(registry))
    assert GpuBatchContext(qs, target, weights).sparse is True
    assert GpuBatchContext(qs, target, weights, sparse=True).sparse is True
    assert GpuBatchContext(qs, target, weights, sparse=False).sparse is False
    assert GpuBatchContext(qs, target, weights, direct=False).sparse is False
    with pytest.raises(ValueError):
        GpuBatchContext(qs, target, weights, direct=False, sparse=True)
    registry_hs, _, weights_hs, _ = _cell_scene(1, halfspace=True)
    qs_hs = build_query_structure(registry_hs, CodeBook(registry_hs))
    target_hs = target_from_specs(registry_hs.specs)
    assert GpuBatchContext(qs_hs, target_hs, weights_hs).sparse is False
    with pytest.raises(ValueError):
        GpuBatchContext(qs_hs, target_hs, weights_hs, sparse=True)
    registry_r, _, weights_r, _ = _scene(0)
    qs_r = build_query_structure(registry_r, CodeBook(registry_r))
    assert GpuBatchContext(qs_r, target_from_specs(registry_r.specs), weights_r).sparse is False


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kw", _KWS)
def test_sparse_state_terms(seed, kw):
    """答案整数计数与 gemv 逐位同，残差损失随之逐位同；组错位分 1e-12 内。"""
    registry, ids, weights, rng = _cell_scene(seed, **kw)
    qs, grouped, codes, _ = _inputs(registry, ids, rng)
    dense, sparse = _pair(registry, weights)
    codes_g = cp.asarray(codes[grouped.unique_ids])
    counts = cp.asarray(grouped.counts.astype(np.float64))
    r1, l1, s1, hs1, _ = dense._state_terms(codes_g, counts)
    r2, l2, s2, hs2, _ = sparse._state_terms(codes_g, counts)
    np.testing.assert_array_equal(cp.asnumpy(r2), cp.asnumpy(r1))
    assert l2 == l1
    np.testing.assert_allclose(cp.asnumpy(s2), cp.asnumpy(s1), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(cp.asnumpy(hs2), cp.asnumpy(hs1))
    assert dense.loss_of(codes, ids) == sparse.loss_of(codes, ids)
    # 稀疏组分与主机端按特征矩阵逐组求和一致
    feats, _ = dense._feats_direct(codes_g)
    wr = cp.asnumpy(dense.weights * cp.abs(r1))
    ref = cp.asnumpy(feats) @ wr
    np.testing.assert_allclose(cp.asnumpy(s2), ref, rtol=1e-12, atol=0)


def _dense_drift_reference(parts, rates_paths, counts, weights, num_groups, num_queries):
    """主机端按条目散射累加的稠密漂移参考，与两条 GPU 通路只差浮点顺序。"""
    drifts = np.zeros((num_groups, num_queries))
    rp = cp.asnumpy(rates_paths)
    for part in parts:
        data = cp.asnumpy(part["data"])
        if data.size == 0:
            continue
        row_of = cp.asnumpy(part["row_of"])
        g = cp.asnumpy(part["grp_of"])[row_of]
        q = cp.asnumpy(part["indices"])
        val = data * rp[cp.asnumpy(part["path_idx"])][row_of]
        np.add.at(drifts, (g, q), val)
    drift_total = counts @ drifts
    self_cross = float(counts @ ((drifts * drifts) @ weights))
    return drifts, drift_total, self_cross


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("budget", [0, 37, 5])
def test_sparse_drift_terms(seed, budget):
    """稀疏漂移项对主机端稠密参考 1e-12 内，分块（含极小预算多块）同口径，两遍逐位同。"""
    registry, ids, weights, rng = _cell_scene(seed)
    qs, grouped, codes, menu = _inputs(registry, ids, rng)
    _, sparse = _pair(registry, weights)
    codes_g = cp.asarray(codes[grouped.unique_ids])
    counts = cp.asarray(grouped.counts.astype(np.float64))
    menu_gpu = _menu_gpu(menu)
    parts = sparse._delta_direct(menu_gpu, codes_g, budget)
    if budget == 5:
        assert len(parts) > 2
    rates_paths = cp.asarray(rng.uniform(0.0, 0.05, menu.num_paths))
    num_groups, num_queries = grouped.num_groups, sparse.num_queries
    got_total, got_cross = sparse._drift_sparse(
        parts, rates_paths, counts, num_groups, num_queries, budget,
    )
    _, ref_total, ref_cross = _dense_drift_reference(
        parts, rates_paths, grouped.counts.astype(np.float64), np.asarray(weights),
        num_groups, num_queries,
    )
    np.testing.assert_allclose(cp.asnumpy(got_total), ref_total, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(got_cross, ref_cross, rtol=1e-12)
    again_total, again_cross = sparse._drift_sparse(
        parts, rates_paths, counts, num_groups, num_queries, budget,
    )
    np.testing.assert_array_equal(cp.asnumpy(again_total), cp.asnumpy(got_total))
    assert again_cross == got_cross
    # 预算不同只改合并分段，不改每个 (组, 题) 项的集合，结果仍在 1e-12 内
    big_total, big_cross = sparse._drift_sparse(
        parts, rates_paths, counts, num_groups, num_queries, 0,
    )
    np.testing.assert_allclose(cp.asnumpy(big_total), ref_total, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(big_cross, ref_cross, rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kw", _KWS)
@pytest.mark.parametrize("budget", [0, 37])
def test_sparse_build_close(seed, kw, budget):
    """整轮构核：损失/beta/方向增益逐位同，交叉项/步长/概率/组分 1e-12 内，两遍逐位同。"""
    registry, ids, weights, rng = _cell_scene(seed, **kw)
    qs, grouped, codes, menu = _inputs(registry, ids, rng)
    dense, sparse = _pair(registry, weights)
    a = dense.build(grouped, menu, codes, entry_budget=budget)
    b = sparse.build(grouped, menu, codes, entry_budget=budget)
    c = sparse.build(grouped, menu, codes, entry_budget=budget)
    assert a.status == b.status == c.status
    assert a.old_loss == b.old_loss
    assert a.beta == b.beta
    assert a.direction_gain == b.direction_gain
    assert a.max_gain_sum == b.max_gain_sum
    np.testing.assert_allclose(b.interaction, a.interaction, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(b.step, a.step, rtol=1e-12)
    np.testing.assert_allclose(b.expected_loss, a.expected_loss, rtol=1e-12)
    np.testing.assert_allclose(b.probabilities, a.probabilities, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b.group_scores, a.group_scores, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(c.probabilities, b.probabilities)
    assert c.interaction == b.interaction and c.step == b.step
    np.testing.assert_array_equal(c.group_scores, b.group_scores)


def test_sparse_build_frozen_matches_dense():
    """无正方向时两通路同样冻结，概率向量逐位同（保持项全 1）。"""
    registry, ids, weights, rng = _cell_scene(2)
    qs, grouped, codes, menu = _inputs(registry, ids, rng)
    probe = registry.build_workload(np.zeros(len(weights)), weights)
    target = probe.features[ids].sum(axis=0)
    dense = GpuBatchContext(qs, target, weights, sparse=False)
    sparse = GpuBatchContext(qs, target, weights)
    assert sparse.sparse
    a = dense.build(grouped, menu, codes)
    b = sparse.build(grouped, menu, codes)
    assert a.status == b.status == "no_positive_direction"
    assert a.old_loss == b.old_loss == 0.0
    np.testing.assert_array_equal(b.probabilities, a.probabilities)


def test_evolve_batch_sparse_trajectory():
    """端到端：首轮损失与稠密逐位同、同种子两遍逐位同轨迹、损失下降；关掉即老路。"""
    def run(gpu_sparse):
        registry, ids, weights, _ = _cell_scene(3, num_rows=80)
        target = target_from_specs(registry.specs)
        provider = make_batch_provider(
            registry, target, weights, np.random.default_rng(5),
            joint_field_sets=[(0, 1), (3, 4)], defer_workload=True,
            work_rows=30, work_below_step=0.5, select_rng=np.random.default_rng(1),
        )
        return evolve_batch(
            ids, 40, np.random.default_rng(9), provider, registry,
            max_frozen_retries=5, backend="gpu", gpu_entry_budget=60,
            gpu_sparse=gpu_sparse,
        )

    off = run(False)
    on1 = run(None)
    on2 = run(True)
    assert on1.records[0].old_loss == off.records[0].old_loss
    assert on1.records[0].beta == off.records[0].beta
    np.testing.assert_array_equal(on1.state_ids, on2.state_ids)
    assert [r.old_loss for r in on1.records] == [r.old_loss for r in on2.records]
    assert on1.records[0].old_loss > min(r.old_loss for r in on1.records)
    assert off.records[0].old_loss > min(r.old_loss for r in off.records)
