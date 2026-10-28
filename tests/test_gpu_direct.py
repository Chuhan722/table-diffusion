"""GPU 单元格直算通路测试，与通用通路逐位对拍，含分块、缺格、半空间混合与回退。"""
import itertools

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

from resevo.batchkernel import evolve_batch, make_batch_provider  # noqa: E402
from resevo.batchmenu import CodeBook, build_query_structure, generate_batch_menu  # noqa: E402
from resevo.dataset import QuerySpec, StateRegistry, TableSchema, target_from_specs  # noqa: E402
from resevo.editspace import EditBudget  # noqa: E402
from resevo.gpukernel import GpuBatchContext, build_marginal_layout  # noqa: E402
from resevo.grouping import group_state_ids  # noqa: E402

_SCHEMA = TableSchema(
    ("a", "b", "c", "d", "e"),
    (("0", "1"), ("0", "1", "2"), ("0", "1"), ("0", "1", "2", "3"), ("0", "1", "2")),
)


def _cell_specs(field_sets, rng, drop=0, halfspace=False, shuffle=False):
    """按字段集铺全格的等值题，可随机丢格、混半空间题、打乱题序。"""
    specs = []
    for fs in field_sets:
        doms = [_SCHEMA.domains[_SCHEMA.field_index(f)] for f in fs]
        for combo in itertools.product(*[range(len(d)) for d in doms]):
            conds = tuple(
                {"attribute": f, "operator": "==", "value": doms[i][v]}
                for i, (f, v) in enumerate(zip(fs, combo))
            )
            specs.append(QuerySpec(f"q{len(specs)}", conds, float(rng.integers(1, 9))))
    if drop:
        keep = sorted(rng.choice(len(specs), len(specs) - drop, replace=False))
        specs = [specs[i] for i in keep]
    if halfspace:
        specs.append(QuerySpec(
            "hs", ({"operator": "halfspace",
                    "scores": {"a": {"0": 2, "1": -1}, "d": {"1": 3, "3": -2}},
                    "threshold": 2},),
            3.0,
        ))
    if shuffle:
        perm = rng.permutation(len(specs))
        specs = [specs[i] for i in perm]
    return [QuerySpec(f"q{i}", s.conditions, s.result) for i, s in enumerate(specs)]


def _cell_scene(seed, num_rows=60, nofield=False, **kw):
    """nofield 为真时字段 c 不进任何题，考察无题字段上的路径不出条目。"""
    rng = np.random.default_rng(seed)
    field_sets = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "e"), ("a", "d", "e"), ("e",)]
    if nofield:
        field_sets = [fs for fs in field_sets if "c" not in fs]
    specs = _cell_specs(field_sets, rng, **kw)
    registry = StateRegistry(_SCHEMA, specs)
    base = [tuple(str(rng.integers(len(dom))) for dom in _SCHEMA.domains) for _ in range(14)]
    rows = [base[int(rng.integers(len(base)))] for _ in range(num_rows)]
    ids = registry.register_table(rows)
    weights = rng.uniform(0.5, 2.0, len(specs))
    return registry, ids, weights, rng


def _inputs(registry, ids, rng):
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=4, donor_copies=4, joint_edits=3, explore_edits=2),
        [(0, 1), (3, 4), (0, 3, 4)],
    )
    return qs, grouped, codes, menu


def _menu_gpu(menu):
    return {
        "group": cp.asarray(menu.group),
        "fields": cp.asarray(menu.fields),
        "values": cp.asarray(menu.values),
        "mass": cp.asarray(menu.mass),
        "offsets": cp.asarray(menu.offsets),
        "num_paths": menu.num_paths,
    }


def _assert_parts_equal(got, ref):
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        for key in ("data", "indices", "indptr", "row_of", "path_idx", "grp_of"):
            np.testing.assert_array_equal(
                cp.asnumpy(g[key]), cp.asnumpy(r[key]), err_msg=key,
            )


def test_layout_detects_cells_and_rejects_ranges():
    """全格考卷识别为直算布局，含区间题的场景回退，强制直算报错。"""
    registry, _, _, _ = _cell_scene(0)
    qs = build_query_structure(registry, CodeBook(registry))
    lay = build_marginal_layout(qs)
    assert lay is not None and lay.num_marginals == 7 and lay.kmax == 3
    assert lay.ranges_sorted
    registry2, _, weights2, _ = _scene(0)
    qs2 = build_query_structure(registry2, CodeBook(registry2))
    assert build_marginal_layout(qs2) is None
    target2 = target_from_specs(registry2.specs)
    assert GpuBatchContext(qs2, target2, weights2).direct is False
    with pytest.raises(ValueError):
        GpuBatchContext(qs2, target2, weights2, direct=True)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kw", [{}, {"drop": 7}, {"halfspace": True}, {"shuffle": True, "drop": 3}, {"nofield": True}])
@pytest.mark.parametrize("budget", [0, 37])
def test_direct_parts_bitwise(seed, kw, budget):
    """直算答案差块与通用通路逐条同序同值，分块边界同切。"""
    registry, ids, weights, rng = _cell_scene(seed, **kw)
    qs, grouped, codes, menu = _inputs(registry, ids, rng)
    target = target_from_specs(registry.specs)
    ctx = GpuBatchContext(qs, target, weights)
    assert ctx.direct
    if kw.get("shuffle"):
        assert not ctx.layout.ranges_sorted
    codes_g = cp.asarray(codes[grouped.unique_ids])
    menu_gpu = _menu_gpu(menu)
    cnt, score_g = ctx._condition_counts(codes_g)
    ref = ctx._delta_parts(menu_gpu, cnt, codes_g, budget)
    got = ctx._delta_direct(menu_gpu, codes_g, budget)
    _assert_parts_equal(got, ref)
    assert sum(int(p["data"].size) for p in got) > 0


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kw", [{}, {"drop": 5, "halfspace": True}])
def test_direct_feats_bitwise(seed, kw):
    """散射特征矩阵与条件计数重构的特征逐位相同，残差损失组分一致。"""
    registry, ids, weights, rng = _cell_scene(seed, **kw)
    qs, grouped, codes, _ = _inputs(registry, ids, rng)
    target = target_from_specs(registry.specs)
    ctx = GpuBatchContext(qs, target, weights)
    codes_g = cp.asarray(codes[grouped.unique_ids])
    counts = cp.asarray(grouped.counts.astype(np.float64))
    cnt, score_g = ctx._condition_counts(codes_g)
    feats_ref = ctx._feats_from_counts(cnt, score_g)
    feats_got, score_got = ctx._feats_direct(codes_g)
    np.testing.assert_array_equal(cp.asnumpy(feats_got), cp.asnumpy(feats_ref))
    np.testing.assert_array_equal(cp.asnumpy(score_got), cp.asnumpy(score_g))
    r1, l1, s1, answers1 = ctx._residual_from_feats(feats_ref, counts)
    r2, l2, s2, answers2 = ctx._residual_from_feats(feats_got, counts)
    np.testing.assert_array_equal(cp.asnumpy(answers2), cp.asnumpy(answers1))
    assert l1 == l2
    np.testing.assert_array_equal(cp.asnumpy(r1), cp.asnumpy(r2))
    np.testing.assert_array_equal(cp.asnumpy(s1), cp.asnumpy(s2))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("budget", [0, 50])
def test_direct_build_bitwise(seed, budget):
    """整轮构核直算与通用逐位相同，损失也相同。"""
    registry, ids, weights, rng = _cell_scene(seed, halfspace=(seed % 2 == 1))
    qs, grouped, codes, menu = _inputs(registry, ids, rng)
    target = target_from_specs(registry.specs)
    gen = GpuBatchContext(qs, target, weights, direct=False)
    # 稀疏整表项/漂移项求和顺序异于稠密，逐位对拍须关掉（见 test_gpu_sparse）
    dir_ = GpuBatchContext(qs, target, weights, direct=True, sparse=False)
    a = gen.build(grouped, menu, codes, entry_budget=budget, probe_grid=8)
    b = dir_.build(grouped, menu, codes, entry_budget=budget, probe_grid=8)
    assert a.status == b.status
    assert a.old_loss == b.old_loss
    assert a.beta == b.beta
    assert a.direction_gain == b.direction_gain
    assert a.interaction == b.interaction
    assert a.step == b.step
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
    np.testing.assert_array_equal(a.group_scores, b.group_scores)
    assert a.probe == b.probe
    assert gen.loss_of(codes, ids) == dir_.loss_of(codes, ids)


def test_evolve_batch_direct_trajectory_bitwise():
    """端到端多轮演化，直算与通用同种子逐位同轨迹（含工作批挑组与重试）。"""
    outs = []
    for direct in (False, True):
        registry, ids, weights, _ = _cell_scene(3, num_rows=80)
        target = target_from_specs(registry.specs)
        provider = make_batch_provider(
            registry, target, weights, np.random.default_rng(5),
            joint_field_sets=[(0, 1), (3, 4)], defer_workload=True,
            work_rows=30, work_below_step=0.5, select_rng=np.random.default_rng(1),
        )
        out = evolve_batch(
            ids, 40, np.random.default_rng(9), provider, registry,
            max_frozen_retries=5, backend="gpu", gpu_entry_budget=60,
            gpu_direct=direct, gpu_sparse=False,
        )
        outs.append(out)
    np.testing.assert_array_equal(outs[0].state_ids, outs[1].state_ids)
    assert [r.old_loss for r in outs[0].records] == [r.old_loss for r in outs[1].records]
    assert [r.beta for r in outs[0].records] == [r.beta for r in outs[1].records]
    assert outs[0].records[0].old_loss > min(r.old_loss for r in outs[0].records)
