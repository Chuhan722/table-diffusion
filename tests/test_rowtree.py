"""整行树采样棋步测试，Chow-Liu 骨架，外援候选注入，输血轮分支。

覆盖树选边正确性，条件分布保真，负答案截零，覆盖不连通拒收，
外援行注入结构，输血轮触发节奏与撞车合并，配方校验，
以及默认关闭对旧路径逐位零改变。
"""
import numpy as np
import pytest

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import QuerySpec, StateRegistry, TableSchema, target_from_specs
from resevo.editspace import EditBudget
from resevo.pairing import _inject_extra_rows, build_rescue_menu
from resevo.candidates import BlockSupport
from resevo.rowtree import TreeRowSampler

from test_batchkernel import _scene

BUDGET = EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2)


def _grid_specs(schema: TableSchema, joint) -> list[QuerySpec]:
    """按给定联合分布函数造全二阶全格子等值考卷。"""
    specs = []
    names, doms = schema.fields, schema.domains
    F = len(names)
    for a in range(F):
        for b in range(a + 1, F):
            for va in doms[a]:
                for vb in doms[b]:
                    specs.append(QuerySpec(
                        f"g{a}{b}{va}{vb}",
                        ({"attribute": names[a], "operator": "==", "value": va},
                         {"attribute": names[b], "operator": "==", "value": vb}),
                        joint(a, b, va, vb),
                    ))
    return specs


def _tree_scene(seed: int, num_rows: int = 30):
    """三字段场景，a 与 b 强绑定 c 独立，考卷为全二阶格子。"""
    rng = np.random.default_rng(seed)
    schema = TableSchema(("a", "b", "c"), (("0", "1"), ("0", "1"), ("0", "1")))
    n = 1000.0

    def joint(a, b, va, vb):
        # a 与 b 同值率 0.9，c 均匀独立，全部边际各半
        if (a, b) == (0, 1):
            return n * (0.45 if va == vb else 0.05)
        return n * 0.25

    specs = _grid_specs(schema, joint)
    registry = StateRegistry(schema, specs)
    rows = [tuple(str(rng.integers(2)) for _ in range(3)) for _ in range(num_rows)]
    ids = registry.register_table(rows)
    weights = np.ones(len(specs))
    return schema, specs, registry, ids, weights


def test_tree_picks_strong_edge_and_cpt():
    """骨架必选强绑定边 a-b，条件概率表按联表行归一。"""
    schema, specs, *_ = _tree_scene(0)
    t = TreeRowSampler().fit(specs, schema)
    edges = {(min(c, t.parent[c]), max(c, t.parent[c])) for c in t.cpt}
    assert (0, 1) in edges
    # a-b 边的条件分布应为 0.9/0.1
    child = 1 if t.parent[1] == 0 else 0
    if t.parent[child] in (0, 1) and child in (0, 1):
        cp = t.cpt[child]
        np.testing.assert_allclose(cp[0], [0.9, 0.1], atol=1e-9)
        np.testing.assert_allclose(cp[1], [0.1, 0.9], atol=1e-9)


def test_tree_sampling_matches_distribution():
    """祖先采样的同值率与独立字段边际收敛到考卷口径。"""
    schema, specs, *_ = _tree_scene(1)
    t = TreeRowSampler().fit(specs, schema)
    rows = t.sample(np.random.default_rng(3), 20000)
    assert all(len(r) == 3 for r in rows)
    same = sum(1 for r in rows if r[0] == r[1]) / len(rows)
    c1 = sum(1 for r in rows if r[2] == "1") / len(rows)
    assert abs(same - 0.9) < 0.02
    assert abs(c1 - 0.5) < 0.02


def test_tree_negative_answers_truncated():
    """噪声卷负答案截零后照常搭树采样不崩。"""
    schema, specs, *_ = _tree_scene(2)
    noisy = [
        QuerySpec(s.query_id, s.conditions, s.result - 260.0) for s in specs
    ]
    t = TreeRowSampler().fit(noisy, schema)
    rows = t.sample(np.random.default_rng(0), 500)
    assert len(rows) == 500


def test_tree_disconnected_raises():
    """考卷二阶覆盖不连通时树搭不起来必须明说。"""
    schema = TableSchema(("a", "b", "c"), (("0", "1"), ("0", "1"), ("0", "1")))
    specs = [QuerySpec(
        f"q{va}{vb}",
        ({"attribute": "a", "operator": "==", "value": va},
         {"attribute": "b", "operator": "==", "value": vb}),
        10.0,
    ) for va in ("0", "1") for vb in ("0", "1")]
    with pytest.raises(ValueError, match="不连通"):
        TreeRowSampler().fit(specs, schema)


def test_inject_extra_rows_structure():
    """外援注入每行追加恰好 per_row 个单状态候选，迁移率 1 份。"""
    singles = [
        BlockSupport((0,), ((5,), (6,)), (1.0, 1.0)),
        BlockSupport((1,), (), ()),
    ]
    out = _inject_extra_rows(singles, np.array([7, 8, 9]), 2, np.random.default_rng(0))
    for old, new in zip(singles, out):
        assert new.rows == old.rows
        assert len(new.outcomes) == len(old.outcomes) + 2
        assert set(o[0] for o in new.outcomes[len(old.outcomes):]) <= {7, 8, 9}
        assert new.mobility == tuple(old.mobility) + (1.0, 1.0)


def test_rescue_menu_extra_rows_registered():
    """外援行注册进小注册表，配对菜单能吃整行外援候选。"""
    schema, specs, registry, ids, weights = _tree_scene(3)
    y = target_from_specs(specs)
    t = TreeRowSampler().fit(specs, schema)
    extra = t.sample(np.random.default_rng(1), 16)
    small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
        registry, ids, y, weights, np.random.default_rng(5), 6, BUDGET,
        extra_rows=extra, extra_per_row=2,
    )
    assert sum(len(sp.rows) for sp in sups) == 6
    tuples = {small.state_tuple(int(o[0])) for sp in sups for o in sp.outcomes}
    assert tuples & set(extra), "菜单里必须出现外援整行"


def test_tree_turn_schedule_and_merge():
    """输血轮按周期触发记名 tree，撞上连败救援合并出场记名 rescue。"""
    schema, specs, registry, ids, weights = _tree_scene(4)
    y = target_from_specs(specs)
    t = TreeRowSampler().fit(specs, schema)
    on = make_batch_provider(
        registry, y, weights, np.random.default_rng(11), budget=BUDGET,
        pairing=True, defer_workload=True, pair_rescue_rows=6,
        rescue_after=6, pairing_backoff=4,
        tree_sampler=t, tree_every=5, tree_pool=8, tree_per_row=1,
    )
    assert on(ids, 0, frozen_streak=0).mode == "batch"
    assert on(ids, 4, frozen_streak=0).mode == "batch"
    assert on(ids, 5, frozen_streak=0).mode == "tree"
    assert on(ids, 6, frozen_streak=0).mode == "batch"
    merged = on(ids, 10, frozen_streak=6)
    assert merged.mode == "rescue"


def test_tree_turn_config_validation():
    """开输血轮缺树或缺救援管线都要拒收。"""
    schema, specs, registry, ids, weights = _tree_scene(5)
    y = target_from_specs(specs)
    t = TreeRowSampler().fit(specs, schema)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="树采样器"):
        make_batch_provider(
            registry, y, weights, rng, pairing=True,
            defer_workload=True, pair_rescue_rows=6, tree_every=5,
        )
    with pytest.raises(ValueError, match="救援管线"):
        make_batch_provider(
            registry, y, weights, rng, tree_sampler=t, tree_every=5,
        )


def test_tree_off_bitwise_identical():
    """tree_every 取 0 与旧路径演化轨迹逐位一致。"""
    outs = []
    for kwargs in ({}, {"tree_every": 0, "tree_pool": 9, "tree_per_row": 3}):
        registry, ids, weights, _ = _scene(6)
        y = target_from_specs(registry.specs)
        prov = make_batch_provider(
            registry, y, weights, np.random.default_rng(2), budget=BUDGET,
            pairing=True, defer_workload=False, **kwargs,
        )
        out = evolve_batch(
            ids.copy(), 6, np.random.default_rng(9), prov, registry,
            max_frozen_retries=2,
        )
        outs.append(out)
    np.testing.assert_array_equal(outs[0].state_ids, outs[1].state_ids)


def test_tree_turn_evolve_smoke():
    """输血轮计划过核抽样落表全链跑通，行数不变损失口径全表。"""
    from resevo.engine import build_kernel, sample_next
    from resevo.state import table_loss

    schema, specs, registry, ids, weights = _tree_scene(7)
    y = target_from_specs(specs)
    t = TreeRowSampler().fit(specs, schema)
    prov = make_batch_provider(
        registry, y, weights, np.random.default_rng(4), budget=BUDGET,
        pairing=True, defer_workload=True, pair_rescue_rows=8,
        tree_sampler=t, tree_every=3, tree_pool=12, tree_per_row=2,
    )
    plan = prov(ids, 3, frozen_streak=0)
    assert plan.mode == "tree"
    res = build_kernel(plan.workload, plan.rescue_ids, plan.supports)
    full_loss = table_loss(registry.build_workload(y, weights), ids)
    np.testing.assert_allclose(res.old_loss, full_loss, rtol=1e-12)
    new_small = sample_next(res, plan.rescue_ids, np.random.default_rng(5))
    moved = new_small != plan.rescue_ids
    current = ids.copy()
    if moved.any():
        tuples = [
            plan.rescue_registry.state_tuple(int(i)) for i in new_small[moved]
        ]
        gids = registry.register_table(tuples)
        current[plan.rescue_sub_idx[moved]] = gids
    assert len(current) == len(ids)
