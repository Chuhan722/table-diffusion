"""子集配对救援测试，惰性长跑冻结轮的解锁通道。

覆盖伪目标恒等式，损失口径一致，provider 分支选择，
关闭时对惰性老行为零改变，以及救援轮抽样落回全表的翻译正确性。
"""
import numpy as np
import pytest

from resevo.batchkernel import BatchRoundPlan, evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs
from resevo.editspace import EditBudget
from resevo.engine import build_kernel
from resevo.pairing import build_rescue_menu
from resevo.state import table_loss, table_residual

from test_batchkernel import _scene

BUDGET = EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2)


def test_rescue_menu_residual_identity():
    """伪目标修正后小注册表口径的子表残差逐位等于全表残差。"""
    registry, ids, weights, rng = _scene(0)
    y = target_from_specs(registry.specs)
    full_resid = table_residual(registry.build_workload(y, weights), ids)
    small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
        registry, ids, y, weights, np.random.default_rng(7), 8, BUDGET
    )
    assert len(sub_ids) == 8 and len(sub_idx) == 8
    sub_resid = table_residual(wl, sub_ids)
    np.testing.assert_allclose(sub_resid, full_resid, rtol=1e-12, atol=1e-12)


def test_rescue_kernel_full_table_loss():
    """救援核报的起点损失就是全表损失，曲线不断档。"""
    registry, ids, weights, rng = _scene(1)
    y = target_from_specs(registry.specs)
    full_loss = table_loss(registry.build_workload(y, weights), ids)
    small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
        registry, ids, y, weights, np.random.default_rng(5), 8, BUDGET
    )
    res = build_kernel(wl, sub_ids, sups)
    np.testing.assert_allclose(res.old_loss, full_loss, rtol=1e-12)


def test_rescue_param_validation():
    """救援行数为负拒收，为零关闭不进分支。"""
    registry, ids, weights, rng = _scene(2)
    y = target_from_specs(registry.specs)
    with pytest.raises(ValueError):
        make_batch_provider(
            registry, y, weights, rng, pairing=True,
            defer_workload=True, pair_rescue_rows=-1,
        )
    with pytest.raises(ValueError):
        build_rescue_menu(registry, ids, y, weights, rng, 0)


def test_provider_branch_selection():
    """惰性模式连败达门槛才走 rescue，之前只刷菜单，节奏按退避周期。"""
    registry, ids, weights, _ = _scene(3)
    y = target_from_specs(registry.specs)
    on = make_batch_provider(
        registry, y, weights, np.random.default_rng(11), budget=BUDGET,
        pairing=True, defer_workload=True, pair_rescue_rows=8,
        rescue_after=6, pairing_backoff=4,
    )
    # 平时轮与门槛前的零星冻结都只刷菜单
    for streak in (0, 1, 3, 5):
        assert on(ids, 0, frozen_streak=streak).mode == "batch"
    plan = on(ids, 0, frozen_streak=6)
    assert plan.mode == "rescue"
    assert plan.rescue_registry is not None and len(plan.rescue_ids) == 8
    assert plan.supports is not None
    # 门槛后按退避周期，7 到 9 刷菜单，10 再救
    for streak in (7, 8, 9):
        assert on(ids, 0, frozen_streak=streak).mode == "batch"
    assert on(ids, 0, frozen_streak=10).mode == "rescue"


def test_rescue_after_validation():
    """救援门槛非正拒收。"""
    registry, ids, weights, rng = _scene(3)
    y = target_from_specs(registry.specs)
    with pytest.raises(ValueError):
        make_batch_provider(
            registry, y, weights, rng, pairing=True,
            defer_workload=True, pair_rescue_rows=8, rescue_after=0,
        )


def test_rescue_disabled_keeps_lazy_behavior():
    """救援关闭时惰性模式冻结轮与不带配对的老行为出同一份菜单。"""
    registry, ids, weights, _ = _scene(4)
    y = target_from_specs(registry.specs)
    a = make_batch_provider(
        registry, y, weights, np.random.default_rng(9), budget=BUDGET,
        pairing=True, defer_workload=True, pair_rescue_rows=0,
    )
    b = make_batch_provider(
        registry, y, weights, np.random.default_rng(9), budget=BUDGET,
        pairing=False, defer_workload=True,
    )
    pa = a(ids, 0, frozen_streak=2)
    pb = b(ids, 0, frozen_streak=2)
    assert pa.mode == pb.mode == "batch"
    np.testing.assert_array_equal(pa.menu.offsets, pb.menu.offsets)
    np.testing.assert_array_equal(pa.menu.fields, pb.menu.fields)
    np.testing.assert_array_equal(pa.menu.values, pb.menu.values)


def test_rescue_writeback_translation():
    """救援轮抽样后只有抽中的行会动，动过的行编号在全局注册表可反查。"""
    registry, ids, weights, _ = _scene(5)
    y = target_from_specs(registry.specs)

    def stub_provider(state_ids, round_index, frozen_streak=0):
        small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
            registry, state_ids, y, weights, np.random.default_rng(13), 8, BUDGET
        )
        return BatchRoundPlan(
            "rescue", wl, supports=sups, rescue_ids=sub_ids,
            rescue_sub_idx=sub_idx, rescue_registry=small,
        )

    before = ids.copy()
    out = evolve_batch(ids, 1, np.random.default_rng(3), stub_provider, registry)
    after = out.state_ids
    assert after.shape == before.shape
    probe = build_rescue_menu(
        registry, before, y, weights, np.random.default_rng(13), 8, BUDGET
    )
    sub_idx = probe[2]
    changed = np.flatnonzero(after != before)
    assert np.isin(changed, sub_idx).all(), "只有抽中的行允许变动"
    for r in changed:
        tup = registry.state_tuple(int(after[r]))
        assert len(tup) == registry.schema.num_fields
    # 落回后的全表损失可在全局注册表口径重算，编号翻译无缝
    loss_after = table_loss(registry.build_workload(y, weights), after)
    assert np.isfinite(loss_after)


def test_register_edited_many_matches_full():
    """编辑行增量注册与全量重算的编号与特征逐位一致。"""
    from resevo.dataset import StateRegistry

    registry, ids, weights, _ = _scene(3)
    y = target_from_specs(registry.specs)
    base_rows = [registry.state_tuple(int(i)) for i in ids]
    mirror = StateRegistry(registry.schema, registry.specs)
    mirror.register_many(
        [registry.state_tuple(i) for i in range(registry.num_states)]
    )
    assert mirror.num_states == registry.num_states
    g = np.random.default_rng(11)
    bases, edits = [], []
    for _ in range(60):
        b = base_rows[int(g.integers(len(base_rows)))]
        e = list(b)
        for j in g.permutation(len(b))[: int(g.integers(1, 4))]:
            dom = registry.schema.domains[int(j)]
            e[int(j)] = dom[int(g.integers(len(dom)))]
        bases.append(b)
        edits.append(tuple(e))
    ids_inc = registry.register_edited_many(bases, edits)
    ids_full = mirror.register_many(edits)
    np.testing.assert_array_equal(ids_inc, ids_full)
    f_inc = registry.build_workload(y, weights).features
    f_full = mirror.build_workload(y, weights).features
    np.testing.assert_array_equal(f_inc, f_full)


def test_swap_sparse_scores_match_dense():
    """交换免物化打分与注册物化后的稠密打分同值。"""
    from resevo.pairing import _score_swaps_sparse

    registry, ids, weights, _ = _scene(4)
    y = target_from_specs(registry.specs)
    wl = registry.build_workload(y, weights)
    resid = table_residual(wl, ids)
    we = wl.weights * resid
    s = np.asarray(ids, dtype=np.int64)
    rows = [registry.state_tuple(int(x)) for x in s]
    pairs = [(0, 1), (2, 3), (4, 7)]
    acts, dense = [], []
    for pidx, (i, k) in enumerate(pairs):
        ri, rk = rows[i], rows[k]
        for j in range(len(ri)):
            if ri[j] == rk[j]:
                continue
            acts.append((pidx, j, i, k, ri[j], rk[j]))
            u, v = list(ri), list(rk)
            u[j], v[j] = rk[j], ri[j]
            dense.append(
                (pidx, registry.register(tuple(u)),
                 registry.register(tuple(v)), int(s[i]), int(s[k]))
            )
    feats = registry.build_workload(y, weights).features
    w = wl.weights
    best_dense = np.zeros(len(pairs))
    for pidx, uid, vid, bi, bk in dense:
        du = feats[uid] - feats[bi]
        dv = feats[vid] - feats[bk]
        tot = (
            du @ we - ((du * du) @ w) / 2
            + dv @ we - ((dv * dv) @ w) / 2
            - (du * dv) @ w
        )
        best_dense[pidx] = max(best_dense[pidx], tot)
    cnt_b, score_b = registry.counts_scores_for_rows(rows)
    best_sparse = _score_swaps_sparse(
        registry, cnt_b, score_b, we, w, len(pairs), acts
    )
    np.testing.assert_allclose(best_sparse, best_dense, rtol=1e-12, atol=1e-12)
