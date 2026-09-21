"""距离衰减参考分布测试，锚定同松紧，结构单调，默认零改变。"""
import numpy as np
import pytest
from test_batchkernel import _scene

from resevo.batchkernel import (
    _distance_reference,
    build_batch_kernel,
    evolve_batch,
    make_batch_provider,
)
from resevo.batchmenu import CodeBook, build_query_structure, generate_batch_menu
from resevo.dataset import query_field_sets, target_from_specs
from resevo.editspace import EditBudget
from resevo.grouping import group_state_ids


def _setup(seed: int, num_rows: int = 36):
    registry, ids, weights, rng = _scene(seed, num_rows)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2),
        [(0, 1), (2, 3)],
    )
    target = target_from_specs(registry.specs)
    workload = registry.build_workload(target, weights)
    return registry, workload, grouped, menu, qs, codes, target, weights


@pytest.mark.parametrize("stay", [0.5, 0.9, 0.99])
def test_anchor_equation_holds(stay):
    """锚定后非空组按行数加权的平均保持概率恰等于 stay。"""
    _, workload, grouped, menu, qs, codes, _, _ = _setup(0)
    counts = grouped.counts.astype(np.float64)
    ref_src, ref_paths = _distance_reference(
        menu, codes[grouped.unique_ids], counts, grouped.num_groups, stay
    )
    nonempty = np.diff(menu.offsets) > 0
    mean_stay = float((counts[nonempty] * ref_src[nonempty]).sum() / counts[nonempty].sum())
    assert abs(mean_stay - stay) < 1e-9
    # 每块归一，源加路径质量和为 1
    seg_mass = ref_src + np.bincount(
        menu.group, weights=ref_paths, minlength=grouped.num_groups
    )
    np.testing.assert_allclose(seg_mass, 1.0, rtol=1e-12)


def test_distance_monotone_within_block():
    """同块内单位质量的参考权重随距离衰减，比值恰为折扣的距离差次幂。"""
    _, workload, grouped, menu, qs, codes, _, _ = _setup(1)
    counts = grouped.counts.astype(np.float64)
    codes_g = codes[grouped.unique_ids]
    ref_src, ref_paths = _distance_reference(
        menu, codes_g, counts, grouped.num_groups, 0.9
    )
    slots = menu.fields >= 0
    src_vals = np.where(slots, codes_g[menu.group[:, None], np.maximum(menu.fields, 0)], -1)
    dist = (slots & (menu.values != src_vals)).sum(axis=1)
    unit = ref_paths / menu.mass / ref_src[menu.group]  # = c^d
    c_est = None
    for d in (1, 2, 3):
        sel = dist == d
        if not np.any(sel):
            continue
        vals = unit[sel]
        np.testing.assert_allclose(vals, vals[0], rtol=1e-12)
        if c_est is None:
            c_est = vals[0] ** (1.0 / d)
        else:
            np.testing.assert_allclose(vals[0], c_est ** d, rtol=1e-10)
    assert c_est is not None and 0 < c_est < 1


def test_kernel_accepts_shape_and_rejects_bogus():
    """distance 形状构核跑通且 beta 与均匀签筒不同，未知形状报错。"""
    _, workload, grouped, menu, qs, codes, _, _ = _setup(2)
    base = build_batch_kernel(workload, grouped, menu, qs, codes)
    got = build_batch_kernel(workload, grouped, menu, qs, codes, ref_shape="distance")
    assert got.status == "ok"
    assert got.beta != base.beta
    np.testing.assert_allclose(got.old_loss, base.old_loss, rtol=1e-12)
    with pytest.raises(ValueError):
        build_batch_kernel(workload, grouped, menu, qs, codes, ref_shape="bogus")


def test_evolve_batch_with_distance_shape():
    """批量演化带 distance 形状端到端跑通，损失单调不升，未知形状报错。"""
    registry, ids, weights, rng = _scene(3, num_rows=30)
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    provider = make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(5), joint_field_sets=fs,
    )
    out = evolve_batch(
        ids, 6, np.random.default_rng(9), provider, registry,
        max_frozen_retries=5, ref_shape="distance",
    )
    losses = [r.old_loss for r in out.records if r.status == "ok"]
    assert len(losses) >= 2
    assert losses[-1] <= losses[0]
    with pytest.raises(ValueError):
        evolve_batch(
            ids, 2, np.random.default_rng(9), provider, registry, ref_shape="bogus"
        )


def test_stay_passthrough_changes_calibration():
    """stay 越高非源先验质量越小，同增益校准解出的首轮 beta 越大。"""
    betas = {}
    for stay in (0.5, 0.99):
        registry, ids, weights, rng = _scene(2, num_rows=30)
        fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
        provider = make_batch_provider(
            registry, target_from_specs(registry.specs), weights,
            np.random.default_rng(5), joint_field_sets=fs,
        )
        out = evolve_batch(
            ids, 1, np.random.default_rng(9), provider, registry,
            max_frozen_retries=3, stay_probability=stay,
        )
        betas[stay] = out.records[0].beta
    assert betas[0.99] > betas[0.5]
