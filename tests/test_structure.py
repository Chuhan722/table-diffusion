"""结构签筒测试，体温计对刻度、两把尺对暴力、改形保质量、默认零改变。"""
import numpy as np
import pytest

from resevo.batchkernel import (
    _reference_arrays,
    build_batch_kernel,
    evolve_batch,
    make_batch_provider,
)
from resevo.batchmenu import CodeBook, build_query_structure, generate_batch_menu
from resevo.dataset import target_from_specs
from resevo.editspace import EditBudget
from resevo.grouping import group_state_ids
from resevo.refine import embed_kurtosis
from resevo.structure import (
    BING_DIM,
    BING_SEED,
    StructShaper,
    _field_weights,
    _moments_kurt,
    shape_reference,
)

from test_batchkernel import _scene


def _menu_scene(seed: int, num_rows: int = 36):
    registry, ids, weights, rng = _scene(seed, num_rows)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng,
        EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2),
        [(0, 1), (2, 3)],
    )
    workload = registry.build_workload(target_from_specs(registry.specs), weights)
    sizes = [len(d) for d in registry.schema.domains]
    return registry, ids, grouped, codes, menu, qs, workload, sizes


def _expand(codes_g, counts):
    return np.repeat(codes_g, counts, axis=0)


@pytest.mark.parametrize("seed", range(4))
def test_temperature_matches_refine_instrument(seed):
    """加权体温计与 refine.embed_kurtosis 在展开表上同读数。"""
    _, _, grouped, codes, _, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    shaper = StructShaper("watch", sizes)
    got = shaper.temperature(codes_g, grouped.counts)
    want = embed_kurtosis(_expand(codes_g, grouped.counts), sizes)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_jia_delta_matches_brute_force(seed):
    """甲尺增量符号与逐路径暴力重算 Σc² 一致。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    shaper = StructShaper("jia", sizes)
    delta = shaper._jia_delta(menu, codes_g, grouped.counts)
    pat = {tuple(row): int(c) for row, c in zip(codes_g.tolist(), grouped.counts)}
    base = sum(c * c for c in pat.values())
    for p in range(menu.num_paths):
        g = int(menu.group[p])
        new_row = list(codes_g[g])
        for s in range(3):
            f = int(menu.fields[p, s])
            if f >= 0:
                new_row[f] = int(menu.values[p, s])
        cnt = dict(pat)
        src = tuple(codes_g[g].tolist())
        cnt[src] -= 1
        if not cnt[src]:
            del cnt[src]
        tgt = tuple(new_row)
        cnt[tgt] = cnt.get(tgt, 0) + 1
        brute = sum(c * c for c in cnt.values()) - base
        assert np.sign(brute) == np.sign(delta[p]), f"路径 {p}"


@pytest.mark.parametrize("seed", range(3))
def test_bing_delta_matches_brute_force(seed):
    """丙尺增量与逐路径暴力重算嵌入峰度一致。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    shaper = StructShaper("bing", sizes)
    delta = shaper._bing_delta(menu, codes_g, grouped.counts)
    bw = _field_weights(sizes, BING_DIM, BING_SEED)

    def kurt_of(rows):
        s = np.zeros((len(rows), BING_DIM))
        for f, w in enumerate(bw):
            s += w.T[rows[:, f]]
        mom = np.stack([(s ** (p + 1)).sum(axis=0) for p in range(4)])
        return float(_moments_kurt(mom, float(len(rows))))

    expanded = _expand(codes_g, grouped.counts)
    row_of_group = np.repeat(np.arange(grouped.num_groups), grouped.counts)
    base = kurt_of(expanded)
    for p in range(0, menu.num_paths, max(1, menu.num_paths // 40)):
        g = int(menu.group[p])
        ridx = int(np.flatnonzero(row_of_group == g)[0])
        mod = expanded.copy()
        for s in range(3):
            f = int(menu.fields[p, s])
            if f >= 0:
                mod[ridx, f] = int(menu.values[p, s])
        brute = kurt_of(mod) - base
        np.testing.assert_allclose(delta[p], brute, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_shape_reference_preserves_group_mass(seed):
    """改形后每组路径质量总和与源概率均精确保持。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    counts = grouped.counts.astype(np.float64)
    rng = np.random.default_rng(seed)
    factors = rng.choice([0.25, 1.0, 4.0], size=menu.num_paths)
    for shape in ("uniform", "distance"):
        src0, paths0 = _reference_arrays(
            menu, codes_g, counts, grouped.num_groups, 0.9, shape
        )
        src1, paths1 = _reference_arrays(
            menu, codes_g, counts, grouped.num_groups, 0.9, shape, factors
        )
        np.testing.assert_array_equal(src0, src1)
        starts = np.minimum(menu.offsets[:-1], len(paths0))
        sum0 = np.add.reduceat(np.r_[paths0, 0.0], starts)
        sum1 = np.add.reduceat(np.r_[paths1, 0.0], starts)
        nonempty = np.diff(menu.offsets) > 0
        np.testing.assert_allclose(sum1[nonempty], sum0[nonempty], rtol=1e-12)
        changed = ~np.isclose(paths0, paths1)
        assert changed.any(), "因子非平凡时形状应当真的变了"


def test_kernel_with_factors_valid_and_residual_first(seed=5):
    """带因子的核概率仍是合法随机矩阵，冻结判定与增益不受签筒影响。"""
    _, _, grouped, codes, menu, qs, workload, sizes = _menu_scene(seed)
    rng = np.random.default_rng(0)
    factors = rng.choice([0.25, 1.0, 4.0], size=menu.num_paths)
    got = build_batch_kernel(
        workload, grouped, menu, qs, codes, path_factors=factors
    )
    ref = build_batch_kernel(workload, grouped, menu, qs, codes)
    assert got.status == ref.status
    np.testing.assert_allclose(got.old_loss, ref.old_loss, rtol=1e-12)
    assert got.max_gain_sum == ref.max_gain_sum  # 增益只看残差，签筒不碰
    for g in range(grouped.num_groups):
        lo, hi = int(got.offsets[g]), int(got.offsets[g + 1])
        np.testing.assert_allclose(got.probabilities[lo:hi].sum(), 1.0, rtol=1e-9)
        assert (got.probabilities[lo:hi] >= 0).all()


def _run_evolve(registry, ids, shaper, seed=11, rounds=6):
    y = target_from_specs(registry.specs)
    w = np.ones(len(registry.specs))
    provider = make_batch_provider(
        registry, y, w, np.random.default_rng(seed),
        budget=EditBudget(max_edit_fields=2, donor_copies=2),
    )
    return evolve_batch(
        ids, rounds, np.random.default_rng(seed), provider, registry,
        max_frozen_retries=3, struct_shaper=shaper,
    )


def test_watch_mode_zero_change_and_temps():
    """watch 只记体温，表轨迹与不装签筒逐位一致，批量轮读数有限。"""
    registry, ids, weights, rng = _scene(6, num_rows=30)
    out0 = _run_evolve(registry, ids, None)
    registry2, ids2, _, _ = _scene(6, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    out1 = _run_evolve(registry2, ids2, StructShaper("watch", sizes))
    np.testing.assert_array_equal(out0.state_ids, out1.state_ids)
    assert out0.temps is None
    assert out1.temps is not None and len(out1.temps) == len(out1.records)
    assert all(np.isfinite(t) for t in out1.temps)


def test_gate_closed_zero_change():
    """门槛设到永不触发时，jia 尺与不装签筒逐位一致。"""
    registry, ids, weights, rng = _scene(7, num_rows=30)
    out0 = _run_evolve(registry, ids, None)
    registry2, ids2, _, _ = _scene(7, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    out1 = _run_evolve(registry2, ids2, StructShaper("jia", sizes, gate=-999.0))
    np.testing.assert_array_equal(out0.state_ids, out1.state_ids)


def test_gate_open_changes_trajectory():
    """门槛设到永远触发时，jia 尺应真的改变演化轨迹（签筒生效）。"""
    registry, ids, weights, rng = _scene(8, num_rows=30)
    out0 = _run_evolve(registry, ids, None, rounds=8)
    registry2, ids2, _, _ = _scene(8, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    out1 = _run_evolve(
        registry2, ids2, StructShaper("jia", sizes, gate=999.0), rounds=8
    )
    assert not np.array_equal(out0.state_ids, out1.state_ids)


def test_shaper_validation():
    with pytest.raises(ValueError):
        StructShaper("bogus", [2, 2])
    with pytest.raises(ValueError):
        StructShaper("jia", [2, 2], boost=1.0)
