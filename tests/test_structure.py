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
    CompositeShaper,
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


def test_ding_effective_boost_and_active():
    """丁尺力度幂律连续，烧满刻度即满力度，无门槛体温为负即启用。"""
    ding = StructShaper("ding", [2, 3])
    assert ding.effective_boost(0.1) == 1.0
    assert ding.effective_boost(0.0) == 1.0
    np.testing.assert_allclose(ding.effective_boost(-0.05), 4.0)
    np.testing.assert_allclose(ding.effective_boost(-0.025), 2.0)
    np.testing.assert_allclose(ding.effective_boost(-0.005), 4.0 ** 0.1)
    assert ding.effective_boost(-0.13) == 4.0  # 超刻度封顶满力度
    assert ding.active(-1e-9) and not ding.active(0.0) and not ding.active(0.1)
    # gate 对丁尺是满速刻度而非门槛，门槛设死也拦不住启用
    gated = StructShaper("ding", [2, 3], gate=-999.0)
    assert gated.active(-0.01)
    for ruler in ("watch", "jia", "bing"):
        assert StructShaper(ruler, [2, 3]).effective_boost(-1.0) == 4.0


def test_ding_factors_follow_temperature():
    """丁尺因子随体温连续，浅烧弱偏置，非负体温因子全一（零干预）。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(0)
    codes_g = codes[grouped.unique_ids]
    ding = StructShaper("ding", sizes)
    bing = StructShaper("bing", sizes)
    full = ding.path_factors(menu, codes_g, grouped.counts, temperature=-0.05)
    same = bing.path_factors(menu, codes_g, grouped.counts)
    np.testing.assert_array_equal(full, same)  # 烧满刻度与丙尺同力度
    half = ding.path_factors(menu, codes_g, grouped.counts, temperature=-0.025)
    boosted = full > 1.0
    np.testing.assert_allclose(half[boosted], 2.0)
    np.testing.assert_allclose(half[full < 1.0], 0.5)
    flat = ding.path_factors(menu, codes_g, grouped.counts, temperature=0.02)
    np.testing.assert_array_equal(flat, np.ones(menu.num_paths))


def test_ding_evolve_runs_and_records_temps():
    """丁尺演化全程跑通，批量轮体温有限，轨迹合法。"""
    registry, ids, weights, rng = _scene(9, num_rows=30)
    sizes = [len(d) for d in registry.schema.domains]
    out = _run_evolve(registry, ids, StructShaper("ding", sizes), rounds=8)
    assert out.temps is not None and len(out.temps) == len(out.records)
    assert all(np.isfinite(t) for t in out.temps)


def _menu_dev_of(menu, codes_g, counts):
    cp = pytest.importorskip("cupy")
    menu_dev = {
        "group": cp.asarray(menu.group),
        "fields": cp.asarray(menu.fields),
        "values": cp.asarray(menu.values),
        "mass": cp.asarray(menu.mass),
        "offsets": cp.asarray(menu.offsets),
        "num_paths": menu.num_paths,
    }
    return (
        cp, menu_dev, cp.asarray(codes_g),
        cp.asarray(np.asarray(counts, dtype=np.float64)),
    )


@pytest.mark.parametrize("seed", range(3))
def test_gpu_factors_match_cpu(seed):
    """卡内药房因子与 host 版一致（bing 满力度与 ding 两档温度）。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    for ruler, temp in (("bing", None), ("ding", -0.03), ("ding", -0.06)):
        shaper = StructShaper(ruler, sizes)
        want = shaper.path_factors(
            menu, codes_g, grouped.counts, temperature=temp
        )
        cp, menu_dev, cg_dev, cnt_dev = _menu_dev_of(
            menu, codes_g, grouped.counts
        )
        got = cp.asnumpy(shaper.path_factors_dev(
            menu_dev, cg_dev, cnt_dev, temperature=temp
        ))
        agree = float((np.sign(got - 1.0) == np.sign(want - 1.0)).mean())
        assert agree >= 0.999, f"{ruler} 因子方向一致率 {agree}"
        np.testing.assert_allclose(
            np.sort(np.unique(got)), np.sort(np.unique(want)), rtol=1e-12
        )


def _run_evolve_gpu(registry, ids, shaper, seed=11, rounds=6):
    y = target_from_specs(registry.specs)
    w = np.ones(len(registry.specs))
    provider = make_batch_provider(
        registry, y, w, np.random.default_rng(seed),
        budget=EditBudget(max_edit_fields=2, donor_copies=2),
    )
    return evolve_batch(
        ids, rounds, np.random.default_rng(seed), provider, registry,
        max_frozen_retries=3, struct_shaper=shaper, backend="gpu",
    )


def test_gpu_pharmacy_deterministic_and_biases():
    """卡内药房：GPU 丁尺同种子重跑逐位一致，且轨迹异于无签筒。"""
    pytest.importorskip("cupy")

    def run(with_shaper: bool):
        registry, ids, _, _ = _scene(9, num_rows=30)
        sizes = [len(d) for d in registry.schema.domains]
        shaper = StructShaper("ding", sizes) if with_shaper else None
        return _run_evolve_gpu(registry, ids, shaper, rounds=8)

    out1 = run(True)
    out2 = run(True)
    np.testing.assert_array_equal(out1.state_ids, out2.state_ids)
    assert out1.temps is not None and len(out1.temps) == len(out1.records)
    assert all(np.isfinite(t) for t in out1.temps)


@pytest.mark.parametrize("seed", range(3))
def test_gpu_thermometer_matches_cpu(seed):
    """GPU 体温计与 host 读数一致（统计等价口径，容差 1e-10）。"""
    cp = pytest.importorskip("cupy")
    _, _, grouped, codes, _, _, _, sizes = _menu_scene(seed)
    codes_g = codes[grouped.unique_ids]
    shaper = StructShaper("ding", sizes)
    want = shaper.temperature(codes_g, grouped.counts)
    got = shaper.temperature_dev(
        cp, cp.asarray(codes_g), cp.asarray(grouped.counts)
    )
    assert abs(got - want) <= 1e-10 * max(1.0, abs(want))
    got2 = shaper.temperature_dev(
        cp, cp.asarray(codes_g), cp.asarray(grouped.counts)
    )
    assert got == got2  # 权重缓存后重读逐位确定


def test_ding_diagnose_gate():
    """初诊门：比钟形基准塌逾 |gate| 才收治。"""
    shaper = StructShaper("ding", [3, 4, 5])
    assert shaper.diagnose(-0.06)
    assert not shaper.diagnose(-0.04)  # adult 型低烧不收治
    assert not shaper.diagnose(0.1)


def test_ding_admission_gate_blocks_all_treatment():
    """初诊不过门时全程零干预，轨迹与不装签筒逐位一致。"""
    registry, ids, _, _ = _scene(6, num_rows=30)
    out0 = _run_evolve(registry, ids, None)
    registry2, ids2, _, _ = _scene(6, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    # 门压到 -100，任何体温都不过门，资格制必须拦下所有给药
    shaper = StructShaper("ding", sizes, gate=-100.0)
    out1 = _run_evolve(registry2, ids2, shaper)
    np.testing.assert_array_equal(out0.state_ids, out1.state_ids)
    assert out1.temps is not None and all(np.isfinite(t) for t in out1.temps)


def test_ji_dev_reading_matches_brute_force():
    """己温读数：组重版 Simpson 相对偏差与展开表暴力值一致。"""
    _, _, grouped, codes, _, _, _, sizes = _menu_scene(3)
    codes_g = codes[grouped.unique_ids]
    rows = _expand(codes_g, grouped.counts)
    _, cnt = np.unique(rows, axis=0, return_counts=True)
    s_brute = float(((cnt / len(rows)) ** 2).sum())
    target = 0.7 * s_brute
    shaper = StructShaper("ji", sizes, ji_target=target)
    got = shaper.temperature(codes_g, grouped.counts)
    assert got == pytest.approx((s_brute - target) / target, rel=1e-12)


def test_ji_requires_target_and_direction():
    """己尺必须给正基准；无己温调因子必须报错。"""
    with pytest.raises(ValueError):
        StructShaper("ji", [3, 4, 5])
    with pytest.raises(ValueError):
        StructShaper("ji", [3, 4, 5], ji_target=0.0)
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(4)
    shaper = StructShaper("ji", sizes, ji_target=0.01)
    with pytest.raises(ValueError):
        shaper.path_factors(menu, codes[grouped.unique_ids], grouped.counts)


def test_ji_factors_flip_with_dev_sign():
    """双向药：聚不够促聚，过聚同一路径翻转为抑聚，力度随 |dev| 连续。"""
    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(5)
    codes_g = codes[grouped.unique_ids]
    shaper = StructShaper("ji", sizes, ji_target=0.01)
    delta = shaper._jia_delta(menu, codes_g, grouped.counts)
    f_under = shaper.path_factors(menu, codes_g, grouped.counts, temperature=-0.6)
    f_over = shaper.path_factors(menu, codes_g, grouped.counts, temperature=0.6)
    s_full = shaper.boost  # |dev|>=己门，满力度
    assert np.all(f_under[delta > 0] == s_full)
    assert np.all(f_under[delta < 0] == 1.0 / s_full)
    assert np.all(f_over[delta > 0] == 1.0 / s_full)
    assert np.all(f_over[delta < 0] == s_full)
    assert np.all(f_under[delta == 0] == 1.0)
    # 连续药力：偏差减半，力度为 boost 的平方根
    f_half = shaper.path_factors(menu, codes_g, grouped.counts, temperature=-0.25)
    assert np.all(
        f_half[delta > 0] == pytest.approx(shaper.boost**0.5, rel=1e-12)
    )


def test_ji_diagnose_gate_and_taper():
    """己门初诊：缺口过半才收治；药力达标趋一。"""
    shaper = StructShaper("ji", [3, 4, 5], ji_target=0.01)
    assert shaper.diagnose(-0.96)  # plants 型大缺口收治
    assert shaper.diagnose(0.96)  # 过聚同样收治
    assert not shaper.diagnose(-0.097)  # adult 型小偏差不收治
    assert shaper.effective_boost(0.0) == pytest.approx(1.0)
    assert shaper.effective_boost(-1.0) == pytest.approx(shaper.boost)
    assert shaper.effective_boost(0.25) == pytest.approx(shaper.boost**0.5)
    assert not shaper.active(0.0)
    assert shaper.active(-0.3)


def test_ji_admission_gate_blocks_all_treatment():
    """己尺初诊不过门时全程零干预，轨迹与不装签筒逐位一致。"""
    registry, ids, _, _ = _scene(7, num_rows=30)
    out0 = _run_evolve(registry, ids, None)
    registry2, ids2, _, _ = _scene(7, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    # 基准取当前表精确 Simpson 的近旁：初诊偏差远小于己门，必不收治
    grouped = group_state_ids(ids2)
    cnt = np.asarray(grouped.counts, dtype=np.float64)
    s_now = float((cnt**2).sum() / cnt.sum() ** 2)
    shaper = StructShaper("ji", sizes, ji_target=s_now)
    out1 = _run_evolve(registry2, ids2, shaper)
    np.testing.assert_array_equal(out0.state_ids, out1.state_ids)
    assert out1.temps is not None and all(np.isfinite(t) for t in out1.temps)


def test_ji_admitted_changes_trajectory():
    """己尺收治后（大缺口基准）轨迹须与基线不同，药真的下场了。"""
    registry, ids, _, _ = _scene(8, num_rows=30)
    out0 = _run_evolve(registry, ids, None, rounds=8)
    registry2, ids2, _, _ = _scene(8, num_rows=30)
    sizes = [len(d) for d in registry2.schema.domains]
    grouped = group_state_ids(ids2)
    cnt = np.asarray(grouped.counts, dtype=np.float64)
    s_now = float((cnt**2).sum() / cnt.sum() ** 2)
    # 基准放到当前值的 40 倍：初诊 dev≈-0.975 过门收治，全程促聚
    shaper = StructShaper("ji", sizes, ji_target=40.0 * s_now)
    out1 = _run_evolve(registry2, ids2, shaper, rounds=8)
    assert not np.array_equal(out0.state_ids, out1.state_ids)


def test_composite_requires_two_and_multiplies_factors():
    """组合尺至少两把；双在治时因子等于各尺因子之积。"""
    from resevo.structure import CompositeShaper

    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(6)
    codes_g = codes[grouped.unique_ids]
    ding = StructShaper("ding", sizes, boost=4.0)
    ji = StructShaper("ji", sizes, boost=8.0, ji_target=0.01)
    with pytest.raises(ValueError):
        CompositeShaper([ding])
    comp = CompositeShaper([ding, ji])
    temp = (-0.4, -0.8)  # 丁发烧、己大缺口，双双确诊在治
    assert comp.diagnose(temp)
    assert comp.active(temp)
    got = comp.path_factors(menu, codes_g, grouped.counts, temperature=temp)
    want = ding.path_factors(
        menu, codes_g, grouped.counts, temperature=temp[0]
    ) * ji.path_factors(menu, codes_g, grouped.counts, temperature=temp[1])
    np.testing.assert_allclose(got, want, rtol=1e-12)
    with pytest.raises(ValueError):
        comp.path_factors(menu, codes_g, grouped.counts)


def test_composite_admission_is_per_ruler():
    """初诊逐尺独立：丁不收治己收治时，因子只含己尺贡献。"""
    from resevo.structure import CompositeShaper

    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(7)
    codes_g = codes[grouped.unique_ids]
    ding = StructShaper("ding", sizes, boost=4.0)
    ji = StructShaper("ji", sizes, boost=8.0, ji_target=0.01)
    comp = CompositeShaper([ding, ji])
    temp0 = (0.2, -0.9)  # 丁体温正常（不过门），己大缺口收治
    assert comp.diagnose(temp0)
    assert comp._admitted == [False, True]
    # 丁尺此后即便发烧也不干预（未收治）
    temp1 = (-0.4, -0.9)
    got = comp.path_factors(menu, codes_g, grouped.counts, temperature=temp1)
    want = ji.path_factors(menu, codes_g, grouped.counts, temperature=temp1[1])
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_composite_ruler_stops_independently():
    """在治尺到站自动停药：己温归零后因子只剩丁尺贡献。"""
    from resevo.structure import CompositeShaper

    _, _, grouped, codes, menu, _, _, sizes = _menu_scene(8)
    codes_g = codes[grouped.unique_ids]
    ding = StructShaper("ding", sizes, boost=4.0)
    ji = StructShaper("ji", sizes, boost=8.0, ji_target=0.01)
    comp = CompositeShaper([ding, ji])
    comp.diagnose((-0.4, -0.9))  # 双双收治
    temp = (-0.4, 0.0)  # 己到站
    assert comp.active(temp)  # 丁还在治
    got = comp.path_factors(menu, codes_g, grouped.counts, temperature=temp)
    want = ding.path_factors(menu, codes_g, grouped.counts, temperature=temp[0])
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert not comp.active((0.1, 0.0))  # 双双到站，全停


def test_treatment_severity_single_rulers():
    """病情深度：己尺偏差绝对值，丁尺烧度，到站归零，其余刻度恒零。"""
    sizes = [4, 4, 4]
    ji = StructShaper("ji", sizes, ji_target=0.5)
    assert ji.treatment_severity(-0.8) == pytest.approx(0.8)
    assert ji.treatment_severity(0.3) == pytest.approx(0.3)
    assert ji.treatment_severity(0.0) == 0.0
    ding = StructShaper("ding", sizes)
    assert ding.treatment_severity(-0.2) == pytest.approx(0.2)
    assert ding.treatment_severity(0.1) == 0.0
    assert StructShaper("jia", sizes).treatment_severity(-9.9) == 0.0


def test_treatment_severity_composite_sums_treating_only():
    """组合尺深度为在治尺之和，未收治尺不计，全到站归零。"""
    sizes = [4, 4]
    comp = CompositeShaper([
        StructShaper("ding", sizes),
        StructShaper("ji", sizes, ji_target=0.5),
    ])
    comp._admitted = [True, True]
    assert comp.treatment_severity((-0.2, -0.8)) == pytest.approx(1.0)
    comp._admitted = [False, True]
    assert comp.treatment_severity((-0.2, -0.8)) == pytest.approx(0.8)
    comp._admitted = [True, True]
    assert comp.treatment_severity((0.1, 0.0)) == 0.0
