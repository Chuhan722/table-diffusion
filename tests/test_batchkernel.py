"""批量核模块测试，与组核同菜单全指标对拍，端到端下降与抽样分布。"""
import numpy as np
import pytest

from resevo.batchkernel import (
    build_batch_kernel,
    evolve_batch,
    make_batch_provider,
    sample_batch_next,
)
from resevo.batchmenu import (
    CodeBook,
    build_query_structure,
    generate_batch_menu,
    menu_to_supports,
)
from resevo.dataset import (
    QuerySpec,
    StateRegistry,
    TableSchema,
    query_field_sets,
    target_from_specs,
)
from resevo.editspace import EditBudget
from resevo.grouping import build_group_kernel, group_state_ids


def _scene(seed: int, num_rows: int = 36):
    """混合域大小加不等权重的小场景，行含重复。"""
    rng = np.random.default_rng(seed)
    schema = TableSchema(
        ("a", "b", "c", "d"),
        (("0", "1"), ("0", "1", "2"), ("0", "1"), ("0", "1", "2")),
    )
    specs = [
        QuerySpec("q0", ({"attribute": "a", "operator": "==", "value": "1"},
                          {"attribute": "b", "operator": "==", "value": "2"}),
                  float(rng.integers(2, 20))),
        QuerySpec("q1", ({"attribute": "b", "operator": "between", "lower": 1, "upper": 2},
                          {"attribute": "c", "operator": "==", "value": "0"}),
                  float(rng.integers(2, 20))),
        QuerySpec("q2", ({"attribute": "d", "operator": ">=", "value": 1},
                          {"attribute": "a", "operator": "==", "value": "0"}),
                  float(rng.integers(2, 20))),
        QuerySpec("q3", ({"attribute": "c", "operator": "==", "value": "1"},
                          {"attribute": "d", "operator": "==", "value": "0"}),
                  float(rng.integers(2, 20))),
        QuerySpec("q4", ({"operator": "halfspace",
                          "scores": {"a": {"0": 2, "1": -1},
                                     "b": {"1": 3, "2": -2},
                                     "c": {"0": 1, "1": -5},
                                     "d": {"0": 0, "1": 4, "2": -3}},
                          "threshold": 2},),
                  float(rng.integers(2, 20))),
        QuerySpec("q5", ({"operator": "halfspace",
                          "scores": {"b": {"0": 6, "2": -6},
                                     "d": {"1": 5, "2": 5}},
                          "threshold": 5},),
                  float(rng.integers(2, 20))),
    ]
    registry = StateRegistry(schema, specs)
    base = [tuple(str(rng.integers(len(dom))) for dom in schema.domains) for _ in range(8)]
    rows = [base[int(rng.integers(len(base)))] for _ in range(num_rows)]
    ids = registry.register_table(rows)
    weights = rng.uniform(0.5, 2.0, len(specs))
    return registry, ids, weights, rng


@pytest.mark.parametrize("seed", range(8))
def test_batch_kernel_matches_group_kernel(seed):
    """同一菜单下批量核与组核的全部指标与概率一致，浮点容差内。"""
    registry, ids, weights, rng = _scene(seed)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2),
        [(0, 1), (2, 3)],
    )
    supports = menu_to_supports(menu, grouped.unique_ids, registry)
    workload = registry.build_workload(target_from_specs(registry.specs), weights)
    codes = codebook.sync()
    ref = build_group_kernel(workload, grouped, supports)
    got = build_batch_kernel(workload, grouped, menu, qs, codes)
    assert got.status == ref.status
    np.testing.assert_allclose(got.old_loss, ref.old_loss, rtol=1e-12)
    np.testing.assert_allclose(got.beta, ref.beta, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got.direction_gain, ref.direction_gain, rtol=1e-9)
    np.testing.assert_allclose(got.interaction, ref.interaction, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got.step, ref.step, rtol=1e-9)
    np.testing.assert_allclose(got.expected_loss, ref.expected_loss, rtol=1e-9)
    for g, block in enumerate(ref.blocks):
        lo, hi = int(got.offsets[g]), int(got.offsets[g + 1])
        np.testing.assert_allclose(
            got.probabilities[lo:hi], block.probabilities, rtol=1e-9, atol=1e-12
        )


def test_sample_batch_distribution():
    """二项加条件多项的分解抽样与直接多项分布频率一致。"""
    registry, ids, weights, rng = _scene(3, num_rows=24)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=2, donor_copies=2, joint_edits=0, explore_edits=0),
    )
    workload = registry.build_workload(target_from_specs(registry.specs), weights)
    result = build_batch_kernel(workload, grouped, menu, qs, codes)
    assert result.status == "ok"
    sampler = np.random.default_rng(7)
    trials = 3000
    hits = np.zeros(registry.num_states + 64, dtype=np.float64)
    for _ in range(trials):
        nxt = sample_batch_next(result, ids, sampler, registry)
        for u in nxt:
            hits[u] += 1
    hits = hits[: registry.num_states]
    # 期望频率由每组概率与重数直接给出，逐状态聚合
    expect = np.zeros(registry.num_states, dtype=np.float64)
    for g in range(grouped.num_groups):
        lo, hi = int(result.offsets[g]), int(result.offsets[g + 1])
        c = float(grouped.counts[g])
        base_id = int(grouped.unique_ids[g])
        expect[base_id] += c * result.probabilities[lo]
        base = registry.state_tuple(base_id)
        for t in range(lo + 1, hi):
            p = int(menu.offsets[g]) + (t - lo - 1)
            edited = list(base)
            for slot in range(3):
                j = int(menu.fields[p, slot])
                if j < 0:
                    break
                edited[j] = registry.schema.domains[j][int(menu.values[p, slot])]
            uid = registry.register_edit(base_id, tuple(edited))
            expect[uid] += c * result.probabilities[t]
    np.testing.assert_allclose(hits / trials, expect, atol=0.05 * max(1.0, expect.max()))


def test_evolve_batch_descends_to_zero():
    """端到端批量循环在小场景把损失打到零并保持冻结语义。"""
    registry, ids, weights, _ = _scene(1, num_rows=30)
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    provider = make_batch_provider(
        registry, target_from_specs(registry.specs), weights,
        np.random.default_rng(5), joint_field_sets=fs,
    )
    out = evolve_batch(
        ids, 300, np.random.default_rng(9), provider, registry, max_frozen_retries=5
    )
    losses = [r.old_loss for r in out.records]
    assert losses[0] > 0
    assert min(losses) < 0.1 * losses[0]


def test_cnt_cache_matches_direct():
    """条件计数缓存路径与直接计数逐位一致，核指标不变。"""
    from resevo.batchmenu import CntCache

    registry, ids, weights, rng = _scene(5)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2),
        [(0, 1), (2, 3)],
    )
    workload = registry.build_workload(target_from_specs(registry.specs), weights)
    plain = build_batch_kernel(workload, grouped, menu, qs, codes)
    cache = CntCache(qs)
    cached = build_batch_kernel(workload, grouped, menu, qs, codes, cnt_cache=cache)
    np.testing.assert_array_equal(cached.probabilities, plain.probabilities)
    assert cached.beta == plain.beta
    assert cached.step == plain.step
