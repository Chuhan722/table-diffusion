"""零空间换位相位测试。

覆盖：换位保全部一阶二阶与课内损失严格不变（金标准），平凡互换
拒绝，恒温器健康跳过，接受率自熄，evolve_batch 校验与默认关对拍。
"""
import numpy as np
import pytest

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import StateRegistry, TableSchema, target_from_specs
from resevo.editspace import EditBudget
from resevo.nullswap import NullSwapPhase
from resevo.state import table_loss

from test_batchkernel import _scene
from test_rowtree import _grid_specs

BUDGET = EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2)


def _swap_schema() -> TableSchema:
    return TableSchema(("a", "b", "c"), (("0", "1"), ("0", "1"), ("0", "1")))


def _codes(rows) -> np.ndarray:
    return np.array([[int(v) for v in r] for r in rows], dtype=np.int32)


def _all_marginals(x: np.ndarray, doms=(2, 2, 2)):
    """全部一阶与二阶联表计数，逐格比对用。"""
    out = []
    f = x.shape[1]
    for j in range(f):
        out.append(np.bincount(x[:, j], minlength=doms[j]))
    for j in range(f):
        for k in range(j + 1, f):
            out.append(np.bincount(
                x[:, j] * doms[k] + x[:, k], minlength=doms[j] * doms[k]
            ))
    return out


def _cross_scene_rows():
    """跨桶交叉图案加厚模式：换位可把细行并入厚模式，辛普森必升。

    c=1 桶：细细反对角 (0,1,1) (1,0,1) 与厚模式 (1,1,1)x3 (0,0,1)x3；
    c=0 桶：对角 (0,0,0) (1,1,0)。
    """
    rows = [("1", "1", "1")] * 3 + [("0", "0", "1")] * 3
    rows += [("0", "1", "1"), ("1", "0", "1"), ("0", "0", "0"), ("1", "1", "0")]
    return rows


def test_swap_preserves_marginals_and_loss():
    """接受后全部一阶二阶逐格不变，课内损失严格不变（金标准）。"""
    schema = _swap_schema()
    rows = _cross_scene_rows()
    x = _codes(rows)
    specs = _grid_specs(schema, lambda a, b, va, vb: 2.0)
    registry = StateRegistry(schema, specs)
    ids = np.asarray(registry.register_table(rows))
    y = np.array([q.result for q in specs])
    w = np.ones(len(specs))
    loss0 = table_loss(registry.build_workload(y, w), ids)
    phase = NullSwapPhase(schema, kurt_gate=999.0, bucket_cap=16, seed=5)
    accepted = 0
    xa = x
    for _ in range(6):
        new_x, stats = phase.run(xa)
        accepted += stats.get("accepts", 0)
        if new_x is not None:
            xa = new_x
    assert accepted > 0, "固定种子下应有接受"
    for got, want in zip(_all_marginals(xa), _all_marginals(x)):
        np.testing.assert_array_equal(got, want)
    new_rows = [tuple(str(int(v)) for v in r) for r in xa]
    new_ids = np.asarray(registry.register_table(new_rows))
    assert table_loss(registry.build_workload(y, w), new_ids) == loss0


def test_trivial_same_bucket_swap_rejected():
    """甲乙同桶时换位只是模式互换，辛普森增量为零应全拒。"""
    schema = _swap_schema()
    rows = [("0", "1", "1"), ("1", "0", "1"), ("0", "0", "1"), ("1", "1", "1")]
    phase = NullSwapPhase(schema, kurt_gate=999.0, bucket_cap=16, seed=3)
    for _ in range(2):
        new_x, stats = phase.run(_codes(rows))
        assert new_x is None
        assert stats["accepts"] == 0 and stats["proposals"] > 0


def test_healthy_table_skipped():
    """峰度不低于门槛整场跳过，表不动不计衰竭。"""
    schema = _swap_schema()
    phase = NullSwapPhase(schema, kurt_gate=-999.0)
    x = _codes(_cross_scene_rows())
    new_x, stats = phase.run(x)
    assert new_x is None
    assert stats["skipped"] == "healthy"
    assert not phase.dead


def test_extinguish_after_two_dry_phases():
    """零候选场接受率为零，连续两场衰竭永久熄火。"""
    schema = _swap_schema()
    rows = [("1", "1", "1")] * 12  # 全同行无对角对
    phase = NullSwapPhase(schema, kurt_gate=999.0)
    _, s1 = phase.run(_codes(rows))
    assert s1["accepts"] == 0 and not s1["dead"]
    _, s2 = phase.run(_codes(rows))
    assert s2["dead"] and phase.dead
    _, s3 = phase.run(_codes(rows))
    assert s3["skipped"] == "dead"


def test_evolve_batch_validation():
    """相位对象与周期必须成对，负参数拒收。"""
    registry, ids, weights, _ = _scene(0)
    y = target_from_specs(registry.specs)
    schema = registry.schema
    provider = make_batch_provider(
        registry, y, weights, np.random.default_rng(0), budget=BUDGET
    )
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="须同时给出"):
        evolve_batch(ids, 2, rng, provider, registry, nullswap_every=3)
    with pytest.raises(ValueError, match="须同时给出"):
        evolve_batch(
            ids, 2, rng, provider, registry,
            nullswap_phase=NullSwapPhase(schema),
        )
    with pytest.raises(ValueError, match="不能为负"):
        evolve_batch(ids, 2, rng, provider, registry, nullswap_every=-1)
    with pytest.raises(ValueError, match="不能为负"):
        evolve_batch(
            ids, 2, rng, provider, registry,
            nullswap_phase=NullSwapPhase(schema), nullswap_every=2,
            nullswap_after=-1,
        )


def test_evolve_batch_default_off_identical():
    """默认参数与显式关闭逐位一致，开关不碰随机流。"""

    def run(**kw):
        registry, ids, weights, _ = _scene(4)
        y = target_from_specs(registry.specs)
        provider = make_batch_provider(
            registry, y, weights, np.random.default_rng(7), budget=BUDGET
        )
        return evolve_batch(
            ids.copy(), 6, np.random.default_rng(9), provider, registry, **kw
        )

    a = run()
    b = run(nullswap_phase=None, nullswap_every=0, nullswap_after=0)
    np.testing.assert_array_equal(a.state_ids, b.state_ids)
    assert [r.old_loss for r in a.records] == [r.old_loss for r in b.records]


def test_evolve_batch_with_phase_runs():
    """开换位相位整链可跑，行数不变，健康表相位零干预逐位一致。"""

    def provider():
        registry, ids, weights, _ = _scene(11)
        y = target_from_specs(registry.specs)
        return registry, ids, make_batch_provider(
            registry, y, weights, np.random.default_rng(3), budget=BUDGET,
        )

    reg_a, ids_a, prov_a = provider()
    base = evolve_batch(
        ids_a.copy(), 8, np.random.default_rng(5), prov_a, reg_a
    )
    reg_b, ids_b, prov_b = provider()
    # 门槛拉到极低使恒温器恒判健康，相位每场跳过，轨迹应逐位一致
    phase = NullSwapPhase(reg_b.schema, kurt_gate=-999.0)
    out = evolve_batch(
        ids_b.copy(), 8, np.random.default_rng(5), prov_b, reg_b,
        nullswap_phase=phase, nullswap_every=2,
    )
    np.testing.assert_array_equal(base.state_ids, out.state_ids)
    assert len(out.state_ids) == len(ids_b)
