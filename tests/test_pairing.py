"""配对板块测试。

锚点来自补充设计文档第二部分的例子，查询 A B AB，目标 (1,1,0)，
权重 diag(2,2,1)，表 (00,11)，所有单行动作增益为负，
交换字段后 (01,10) 联合增益 0.5，Gamma 恰为 0.5。
"""
import numpy as np
import pytest

from resevo.candidates import BlockSupport, validate_partition
from resevo.dataset import (
    QuerySpec,
    StateRegistry,
    TableSchema,
    query_field_sets,
    target_from_specs,
)
from resevo.editspace import EditBudget, make_edit_provider
from resevo.engine import evolve
from resevo.pairing import PairingBudget, build_paired_supports, make_paired_provider
from resevo.state import table_loss


def _toy():
    schema = TableSchema(("A", "B"), (("0", "1"), ("0", "1")))
    specs = [
        QuerySpec("SA", ({"attribute": "A", "operator": "==", "value": "1"},), 1.0),
        QuerySpec("SB", ({"attribute": "B", "operator": "==", "value": "1"},), 1.0),
        QuerySpec(
            "SAB",
            (
                {"attribute": "A", "operator": "==", "value": "1"},
                {"attribute": "B", "operator": "==", "value": "1"},
            ),
            0.0,
        ),
    ]
    registry = StateRegistry(schema, specs)
    return registry, target_from_specs(specs), np.array([2.0, 2.0, 1.0])


def _full_single_menus(registry, rows):
    """全集单行菜单，四个状态全注册，每行给其余三个状态。"""
    all_states = [
        registry.register(t)
        for t in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))
    ]
    ids = registry.register_table(rows)
    menus = []
    for i, sid in enumerate(ids):
        outs = tuple((u,) for u in all_states if u != int(sid))
        menus.append(BlockSupport((i,), outs, (1.0,) * len(outs)))
    return ids, menus


def test_gamma_hand_anchor():
    """文档例逐数字对上，配出 (0,1) 对且 Gamma 恰为 0.5。"""
    registry, y, w = _toy()
    ids, menus = _full_single_menus(registry, [("0", "0"), ("1", "1")])
    menu = build_paired_supports(ids, menus, registry, y, w, np.random.default_rng(0))
    assert len(menu.pairs) == 1
    i, k, gamma = menu.pairs[0]
    assert (i, k) == (0, 1)
    assert gamma == pytest.approx(0.5, abs=1e-12)
    paired = [sp for sp in menu.supports if len(sp.rows) == 2]
    assert len(paired) == 1 and paired[0].rows == (0, 1)


def test_shuffled_rows_still_recover_pair():
    """打乱行顺序后自动配对仍能找出补偿对，文档给的验收标准。"""
    registry, y, w = _toy()
    ids, menus = _full_single_menus(registry, [("1", "1"), ("0", "0")])
    for seed in range(5):
        menu = build_paired_supports(
            ids, menus, registry, y, w, np.random.default_rng(seed)
        )
        assert [(i, k) for i, k, _ in menu.pairs] == [(0, 1)]
        assert menu.pairs[0][2] == pytest.approx(0.5, abs=1e-12)


def test_joint_menu_keeps_all_single_edits():
    """联合菜单必须完整保留两边全部单边动作，不能删掉有益单行方向。"""
    registry, y, w = _toy()
    ids, menus = _full_single_menus(registry, [("0", "0"), ("1", "1")])
    menu = build_paired_supports(ids, menus, registry, y, w, np.random.default_rng(0))
    sp = [b for b in menu.supports if len(b.rows) == 2][0]
    si, sk = int(ids[0]), int(ids[1])
    outs = set(sp.outcomes)
    for (u,) in menus[0].outcomes:
        assert (u, sk) in outs
    for (v,) in menus[1].outcomes:
        assert (si, v) in outs


def test_partition_and_budgets_respected():
    """输出必须是全行不相交划分，块至多两行，对数与菜单长度守预算。"""
    registry, y, w = _toy()
    rows = [("0", "0"), ("1", "1"), ("0", "0"), ("1", "1"), ("0", "1"), ("1", "0")]
    ids, menus = _full_single_menus(registry, rows)
    budget = PairingBudget(max_pairs=2)
    menu = build_paired_supports(
        ids, menus, registry, y, w, np.random.default_rng(3), budget
    )
    validate_partition(menu.supports, len(rows))
    assert len(menu.pairs) <= 2
    for sp in menu.supports:
        assert len(sp.rows) <= 2
        if len(sp.rows) == 2:
            joint_cap = budget.swap_actions + budget.combine_actions
            assert len(sp.outcomes) <= 3 + 3 + joint_cap
    for _, _, gamma in menu.pairs:
        assert gamma > budget.gamma_tol


def test_paired_provider_breaks_deadlock():
    """单行菜单在死角例必冻结停机，配对提供器把损失打到零。"""
    jfs_ref, y, w = None, None, None
    # 纯单行提供器，卡死并停在正损失
    registry_a, y, w = _toy()
    ids_a = registry_a.register_table([("0", "0"), ("1", "1")])
    jfs = query_field_sets(registry_a.specs, registry_a.schema)
    single = make_edit_provider(
        registry_a, y, w, np.random.default_rng(7), EditBudget(), jfs
    )
    out_a = evolve(
        None, ids_a, 60, np.random.default_rng(11),
        support_provider=single, max_frozen_retries=5,
    )
    assert out_a.stop_reason == "no_positive_direction"
    assert table_loss(registry_a.build_workload(y, w), out_a.state_ids) > 0

    # 配对提供器，冻结重试轮启用配对，损失降到零
    registry_b, y, w = _toy()
    ids_b = registry_b.register_table([("0", "0"), ("1", "1")])
    paired = make_paired_provider(
        registry_b, y, w, np.random.default_rng(7), EditBudget(), jfs
    )
    out_b = evolve(
        None, ids_b, 60, np.random.default_rng(11),
        support_provider=paired, max_frozen_retries=10,
    )
    assert table_loss(registry_b.build_workload(y, w), out_b.state_ids) <= 1e-12
