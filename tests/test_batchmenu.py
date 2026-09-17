"""批量菜单模块测试，编码重构对拍与规范化语义。"""
import numpy as np
import pytest

from resevo.batchmenu import (
    BatchMenu,
    CodeBook,
    _normalize_menu,
    build_query_structure,
    condition_counts,
    generate_batch_menu,
)
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.editspace import EditBudget


def _mixed_setup(seed: int):
    """混合域大小与三种算子的小场景，够碰到全部代码路径。"""
    rng = np.random.default_rng(seed)
    schema = TableSchema(
        ("a", "b", "c", "d", "e"),
        (("0", "1"), ("0", "1", "2"), ("0", "1"), ("0", "1", "2"), ("0", "1")),
    )
    specs = [
        QuerySpec("q0", ({"attribute": "a", "operator": "==", "value": "1"},
                          {"attribute": "b", "operator": "==", "value": "2"}), 5.0),
        QuerySpec("q1", ({"attribute": "b", "operator": "between", "lower": 1, "upper": 2},
                          {"attribute": "c", "operator": "==", "value": "0"}), 7.0),
        QuerySpec("q2", ({"attribute": "d", "operator": ">=", "value": 1},
                          {"attribute": "e", "operator": "==", "value": "1"},
                          {"attribute": "a", "operator": "==", "value": "0"}), 3.0),
        QuerySpec("q3", ({"attribute": "b", "operator": "==", "value": "0"},
                          {"attribute": "b", "operator": "between", "lower": 0, "upper": 1},
                          {"attribute": "d", "operator": "==", "value": "2"}), 2.0),
    ]
    registry = StateRegistry(schema, specs)
    rows = [
        tuple(str(rng.integers(len(dom))) for dom in schema.domains) for _ in range(40)
    ]
    ids = registry.register_table(rows)
    return registry, ids, rng


@pytest.mark.parametrize("seed", range(6))
def test_condition_counts_reconstruct_features(seed):
    """条件计数与字段组个数的相等判断逐位重构注册表特征。"""
    registry, ids, _ = _mixed_setup(seed)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    codes = codebook.sync()
    cnt, _ = condition_counts(qs, codes)
    rebuilt = (cnt == qs.ncond[None, :]).astype(np.float64)
    workload = registry.build_workload(
        np.zeros(len(registry.specs)), np.ones(len(registry.specs))
    )
    np.testing.assert_array_equal(rebuilt, np.asarray(workload.features))


def test_normalize_menu_semantics():
    """无效槽清空，落回源丢弃，组内重复合并质量且保持首现顺序。"""
    codes_g = np.array([[0, 1], [1, 0]], dtype=np.int32)
    group = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    fields = np.array(
        [[0, -1, -1], [0, 1, -1], [0, -1, -1], [1, 0, -1], [1, -1, -1]],
        dtype=np.int32,
    )
    values = np.array(
        [[1, -1, -1], [0, 1, -1], [1, -1, -1], [1, 1, -1], [0, -1, -1]],
        dtype=np.int32,
    )
    # 路径 0 与 2 相同，路径 1 两槽全是无效变化应整条丢弃，
    # 路径 3 的字段 1 槽无效只剩字段 0 改 1，与路径 0 相同合并成质量 3，
    # 组 1 的路径替代值等于当前值是落回源，整条丢弃
    menu = _normalize_menu(group, fields, values, codes_g, 2)
    assert menu.offsets.tolist() == [0, 1, 1]
    assert menu.group.tolist() == [0]
    assert menu.fields[0].tolist() == [0, -1, -1]
    assert menu.values[0].tolist() == [1, -1, -1]
    assert menu.mass.tolist() == [3.0]


@pytest.mark.parametrize("seed", range(4))
def test_generate_batch_menu_paths_valid(seed):
    """生成的菜单全部路径有效，无落回源，组内签名唯一，值码在域内。"""
    registry, ids, rng = _mixed_setup(seed)
    codebook = CodeBook(registry)
    codes = codebook.sync()
    from resevo.grouping import group_state_ids

    grouped = group_state_ids(ids)
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(), [(0, 1), (1, 2, 3)],
    )
    codes_g = codes[grouped.unique_ids]
    seen = set()
    for p in range(menu.num_paths):
        g = int(menu.group[p])
        changes = []
        for slot in range(3):
            j = int(menu.fields[p, slot])
            if j < 0:
                assert menu.values[p, slot] < 0
                continue
            v = int(menu.values[p, slot])
            assert 0 <= v < codebook.domain_sizes[j]
            assert v != codes_g[g, j], "替代值不得等于当前值"
            changes.append((j, v))
        assert changes, "路径必须至少一个有效变化"
        assert changes == sorted(changes), "槽位按字段升序"
        key = (g, tuple(changes))
        assert key not in seen, "组内签名必须唯一"
        seen.add(key)
    assert menu.offsets[-1] == menu.num_paths
    assert np.all(np.diff(menu.offsets) >= 0)
