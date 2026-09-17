"""懒注册批量菜单，候选编辑用差分表示，不再逐个登记进注册表。

按文档规模化建议，菜单里的候选改法只需要答案差向量就能算增益，
改动一个字段只影响依赖该字段的查询，差向量可以按字段整列批量算出，
只有真被抽中落地的状态才注册，注册表与特征矩阵不再随候选无界膨胀。

本模块负责三件事，
一，值编码本，把字符串状态元组映射成整数码矩阵供矢量化比较，
二，查询结构表，每字段的依赖查询与域值通过表，条件计数分解，
三，批量菜单生成，四类候选路径全部矢量化抽取并规范化合并，
候选来源与预算和行级菜单一致，随机数消费顺序不同但分布语义相同。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import BlockSupport
from .dataset import StateRegistry, evaluate_compiled
from .editspace import EditBudget

_NO_FIELD = np.int32(-1)


class CodeBook:
    """值编码本，状态元组按字段值域映射成整数码，随注册表增量同步。

    码就是值在 schema 值域元组里的下标，注册表行只增不改，
    因此码矩阵只需要追加新行，已编码行永不变。
    """

    def __init__(self, registry: StateRegistry):
        self.registry = registry
        schema = registry.schema
        self.value_maps = tuple(
            {v: c for c, v in enumerate(domain)} for domain in schema.domains
        )
        self.domain_sizes = np.array(
            [len(domain) for domain in schema.domains], dtype=np.int32
        )
        self._codes = np.empty((0, schema.num_fields), dtype=np.int32)
        self._synced = 0

    def sync(self) -> NDArray[np.int32]:
        """把注册表新增状态补进码矩阵，返回前段只读视图。"""
        total = self.registry.num_states
        if total > self._codes.shape[0]:
            capacity = max(1024, self._codes.shape[0])
            while capacity < total:
                capacity *= 2
            grown = np.empty((capacity, self._codes.shape[1]), dtype=np.int32)
            grown[: self._synced] = self._codes[: self._synced]
            self._codes = grown
        for s in range(self._synced, total):
            row = self.registry.state_tuple(s)
            self._codes[s] = [m[v] for m, v in zip(self.value_maps, row)]
        self._synced = total
        view = self._codes[:total]
        view.setflags(write=False)
        return view


@dataclass(frozen=True)
class QueryStructure:
    """查询结构表，条件计数分解的全部静态信息，构造一次跨轮复用。

    每条查询按涉及字段拆成字段级条件组，ncond 是字段组个数，
    allow[j][t, v] 表示依赖字段 j 的第 t 条查询在 j 上的条件组对域值码 v 是否通过，
    一行通过查询当且仅当它在所有涉及字段上的条件组都通过，
    因此通过数计数矩阵 cnt 与 ncond 的相等判断精确重构查询特征。
    """

    dep: tuple[NDArray[np.int64], ...]  # 每字段依赖查询列表
    allow: tuple[NDArray[np.bool_], ...]  # 每字段 mj 乘 Kj 通过表
    ncond: NDArray[np.int64]  # 每查询的字段组个数


def build_query_structure(registry: StateRegistry, codebook: CodeBook) -> QueryStructure:
    """从预编译条件构造查询结构表，同字段多条件按与合并。"""
    schema = registry.schema
    num_queries = len(registry.specs)
    dep = registry.field_queries
    ncond = np.zeros(num_queries, dtype=np.int64)
    per_field_conds: list[dict[int, list[tuple]]] = [dict() for _ in range(schema.num_fields)]
    for q, conds in enumerate(registry.compiled):
        fields = set()
        for cond in conds:
            j = int(cond[1])
            fields.add(j)
            per_field_conds[j].setdefault(q, []).append(cond)
        ncond[q] = len(fields)
    allow = []
    for j in range(schema.num_fields):
        domain = schema.domains[j]
        table = np.ones((len(dep[j]), len(domain)), dtype=np.bool_)
        for t, q in enumerate(dep[j]):
            conds = tuple(per_field_conds[j][int(q)])
            for v, value in enumerate(domain):
                # 借用编译求值语义，单字段条件组对该域值的与合并
                table[t, v] = evaluate_compiled(conds, _probe_row(schema.num_fields, j, value)) > 0
        allow.append(table)
    return QueryStructure(dep, tuple(allow), ncond)


def _probe_row(num_fields: int, j: int, value: str) -> tuple[str, ...]:
    """探针行，只有第 j 位有真值，其余位不会被单字段条件组读取。"""
    row = [""] * num_fields
    row[j] = value
    return tuple(row)


def condition_counts(
    qs: QueryStructure, codes_g: NDArray[np.int32]
) -> tuple[NDArray[np.int16], list[NDArray[np.bool_]]]:
    """条件计数矩阵与每字段通过表，cnt[g,q] 是组 g 在查询 q 上通过的字段组数。

    cnt 与 ncond 的相等判断逐位重构注册表特征，有测试把守。
    """
    num_groups, num_fields = codes_g.shape
    cnt = np.zeros((num_groups, len(qs.ncond)), dtype=np.int16)
    pass_list: list[NDArray[np.bool_]] = []
    for j in range(num_fields):
        if len(qs.dep[j]) == 0:
            pass_list.append(np.zeros((num_groups, 0), dtype=np.bool_))
            continue
        pj = qs.allow[j][:, codes_g[:, j]].T  # G 乘 mj
        pass_list.append(pj)
        cnt[:, qs.dep[j]] += pj
    return cnt, pass_list


class CntCache:
    """状态级条件计数缓存，行与注册表编号对齐，只增不改跨轮复用。

    每状态的条件计数只依赖状态本身，与轮次无关，
    新注册状态按增量补算，每轮取行只是一次切片拷贝。
    """

    def __init__(self, qs: QueryStructure):
        self.qs = qs
        self._cnt = np.empty((0, len(qs.ncond)), dtype=np.int16)
        self._synced = 0

    def sync(self, codes: NDArray[np.int32]) -> NDArray[np.int16]:
        total = codes.shape[0]
        if total > self._cnt.shape[0]:
            capacity = max(1024, self._cnt.shape[0])
            while capacity < total:
                capacity *= 2
            grown = np.empty((capacity, self._cnt.shape[1]), dtype=np.int16)
            grown[: self._synced] = self._cnt[: self._synced]
            self._cnt = grown
        if total > self._synced:
            fresh, _ = condition_counts(self.qs, codes[self._synced : total])
            self._cnt[self._synced : total] = fresh
            self._synced = total
        view = self._cnt[:total]
        view.setflags(write=False)
        return view


@dataclass(frozen=True)
class BatchMenu:
    """批量菜单，每条路径是至多三个字段的差分，已规范化合并。

    规范化语义与行级菜单一致，落回源的路径丢弃，
    同组产生相同后继的路径合并质量，顺序保持首次出现顺序，
    mass 是合并前的路径条数，对应行级迁移率全一时的参考质量。
    """

    offsets: NDArray[np.int64]  # G+1，每组路径段边界
    group: NDArray[np.int64]  # P，每路径所属组
    fields: NDArray[np.int32]  # P 乘 3，改动字段，-1 填充
    values: NDArray[np.int32]  # P 乘 3，改动新值码
    mass: NDArray[np.float64]  # P，合并质量

    @property
    def num_paths(self) -> int:
        return len(self.group)


def _alt_codes(
    cur: NDArray[np.int32], sizes: NDArray[np.int32], r: NDArray[np.float64]
) -> NDArray[np.int32]:
    """均匀抽一个不同于当前码的替代码，值域退化时返回 -1。"""
    span = np.maximum(sizes - 1, 1)
    v = np.minimum((r * span).astype(np.int32), span - 1)
    v += v >= cur
    return np.where(sizes <= 1, _NO_FIELD, v).astype(np.int32)


def _alt2_codes(
    cur: NDArray[np.int32],
    first: NDArray[np.int32],
    sizes: NDArray[np.int32],
    r: NDArray[np.float64],
) -> NDArray[np.int32]:
    """在当前码与第一替代码之外均匀抽第二替代码，值域不足时返回 -1。"""
    span = np.maximum(sizes - 2, 1)
    v = np.minimum((r * span).astype(np.int32), span - 1)
    lo = np.minimum(cur, first)
    hi = np.maximum(cur, first)
    v += v >= lo
    v += v >= hi
    return np.where(sizes <= 2, _NO_FIELD, v).astype(np.int32)


def _distinct_fields(
    num_fields: int, count: int, rng: np.random.Generator, how_many: int
) -> NDArray[np.int32]:
    """批量抽至多三个互不相同的字段，均匀无放回，列为字段槽。"""
    out = np.full((count, 3), _NO_FIELD, dtype=np.int32)
    f1 = rng.integers(num_fields, size=count).astype(np.int32)
    out[:, 0] = f1
    if how_many >= 2 and num_fields >= 2:
        f2 = rng.integers(num_fields - 1, size=count).astype(np.int32)
        f2 += f2 >= f1
        out[:, 1] = f2
        if how_many >= 3 and num_fields >= 3:
            f3 = rng.integers(num_fields - 2, size=count).astype(np.int32)
            lo = np.minimum(f1, f2)
            hi = np.maximum(f1, f2)
            f3 += f3 >= lo
            f3 += f3 >= hi
            out[:, 2] = f3
    return out


def generate_batch_menu(
    codes_g: NDArray[np.int32],
    counts: NDArray[np.int64],
    domain_sizes: NDArray[np.int32],
    rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
) -> BatchMenu:
    """矢量化生成整轮菜单，候选来源与预算与行级菜单一致。

    单属性修改抽字段每字段至多两个替代值，供体复制按重数加权抽供体行，
    联合修改从查询字段集合抽取，支持探索独立换一到三个字段，
    随机数消费全部批量化，顺序与行级版不同但每类路径的分布语义相同。
    """
    if budget is None:
        budget = EditBudget()
    joint_sets = [fs for fs in (joint_field_sets or []) if len(fs) >= 2]
    num_groups, num_fields = codes_g.shape
    pieces_f: list[NDArray[np.int32]] = []
    pieces_v: list[NDArray[np.int32]] = []
    pieces_g: list[NDArray[np.int64]] = []
    gidx = np.arange(num_groups, dtype=np.int64)

    def _push(groups, fields, values):
        pieces_g.append(groups)
        pieces_f.append(fields)
        pieces_v.append(values)

    # 一，单属性修改，每组抽 m 个字段，每字段至多两个替代值
    m = min(budget.max_edit_fields, num_fields)
    if m > 0:
        perm = np.argsort(rng.random((num_groups, num_fields)), axis=1)[:, :m].astype(np.int32)
        sizes = domain_sizes[perm]
        cur = np.take_along_axis(codes_g, perm, axis=1)
        v1 = _alt_codes(cur, sizes, rng.random((num_groups, m)))
        singles_f = [perm]
        singles_v = [v1]
        if budget.max_values_per_field >= 2:
            v2 = _alt2_codes(cur, v1, sizes, rng.random((num_groups, m)))
            singles_f.append(perm)
            singles_v.append(v2)
        for fcol, vcol in zip(singles_f, singles_v):
            fields = np.full((num_groups * m, 3), _NO_FIELD, dtype=np.int32)
            values = np.full((num_groups * m, 3), _NO_FIELD, dtype=np.int32)
            fields[:, 0] = fcol.ravel()
            values[:, 0] = vcol.ravel()
            _push(np.repeat(gidx, m), fields, values)

    # 二，供体复制，按重数加权抽供体组，改一到两个字段为供体的值
    if budget.donor_copies > 0 and num_groups > 0:
        d = budget.donor_copies
        cumulative = np.cumsum(counts)
        draws = rng.integers(int(cumulative[-1]), size=(num_groups, d))
        donors = np.searchsorted(cumulative, draws, side="right")
        k = rng.integers(1, 3, size=(num_groups, d))
        dfields = _distinct_fields(num_fields, num_groups * d, rng, 2)
        dfields[:, 2] = _NO_FIELD
        dfields[(k.ravel() < 2), 1] = _NO_FIELD
        donor_flat = donors.ravel()
        values = np.full((num_groups * d, 3), _NO_FIELD, dtype=np.int32)
        for slot in range(2):
            live = dfields[:, slot] >= 0
            values[live, slot] = codes_g[donor_flat[live], dfields[live, slot]]
        _push(np.repeat(gidx, d), dfields, values)

    # 三，联合修改，从查询涉及的多字段集合抽一个，各字段换替代值
    if budget.joint_edits > 0 and joint_sets:
        jn = budget.joint_edits
        sets_arr = np.full((len(joint_sets), 3), _NO_FIELD, dtype=np.int32)
        for i, fs in enumerate(joint_sets):
            sets_arr[i, : min(len(fs), 3)] = fs[:3]
        pick = rng.integers(len(joint_sets), size=(num_groups, jn))
        jfields = sets_arr[pick.ravel()]
        jgroups = np.repeat(gidx, jn)
        values = np.full((num_groups * jn, 3), _NO_FIELD, dtype=np.int32)
        for slot in range(3):
            live = jfields[:, slot] >= 0
            if not np.any(live):
                continue
            fj = jfields[live, slot]
            values[live, slot] = _alt_codes(
                codes_g[jgroups[live], fj], domain_sizes[fj], rng.random(int(live.sum()))
            )
        _push(jgroups, jfields, values)

    # 四，支持探索，独立抽一到三个字段换替代值，可产生表里没有的状态
    if budget.explore_edits > 0:
        en = budget.explore_edits
        k = rng.integers(1, 4, size=(num_groups, en)).ravel()
        efields = _distinct_fields(num_fields, num_groups * en, rng, 3)
        for slot in (1, 2):
            efields[(k < slot + 1), slot] = _NO_FIELD
        egroups = np.repeat(gidx, en)
        values = np.full((num_groups * en, 3), _NO_FIELD, dtype=np.int32)
        for slot in range(3):
            live = efields[:, slot] >= 0
            if not np.any(live):
                continue
            fj = efields[live, slot]
            values[live, slot] = _alt_codes(
                codes_g[egroups[live], fj], domain_sizes[fj], rng.random(int(live.sum()))
            )
        _push(egroups, efields, values)

    group = np.concatenate(pieces_g)
    fields = np.concatenate(pieces_f)
    values = np.concatenate(pieces_v)
    return _normalize_menu(group, fields, values, codes_g, num_groups)


def _normalize_menu(
    group: NDArray[np.int64],
    fields: NDArray[np.int32],
    values: NDArray[np.int32],
    codes_g: NDArray[np.int32],
    num_groups: int,
) -> BatchMenu:
    """规范化菜单，清掉无效槽位，丢落回源路径，组内去重合并保首现顺序。

    与行级规范化语义一致，替代值等于当前值的槽是无效变化直接清空，
    全部槽无效等价于落回源，同组相同差分的路径质量相加，
    首现顺序由生成顺序的稳定排序恢复，供对拍时逐段对齐。
    """
    # 清无效槽，值码为负或与当前值相同都不构成变化
    for slot in range(3):
        live = fields[:, slot] >= 0
        idx = np.flatnonzero(live)
        same = codes_g[group[idx], fields[idx, slot]] == values[idx, slot]
        bad = idx[(values[idx, slot] < 0) | same]
        fields[bad, slot] = _NO_FIELD
        values[bad, slot] = _NO_FIELD
    # 槽内按字段升序排列，无效槽排最后，打包排序一次完成
    packed = np.where(
        fields < 0,
        np.iinfo(np.int64).max,
        fields.astype(np.int64) << 32 | values.astype(np.int64),
    )
    packed.sort(axis=1)
    alive = packed[:, 0] != np.iinfo(np.int64).max
    group = group[alive]
    packed = packed[alive]
    # 组内去重合并，lexsort 找重复，首现位置恢复生成顺序
    order = np.lexsort((packed[:, 2], packed[:, 1], packed[:, 0], group))
    sg = group[order]
    sp = packed[order]
    new_key = np.ones(len(order), dtype=np.bool_)
    if len(order) > 1:
        new_key[1:] = (sg[1:] != sg[:-1]) | np.any(sp[1:] != sp[:-1], axis=1)
    key_ids = np.cumsum(new_key) - 1
    mass = np.bincount(key_ids).astype(np.float64)
    # lexsort 稳定，同键内保持生成顺序，键首成员就是首现位置
    firsts = order[new_key]
    # 输出必须组内连续才能按段边界切组，组间按组号排组内保持首现顺序
    grp_keys = sg[new_key]
    keep = np.lexsort((firsts, grp_keys))
    out_packed = sp[new_key][keep]
    out_group = sg[new_key][keep]
    out_mass = mass[keep]
    dead = out_packed == np.iinfo(np.int64).max
    out_fields = np.where(dead, _NO_FIELD, (out_packed >> 32).astype(np.int32))
    out_values = np.where(dead, _NO_FIELD, (out_packed & 0xFFFFFFFF).astype(np.int32))
    offsets = np.zeros(num_groups + 1, dtype=np.int64)
    np.add.at(offsets, out_group + 1, 1)
    offsets = np.cumsum(offsets)
    return BatchMenu(offsets, out_group, out_fields, out_values, out_mass)


def menu_to_supports(
    menu: BatchMenu,
    grouped_unique_ids: NDArray[np.int64],
    registry: StateRegistry,
) -> list[BlockSupport]:
    """把批量菜单展开成组级块支持，逐路径注册，仅供对拍测试使用。"""
    schema = registry.schema
    supports = []
    for g in range(len(grouped_unique_ids)):
        lo, hi = int(menu.offsets[g]), int(menu.offsets[g + 1])
        base_id = int(grouped_unique_ids[g])
        base = registry.state_tuple(base_id)
        outcomes = []
        mobility = []
        for p in range(lo, hi):
            edited = list(base)
            for slot in range(3):
                j = int(menu.fields[p, slot])
                if j < 0:
                    break
                edited[j] = schema.domains[j][int(menu.values[p, slot])]
            outcomes.append((registry.register_edit(base_id, tuple(edited)),))
            mobility.append(float(menu.mass[p]))
        supports.append(BlockSupport((g,), tuple(outcomes), tuple(mobility)))
    return supports
