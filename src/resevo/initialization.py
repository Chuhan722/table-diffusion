"""起点表初始化，从全二阶考卷答案边缘化出一阶比例并按比例抽样。

一阶不出题是考卷设计决策，初始化需要的一阶信息从二阶答案白拿，
对每个字段找一个配对字段，把二阶格子答案按本字段原子求和就是一阶计数，
覆盖校验要求求和恰等于表行数，配对字段任选结果精确相同。
分箱原子内部的取值分布考卷不提供，箱内均匀抽样，最简占位。
"""
from __future__ import annotations

import numpy as np

from .dataset import QuerySpec, TableSchema


def _atom_signature(cond: dict) -> tuple:
    """单字段条件的语义签名，同一原子不同写法归一。"""
    if cond["operator"] == "==":
        return ("==", str(cond["value"]))
    if cond["operator"] == "between":
        return ("between", float(cond["lower"]), float(cond["upper"]))
    if cond["operator"] == ">=":
        return (">=", float(cond["value"]))
    raise ValueError(f"一阶边缘化不支持算子 {cond['operator']}")


def _atom_values(cond: dict, domain: tuple[str, ...]) -> tuple[str, ...]:
    """原子条件覆盖的域值列表，分箱按数值闭区间筛。"""
    if cond["operator"] == "==":
        v = str(cond["value"])
        return (v,) if v in domain else ()
    if cond["operator"] == "between":
        lo, hi = float(cond["lower"]), float(cond["upper"])
        return tuple(v for v in domain if lo <= float(v) <= hi)
    if cond["operator"] == ">=":
        t = float(cond["value"])
        return tuple(v for v in domain if float(v) >= t)
    raise ValueError(f"一阶边缘化不支持算子 {cond['operator']}")


def _collect_pair_cells(
    specs: list[QuerySpec], schema: TableSchema,
) -> dict[tuple[int, int], dict[tuple, tuple[dict, dict, float]]]:
    """按字段对分桶收集全部两单字段条件查询的格子答案。

    只看恰好两个单字段条件的查询，半空间跳过，
    键是排好序的字段下标对，值是原子签名对到条件与答案的映射。
    """
    pair_cells: dict[tuple[int, int], dict[tuple, tuple[dict, dict, float]]] = {}
    for spec in specs:
        if len(spec.conditions) != 2:
            continue
        if any(c["operator"] == "halfspace" for c in spec.conditions):
            continue
        ca, cb = spec.conditions
        ja = schema.field_index(ca["attribute"])
        jb = schema.field_index(cb["attribute"])
        if ja == jb:
            continue
        if ja > jb:
            ja, jb, ca, cb = jb, ja, cb, ca
        key = (_atom_signature(ca), _atom_signature(cb))
        pair_cells.setdefault((ja, jb), {})[key] = (ca, cb, float(spec.result))
    return pair_cells


def derive_first_order(
    specs: list[QuerySpec], schema: TableSchema, total_rows: int,
    tol_rows: float = 0.0,
) -> list[list[tuple[dict, float]]]:
    """从二阶等值与分箱查询答案边缘化出每字段的一阶原子计数。

    按字段对分桶，对字段 A 取包含 A 的第一个覆盖完整的字段对，
    固定 A 侧原子对 B 侧全部原子求和得到 A 的原子计数，
    覆盖校验，全部原子计数之和必须恰等于表行数，否则拒绝，
    tol_rows 是相对容差，噪声考卷答案带噪总和不再精确等于行数，
    调用方显式放宽，默认零保持零噪声路径精确校验。
    """
    pair_cells = _collect_pair_cells(specs, schema)

    marginals: list[list[tuple[dict, float]]] = []
    for j in range(schema.num_fields):
        found = None
        for (ja, jb), cells in sorted(pair_cells.items()):
            if j not in (ja, jb):
                continue
            side = 0 if j == ja else 1
            counts: dict[tuple, tuple[dict, float]] = {}
            for (sa, sb), (ca, cb, r) in cells.items():
                sig = sa if side == 0 else sb
                cond = ca if side == 0 else cb
                prev = counts.get(sig)
                counts[sig] = (cond, (prev[1] if prev else 0.0) + r)
            total = sum(v for _, v in counts.values())
            if abs(total - total_rows) <= tol_rows * total_rows:
                found = sorted(counts.values(), key=lambda t: _atom_signature(t[0]))
                break
        if found is None:
            raise ValueError(
                f"字段 {schema.fields[j]} 找不到覆盖完整的二阶配对，无法边缘化一阶"
            )
        marginals.append(found)
    return marginals


def derive_first_order_avg(
    specs: list[QuerySpec], schema: TableSchema, total_rows: int,
) -> list[list[tuple[dict, float]]]:
    """全表逆方差加权平均边缘化一阶，替代贪心选首张过关表。

    对字段 A，每个配对字段 B 的联表按 A 侧原子求和各得一份 A 的计数，
    每份的噪声方差与 B 侧原子数成正比，按一比伙伴原子数加权平均，
    即独立高斯异方差下的最大似然融合，全二值考卷自动退化为等权，
    结构校验取全部覆盖表 A 侧原子集合的并集作参照，
    集合不等于并集的缺格表跳过不进平均，全部被跳过才拒绝，
    不做行数总和校验，总和偏差交给抽样归一与可选清洗消化，
    纯后处理零隐私预算，total_rows 仅作接口对齐不参与计算。
    """
    pair_cells = _collect_pair_cells(specs, schema)
    marginals: list[list[tuple[dict, float]]] = []
    for j in range(schema.num_fields):
        tables: list[tuple[frozenset, dict[tuple, tuple[dict, float]], int]] = []
        for (ja, jb), cells in sorted(pair_cells.items()):
            if j not in (ja, jb):
                continue
            side = 0 if j == ja else 1
            counts: dict[tuple, tuple[dict, float]] = {}
            partner_sigs: set = set()
            for (sa, sb), (ca, cb, r) in cells.items():
                sig = sa if side == 0 else sb
                cond = ca if side == 0 else cb
                partner_sigs.add(sb if side == 0 else sa)
                prev = counts.get(sig)
                counts[sig] = (cond, (prev[1] if prev else 0.0) + r)
            tables.append((frozenset(counts), counts, len(partner_sigs)))
        if not tables:
            raise ValueError(
                f"字段 {schema.fields[j]} 找不到任何二阶配对，无法平均边缘化一阶"
            )
        union: frozenset = frozenset().union(*(t[0] for t in tables))
        acc = {sig: 0.0 for sig in union}
        conds: dict[tuple, dict] = {}
        weight_sum = 0.0
        for sigs, counts, partner_n in tables:
            if sigs != union:
                continue
            w = 1.0 / max(1, partner_n)
            for sig, (cond, v) in counts.items():
                acc[sig] += w * v
                conds.setdefault(sig, cond)
            weight_sum += w
        if weight_sum <= 0.0:
            raise ValueError(
                f"字段 {schema.fields[j]} 无覆盖完整的二阶配对，缺格表全部被跳过"
            )
        merged = [(conds[sig], acc[sig] / weight_sum) for sig in union]
        merged.sort(key=lambda t: _atom_signature(t[0]))
        marginals.append(merged)
    return marginals


def _project_simplex(v: np.ndarray, total: float) -> np.ndarray:
    """欧氏投影到总和 total 的非负单纯形，Duchi 排序法，输入可含负值。"""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    idx = np.arange(1, len(u) + 1)
    cond = u - cssv / idx > 0
    rho = idx[cond][-1]
    tau = cssv[rho - 1] / rho
    return np.maximum(v - tau, 0.0)


def clean_first_order(
    marginals: list[list[tuple[dict, float]]], total_rows: int,
) -> list[list[tuple[dict, float]]]:
    """噪声一阶计数清洗，逐字段投影到总和恰等行数的非负单纯形。

    截负加等额摊归一，等价于欧氏投影，高斯噪声每格同方差所以等额摊对口，
    只作用于初始化用的一阶比例，考卷答案与评分完全不动，零隐私预算。
    """
    cleaned = []
    for atoms in marginals:
        counts = np.array([c for _, c in atoms], dtype=np.float64)
        fixed = _project_simplex(counts, float(total_rows))
        cleaned.append([(cond, float(v)) for (cond, _), v in zip(atoms, fixed)])
    return cleaned


def sample_initial_rows(
    marginals: list[list[tuple[dict, int]]],
    schema: TableSchema,
    num_rows: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """按一阶原子比例逐字段独立抽样出起点表。

    原子按计数比例抽，分箱原子内部对覆盖的域值均匀抽，最简占位，
    字段之间独立，关联结构留给演化去修。
    """
    columns = []
    for j, atoms in enumerate(marginals):
        weights = np.array([c for _, c in atoms], dtype=np.float64)
        probs = weights / weights.sum()
        picks = rng.choice(len(atoms), size=num_rows, p=probs)
        domain = schema.domains[j]
        value_pool = [_atom_values(cond, domain) for cond, _ in atoms]
        col = []
        for a in picks:
            pool = value_pool[int(a)]
            if not pool:
                raise ValueError("原子在域内无覆盖值，考卷与表不一致")
            col.append(pool[int(rng.integers(len(pool)))])
        columns.append(col)
    return [tuple(columns[j][i] for j in range(schema.num_fields)) for i in range(num_rows)]
