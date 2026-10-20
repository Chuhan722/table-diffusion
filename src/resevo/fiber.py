"""纤维洗牌平均：引擎内置的"没测的对"目标推断（工作笔记第五十八、五十九步）。

原则：满足全部测过统计量的表构成一个纤维，纤维的平均≈最大熵分布；引擎沿下降路径停在纤维里
一张带路径指纹的少数派表上。做法：对属性 a，把行按 a 的全部测过邻居（考卷里与 a 同题的属性）
取值分组，组内随机重排 a 列——每个测过集合的边际逐格不变，变的只有 a 与非邻居（没测的对）。
一条链连洗若干趟，从第 2 趟起对每对没测字段的二阶边际取平均，平均值当伪测量补进考卷，
引擎从收敛表热启动再收敛落表。只用自己的收敛表 + 考卷里"哪些属性组合测过"，零对手产物、零预算。

与 scripts/fiber_shuffle.py / fiber_targets.py 的数学逐步相同（同种子同表结果逐格一致），
这里以字段编码（schema.domains 的下标）工作，与取值标签无关。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .dataset import QuerySpec, TableSchema

PSEUDO_PREFIX = "P2_"


def measured_sets(specs: list[QuerySpec], schema: TableSchema, ignore_prefix: str = PSEUDO_PREFIX) -> list[tuple[int, ...]]:
    """考卷里测过的字段集合（去重、按字段序号排序），只认等值格子题；题号以 ignore_prefix 开头的伪目标不算。"""
    out: set[tuple[int, ...]] = set()
    for q in specs:
        if ignore_prefix and q.query_id.startswith(ignore_prefix):
            continue
        for c in q.conditions:
            if c["operator"] != "==":
                raise ValueError(f"考卷含非等值题 {q.query_id}（{c['operator']}），纤维洗牌只支持边际格子题")
        out.add(tuple(sorted(schema.field_index(c["attribute"]) for c in q.conditions)))
    return sorted(out)


def neighbors(num_fields: int, sets: list[tuple[int, ...]]) -> list[list[int]]:
    """每个字段的测过邻居：与它出现在同一测过集合里的其他字段。"""
    nb: list[set[int]] = [set() for _ in range(num_fields)]
    for s in sets:
        for a in s:
            nb[a].update(b for b in s if b != a)
    return [sorted(v) for v in nb]


def group_ids(X: NDArray[np.int64], cols: list[int]) -> NDArray[np.int64]:
    if not cols:
        return np.zeros(len(X), dtype=np.int64)
    _, inv = np.unique(X[:, cols], axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def shuffle_column(X: NDArray[np.int64], a: int, gid: NDArray[np.int64], rng: np.random.Generator) -> None:
    """组内随机重排第 a 列：按（组号，随机数）排序取值，按（组号，原序）排序落位。"""
    n = len(X)
    src = np.lexsort((rng.random(n), gid))
    dst = np.lexsort((np.arange(n), gid))
    X[dst, a] = X[src, a].copy()


def marg(X: NDArray[np.int64], cols: tuple[int, ...], doms: list[int]) -> NDArray[np.float64]:
    key = X[:, cols[0]].astype(np.int64)
    size = doms[cols[0]]
    for c in cols[1:]:
        key = key * doms[c] + X[:, c]
        size *= doms[c]
    return np.bincount(key, minlength=size).astype(np.float64)


def encode_rows(rows: list[tuple[str, ...]], schema: TableSchema) -> NDArray[np.int64]:
    """字段值元组表 → 值域下标编码矩阵。"""
    maps = [{v: i for i, v in enumerate(d)} for d in schema.domains]
    X = np.empty((len(rows), schema.num_fields), dtype=np.int64)
    for j, m in enumerate(maps):
        X[:, j] = [m[r[j]] for r in rows]
    return X


@dataclass
class FiberResult:
    measured_pairs: list[tuple[int, int]]
    unmeasured_pairs: list[tuple[int, int]]
    averages: dict[tuple[int, int], NDArray[np.float64]]  # 没测的对 → 长 doms[a]*doms[b] 的平均二阶边际
    passes: int
    averaged: int


def fiber_average(
    X: NDArray[np.int64], doms: list[int], sets: list[tuple[int, ...]],
    passes: int, rng: np.random.Generator, start_pass: int = 2,
) -> FiberResult:
    """一条链连洗 passes 趟，从 start_pass 趟起累加没测的对的二阶边际并平均；X 原地被洗。

    结束时自检：每个测过集合的边际须逐格不变，否则抛错。
    """
    if passes < start_pass:
        raise ValueError(f"趟数 {passes} 须不少于起算趟 {start_pass}")
    F = X.shape[1]
    nb = neighbors(F, sets)
    pairs = list(combinations(range(F), 2))
    mset = {s for s in sets if len(s) == 2}
    measured = [p for p in pairs if p in mset]
    unmeas = [p for p in pairs if p not in mset]
    before = {s: marg(X, s, doms) for s in sets}
    acc = {p: np.zeros(doms[p[0]] * doms[p[1]], dtype=np.float64) for p in unmeas}
    n = 0
    for t in range(1, passes + 1):
        for a in rng.permutation(F):
            shuffle_column(X, int(a), group_ids(X, nb[int(a)]), rng)
        if t >= start_pass:
            for p in unmeas:
                acc[p] += marg(X, p, doms)
            n += 1
    for s in sets:
        if not np.array_equal(marg(X, s, doms), before[s]):
            raise RuntimeError(f"纤维洗牌自检失败：测过集合 {s} 的边际变了")
    return FiberResult(measured, unmeas, {p: acc[p] / n for p in unmeas}, passes, n)


def pseudo_sigma(specs: list[QuerySpec], mult: float = 1.0) -> float | None:
    """伪测量 σ：考卷二阶题 σ 的中位数 × mult；考卷没有带 σ 的二阶题则返回 None。"""
    s2 = [q.sigma for q in specs if len(q.conditions) == 2 and q.sigma is not None]
    if not s2:
        return None
    return float(np.median(np.asarray(s2, dtype=np.float64))) * mult


def fiber_specs(result: FiberResult, schema: TableSchema, sigma: float | None, prefix: str = PSEUDO_PREFIX) -> list[QuerySpec]:
    """没测的对的平均边际逐格写成伪测量题（题号 prefix + 五位序号，与 scripts/add_infer_targets.py 同式）。"""
    out: list[QuerySpec] = []
    k = 0
    for a, b in result.unmeasured_pairs:
        m = result.averages[(a, b)].reshape(len(schema.domains[a]), len(schema.domains[b]))
        for ia, va in enumerate(schema.domains[a]):
            for ib, vb in enumerate(schema.domains[b]):
                k += 1
                out.append(QuerySpec(
                    f"{prefix}{k:05d}",
                    (
                        {"attribute": schema.fields[a], "operator": "==", "value": va},
                        {"attribute": schema.fields[b], "operator": "==", "value": vb},
                    ),
                    float(m[ia, ib]), sigma,
                ))
    return out
