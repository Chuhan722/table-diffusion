"""
按 1-way 边缘确定性初始化（做法 B）

## 这个模块做什么

给扩散演化的初始表 S_0 一个更好的起点：不再纯随机，而是让每个属性自己的
分布一开始就贴合 1-way 边缘测量。属性间的联合关系交给后续 3-way 演化去修。

设计详见 docs/设计/初始化设计_1way边缘确定性初始化.md。

## 三个职责

1. derive_bins_from_queries(queries, schema)
   —— 纯函数，不读数据。从查询里作用在数值属性上的条件提取"切点"，造出分箱。
      类别属性每个合法值天然是一个箱。这是"workload 诱导分箱"：箱由查询决定，
      正好等于查询能区分的最粗粒度。

2. load_marginals(path)
   —— 读 build_marginals.py 离线产出的 init_marginals.json（含箱定义 + 计数）。

3. init_from_marginals(n_records, schema, marginals, rng)
   —— 运行时，不读源数据。按每属性的边缘计数确定性填配额 + 箱内均摊
      + 每列独立打乱，生成 S_0。

## 离线 / 运行时的分界（守铁律 6）

"数每箱人数"是离线预计算的 DP 测量接口（现在噪声=0），放在 scripts/build_marginals.py，
那里会读源数据。运行时只读 marginals 文件，从不碰源数据。

## 算子白名单与回退（沿用 vectorized_eval 的既定模式）

造箱只对白名单算子精确推切点。白名单外算子 → 跳过该查询（不贡献切点）+ 打印提醒，
不报错：后果只是"这条查询的 1-way 信息没烤进初始化"，起点差一点，评价仍正确。
"""
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from table_diffevo.schema import Schema


# 造箱支持的数值算子白名单。白名单外算子 → 跳过该查询（不贡献切点）+ 提醒。
# 与 vectorized_eval.VECTORIZED_OPS 各自独立：那个管"评价加速"，这个管"造箱切点"。
BIN_OPS = {">=", ">", "<=", "<", "between", "=="}


def _cut_points_from_condition(cond: Dict[str, Any]) -> List[int]:
    """
    从单个查询条件提取该数值属性上的切点（整数域、闭箱语义）。

    切点定义为"半开区间 [prev, cut) 的右端点"，即 cut 处新开一个箱。
    端点语义严格对齐 queries.eval_condition：

    | 算子              | 切点        | 理由                         |
    |-------------------|-------------|------------------------------|
    | >= t              | t           | [.., t-1] | [t, ..]          |
    | > t               | t+1         | 等价 >= t+1                  |
    | <= t              | t+1         | [.., t] | [t+1, ..]          |
    | < t               | t           | [.., t-1] | [t, ..]          |
    | between [lo, hi]  | lo, hi+1    | 闭区间，hi 含在箱内          |
    | == v              | v, v+1      | 退化箱 [v, v] 单独成箱       |

    白名单外算子返回 None（调用方据此跳过该查询）。
    """
    op = cond["operator"]
    if op == ">=":
        return [int(cond["value"])]
    if op == ">":
        return [int(cond["value"]) + 1]
    if op == "<=":
        return [int(cond["value"]) + 1]
    if op == "<":
        return [int(cond["value"])]
    if op == "between":
        return [int(cond["lower"]), int(cond["upper"]) + 1]
    if op == "==":
        v = int(cond["value"])
        return [v, v + 1]
    return None  # 白名单外


def derive_bins_from_queries(
    queries: List[Dict[str, Any]],
    schema: Schema,
    verbose: bool = True,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    从查询造出每个数值属性的分箱（workload 诱导分箱）。

    对每个数值属性，扫描全部查询里作用在它身上的条件，收集切点，并入 schema
    域端点，排序去重，切成一组整数闭箱 [lo, hi]（覆盖整个域，箱之间无缝无叠）。

    Parameters
    ----------
    queries : List[Dict]
        查询定义列表。每条含 conditions（[{attribute, operator, ...}, ...]）。
    schema : Schema
        属性 schema，提供数值属性的域端点 [dmin, dmax]。
    verbose : bool, default True
        遇到白名单外算子、或数值属性无查询覆盖时是否打印提醒。

    Returns
    -------
    Dict[str, List[Tuple[int, int]]]
        {数值属性名: [(lo, hi), ...]}，每个箱是整数闭区间。
        - 只包含"有查询覆盖且切点有效"的数值属性。
        - 无任何白名单内条件覆盖的数值属性不在返回值里（运行时回退随机）。
        - 类别属性不在此函数职责内（每合法值一箱，由计数环节直接处理）。

    Notes
    -----
    **永远铺满整个域**：总把 schema 域端点 [dmin, dmax+1] 并入切点，因此无论
    查询给多少切点，箱都铺满 [dmin, dmax]、计数之和恒为 N。

    **白名单外算子**：跳过该条件（不贡献切点）+ 提醒；不报错。
    """
    # 收集每个数值属性名，及其域端点
    numeric_domains: Dict[str, Tuple[int, int]] = {}
    for attr in schema.get_numeric_blocks():
        low, high = attr.range
        numeric_domains[attr.name] = (int(low), int(high))

    # 每个数值属性的切点集合（先并入域端点：dmin 和 dmax+1）
    cuts: Dict[str, set] = {
        name: {dmin, dmax + 1} for name, (dmin, dmax) in numeric_domains.items()
    }
    covered: set = set()          # 至少被一条白名单内条件覆盖的数值属性
    skipped_ops: set = set()      # 遇到的白名单外算子（用于提醒）

    for q in queries:
        for cond in q["conditions"]:
            attr = cond["attribute"]
            if attr not in numeric_domains:
                continue  # 非数值属性不在此造箱
            pts = _cut_points_from_condition(cond)
            if pts is None:
                skipped_ops.add(cond["operator"])
                continue
            dmin, dmax = numeric_domains[attr]
            for p in pts:
                # 切点裁剪到域内 [dmin, dmax+1]（域外切点无意义）
                p = max(dmin, min(p, dmax + 1))
                cuts[attr].add(p)
            covered.add(attr)

    if verbose and skipped_ops:
        print(
            f"提示：造箱遇到白名单外算子 {sorted(skipped_ops)}，已跳过这些条件"
            f"（该查询的 1-way 信息不烤进初始化，评价不受影响）。"
        )

    bins: Dict[str, List[Tuple[int, int]]] = {}
    for name, (dmin, dmax) in numeric_domains.items():
        if name not in covered:
            if verbose:
                print(
                    f"提示：数值属性 {name!r} 无查询覆盖，初始化时该列回退随机。"
                )
            continue
        sorted_cuts = sorted(cuts[name])
        # 相邻切点围出半开区间 [c_k, c_{k+1})，转成整数闭箱 [c_k, c_{k+1}-1]
        attr_bins: List[Tuple[int, int]] = []
        for a, b in zip(sorted_cuts[:-1], sorted_cuts[1:]):
            lo, hi = a, b - 1
            if lo <= hi:  # 跳过空箱（相邻切点相等时不会发生，去重后 b>a）
                attr_bins.append((lo, hi))
        bins[name] = attr_bins

    return bins


def load_marginals(path: str) -> Dict[str, Any]:
    """
    加载 scripts/build_marginals.py 离线产出的 init_marginals.json。

    文件格式（见 build_marginals.py 的写出逻辑）：
        {
          "dataset": str,
          "n_records": int,
          "queries_source": str,       # 造箱依据的查询文件，便于审计/复现
          "attributes": {
            "<numeric attr>":     {"type": "numeric",
                                   "bins": [[lo, hi], ...],
                                   "counts": [c0, c1, ...]},
            "<categorical attr>": {"type": "categorical",
                                   "values": [v0, v1, ...],
                                   "counts": [c0, c1, ...]},
            ...
          }
        }

    运行时只读此文件，从不碰源数据（守铁律 6）。

    Returns
    -------
    Dict[str, Any]
        解析后的 dict，结构同上。
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _largest_remainder_quota(counts: np.ndarray, n_records: int) -> np.ndarray:
    """
    把（可能非整数、可能不和为 N 的）计数调成正好和为 n_records 的整数配额。

    用最大余数法（Hamilton 法）：先各自向下取整，再把剩余名额按小数部分从大到小
    逐个 +1，直到凑满 n_records。

    无噪声阶段计数本就是和为 N 的整数，此函数返回原值（floor 后余 0）。
    保留此步是为将来 DP 加噪：那时计数会是带小数、可能不和为 N 的测量值。

    Parameters
    ----------
    counts : np.ndarray, shape (K,)
        各箱/各值的目标计数（float，可含小数，可为负——负数先截到 0）。
    n_records : int
        目标总记录数 N。

    Returns
    -------
    np.ndarray, shape (K,), int
        非负整数配额，和恰为 n_records。
    """
    counts = np.asarray(counts, dtype=float)
    counts = np.clip(counts, 0.0, None)  # 负计数（加噪可能出现）截到 0
    total = counts.sum()
    if total <= 0:
        # 全 0（无信息）：均匀摊到各箱，保证和为 N
        K = len(counts)
        base = n_records // K
        quota = np.full(K, base, dtype=int)
        quota[: n_records - base * K] += 1
        return quota

    scaled = counts * (n_records / total)  # 归一化到总量 N
    floor = np.floor(scaled).astype(int)
    remainder = n_records - int(floor.sum())
    if remainder > 0:
        # 按小数部分从大到小，给前 remainder 个 +1
        frac = scaled - floor
        order = np.argsort(-frac)
        floor[order[:remainder]] += 1
    return floor


def _fill_numeric_column(
    bins: List[Tuple[int, int]], quota: np.ndarray
) -> np.ndarray:
    """
    数值列：把每个箱的配额确定性等距铺在该箱的整数值上（箱内均摊，无随机）。

    因箱边界对齐查询切点，箱内怎么铺都不改查询计数，只改善分布形状。

    Parameters
    ----------
    bins : List[(lo, hi)]
        整数闭箱列表。
    quota : np.ndarray, shape (len(bins),), int
        每个箱要填的记录数。

    Returns
    -------
    np.ndarray, shape (sum(quota),), int
        未打乱的数值序列（块状，调用方再统一打乱）。
    """
    parts = []
    for (lo, hi), cnt in zip(bins, quota):
        if cnt <= 0:
            continue
        width = hi - lo + 1
        # 等距均摊：第 i 条落在 lo + (i % width)，确定性、无随机
        vals = lo + (np.arange(cnt) % width)
        parts.append(vals)
    if not parts:
        return np.empty(0, dtype=int)
    return np.concatenate(parts).astype(int)


def init_from_marginals(
    n_records: int,
    schema: Schema,
    marginals: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    按 1-way 边缘确定性生成初始表 S_0（做法 B）。

    每列独立：按边缘计数确定性填配额 → 数值列箱内均摊 → 每列独立随机打乱。
    没有边缘信息的属性（marginals 里没有、或数值属性无查询覆盖）回退到随机填充。

    Parameters
    ----------
    n_records : int
        记录条数 N。
    schema : Schema
        属性 schema，决定列结构、列顺序、类别合法值、数值域。
    marginals : Dict
        load_marginals 的返回值（含 attributes 字段）。
    rng : np.random.Generator or None
        随机数生成器。用于"每列独立打乱"和"回退列的随机填充"。
        固定种子 → 完全可复现（铁律 5）。

    Returns
    -------
    pd.DataFrame, shape (n_records, n_attributes)
        初始合成表，列顺序与 schema.attribute_names() 一致。

    Notes
    -----
    **不读源数据**：只用 schema（公开）+ marginals 文件（离线测量）。

    **每列独立打乱的必要性**：填完是块状 [v0,v0,...,v1,v1,...]，若不打乱，各列
    同位置会对齐、凭空造出属性间关联。用各列各自的置换打散，保证列间联合随机。
    """
    if n_records <= 0:
        raise ValueError(f"n_records 必须 > 0，得到 {n_records}")
    if rng is None:
        rng = np.random.default_rng()

    attr_marg = marginals.get("attributes", {})
    columns: Dict[str, np.ndarray] = {}

    for attr in schema.attributes:
        name = attr.name
        spec = attr_marg.get(name)

        if spec is None:
            # 无边缘信息 → 回退随机（口径与 generator.init_synthetic_table 一致）
            columns[name] = _random_column(attr, n_records, rng)
            continue

        counts = np.asarray(spec["counts"], dtype=float)
        quota = _largest_remainder_quota(counts, n_records)

        if spec["type"] == "numeric":
            bins = [tuple(b) for b in spec["bins"]]
            col = _fill_numeric_column(bins, quota)
        else:
            # 类别：每个值按配额重复
            values = np.asarray(spec["values"])
            col = np.repeat(values, quota)

        # 每列独立打乱（打散块状结构，避免列间人为关联）
        rng.shuffle(col)
        columns[name] = col

    return pd.DataFrame(columns, columns=schema.attribute_names())


def _random_column(attr, n_records: int, rng: np.random.Generator) -> np.ndarray:
    """
    回退随机填充单列（口径与 generator.init_synthetic_table 完全一致）。

    数值列：域内均匀整数（含端点）。类别列：合法值集合上均匀抽样。
    """
    if attr.is_numeric():
        low, high = attr.range
        return rng.integers(int(low), int(high) + 1, size=n_records)
    idx = rng.integers(0, len(attr.values), size=n_records)
    values = np.asarray(attr.values)
    return values[idx]
