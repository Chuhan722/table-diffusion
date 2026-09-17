"""基线共用工具，从考卷重建分箱域与全二阶列联表，与 age 还原。

红线说明，本模块只依赖标准库与 numpy，不 import 旧仓任何代码，
供 run_baseline_pgm.py 与 run_baseline_gsd.py 在各自第三方环境里加载。
喂料协议，两家基线只吃考卷里 548 条全二阶答案，
一阶边缘由二阶精确边缘化推出，不直接读真实表统计量，
真实表只用于两处公开元数据，字段取值域与行数校验。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_CSV = REPO / "data" / "test_300x10" / "test_300x10.csv"
EXAM_JSON = REPO / "data" / "test_300x10" / "measured_698query.json"
EXPECTED_PAIRS = 45
EXPECTED_CELLS = 548


def load_real_domains() -> tuple[list[str], dict[str, list[str]], int]:
    """读真实表的字段顺序与每字段取值域，域视为公开元数据。"""
    with open(DATA_CSV, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    domains = {f: sorted({r[f] for r in rows}) for f in fields}
    return fields, domains, len(rows)


def load_exam_structure():
    """从考卷重建，分箱类别与 45 个字段对的完整二阶列联表。

    返回 fields 字段顺序，cats 每字段类别标签列表，
    bins age 的闭区间列表，pair_tables 字段对到整数计数矩阵，total 行数。
    """
    fields, domains, n_rows = load_real_domains()
    exam = json.loads(EXAM_JSON.read_text(encoding="utf-8"))
    total = int(exam["record_count"])
    if total != n_rows:
        raise RuntimeError(f"考卷行数 {total} 与真实表 {n_rows} 不一致")

    two_way = [q for q in exam["queries"] if q.get("type") == "eq2way"]
    if len(two_way) != EXPECTED_CELLS:
        raise RuntimeError(f"二阶题数漂移 {len(two_way)} != {EXPECTED_CELLS}")

    bins: list[tuple[int, int]] = []
    cat_values: dict[str, set] = {f: set() for f in fields}
    for q in two_way:
        for c in q["conditions"]:
            attr = c["attribute"]
            if c["operator"] == "between":
                if attr != "age":
                    raise RuntimeError(f"意外的分箱字段 {attr}")
                b = (int(c["lower"]), int(c["upper"]))
                if b not in bins:
                    bins.append(b)
            elif c["operator"] == "==":
                cat_values[attr].add(str(c["value"]))
            else:
                raise RuntimeError(f"二阶题出现意外算子 {c['operator']}")
    bins.sort()
    if len(bins) != 5:
        raise RuntimeError(f"age 箱数漂移 {len(bins)} != 5")

    cats: dict[str, list[str]] = {}
    for f in fields:
        if f == "age":
            cats[f] = [f"{lo}_{hi}" for lo, hi in bins]
        else:
            vals = sorted(cat_values[f])
            if vals != domains[f]:
                raise RuntimeError(f"字段 {f} 考卷取值与公开域不一致")
            cats[f] = vals

    def cell_index(cond: dict) -> tuple[str, int]:
        attr = cond["attribute"]
        if cond["operator"] == "between":
            return attr, bins.index((int(cond["lower"]), int(cond["upper"])))
        return attr, cats[attr].index(str(cond["value"]))

    order = {f: i for i, f in enumerate(fields)}
    pair_tables: dict[tuple[str, str], np.ndarray] = {}
    filled: dict[tuple[str, str], np.ndarray] = {}
    for q in two_way:
        (fa, ia), (fb, ib) = (cell_index(c) for c in q["conditions"])
        if order[fa] > order[fb]:
            fa, ia, fb, ib = fb, ib, fa, ia
        key = (fa, fb)
        if key not in pair_tables:
            shape = (len(cats[fa]), len(cats[fb]))
            pair_tables[key] = np.zeros(shape, dtype=np.int64)
            filled[key] = np.zeros(shape, dtype=bool)
        if filled[key][ia, ib]:
            raise RuntimeError(f"字段对 {key} 格 ({ia},{ib}) 重复出题")
        pair_tables[key][ia, ib] = int(q["result"])
        filled[key][ia, ib] = True

    if len(pair_tables) != EXPECTED_PAIRS:
        raise RuntimeError(f"字段对数漂移 {len(pair_tables)} != {EXPECTED_PAIRS}")
    for key, mask in filled.items():
        if not mask.all():
            raise RuntimeError(f"字段对 {key} 列联表不完整")
        if int(pair_tables[key].sum()) != total:
            raise RuntimeError(f"字段对 {key} 计数和不等于 {total}")

    return fields, cats, bins, pair_tables, total


def marginalize_first_order(
    fields: list[str],
    cats: dict[str, list[str]],
    pair_tables: dict[tuple[str, str], np.ndarray],
    total: int,
) -> dict[str, np.ndarray]:
    """从全二阶表边缘化一阶计数，并断言所有字段对给出的一阶完全一致。"""
    margins: dict[str, np.ndarray] = {}
    for (fa, fb), table in pair_tables.items():
        for f, m in ((fa, table.sum(axis=1)), (fb, table.sum(axis=0))):
            if f in margins:
                if not np.array_equal(margins[f], m):
                    raise RuntimeError(f"字段 {f} 各字段对一阶边缘不一致")
            else:
                margins[f] = m
    for f in fields:
        if int(margins[f].sum()) != total:
            raise RuntimeError(f"字段 {f} 一阶计数和不等于 {total}")
        if len(margins[f]) != len(cats[f]):
            raise RuntimeError(f"字段 {f} 一阶长度漂移")
    return margins


def decode_rows(
    codes: dict[str, np.ndarray],
    fields: list[str],
    cats: dict[str, list[str]],
    bins: list[tuple[int, int]],
    age_domain: list[str],
    seed: int,
) -> list[dict[str, str]]:
    """整数码还原为原始格式行，age 在箱内对公开域取值均匀抽，最简占位。"""
    rng = np.random.default_rng(seed)
    bin_values = []
    for lo, hi in bins:
        vals = [v for v in age_domain if lo <= int(v) <= hi]
        if not vals:
            raise RuntimeError(f"age 箱 {lo}_{hi} 在公开域内无取值")
        bin_values.append(vals)
    n = len(next(iter(codes.values())))
    rows = []
    for i in range(n):
        row = {}
        for f in fields:
            c = int(codes[f][i])
            if f == "age":
                row[f] = str(rng.choice(bin_values[c]))
            else:
                row[f] = cats[f][c]
        rows.append(row)
    return rows


def write_table(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def binned_counts(
    rows_codes: dict[str, np.ndarray],
    pair_tables: dict[tuple[str, str], np.ndarray],
    cats: dict[str, list[str]],
) -> tuple[float, float]:
    """合成码表与考卷二阶目标的偏差回执，返回平均与最大绝对差。"""
    diffs = []
    for (fa, fb), target in pair_tables.items():
        ka, kb = len(cats[fa]), len(cats[fb])
        flat = rows_codes[fa].astype(np.int64) * kb + rows_codes[fb].astype(np.int64)
        counts = np.bincount(flat, minlength=ka * kb).reshape(ka, kb)
        diffs.append(np.abs(counts - target).ravel())
    all_d = np.concatenate(diffs)
    return float(all_d.mean()), float(all_d.max())
