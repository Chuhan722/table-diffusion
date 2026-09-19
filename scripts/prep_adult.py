"""adult 数据预处理，UCI 原始文件到纯离散整数码表。

口径拍板，
一，15 字段全保留，train 切分 32561 行，缺失问号当独立类别不丢行，
二，education-num 本就 1 到 16 整数直接码即值，
三，age，fnlwgt，hours-per-week 等频 16 桶，分位边界去重防空桶，
四，capital-gain 与 capital-loss 零值扎堆，零单独一桶，非零部分等频 8 桶，
五，类别字段去空格后按字典序因子化，
六，输出 attr_1 到 attr_15 表头的整数码表，两家基线吃同一张表码即值。
用法，./.venv/bin/python scripts/prep_adult.py 原始adult.data路径
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

RAW_FIELDS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
NUMERIC_EQ16 = {"age", "fnlwgt", "hours-per-week"}
ZERO_HEAVY = {"capital-gain", "capital-loss"}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "adult"


def bin_equal_freq(values: np.ndarray, bins: int) -> np.ndarray:
    """等频分箱，分位边界去重，返回桶号。"""
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1], method="higher")
    edges = np.unique(qs)
    return np.searchsorted(edges, values, side="left").astype(np.int64)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法，prep_adult.py 原始adult.data路径")
    raw_path = Path(sys.argv[1])
    rows = []
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(RAW_FIELDS):
                raise AssertionError(f"字段数 {len(parts)} 不等于 {len(RAW_FIELDS)}")
            rows.append(parts)
    n = len(rows)
    print(f"原始行数 {n}")

    codes = np.zeros((n, len(RAW_FIELDS)), dtype=np.int64)
    for j, name in enumerate(RAW_FIELDS):
        col = [r[j] for r in rows]
        if name in NUMERIC_EQ16:
            vals = np.array([int(v) for v in col], dtype=np.int64)
            codes[:, j] = bin_equal_freq(vals, 16)
            kind = "等频16桶"
        elif name in ZERO_HEAVY:
            vals = np.array([int(v) for v in col], dtype=np.int64)
            nz = vals > 0
            out = np.zeros(n, dtype=np.int64)
            if nz.any():
                out[nz] = 1 + bin_equal_freq(vals[nz], 8)
            codes[:, j] = out
            kind = "零桶加非零等频8桶"
        elif name == "education-num":
            vals = np.array([int(v) for v in col], dtype=np.int64)
            uniq = np.unique(vals)
            codes[:, j] = np.searchsorted(uniq, vals)
            kind = "整数码即值"
        else:
            uniq = sorted(set(col))
            lut = {v: k for k, v in enumerate(uniq)}
            codes[:, j] = [lut[v] for v in col]
            kind = "字典序因子化"
        width = int(codes[:, j].max()) + 1
        print(f"attr_{j + 1} {name} {kind} 域宽 {width}")

    widths = codes.max(axis=0) + 1
    total = int(sum(
        int(widths[a]) * int(widths[b])
        for a in range(len(RAW_FIELDS)) for b in range(a + 1, len(RAW_FIELDS))
    ))
    print(f"全二阶格子总条数 {total}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "adult.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"attr_{j + 1}" for j in range(len(RAW_FIELDS))])
        w.writerows(codes.tolist())
    print(f"写入 {out_csv}，{n} 行 {len(RAW_FIELDS)} 字段")


if __name__ == "__main__":
    main()
