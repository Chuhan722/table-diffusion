"""ACSmobility-CA 预处理，folktables 原始混合域到纯离散整数码表。

口径拍板（对齐 GSD 论文 dp-data 管道 + 本仓库 adult 先例），
一，数据源 folktables ACS 2018 1-Year person 加州，ACSMobility 任务
    21 特征加 MIG 标签，与 GSD 论文（ICML 2023）表 1 同任务同州同年，
二，行数 80329 与论文 64263 不一致系 folktables 版本演化（旧版
    df_to_numpy 丢弃含缺失行，今版缺失填充后保留），两家基线吃
    同一张码表，口径公平不受影响，笔记留案，
三，ESP 单值列删除（dp-data 同规矩：单值类别列不进域），
四，AGEP 任务过滤后只剩 16 个值（19-34），整数码即值，
五，WKHP 与 JWMNP 零值扎堆（缺失填充与不适用人群），零单独一桶，
    非零部分等频 15 桶，与 adult 的 capital-gain 同律，
六，PINCP 含负值重尾，整体等频 16 桶，分位边界去重防空桶，
七，类别字段按数值升序因子化，
八，输出 attr_1 到 attr_21 表头整数码表 data/acsmob/acsmob.csv。
用法，./.venv/bin/python scripts/prep_acsmob.py
（需先有 /tmp/acs_raw 下的 2018 加州原始数据，folktables 自动下载）
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "acsmob"
RAW_ROOT = "/tmp/acs_raw"

# 4 数值列 + 17 类别列（含标签），ESP 单值删除
NUMERIC_EQ16 = {"PINCP"}
ZERO_HEAVY = {"WKHP", "JWMNP"}
CODE_AS_IS = {"AGEP"}
DROP_SINGLE = {"ESP"}


def bin_equal_freq(values: np.ndarray, bins: int) -> np.ndarray:
    """等频分箱，分位边界去重，返回桶号。"""
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1], method="higher")
    edges = np.unique(qs)
    return np.searchsorted(edges, values, side="left").astype(np.int64)


def main() -> None:
    from folktables import ACSDataSource, ACSMobility

    ds = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=RAW_ROOT
    )
    data = ds.get_data(states=["CA"], download=False)
    feats, target, _ = ACSMobility.df_to_numpy(data)
    names = list(ACSMobility.features) + ["MIG"]
    mat = np.column_stack([feats, target.astype(np.float64)])
    n = mat.shape[0]
    print(f"原始行数 {n}，特征列 {len(names)}")

    keep, kept_names = [], []
    for j, name in enumerate(names):
        col = mat[:, j]
        if name in DROP_SINGLE:
            assert len(np.unique(col)) == 1, f"{name} 不再是单值列，口径需复查"
            print(f"删除单值列 {name}")
            continue
        if name in CODE_AS_IS:
            vals = col.astype(np.int64)
            uniq = np.unique(vals)
            out = np.searchsorted(uniq, vals)
            kind = "整数码即值"
        elif name in NUMERIC_EQ16:
            out = bin_equal_freq(col, 16)
            kind = "等频16桶"
        elif name in ZERO_HEAVY:
            vals = col.astype(np.int64)
            nz = vals > 0
            out = np.zeros(n, dtype=np.int64)
            if nz.any():
                out[nz] = 1 + bin_equal_freq(col[nz], 15)
            kind = "零桶加非零等频15桶"
        else:
            vals = col.astype(np.int64)
            uniq = np.unique(vals)
            out = np.searchsorted(uniq, vals)
            kind = "升序因子化"
        keep.append(out.astype(np.int64))
        kept_names.append(name)
        print(f"attr_{len(keep)} {name} {kind} 域宽 {int(out.max()) + 1}")

    codes = np.column_stack(keep)
    widths = codes.max(axis=0) + 1
    k = codes.shape[1]
    total = int(
        sum(
            int(widths[a]) * int(widths[b])
            for a in range(k) for b in range(a + 1, k)
        )
    )
    print(f"保留 {k} 列，全二阶格子总条数 {total}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "acsmob.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"attr_{j + 1}" for j in range(k)])
        w.writerows(codes.tolist())
    print(f"写入 {out_csv}，{n} 行 {k} 字段")
    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as fh:
        fh.write(
            "# acsmob（ACSmobility-CA 2018）\n\n"
            "folktables ACS 2018 1-Year 加州 ACSMobility 任务，GSD 论文"
            "（ICML 2023）主场基准之一。20 特征加 MIG 标签共 21 列整数码，"
            "口径见 scripts/prep_acsmob.py 档头。行数 80329 与论文 64263 "
            "之差系 folktables 版本缺失值处理演化，两家基线同表公平。\n\n"
            "原始列名对照：" + "，".join(
                f"attr_{j + 1}={nm}" for j, nm in enumerate(kept_names)
            ) + "\n"
        )
    print("README 已写")


if __name__ == "__main__":
    main()
