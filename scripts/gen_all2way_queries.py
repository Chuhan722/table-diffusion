"""生成 all-2way 完整考卷（GSD 对齐的正式实验主考卷形态，双数据集参数化）。

## 设计依据（2026-09-06 讨论定稿）

AIM (VLDB'22) 与 Private-GSD (ICML'23) 的正式实验考卷均为"k-way 边缘全家族"
（GSD 的 all-2way 主实验为一次性全测形态）。plants 考卷把二维查询从 980 池
中的 460 格（4.9% 覆盖）补成全家族 9384 格（100% 覆盖）；nltcs 考卷为标准
all-2way 480 格（旧 1001 池的 479 double 已信息等效覆盖 120/120 对，本考卷
是其唯一化、标准化形态，供 PGM-on-all2way 重拟合与纯二维池跑统一使用）。

## 构造规则（零随机、纯纸面枚举）

- 属性对：按数据 csv 列序取全部 C(M,2) 个 (i<j) 对；
- 每对纳入完整 2×2 列联表的全部 4 个 cell（完整 marginal 语义，
  与 measured_1000query.json 的"完整成组"口径一致）；
- 共 C(M,2)×4 条 double 查询，无 single、无 triple；
- 目标值 result = 数据集上的精确计数（无噪声内层口径：测量答案视为
  已发布统计量；正式加噪跑只替换这一列）。

## 自检（生成时强制执行，任一失败即拒绝写盘）

1. 每对 4 cell 计数之和 == 总记录数 N；
2. 查询指纹全体唯一（query_fingerprint 语义级判重）；
3. 与 measured_1000query.json 的旧池 double 逐指纹对账：
   凡出现在旧池中的二维格，目标值必须逐位一致（plants 460 / nltcs 479 条）。

## 数据来源

公开密度估计基准系列（Lowd & Davis / Van Haaren & Davis 文献族）。
来源与哈希见 data/<dataset>/README.md。
"""
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from table_diffevo.quality import query_fingerprint

DATASETS = {
    "plants": {
        "data_path": "data/plants/plants.csv",
        "measured_path": "configs/plants/measured_1000query.json",
        "out_path": "configs/plants/all2way_issue53_v1.json",
        "dataset_label": "plants.csv",
        "expected_old_pool_doubles": 460,
        "description": (
            "plants all-2way 完整考卷（GSD 对齐）：按列序枚举全部 "
            "C(69,2)=2346 个属性对，每对完整 2×2 列联表 4 cell 成组，"
            "共 9384 条 double；零随机零筛选；目标值为无噪声精确计数；"
            "与 measured_1000query.json 的 460 条 double 逐指纹对账一致"
        ),
    },
    "nltcs": {
        "data_path": "data/nltcs/nltcs.csv",
        "measured_path": "configs/nltcs/measured_1000query.json",
        "out_path": "configs/nltcs/all2way_issue53_v1.json",
        "dataset_label": "nltcs.csv",
        "expected_old_pool_doubles": 479,
        "description": (
            "nltcs all-2way 完整考卷（GSD 对齐）：按列序枚举全部 "
            "C(16,2)=120 个属性对，每对完整 2×2 列联表 4 cell 成组，"
            "共 480 条 double；零随机零筛选；目标值为无噪声精确计数；"
            "与 measured_1000query.json 的 479 条 double 逐指纹对账一致"
        ),
    },
}


def build_payload(spec):
    df = pd.read_csv(spec["data_path"])
    columns = list(df.columns)
    n_records = len(df)
    values = df.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise RuntimeError("数据必须全部为二值 0/1")

    queries = []
    index = 0
    for i, j in itertools.combinations(range(len(columns)), 2):
        a, b = columns[i], columns[j]
        cell_counts = np.bincount(values[:, i] * 2 + values[:, j], minlength=4)
        if int(cell_counts.sum()) != n_records:
            raise RuntimeError(f"pair ({a},{b}) 四格计数之和不等于 N")
        for va in (0, 1):
            for vb in (0, 1):
                index += 1
                count = int(cell_counts[va * 2 + vb])
                queries.append({
                    "id": f"AW{index:05d}",
                    "type": "double",
                    "expression": f"{a} == {va} AND {b} == {vb}",
                    "conditions": [
                        {"attribute": a, "operator": "==", "value": int(va)},
                        {"attribute": b, "operator": "==", "value": int(vb)},
                    ],
                    "result": count,
                    "group": f"pair_{a}_{b}",
                })

    expected_count = len(columns) * (len(columns) - 1) // 2 * 4
    if len(queries) != expected_count:
        raise RuntimeError(
            f"查询数量错误: {len(queries)} != {expected_count}"
        )

    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("all-2way 考卷存在重复语义查询")

    with open(spec["measured_path"], "r", encoding="utf-8") as f:
        measured = json.load(f)
    all2way_by_fingerprint = dict(zip(fingerprints, queries))
    checked = 0
    for query in measured["queries"]:
        if query["type"] != "double":
            continue
        fingerprint = query_fingerprint(query)
        if fingerprint not in all2way_by_fingerprint:
            raise RuntimeError(
                f"旧池 double 查询不在全家族中: {query['id']}"
            )
        if int(all2way_by_fingerprint[fingerprint]["result"]) != int(
            query["result"]
        ):
            raise RuntimeError(
                f"旧池 double 目标值不一致: {query['id']}"
            )
        checked += 1
    if checked != spec["expected_old_pool_doubles"]:
        raise RuntimeError(
            f"旧池 double 对账数量错误: {checked} != "
            f"{spec['expected_old_pool_doubles']}"
        )

    return {
        "dataset": spec["dataset_label"],
        "record_count": n_records,
        "query_count": len(queries),
        "result_unit": "records",
        "workload_version": 1,
        "description": spec["description"],
        "queries": queries,
    }, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="确定性重建并与已存在的正式文件逐字段比较",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        out, checked = build_payload(spec)
        payload = json.dumps(out, ensure_ascii=False, indent=2)
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if args.verify_existing:
            with open(spec["out_path"], "r", encoding="utf-8") as f:
                if f.read() != payload:
                    raise RuntimeError(
                        f"{spec['out_path']} 无法由冻结构造器逐字重建"
                    )
            print(f"verified {spec['out_path']}: sha256 = {sha}")
        else:
            with open(spec["out_path"], "x", encoding="utf-8") as f:
                f.write(payload)
            print(f"写出 {spec['out_path']}: {out['query_count']} 条查询")
            print(f"file sha256 = {sha}")
        counts = np.array([q["result"] for q in out["queries"]], dtype=float)
        print(
            f"counts: min={counts.min():.0f} max={counts.max():.0f} "
            f"mean={counts.mean():.1f} zero_cells={(counts == 0).sum():.0f}"
        )
        print(
            f"旧池 double 对账通过: {checked}/"
            f"{spec['expected_old_pool_doubles']}"
        )


if __name__ == "__main__":
    main()
