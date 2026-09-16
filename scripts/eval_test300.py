"""评价脚本，AIM 式与 GSD 式指标对比任意合成表与真实表。

评价与生成严格分离，本脚本只读表文件与查询文件，不触碰引擎，
保留查询只在这里出现，加载时强制校验与生成查询语义零交集。
用法，./.venv/bin/python scripts/eval_test300.py 表1.csv 表2.csv ...
不给表时只报真实表自评与随机表基线。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.dataset import load_queries, load_table  # noqa: E402
from resevo.metrics import (  # noqa: E402
    aim_workload_error,
    assert_disjoint_workloads,
    gsd_query_errors,
    load_heldout_queries,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_300x10"


def evaluate_one(name, synth_rows, real_rows, schema, measured, heldout):
    n = len(real_rows)
    gm = gsd_query_errors(measured, synth_rows, schema, n)
    gh = gsd_query_errors(heldout, synth_rows, schema, n)
    a2 = aim_workload_error(real_rows, synth_rows, schema, 2)
    a3 = aim_workload_error(real_rows, synth_rows, schema, 3)
    print(f"\n== {name}，行数 {len(synth_rows)} ==")
    print(f"GSD 式 measured 50   平均 {gm.average_error:.6f}  最大 {gm.max_error:.6f}")
    print(f"GSD 式 heldout 1024  平均 {gh.average_error:.6f}  最大 {gh.max_error:.6f}")
    print(f"AIM 式 全部二阶边缘 45   平均 {a2.average_error:.6f}  最大 {a2.max_error:.6f}")
    print(f"AIM 式 全部三阶边缘 120  平均 {a3.average_error:.6f}  最大 {a3.max_error:.6f}")
    return gm, gh, a2, a3


def random_table(schema, n, rng):
    return [
        tuple(
            schema.domains[j][int(rng.integers(len(schema.domains[j])))]
            for j in range(schema.num_fields)
        )
        for _ in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="AIM 与 GSD 式评价对比")
    parser.add_argument("tables", nargs="*", help="待评合成表 CSV 路径列表")
    parser.add_argument("--baseline-seeds", type=int, default=3, help="随机基线种子数")
    args = parser.parse_args()

    schema, real_rows = load_table(str(DATA_DIR / "test_300x10.csv"))
    measured = load_queries(str(DATA_DIR / "measured_50query.json"))
    heldout = load_heldout_queries(str(DATA_DIR / "heldout_issue53_v1.json"))
    assert_disjoint_workloads(measured, heldout)
    print(f"评价集校验通过，measured {len(measured)} 条与 heldout {len(heldout)} 条语义零交集")

    evaluate_one("真实表自评，全零基准", real_rows, real_rows, schema, measured, heldout)

    baselines = []
    for seed in range(args.baseline_seeds):
        rng = np.random.default_rng(seed)
        rows = random_table(schema, len(real_rows), rng)
        reports = evaluate_one(
            f"随机表基线 种子 {seed}", rows, real_rows, schema, measured, heldout
        )
        baselines.append(reports)
    if baselines:
        print("\n== 随机基线均值 ==")
        labels = ["GSD measured 平均", "GSD heldout 平均", "AIM 二阶平均", "AIM 三阶平均"]
        for k, label in enumerate(labels):
            vals = [b[k].average_error for b in baselines]
            print(f"{label}  {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    for path in args.tables:
        _, synth_rows = load_table(path)
        evaluate_one(path, synth_rows, real_rows, schema, measured, heldout)


if __name__ == "__main__":
    main()
