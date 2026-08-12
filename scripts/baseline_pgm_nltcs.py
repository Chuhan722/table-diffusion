"""Tier-1 同信息条件基线：Private-PGM（mbi）无噪声估计 + 合成（Issue #46）。

信息条件与本仓库生成器严格一致：

- 公开 schema（16 个二值属性）与公开记录数；
- 1-way 初始化边缘（configs/nltcs/init_marginals.json 的公开计数）；
- workload 的 1001 个公开单元格计数（不提供完整 2/3-way 边缘表——
  workload 只测量了每个属性组合的部分单元格）。

方法：mbi 的 MirrorDescent 在图模型边缘多面体上做无噪声（stddev→极小）
线性测量拟合，测量向量 = 每个 workload 单元格的指示投影 + 每属性 1-way
全边缘。之后从学到的 MarkovRandomField 采样合成表。

运行环境：独立 conda 环境 td_baseline_pgm（python>=3.11 + private-pgm），
不依赖本仓库 torch 环境。用法：

    conda run -n td_baseline_pgm python scripts/baseline_pgm_nltcs.py

输出 JSON 与合成表 CSV 到 outputs/tier1_pgm_baseline/（git 忽略）。评价与
本仓库生成器共用离线协议（由 gsd 环境的 evaluate 脚本另行执行，保持评价器
单一来源）。
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_ATTRS = [f"attr_{i}" for i in range(1, 17)]
QUERY_PATH = Path("configs/nltcs/measured_1000query.json")
MARGINALS_PATH = Path("configs/nltcs/init_marginals.json")
OUTPUT_DIR = Path("outputs/tier1_pgm_baseline")
NOISELESS_STDDEV = 1e-6
SEEDS = [100, 101, 102, 103, 104]
ITERS = 5000


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_measurements():
    """把公开输入转成 mbi LinearMeasurement 列表（无噪声）。"""
    from mbi import Domain, LinearMeasurement

    workload = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    marginals = json.loads(MARGINALS_PATH.read_text(encoding="utf-8"))
    n_records = int(marginals["n_records"])
    domain = Domain(SCHEMA_ATTRS, [2] * len(SCHEMA_ATTRS))

    measurements = []
    # 1-way 公开边缘（初始化信息，与生成器一致）。
    for attr in SCHEMA_ATTRS:
        info = marginals["attributes"][attr]
        counts = np.zeros(2, dtype=float)
        for value, count in zip(info["values"], info["counts"]):
            counts[int(value)] = float(count)
        measurements.append(LinearMeasurement(
            counts, clique=(attr,), stddev=NOISELESS_STDDEV,
        ))

    # workload 单元格计数：每个查询是若干属性=值的合取，对应 clique 边缘
    # 展平向量中的一个索引。按 clique 聚合成稀疏指示矩阵查询。
    class CellQuery:
        """从 clique 边缘 datavector 中取出指定单元格子集。"""

        def __init__(self, indices, size):
            self.indices = np.asarray(indices, dtype=int)
            self.size = int(size)

        def __call__(self, factor):
            return factor.datavector()[self.indices]

        def op_norm_sq(self):
            return 1.0

    grouped = {}
    for query in workload["queries"]:
        conditions = sorted(
            (c["attribute"], int(c["value"])) for c in query["conditions"]
        )
        clique = tuple(name for name, _ in conditions)
        values = tuple(value for _, value in conditions)
        # datavector 按 domain 顺序展平（行主序，2 值属性）。
        flat = 0
        for value in values:
            flat = flat * 2 + value
        grouped.setdefault(clique, []).append(
            (flat, float(query["result"]))
        )

    for clique, cells in sorted(grouped.items()):
        cells.sort()
        indices = [flat for flat, _ in cells]
        answers = np.asarray([answer for _, answer in cells], dtype=float)
        size = 2 ** len(clique)
        measurements.append(LinearMeasurement(
            answers, clique=clique, stddev=NOISELESS_STDDEV,
            query=CellQuery(indices, size),
        ))
    return domain, measurements, n_records, workload


def workload_l1(table, workload, n_records):
    total_abs = 0.0
    for query in workload["queries"]:
        mask = np.ones(len(table), dtype=bool)
        for cond in query["conditions"]:
            mask &= table[cond["attribute"]].to_numpy() == int(cond["value"])
        total_abs += abs(float(query["result"]) - float(mask.sum()))
    return total_abs / (len(workload["queries"]) * n_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    from mbi.estimation import MirrorDescent

    domain, measurements, n_records, workload = build_measurements()
    started = time.perf_counter()
    model = MirrorDescent().estimate(
        domain, measurements, known_total=float(n_records),
        iters=args.iters,
    )
    fit_elapsed = time.perf_counter() - started

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        # mbi 的采样使用全局 numpy 随机状态；固定种子保证可复现。
        np.random.seed(seed)
        start = time.perf_counter()
        synthetic = model.synthetic_data(rows=n_records, method="round")
        elapsed = time.perf_counter() - start
        frame = pd.DataFrame(dict(synthetic.data))[SCHEMA_ATTRS].astype(int)
        csv_path = OUTPUT_DIR / f"pgm_synthetic_seed{seed}.csv"
        frame.to_csv(csv_path, index=False)
        rows.append({
            "seed": int(seed),
            "measured_workload_l1": float(
                workload_l1(frame, workload, n_records)
            ),
            "sample_elapsed_sec": float(elapsed),
            "table_sha256": _sha256_file(csv_path),
            "table_path": str(csv_path),
        })
        print(f"seed={seed} measured_L1={rows[-1]['measured_workload_l1']:.6f} "
              f"({elapsed:.1f}s)", flush=True)

    report = {
        "experiment": "tier1_pgm_baseline_nltcs",
        "method": (
            "private-pgm MirrorDescent, noiseless linear measurements "
            "(1-way marginals + 1001 workload cells), synthetic_data(round)"
        ),
        "information_condition": (
            "public schema + public N + public 1-way marginals + "
            "public workload cell answers only; no raw data access"
        ),
        "iters": int(args.iters),
        "fit_elapsed_sec": float(fit_elapsed),
        "public_input_sha256": {
            "queries": _sha256_file(QUERY_PATH),
            "marginals": _sha256_file(MARGINALS_PATH),
        },
        "seeds": rows,
        "measured_l1_mean": float(np.mean(
            [row["measured_workload_l1"] for row in rows]
        )),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    out = OUTPUT_DIR / "pgm_baseline_report.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"mean measured_L1 = {report['measured_l1_mean']:.6f}")
    print(f"output={out}")
    print(f"sha256={_sha256_file(out)}")


if __name__ == "__main__":
    main()
