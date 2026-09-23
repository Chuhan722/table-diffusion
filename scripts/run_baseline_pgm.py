"""Private-PGM 同卷基线，读带噪考卷 JSON，喂 mbi 2.0 MirrorDescent 拟合图模型再采样成表。

同卷公平性说明，考卷全部是二字段等值计数查询且每字段对满格覆盖，
恰好等价于 PGM 需要的全套二阶带噪边际，零信息损失。
AIM 剥掉自适应选择过程就是本脚本跑的 Private-PGM，纯比生成器官。
旧版零噪声文献对照脚本见 git 历史 bb8649c，本版把考卷路径与噪声参数化后两者通吃。

用法，需在 /home/chuhan/projects/private-pgm 的 venv 里跑，
  .venv/bin/python run_baseline_pgm.py --exam data/nltcs_eps1_s1/measured_480query.json \
      --sigma 62.6669 --seed 1 --out results/nltcs_eps1_pgm_seed1.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from mbi import Domain, estimation, junction_tree, marginal_loss


def build_measurements(exam: dict, sigma: float):
    queries = exam["queries"]
    values: dict[str, set] = defaultdict(set)
    for q in queries:
        for c in q["conditions"]:
            if c["operator"] != "==":
                raise ValueError(f"非等值算子无法转边际: {c}")
            values[c["attribute"]].add(c["value"])

    def attr_key(name: str):
        head, _, tail = name.rpartition("_")
        return (head, int(tail)) if tail.isdigit() else (name, 0)

    attrs = sorted(values, key=attr_key)
    value_order = {a: sorted(values[a], key=lambda v: int(v)) for a in attrs}
    value_index = {a: {v: i for i, v in enumerate(value_order[a])} for a in attrs}
    domain = Domain(attrs, [len(value_order[a]) for a in attrs])

    pair_cells: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for q in queries:
        c1, c2 = sorted(q["conditions"], key=lambda c: attr_key(c["attribute"]))
        a, b = c1["attribute"], c2["attribute"]
        ia, ib = value_index[a][c1["value"]], value_index[b][c2["value"]]
        pair_cells[(a, b)][(ia, ib)] = float(q["result"])

    measurements = []
    skipped = []
    for (a, b), cells in sorted(pair_cells.items()):
        na, nb = len(value_order[a]), len(value_order[b])
        if len(cells) != na * nb:
            skipped.append((a, b, len(cells), na * nb))
            continue
        mat = np.zeros((na, nb))
        for (ia, ib), y in cells.items():
            mat[ia, ib] = y
        measurements.append(
            marginal_loss.LinearMeasurement(mat.flatten(), (a, b), stddev=sigma)
        )
    return domain, measurements, value_order, skipped


def main():
    ap = argparse.ArgumentParser(description="Private-PGM 同卷基线")
    ap.add_argument("--exam", required=True, help="带噪考卷 JSON 路径")
    ap.add_argument("--sigma", type=float, required=True, help="考卷计数噪声标准差")
    ap.add_argument("--seed", type=int, default=1, help="采样种子")
    ap.add_argument("--iters", type=int, default=1000, help="MirrorDescent 迭代数")
    ap.add_argument("--out", required=True, help="输出合成表 CSV 路径")
    ap.add_argument(
        "--max-model-mb",
        type=float,
        default=32768,
        help="联结树参数表内存上限 MB，超限直接判树宽爆炸退出",
    )
    args = ap.parse_args()

    with open(args.exam, "r", encoding="utf-8") as fh:
        exam = json.load(fh)
    total = int(exam["record_count"])
    domain, measurements, value_order, skipped = build_measurements(exam, args.sigma)
    print(f"字段 {len(domain)} 个，边际 {len(measurements)} 对，known_total {total}")
    if skipped:
        print(f"缺格跳过 {len(skipped)} 对: {skipped[:5]}")

    cliques = [m.clique for m in measurements]
    model_mb = junction_tree.hypothetical_model_size(domain, cliques)
    print(f"联结树参数表 {model_mb:.1f} MB")
    if model_mb > args.max_model_mb:
        raise SystemExit(
            f"树宽爆炸: 参数表 {model_mb:.1f} MB 超上限 {args.max_model_mb:.0f} MB，"
            "精确图模型推断在全二阶边际下不可行，记录为证据"
        )

    loss_fn = marginal_loss.from_linear_measurements(measurements, domain)
    t0 = time.time()
    model = estimation.MirrorDescent().estimate(
        domain, loss_fn, known_total=total, iters=args.iters
    )
    fit_s = time.time() - t0

    np.random.seed(args.seed)
    syn = model.synthetic_data(total)
    df = pd.DataFrame(
        {a: [value_order[a][i] for i in syn.data[a]] for a in domain.attributes}
    )
    df.to_csv(args.out, index=False)
    print(f"拟合 {fit_s:.0f} 秒，合成 {len(df)} 行 -> {args.out}")


if __name__ == "__main__":
    main()
