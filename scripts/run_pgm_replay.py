"""独立 PGM 回放：把 AIM 测量录像原样喂给 Private-PGM（mbi MirrorDescent）出表。

第一阶段 1B 消融的核对臂。AIM = 选测器 + PGM，录像里的测量列表就是 AIM 末次
estimate 收到的那一列（含重复团、各自 sigma），本脚本复刻 AIM 收尾那两行：
    model = MirrorDescent().estimate(domain, measurements, iters=max_iters)
    synth = model.synthetic_data(rows=N)
区别仅在没有前面几十轮的 warm_start 势能（从零起步跑 iters 轮）。known_total 默认喂真表
行数 N（我们引擎与 GSD 都拿 N，三方同口径）；--known-total auto 则与 AIM 原样留空自估。
若此臂与同一跑的 AIM 终表各轴相差 ≤2%，即证实「AIM 终表 ≈ 独立 PGM」，1B 里的
对照臂可直接用 AIM 终表；差得多则以本臂为 PGM 对照（同录像、同迭代预算）。

用法（A6000 aim2 环境，纯 CPU）：
  PYTHONPATH=/root/private-pgm/src /root/miniconda3/envs/aim2/bin/python scripts/run_pgm_replay.py \
      --dump results/aimreplay/adult_eps0p1_aimfull_seed1_measurements.json \
      --seed 1 --out results/stage1/adult_eps0p1_pgmrep_seed1.csv
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="AIM 测量录像 -> 独立 Private-PGM 出表")
    ap.add_argument("--dump", required=True, help="run_aim_full.py --dump-measurements 产物")
    ap.add_argument("--seed", type=int, default=1, help="合成采样随机种子（numpy 全局）")
    ap.add_argument("--iters", type=int, default=1000, help="MirrorDescent 迭代数，AIM 默认 max_iters=1000")
    ap.add_argument("--rows", type=int, default=0, help="合成行数，0 为录像里的真表行数")
    ap.add_argument("--known-total", choices=["rows", "auto"], default="rows",
                    help="模型总量：rows 喂真表行数（与我们引擎、GSD 同口径）；auto 留空由 mbi 从噪声测量自估（AIM 原样）")
    ap.add_argument("--max-model-mb", type=float, default=4096.0,
                    help="联结树参数表内存上限 MB，超限直接退出记录为证据")
    ap.add_argument("--out", required=True, help="输出合成表 CSV")
    args = ap.parse_args()

    from mbi import Domain, estimation, junction_tree
    from mbi.marginal_loss import LinearMeasurement

    with open(args.dump, "r", encoding="utf-8") as fh:
        dump = json.load(fh)
    columns = list(dump["columns"])
    domain = Domain(columns, [int(s) for s in dump["domain_sizes"]])
    rows = args.rows or int(dump["num_rows"])

    measurements = []
    for rec in dump["measurements"]:
        cl = tuple(rec["clique"])
        shape = [int(domain[a]) for a in cl]
        if shape != [int(s) for s in rec["shape"]]:
            raise SystemExit(f"录像团 {cl} 形状 {rec['shape']} 与域 {shape} 不符")
        y = np.asarray(rec["y"], dtype=np.float64)
        if y.size != int(np.prod(shape)):
            raise SystemExit(f"录像团 {cl} 长度 {y.size} 与域 {shape} 不符")
        measurements.append(LinearMeasurement(y, cl, stddev=float(rec["sigma"])))
    n1 = sum(1 for m in measurements if len(m.clique) == 1)
    cliques = [m.clique for m in measurements]
    model_mb = junction_tree.hypothetical_model_size(domain, cliques)
    print(f"{dump['data']} eps={dump['epsilon']} 录像种子={dump['seed']} 测量 {len(measurements)} 条"
          f"（一阶 {n1}，高阶 {len(measurements) - n1}），联结树参数表 {model_mb:.1f} MB，"
          f"iters={args.iters} rows={rows} known_total={args.known_total}", flush=True)
    if model_mb > args.max_model_mb:
        raise SystemExit(f"树宽爆炸: 参数表 {model_mb:.1f} MB 超上限 {args.max_model_mb:.0f} MB")

    t0 = time.time()
    known_total = float(rows) if args.known_total == "rows" else None
    model = estimation.MirrorDescent().estimate(domain, measurements, known_total=known_total, iters=args.iters)
    fit_s = time.time() - t0
    np.random.seed(args.seed)
    synth = model.synthetic_data(rows=rows)
    out = pd.DataFrame({c: np.asarray(synth.data[c]) for c in columns})
    out.to_csv(args.out, index=False)
    print(f"拟合 {fit_s:.0f} 秒，合成 {len(out)} 行 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
