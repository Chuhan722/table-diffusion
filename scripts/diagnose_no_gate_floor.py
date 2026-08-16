#!/usr/bin/env python
"""诊断无门恒定动力学的瓶颈类型（探索性，非正式协议）。

问题：rho=0.01 无门在 nltcs 2000 轮终点 3.28M 平方 loss，是
(a) 噪声地板（已平台，地板高度由每轮扰动注入决定），还是
(b) 预算不足（仍在下降）？

设计：无门（tol=inf）恒定扰动，dev 种子，4000 轮（正式预算 2 倍），
扫 rho ∈ {0.003, 0.01, 0.03, 0.1}，另加 rho=0.01+mu=0（关闭均匀变异
噪声源，隔离"变异 vs 复制搅动"两类噪声的地板贡献）。

输出：每臂 loss 轨迹（下采样）、final/best/tail loss、final L1。
结论用于选择机制方向：平台 → 降扰动破坏性（守恒扰动/噪声退火）；
未收敛 → 加速动力学（退火起点更高）。

用法：CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd \
  python scripts/diagnose_no_gate_floor.py --arms rho0.003 rho0.01
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

DEV_SEED = 42
ROUNDS = 4000
SAMPLE_EVERY = 20

SHARED = dict(
    beta=1.0, h=0.8, eta=0.5, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    alpha_min=2.0, alpha_max=10.0, exclude_self=True,
    tol=float("inf"),
)

ARMS = {
    "rho0.003": dict(rho=0.003, mu=0.01),
    "rho0.01": dict(rho=0.01, mu=0.01),
    "rho0.03": dict(rho=0.03, mu=0.01),
    "rho0.1": dict(rho=0.1, mu=0.01),
    "rho0.01_mu0": dict(rho=0.01, mu=0.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument(
        "--out", default="outputs/gate_free_self_cooling/diagnose_floor.json"
    )
    args = parser.parse_args()

    schema = load_schema("configs/nltcs/schema.yaml")
    queries = load_queries("configs/nltcs/measured_1000query.json")
    marginals = load_marginals("configs/nltcs/init_marginals.json")
    target = np.asarray([q["result"] for q in queries], dtype=float)

    results = {}
    for arm in args.arms:
        extra = ARMS[arm]
        t0 = time.time()
        _best, diag = run_evolution(
            target=target,
            schema=schema,
            queries=queries,
            marginals=marginals,
            n_records=16181,
            n_rounds=args.rounds,
            seed=args.seed,
            device="cuda",
            log_every=0,
            return_final_table=True,
            **SHARED,
            **extra,
        )
        final_table = diag.pop("final_table")
        final_answers = evaluate_table(final_table, queries)
        final_loss = float(compute_loss(final_answers, target))
        final_l1 = float(
            compute_normalized_l1(final_answers, target, n_records=16181)
        )
        hist = np.asarray(diag["loss_history"], dtype=float)
        results[arm] = {
            "params": {**{k: extra.get(k) for k in ("rho", "mu")}},
            "rounds": args.rounds,
            "seed": args.seed,
            "loss_at_2000_proposal_view": float(hist[2000])
            if len(hist) > 2000 else None,
            "final_loss": final_loss,
            "final_l1": final_l1,
            "best_loss": float(hist.min()),
            "best_round": int(hist.argmin()),
            "tail500_mean": float(hist[-500:].mean()),
            "tail500_slope_per_1k": float(
                np.polyfit(np.arange(len(hist[-500:])), hist[-500:], 1)[0] * 1000
            ),
            "trajectory_every%d" % SAMPLE_EVERY: [
                float(x) for x in hist[::SAMPLE_EVERY]
            ],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"[{arm}] final={final_loss:.4g} L1={final_l1:.6f} "
            f"best={hist.min():.4g}@{hist.argmin()} "
            f"tail_slope={results[arm]['tail500_slope_per_1k']:.4g}/1k "
            f"({results[arm]['elapsed_sec']}s)",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=1, ensure_ascii=False))
    print("saved ->", out)


if __name__ == "__main__":
    main()
