"""
exclude_self（对角线屏蔽）回归实验：屏蔽前(False) vs 屏蔽后(True)

目的：确认对角屏蔽没搞坏主战场、修好了小表隐患。
- nltcs α2→10：baseline(False) 已有 outputs/alpha_multiseed_1500/a2_10/（写死 True 之前跑的），
  本脚本只补 True 侧，与之对比 loss/L1/接受率。预期几乎不变（自身率本来 0.04%）。
- test_300x10 α2→10：无对齐 baseline，本脚本跑 False+True 两侧。预期 True 侧末轮
  自身率归零、多样性略升。

自身率现已是常驻诊断字段 donor_self_rate_history（evolution.py），本脚本直接读它。

用法（在空闲卡上跑，1 号卡）：
    CUDA_VISIBLE_DEVICES=1 python scripts/exclude_self_regression.py --dataset nltcs --exclude both --seeds 0 1 2
    CUDA_VISIBLE_DEVICES=1 python scripts/exclude_self_regression.py --dataset small --exclude both --seeds 42 43 44 45 46
"""
import os
import json
import time
import argparse
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.io import save_run

DATASETS = {
    "nltcs": {
        "schema": "configs/nltcs/schema.yaml",
        "query": "configs/nltcs/measured_1000query.json",
        "marginals": "configs/nltcs/init_marginals.json",
        "n_records": 16181,
        "n_rounds": 1500,
        "log_every": 100,
    },
    "small": {
        "schema": "configs/test_300x10/schema.yaml",
        "query": "configs/test_300x10/measured_50query.json",
        "marginals": "configs/test_300x10/init_marginals.json",
        "n_records": 300,
        "n_rounds": 100,
        "log_every": 50,
    },
}

# 推荐配置 α2→10, λ0.5，其余固定值与 alpha_multiseed_1500 对齐
PARAMS = {
    "beta": 1.0, "h": 0.8, "rho": 0.01, "eta": 0.5, "mu": 0.01,
    "device": "cuda", "eval_method": "vectorized", "batch_size": 256,
    "init_method": "marginal",
    "distance_mode": "geometric", "lambda_param": 0.5,
    "alpha_min": 2.0, "alpha_max": 10.0, "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def diversity(best_S):
    """合成表多样性：唯一记录数 + 重复率 + 最高频占比。"""
    n = len(best_S)
    vc = best_S.value_counts()
    uniq = int(len(vc))
    return {
        "unique": uniq,
        "unique_pct": uniq / n,
        "dup_rate": 1.0 - uniq / n,
        "top1_pct": float(vc.iloc[0] / n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--exclude", choices=["true", "false", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    schema = load_schema(ds["schema"])
    queries = load_queries(ds["query"])
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(ds["marginals"])

    sides = {"true": [True], "false": [False], "both": [False, True]}[args.exclude]
    out_root = f"outputs/exclude_self_regression/{args.dataset}"
    os.makedirs(out_root, exist_ok=True)

    records = []
    for excl in sides:
        tag = "excl_true" if excl else "excl_false"
        for seed in args.seeds:
            print(f"\n{'='*60}\n{args.dataset} α2→10 exclude_self={excl}  "
                  f"seed={seed} @ {ds['n_rounds']} 轮\n{'='*60}")
            start = time.perf_counter()
            best_S, diag = run_evolution(
                target, queries, schema,
                n_records=ds["n_records"], n_rounds=ds["n_rounds"], seed=seed,
                marginals=marginals, log_every=ds["log_every"],
                exclude_self=excl, **PARAMS,
            )
            elapsed = time.perf_counter() - start
            run_dir = os.path.join(out_root, f"{tag}_seed{seed}")
            save_run(best_S, diag, run_dir=run_dir)

            sr = diag["donor_self_rate_history"]
            div = diversity(best_S)
            rec = {
                "dataset": args.dataset, "exclude_self": excl, "seed": seed,
                "best_loss": float(diag["best_loss"]),
                "normalized_l1_error": float(diag["normalized_l1_error"]),
                "accept_rate": float(sum(diag["accept_history"]) / len(diag["accept_history"])),
                "self_rate_mean": float(np.mean(sr)),
                "self_rate_last100_mean": float(np.mean(sr[-100:])),
                "self_rate_max": float(np.max(sr)),
                **div,
                "elapsed_sec": float(elapsed),
            }
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            records.append(rec)
            print(f"best_loss: {rec['best_loss']:.4e}  L1: {rec['normalized_l1_error']:.4f}  "
                  f"接受率: {rec['accept_rate']*100:.1f}%  "
                  f"自身率(末100均值): {rec['self_rate_last100_mean']*100:.2f}%  "
                  f"唯一: {div['unique']}  耗时: {elapsed/60:.1f}min")

    with open(os.path.join(out_root, "records.json"), "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，汇总: {out_root}/records.json")


if __name__ == "__main__":
    main()
