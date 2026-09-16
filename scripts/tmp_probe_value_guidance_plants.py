#!/usr/bin/env python3
"""值引导核 plants 摸底（临时诊断，不入正式协议）。

nltcs 上 V9(max 自适应) 全场最优（loss 505 / fam480 mean 6.57e-5 首破
1e-4）。本脚本把同款配置搬到 plants（9384 all-2way + 138 one-way =
9522 池，N=17412，69 属性）验证跨数据集成立性：

- L1 : lottery + 深退火，无引导（plants 上的新基线对照）
- V8 : L1 + 值引导 λ=8000 固定（nltcs 手调最优——验证手调 λ 不可迁移）
- V9 : L1 + 值引导 λ=4 + 自适应 max 尺度（λ_eff = λ_t / max|wr_t|）

评估口径与 GSD plants 局对齐：fam9384（all-2way 家族）normalized L1
（|q−y|/N）mean/max 为主指标 + in-pool 9522 + 冻结 heldout 3/4-way。
参照：GSD 官方 fam 1.4e-5 / heldout 0.003514；引擎旧法（all2way pool
残差臂）fam 0.001292 / heldout 0.006171。

用法：CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=4 \
    .venv/bin/python scripts/tmp_probe_value_guidance_plants.py --arm V9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tmp_value_guidance_probe_plants_seed9908"

SEED = 9908
N_RECORDS = 17412
N_ROUNDS = 20000
EXPECTED_ALL2WAY = 9384
EXPECTED_POOL = 9522

_DEEP_ANNEAL = dict(
    # 与 nltcs 摸底同款深退火：rho 几何 0.01→1e-5、mu 0.01→1e-6，
    # 上限 2 万轮 patience 6 早停兜底。
    n_rounds=N_ROUNDS,
    rho_anneal_rounds=2000,
    rho_anneal_end=1e-5,
    mu_anneal_start_round=1050,
    mu_anneal_rounds=2000,
    mu_anneal_end=1e-6,
)
_VALUE_GUIDANCE = dict(
    value_guidance_strength=4.0,
    value_guidance_warmup_start_round=1050,
    value_guidance_warmup_rounds=2000,
)
ARMS = {
    "L1": dict(lottery_first_donor_selection=True, **_DEEP_ANNEAL),
    "V8": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_strength": 8000.0,
    },
    "V9": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_adaptive_scale": True,
    },
}


def build_pool():
    """9522 池：9384 all-2way 真答案 + 138 one-way 边缘。"""
    marginals = load_marginals(str(ROOT / "configs/plants/init_marginals.json"))
    payload = json.loads(
        (ROOT / "configs/plants/all2way_issue53_v1.json").read_text()
    )
    queries = list(payload["queries"])
    assert len(queries) == EXPECTED_ALL2WAY
    targets = [float(q["result"]) for q in queries]

    index = 0
    for attribute, spec in marginals["attributes"].items():
        for value, count in zip(spec["values"], spec["counts"]):
            index += 1
            queries.append({
                "id": f"OW{index:04d}",
                "type": "single",
                "expression": f"{attribute} == {value}",
                "conditions": [
                    {"attribute": attribute, "operator": "==", "value": value}
                ],
                "result": float(count),
            })
            targets.append(float(count))
    assert len(queries) == EXPECTED_POOL
    return queries, np.asarray(targets, dtype=float), marginals


def normalized_l1_stats(target, current, n_records):
    err = np.abs(np.asarray(current, dtype=float) - np.asarray(target)) / n_records
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(err.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-rounds", type=int, default=None,
                        help="冒烟用小轮数覆盖")
    args = parser.parse_args()
    arm = dict(ARMS[args.arm])
    if args.n_rounds is not None:
        arm["n_rounds"] = args.n_rounds

    schema = load_schema(str(ROOT / "configs/plants/schema.yaml"))
    queries, targets, marginals = build_pool()

    base_fields = dict(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device=args.device,
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=500,
        rho=0.01,
        eta=0.5,
        mu=0.01,
        rho_anneal_start_round=1050,
        rho_anneal_rounds=700,
        rho_anneal_end=0.001,
        lambda_param=0.5,
        fixed_alpha=16.0,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
        exclude_self=True,
        inner_early_stopping_patience_ticks=6,
    )
    base_fields.update(arm)
    config = FitnessOnlyConfig(**base_fields)

    start = time.perf_counter()
    table, diagnostics = run_fitness_only_evolution(
        targets, queries, schema, N_RECORDS,
        config=config, fitness_mode="residual", marginals=marginals,
    )
    walltime = time.perf_counter() - start

    q_pool = evaluate_vectorized(
        table, queries, schema, want_fitness=False, device=args.device,
        verbose=False,
    )[0]
    heldout = json.loads(
        (ROOT / "configs/plants/heldout_issue53_v1.json").read_text()
    )
    heldout_queries = list(heldout["queries"])
    heldout_targets = np.asarray(
        [float(q["result"]) for q in heldout_queries], dtype=float
    )
    q_heldout = evaluate_vectorized(
        table, heldout_queries, schema, want_fitness=False,
        device=args.device, verbose=False,
    )[0]
    kinds = np.asarray([q["type"] for q in heldout_queries])

    vg = diagnostics["value_guidance"]
    vg_summary = {
        "enabled": vg["enabled"],
        "strength": vg["strength"],
        "warmup_start_round": vg["warmup_start_round"],
        "warmup_rounds": vg["warmup_rounds"],
        "drop_donor": vg["drop_donor"],
    }
    if vg["enabled"]:
        lam = vg["lambda_history"]
        vg_summary.update({
            "lambda_first_last": (lam[0], lam[-1]),
            "gain_recompute_count": vg["gain_recompute_count"],
            "structure_compile_elapsed_sec": round(
                vg["structure_compile_elapsed_sec"], 4
            ),
        })
        if vg.get("adaptive_scale"):
            sc = vg["scale_history"]
            lam_eff = [l / s for l, s in zip(lam, sc)]
            vg_summary.update({
                "adaptive_scale": True,
                "scale_first_last": (sc[0], sc[-1]),
                "lambda_eff_first_last": (lam_eff[0], lam_eff[-1]),
                "lambda_eff_max": max(lam_eff),
            })

    report = {
        "arm": args.arm,
        "config_overrides": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in arm.items()
        },
        "seed": SEED,
        "dataset": "plants",
        "device": args.device,
        "rounds_run": diagnostics.get("rounds_run"),
        "termination_reason": diagnostics.get("termination_reason"),
        "walltime_sec": round(walltime, 1),
        "final_loss": diagnostics["loss_history"][-1],
        # GSD 口径主指标：9384 格 all-2way 家族（build_pool 前 9384 条）
        "family9384_normalized_l1": normalized_l1_stats(
            targets[:EXPECTED_ALL2WAY], q_pool[:EXPECTED_ALL2WAY], N_RECORDS
        ),
        "in_pool_normalized_l1": normalized_l1_stats(
            targets, q_pool, N_RECORDS
        ),
        "heldout_normalized_l1": {
            label: normalized_l1_stats(
                heldout_targets[kinds == kind],
                np.asarray(q_heldout)[kinds == kind],
                N_RECORDS,
            )
            for label, kind in (
                ("3way", "heldout_3way"), ("4way", "heldout_4way"),
            )
            if (kinds == kind).any()
        },
        "value_guidance": vg_summary,
        "references": {
            "gsd_official": {"family_mean": 1.4e-05, "heldout_comb": 0.003514},
            "engine_all2way_pool_residual": {
                "family_mean": 0.001292, "heldout_comb": 0.006171,
            },
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.arm}.json"
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in (
        "arm", "rounds_run", "termination_reason", "walltime_sec",
        "final_loss", "family9384_normalized_l1",
    )}, indent=1, ensure_ascii=False))
    print(f"已写出 {out_path}")


if __name__ == "__main__":
    main()
