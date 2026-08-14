#!/usr/bin/env python
"""dev 定标：时间驱动几何 rho 退火 vs 无门恒定（探索性，非正式协议）。

诊断结论（scripts/diagnose_no_gate_floor.py，dev seed 42、4000 轮）：
无门恒定动力学存在 rho 依赖的噪声地板与时间尺度权衡——rho=0.1 快速
到达 ~1.8e7 平台；rho=0.003 地板低得多但 4000 轮预算内远未收敛。
几何退火 rho_t = rho_start·(rho_end/rho_start)^(t/T) 应能先快降后深潜。

臂设计（全部 tol=inf 无门、mu=0.01、正式预算 2000 轮）：
  anneal_A: rho 0.1 → 0.01（终点=项目标准）
  anneal_B: rho 0.1 → 0.003（更深终点）
  anneal_C: rho 0.3 → 0.003（更高起点+深终点）
  const_001: rho=0.01 恒定（正式对照臂在 dev seed 的镜像）
参考锚点：正式 rho=0.01 恒定（seed 100..104）终点 3.28M；
有门 rho=0.01（同）1.98M —— dev 目标是终点 loss 显著低于恒定臂，
并逼近/越过有门水位（注意 dev seed 与正式 seed 不同，只看方向）。

用法：CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd \
  python scripts/dev_rho_anneal_calibration.py --arms anneal_A const_001
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

DEV_SEEDS = [42, 43, 44]
ROUNDS = 2000

SHARED = dict(
    eta=0.5, mu=0.01, beta=1.0, h=0.8, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    alpha_min=2.0, alpha_max=10.0, exclude_self=True,
    tol=float("inf"),
)

ARMS = {
    "anneal_A": dict(rho=0.1, rho_anneal_end=0.01),
    "anneal_B": dict(rho=0.1, rho_anneal_end=0.003),
    "anneal_C": dict(rho=0.3, rho_anneal_end=0.003),
    "const_001": dict(rho=0.01),
    # 两段式：前 K 轮快降到终点，其后恒定深潜（诊断显示高温地板 ~200-400
    # 轮即到，几何全程调度在高温段停留过久）。
    "twophase_A": dict(rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300),
    "twophase_B": dict(rho=0.1, rho_anneal_end=0.005, rho_anneal_rounds=400),
    # 漂移场强度扫描（低温区净下降速率是无门 vs 有门的主要差距来源；
    # 以下全为分布侧参数，不读取候选评价）。
    "twophase_ds4": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=4.0,
    ),
    "twophase_ds8": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=8.0,
    ),
    "twophase_alpha_hi": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        alpha_min=6.0, alpha_max=20.0,
    ),
    "twophase_ds4_alpha_hi": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=4.0, alpha_min=6.0, alpha_max=20.0,
    ),
    # 归因：ds8 增益是否依赖退火（恒定 rho=0.01 + ds8）；拐点探测 ds16。
    "const_001_ds8": dict(rho=0.01, diffusion_direction_strength=8.0),
    "twophase_ds16": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=16.0,
    ),
    "twophase_ds8_alpha_hi": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=8.0, alpha_min=6.0, alpha_max=20.0,
    ),
    "twophase_ds4_alpha_xhi": dict(
        rho=0.1, rho_anneal_end=0.01, rho_anneal_rounds=300,
        diffusion_direction_strength=4.0, alpha_min=12.0, alpha_max=40.0,
    ),
    # 最优组合去退火：若与 twophase_ds4_alpha_hi 打平，机制收敛为
    # 恒定 rho + 强漂移 + 陡选择（最简形式，无调度组件）。
    "const_001_ds4_alpha_hi": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        alpha_min=6.0, alpha_max=20.0,
    ),
    # 尺度不变选择（结构性方案）：行内标准化后有效温度恒等于 alpha，
    # 无 alpha 调度自由度（alpha_min==alpha_max）。判定标准：恒定标准分
    # alpha 打平/超过调出来的最优递增 alpha 谱系 → 证明"结构而非调参"。
    "si_a2": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=2.0, alpha_max=2.0,
    ),
    "si_a4": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=4.0, alpha_max=4.0,
    ),
    "si_a6": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=6.0, alpha_max=6.0,
    ),
    "si_a8": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=8.0, alpha_max=8.0,
    ),
    "si_a10": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=10.0, alpha_max=10.0,
    ),
    "si_a12": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=12.0, alpha_max=12.0,
    ),
    "si_a16": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=16.0, alpha_max=16.0,
    ),
    # ds 归因：si 下方向强度回到默认 2.0，检验 ds4 是否仍必要（更简配置）。
    "si_a10_ds2": dict(
        rho=0.01, diffusion_direction_strength=2.0,
        selection_scale_invariant=True, alpha_min=10.0, alpha_max=10.0,
    ),
    "si_a12_ds2": dict(
        rho=0.01, diffusion_direction_strength=2.0,
        selection_scale_invariant=True, alpha_min=12.0, alpha_max=12.0,
    ),
    "si_a16_ds2": dict(
        rho=0.01, diffusion_direction_strength=2.0,
        selection_scale_invariant=True, alpha_min=16.0, alpha_max=16.0,
    ),
    "si_a20_ds2": dict(
        rho=0.01, diffusion_direction_strength=2.0,
        selection_scale_invariant=True, alpha_min=20.0, alpha_max=20.0,
    ),
    "si_a24_ds2": dict(
        rho=0.01, diffusion_direction_strength=2.0,
        selection_scale_invariant=True, alpha_min=24.0, alpha_max=24.0,
    ),
    "gate_si_a10": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        selection_scale_invariant=True, alpha_min=10.0, alpha_max=10.0,
        tol=1e-9,
    ),
    # 公平性对照（rho 混淆教训）：有门同配置。"无门越过有门"必须在双方
    # 同参数下成立，否则是配置混淆而非机制差异。
    "gate_ds4_alpha_hi": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        alpha_min=6.0, alpha_max=20.0, tol=1e-9,
    ),
    "gate_ds4_alpha_xhi": dict(
        rho=0.01, diffusion_direction_strength=4.0,
        alpha_min=12.0, alpha_max=40.0, tol=1e-9,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEV_SEEDS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--out", default="outputs/gate_free_self_cooling/dev_rho_anneal.json"
    )
    args = parser.parse_args()

    schema = load_schema("configs/nltcs/schema.yaml")
    queries = load_queries("configs/nltcs/measured_1000query.json")
    marginals = load_marginals("configs/nltcs/init_marginals.json")
    target = np.asarray([q["result"] for q in queries], dtype=float)

    results = {}
    for arm in args.arms:
        per_seed = []
        arm_kwargs = {**SHARED, **ARMS[arm]}  # 臂参数覆盖共享默认
        for seed in args.seeds:
            t0 = time.time()
            _best, diag = run_evolution(
                target=target,
                schema=schema,
                queries=queries,
                marginals=marginals,
                n_records=16181,
                n_rounds=args.rounds,
                seed=seed,
                device="cuda",
                log_every=0,
                return_final_table=True,
                **arm_kwargs,
            )
            final_table = diag.pop("final_table")
            final_answers = evaluate_table(final_table, queries)
            final_loss = float(compute_loss(final_answers, target))
            final_l1 = float(
                compute_normalized_l1(final_answers, target, n_records=16181)
            )
            hist = np.asarray(diag["loss_history"], dtype=float)
            per_seed.append({
                "seed": seed,
                "final_loss": final_loss,
                "final_l1": final_l1,
                "best_loss": float(hist.min()),
                "tail200_mean": float(hist[-200:].mean()),
                "elapsed_sec": round(time.time() - t0, 1),
            })
            print(
                f"[{arm} seed={seed}] final={final_loss:.4g} "
                f"L1={final_l1:.6f} best={hist.min():.4g} "
                f"({per_seed[-1]['elapsed_sec']}s)",
                flush=True,
            )
        results[arm] = {
            "params": ARMS[arm],
            "rounds": args.rounds,
            "seeds": list(args.seeds),
            "per_seed": per_seed,
            "mean_final_loss": float(
                np.mean([r["final_loss"] for r in per_seed])
            ),
            "mean_final_l1": float(np.mean([r["final_l1"] for r in per_seed])),
        }
        print(
            f"== {arm}: mean_final={results[arm]['mean_final_loss']:.4g} "
            f"mean_L1={results[arm]['mean_final_l1']:.6f}",
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
