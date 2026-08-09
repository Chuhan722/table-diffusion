#!/usr/bin/env python3
"""A2：Probe 模式收敛验证（test_300x10, 单 seed 探路）

目的
----
验证 probe 模式在足够长轮数（4000 轮）下能否触发"经验平台"
（连续 2 次探测全失败），自动检测到收敛并停机。

冒烟（A1）结论：W=30/P=2/eps_L1=1e-3 参数稳健，1500 轮触发 2 次探测但
未触发经验平台（系统还在缓慢下降）。A2 加长到 4000 轮，观察平台能否触发。

参数（冒烟已验证，本次不变）
------------------------------
    probe_block_candidate_budget = 30    块大小（候选评估次数）
    probe_P                      = 2     停滞判定块数
    probe_H_candidate_budget     = 100   探测预算（候选评估次数/分支）
    probe_s                      = 0.2   归一化步长
    probe_C                      = 1     冷却块数
    probe_eps_L1                 = 1e-3  停滞/平台判定阈值
    alpha_min/max                = 2/10  α 范围

用法
----
    CUDA_VISIBLE_DEVICES=1 python experiments/probe_convergence_a2.py
    CUDA_VISIBLE_DEVICES=1 python experiments/probe_convergence_a2.py --seed 42 --rounds 4000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema

# ── 数据集路径 ────────────────────────────────────────────────
SCHEMA_PATH   = "configs/test_300x10/schema.yaml"
QUERY_PATH    = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH= "configs/test_300x10/init_marginals.json"
N_RECORDS     = 300

# ── 冻结参数（与阶段 3 / 冒烟一致）────────────────────────────
FROZEN_PARAMS = dict(
    beta=1.0,
    h=0.8,
    rho=0.01,
    eta=0.5,
    mu=0.01,
    eval_method="vectorized",
    batch_size=256,
    init_method="marginal",
    distance_mode="geometric",
    lambda_param=0.5,
    delta=0.05,
    winsorize_quantiles=(0.01, 0.99),
    exclude_self=True,
    max_retries=0,
    residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
)

# ── Probe 参数（用户调整：α ∈ [5, 12]）────────────────────────────
PROBE_PARAMS = dict(
    alpha_schedule_mode="probe",
    alpha_min=5.0,         # 起点更贪（2→5）
    alpha_max=12.0,        # 上限更高（10→12）
    probe_block_candidate_budget=30,   # W=30 候选评估次数/块
    probe_P=2,             # 停滞阈值：连续 2 块无改善
    probe_H_candidate_budget=100,      # 探测预算：100 候选评估次数/分支
    probe_s=0.2,           # 步长：0.2（α 每步约 ±1.4）
    probe_C=1,             # 冷却：1 块
    probe_eps_L1=1e-3,     # 停滞/平台判定阈值
)


def load_dataset():
    schema    = load_schema(SCHEMA_PATH)
    queries   = load_queries(QUERY_PATH)
    target    = np.asarray([q["result"] for q in queries], dtype=float)
    marginals = load_marginals(MARGINALS_PATH)
    return schema, queries, target, marginals


def main():
    parser = argparse.ArgumentParser(description="A2 Probe 收敛验证")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--rounds",     type=int, default=4000)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/probe_convergence_test/a2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A2：Probe 收敛验证  test_300x10")
    print("=" * 70)
    print(f"种子:        {args.seed}")
    print(f"轮数上限:    {args.rounds}")
    print(f"设备:        {args.device}")
    print(f"块大小 W:    {PROBE_PARAMS['probe_block_size']}")
    print(f"停滞阈值 P:  {PROBE_PARAMS['probe_P']}")
    print(f"探测预算 H:  {PROBE_PARAMS['probe_H']} 轮/分支")
    print(f"步长 s:      {PROBE_PARAMS['probe_s']}")
    print(f"eps_L1:      {PROBE_PARAMS['probe_eps_L1']}")
    print(f"α 范围:      [{PROBE_PARAMS['alpha_min']}, {PROBE_PARAMS['alpha_max']}]")
    print(f"输出目录:    {output_dir}")
    print()

    print("加载数据集...")
    schema, queries, target, marginals = load_dataset()
    print(f"  Schema:  {len(schema.attributes)} 属性")
    print(f"  Queries: {len(queries)} 个")
    print(f"  Records: {N_RECORDS}")
    print()

    print("开始运行 probe 模式...")
    start_time = datetime.now()

    syn_table, diag = run_evolution(
        target,
        queries,
        schema,
        n_records=N_RECORDS,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
        marginals=marginals,
        log_every=100,        # 每 100 轮打一行进度
        **FROZEN_PARAMS,
        **PROBE_PARAMS,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── 结果汇总 ─────────────────────────────────────────────
    probe_history  = diag.get("probe_history", [])
    alpha_history  = diag.get("alpha_history", [])
    rounds_run     = diag.get("rounds_run", args.rounds)
    final_L1       = float(diag["normalized_l1_error"])
    best_loss_val  = float(diag["best_loss"])
    stopped_early  = diag.get("stopped_early", False)

    # 判断是否因平台触发：stopped_early=True 且轮数 < n_rounds
    converged_by_plateau = stopped_early and (rounds_run < args.rounds)

    n_probes        = len(probe_history)
    n_failed_probes = sum(1 for p in probe_history if p.get("all_failed", False))

    print()
    print("=" * 70)
    print("运行完成")
    print("=" * 70)
    print(f"实际轮数:        {rounds_run}/{args.rounds}")
    print(f"墙钟时间:        {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"最终 L1:         {final_L1:.6f}")
    print(f"Best loss:       {best_loss_val:.2f}")
    print(f"探测触发次数:    {n_probes}")
    print(f"探测失败次数:    {n_failed_probes}")
    print(f"触发经验平台:    {'✅ 是（自动停机）' if converged_by_plateau else '❌ 否（跑完 n_rounds 上限）'}")
    print()

    if probe_history:
        print("探测历史:")
        for i, p in enumerate(probe_history, 1):
            print(f"  [{i}] Round {p['round']:4d} | "
                  f"α {p['trigger_alpha']:.2f}→{p.get('winner_alpha', '?'):.2f} | "
                  f"Winner: {p['winner']:4s} | "
                  f"All failed: {p.get('all_failed', '?')} | "
                  f"累计失败: {p.get('failed_probes', '?')}")
    else:
        print("  ⚠️  没有触发过探测")

    if alpha_history:
        alphas = alpha_history
        print(f"\nα 轨迹:  初始={alphas[0]:.2f}  最终={alphas[-1]:.2f}  "
              f"min={min(alphas):.2f}  max={max(alphas):.2f}")

    # ── A2 判定 ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("A2 判定")
    print("=" * 70)
    if converged_by_plateau:
        print("✅ 经验平台触发，probe 自动收敛停机")
        print(f"   停机轮数: {rounds_run}")
        print(f"   停机 L1:  {final_L1:.6f}")
        print("   → 下一步：补充 seed 43/44，确认多 seed 稳定性，或直接上 nltcs")
    elif n_probes == 0:
        print(f"⚠️  {args.rounds} 轮内未触发任何探测")
        print("   → 说明系统一直在实质改善，eps_L1=1e-3 可能偏紧或轮数仍不够")
        print("   → 建议：检查 L1 曲线末段降幅，考虑调大 eps_L1 或再加轮数")
    else:
        print(f"⚠️  触发了 {n_probes} 次探测（含 {n_failed_probes} 次失败），但未到平台")
        print(f"   末段 L1={final_L1:.6f}，loss 还在缓慢下降")
        print(f"   → 说明 {args.rounds} 轮不够收敛，或 eps_L1=1e-3 偏紧")
        print("   → 建议：查末段每块降幅，决定调大 eps_L1 还是再加轮数")

    # ── 保存结果 ─────────────────────────────────────────────
    result = {
        "experiment": "probe_convergence_a2",
        "dataset":    "test_300x10",
        "seed":       args.seed,
        "n_rounds_limit":  args.rounds,
        "rounds_run":      rounds_run,
        "elapsed_sec":     elapsed,
        "params": {**PROBE_PARAMS, "W": PROBE_PARAMS["probe_block_size"]},
        "final_L1":        final_L1,
        "best_loss":       best_loss_val,
        "n_probes":        n_probes,
        "n_failed_probes": n_failed_probes,
        "converged_by_plateau": converged_by_plateau,
        "stopped_early":   stopped_early,
        "probe_history":   probe_history,
        "alpha_initial":   alpha_history[0] if alpha_history else None,
        "alpha_final":     alpha_history[-1] if alpha_history else None,
    }

    out_file = output_dir / f"seed_{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "__float__") else str(o))

    print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
