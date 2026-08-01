"""
nltcs 上 single_block 参数扫描（ρ 和 ε）。

用法（先设环境变量选空闲卡）：
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_nltcs.py <phase>

phase:
    time    阶段0：ρ=0.08 跑 20 轮，实测每轮耗时
    rho     阶段1：固定 ε=0.05，扫 ρ，500 轮
    eps     阶段2：固定 ρ=RHO_STAR，扫 ε，500 轮
"""
import sys
import json
import numpy as np
from pathlib import Path

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals


SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
N_RECORDS = 16181
SEED = 0

# 扫描网格（往低侧多覆盖：nltcs 接受率可能比小数据更早崩）
RHO_GRID = [0.03, 0.05, 0.08, 0.10]
EPS_FIXED_FOR_RHO = 0.05
RHO_STAR = 0.10          # 阶段1定：nltcs 接受率不崩，L1 单调降到 0.10；空集高是收手信号
EPS_GRID = [0.0, 0.05, 0.10, 0.20]  # 往上探：空集高，看变异能否把浪费转成有效改动

SWEEP_ROUNDS = 500
OUTPUT_DIR = Path("outputs/nltcs_sweep")


def run_one(rho, epsilon, n_rounds, seed=SEED):
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    _, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS, n_rounds=n_rounds, seed=seed,
        rho=rho, eta=0.5, mu=0.01,
        update_mode='single_block', epsilon=epsilon,
        device='cuda',
        init_method='marginal', marginals=marginals,
        distance_mode='geometric',
        log_every=99999,
    )
    return diag


def summarize(diag, rho, epsilon):
    losses = diag["loss_history"]
    accepts = diag["accept_history"]
    half = len(accepts) // 2
    return {
        "rho": rho, "epsilon": epsilon,
        "initial_loss": losses[0],
        "final_loss": diag["best_loss"],
        "loss_reduction_pct": (1 - diag["best_loss"] / losses[0]) * 100,
        "acceptance_rate": sum(accepts) / len(accepts),
        "acceptance_rate_late": sum(accepts[half:]) / len(accepts[half:]),
        "normalized_l1_error": diag["normalized_l1_error"],
        "mutation_attempt_rate_mean": float(
            np.mean(diag["mutation_attempt_rate_history"])),
        "empty_copy_set_total": sum(diag["empty_copy_set_count_history"]),
        "sec_per_round": diag["sec_per_round"],
        "loss_history": losses,
        "accept_history": accepts,
    }


def phase_time():
    print("阶段0：ρ=0.08, ε=0.05, 20 轮，实测耗时\n")
    diag = run_one(0.08, 0.05, n_rounds=20)
    spr = diag["sec_per_round"]
    print(f"每轮 {spr*1000:.0f}ms | 500轮≈{spr*500/60:.1f}分 | "
          f"1000轮≈{spr*1000/60:.1f}分")
    print(f"扫描一阶段(4组×{SWEEP_ROUNDS}轮)≈{spr*SWEEP_ROUNDS*4/60:.0f}分")


def phase_rho():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"阶段1：ρ 扫描 | ε={EPS_FIXED_FOR_RHO} | {SWEEP_ROUNDS}轮 | seed={SEED}\n")
    print(f"{'ρ':>6} | {'最终loss':>10} | {'降幅':>6} | {'接受率':>7} | "
          f"{'后半接受率':>9} | {'归一L1':>7} | {'变异率':>7} | {'空集':>5}")
    print("-" * 82)
    results = []
    for rho in RHO_GRID:
        r = summarize(run_one(rho, EPS_FIXED_FOR_RHO, SWEEP_ROUNDS),
                      rho, EPS_FIXED_FOR_RHO)
        results.append(r)
        print(f"{r['rho']:>6.2f} | {r['final_loss']:>10.2e} | "
              f"{r['loss_reduction_pct']:>5.1f}% | {r['acceptance_rate']:>6.1%} | "
              f"{r['acceptance_rate_late']:>8.1%} | {r['normalized_l1_error']:>7.4f} | "
              f"{r['mutation_attempt_rate_mean']:>7.4f} | {r['empty_copy_set_total']:>5d}")
    with open(OUTPUT_DIR / "rho_sweep.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {OUTPUT_DIR / 'rho_sweep.json'}")


def phase_eps():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"阶段2：ε 扫描 | ρ={RHO_STAR} | {SWEEP_ROUNDS}轮 | seed={SEED}\n")
    print(f"{'ε':>6} | {'最终loss':>10} | {'降幅':>6} | {'接受率':>7} | "
          f"{'后半接受率':>9} | {'归一L1':>7} | {'变异率':>7} | {'空集':>5}")
    print("-" * 82)
    results = []
    for eps in EPS_GRID:
        r = summarize(run_one(RHO_STAR, eps, SWEEP_ROUNDS), RHO_STAR, eps)
        results.append(r)
        print(f"{r['epsilon']:>6.2f} | {r['final_loss']:>10.2e} | "
              f"{r['loss_reduction_pct']:>5.1f}% | {r['acceptance_rate']:>6.1%} | "
              f"{r['acceptance_rate_late']:>8.1%} | {r['normalized_l1_error']:>7.4f} | "
              f"{r['mutation_attempt_rate_mean']:>7.4f} | {r['empty_copy_set_total']:>5d}")
    with open(OUTPUT_DIR / "eps_sweep.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {OUTPUT_DIR / 'eps_sweep.json'}")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "time"
    {"time": phase_time, "rho": phase_rho, "eps": phase_eps}[phase]()
