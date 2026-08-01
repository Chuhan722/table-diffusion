"""
ε 扫描：single_block 模式在 test_300x10 上找合适的变异占比。

固定 ρ=0.08（由 ρ 扫描选出），200 轮，单种子。
判断变异（ε>0）相对纯复制（ε=0）是否有益。
"""
import json
import numpy as np
from pathlib import Path

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
N_RECORDS = 300
N_ROUNDS = 200
SEED = 42
RHO = 0.08

EPSILON_GRID = [0.0, 0.01, 0.05, 0.1, 0.2]

OUTPUT_DIR = Path("outputs/epsilon_sweep")


def run_one(epsilon: float) -> dict:
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])

    _, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=SEED,
        rho=RHO, eta=0.5, mu=0.01,
        update_mode='single_block', epsilon=epsilon,
        device='numpy', log_every=99999,  # 不打印逐轮
    )

    losses = diag["loss_history"]
    accepts = diag["accept_history"]
    half = len(accepts) // 2
    return {
        "epsilon": epsilon,
        "initial_loss": losses[0],
        "final_loss": diag["best_loss"],
        "loss_reduction_pct": (1 - diag["best_loss"] / losses[0]) * 100,
        "acceptance_rate": sum(accepts) / len(accepts),
        "acceptance_rate_late": sum(accepts[half:]) / len(accepts[half:]),
        "normalized_l1_error": diag["normalized_l1_error"],
        "mutation_attempt_rate_mean": float(
            np.mean(diag["mutation_attempt_rate_history"])),
        "copy_attempt_rate_mean": float(
            np.mean(diag["copy_attempt_rate_history"])),
        "loss_history": losses,
        "accept_history": accepts,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ε 扫描 | test_300x10 | ρ={RHO} | {N_ROUNDS} 轮 | seed={SEED}\n")
    print(f"{'ε':>6} | {'最终loss':>10} | {'降幅':>6} | "
          f"{'接受率':>7} | {'后半接受率':>9} | {'归一L1':>7} | {'实际变异率':>9}")
    print("-" * 76)

    results = []
    for eps in EPSILON_GRID:
        r = run_one(eps)
        results.append(r)
        print(f"{r['epsilon']:>6.2f} | {r['final_loss']:>10.2e} | "
              f"{r['loss_reduction_pct']:>5.1f}% | "
              f"{r['acceptance_rate']:>6.1%} | "
              f"{r['acceptance_rate_late']:>8.1%} | "
              f"{r['normalized_l1_error']:>7.4f} | "
              f"{r['mutation_attempt_rate_mean']:>9.4f}")

    best = min(results, key=lambda r: r['final_loss'])
    baseline = next(r for r in results if r['epsilon'] == 0.0)
    print("\n" + "=" * 76)
    print(f"最优 ε = {best['epsilon']} (最终 loss {best['final_loss']:.2e})")
    print(f"纯复制基线 ε=0.0: loss {baseline['final_loss']:.2e}")
    improve = (1 - best['final_loss'] / baseline['final_loss']) * 100
    if best['epsilon'] == 0.0:
        print("→ 变异无益，纯复制最优（当前数据/初始化下多样性已足够）")
    else:
        print(f"→ 变异有益，ε={best['epsilon']} 比纯复制低 {improve:.1f}%")

    with open(OUTPUT_DIR / "epsilon_sweep_results.json", 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {OUTPUT_DIR / 'epsilon_sweep_results.json'}")


if __name__ == "__main__":
    main()
