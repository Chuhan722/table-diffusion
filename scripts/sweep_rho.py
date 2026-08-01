"""
ρ 扫描：single_block 模式在 test_300x10 上找合适的记录参与率。

固定 ε=0.05，200 轮，单种子。
判断标准：接受率维持 60-90% 且最终 loss 最低。
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
EPSILON = 0.05

RHO_GRID = [0.02, 0.05, 0.08, 0.10, 0.15]

OUTPUT_DIR = Path("outputs/rho_sweep")


def run_one(rho: float) -> dict:
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])

    _, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=SEED,
        rho=rho, eta=0.5, mu=0.01,
        update_mode='single_block', epsilon=EPSILON,
        device='numpy', log_every=0,
    )

    losses = diag["loss_history"]
    accepts = diag["accept_history"]
    # 后半程接受率：收敛后期更能反映"是否卡在被拒"
    half = len(accepts) // 2
    return {
        "rho": rho,
        "initial_loss": losses[0],
        "final_loss": diag["best_loss"],
        "loss_reduction_pct": (1 - diag["best_loss"] / losses[0]) * 100,
        "acceptance_rate": sum(accepts) / len(accepts),
        "acceptance_rate_late": sum(accepts[half:]) / len(accepts[half:]),
        "normalized_l1_error": diag["normalized_l1_error"],
        "empty_copy_set_total": sum(diag["empty_copy_set_count_history"]),
        "mutation_attempt_rate_mean": float(
            np.mean(diag["mutation_attempt_rate_history"])),
        "loss_history": losses,
        "accept_history": accepts,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ρ 扫描 | test_300x10 | ε={EPSILON} | {N_ROUNDS} 轮 | seed={SEED}\n")
    print(f"{'ρ':>6} | {'最终loss':>10} | {'降幅':>6} | "
          f"{'接受率':>7} | {'后半接受率':>9} | {'归一L1':>7} | {'空集':>5}")
    print("-" * 72)

    results = []
    for rho in RHO_GRID:
        r = run_one(rho)
        results.append(r)
        print(f"{r['rho']:>6.2f} | {r['final_loss']:>10.2e} | "
              f"{r['loss_reduction_pct']:>5.1f}% | "
              f"{r['acceptance_rate']:>6.1%} | "
              f"{r['acceptance_rate_late']:>8.1%} | "
              f"{r['normalized_l1_error']:>7.4f} | "
              f"{r['empty_copy_set_total']:>5d}")

    # 推荐：接受率后半程在 [0.6, 0.9] 内、loss 最低
    healthy = [r for r in results if 0.6 <= r['acceptance_rate_late'] <= 0.9]
    pool = healthy if healthy else results
    best = min(pool, key=lambda r: r['final_loss'])
    print("\n" + "=" * 72)
    print(f"推荐 ρ = {best['rho']} "
          f"(最终 loss {best['final_loss']:.2e}, "
          f"后半接受率 {best['acceptance_rate_late']:.1%})")
    if not healthy:
        print("⚠ 没有 ρ 的后半接受率落在 [0.6,0.9]，需扩大或平移扫描范围")

    with open(OUTPUT_DIR / "rho_sweep_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {OUTPUT_DIR / 'rho_sweep_results.json'}")


if __name__ == "__main__":
    main()
