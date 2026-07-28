"""
小数据 test_300x10 上 α 三配置多种子对比。

目的：核实文档 3.3 节"α1.5→6 比 α2→10 好 14%"是否稳健，还是单种子噪声。
test_300x10 是 0% 重复（全唯一）的分布，与 nltcs（83% 重复）相反，
用来验证 α 配置的跨数据集稳定性。

小数据 100 轮秒级，直接跑 5 种子，看 α 差异是否稳定 + 配对 t 检验。
"""
import numpy as np
import pandas as pd
from scipy import stats

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals

SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
REAL_DATA_PATH = "data/test_300x10/test_300x10.csv"

N_RECORDS = 300
N_ROUNDS = 100
SEEDS = [0, 1, 2, 3, 42]

COMMON = {
    "beta": 1.0, "h": 0.8, "rho": 0.01, "eta": 0.5, "mu": 0.01,
    "device": "cuda", "eval_method": "vectorized", "batch_size": 256,
    "init_method": "marginal", "log_every": 0,
    "distance_mode": "geometric", "lambda_param": 0.5,
    "delta": 0.05, "winsorize_quantiles": (0.01, 0.99),
}

CONFIGS = [
    {"name": "a0p5_4", "alpha_min": 0.5, "alpha_max": 4.0},
    {"name": "a1p5_6", "alpha_min": 1.5, "alpha_max": 6.0},
    {"name": "a2_10", "alpha_min": 2.0, "alpha_max": 10.0},
]


def diversity(synth_df, real_df):
    n = len(synth_df)
    vc = synth_df.value_counts()
    real_vc = real_df.value_counts()
    real_top = set(real_vc.head(10).index)
    synth_top = set(vc.head(10).index)
    return {
        "n_unique": int(len(vc)),
        "dup_rate_pct": float((n - len(vc)) / n * 100),
        "max_freq_pct": float(vc.iloc[0] / n * 100),
        "top10_cover": int(len(real_top & synth_top)),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)
    real_df = pd.read_csv(REAL_DATA_PATH)

    losses = {c["name"]: [] for c in CONFIGS}
    l1s = {c["name"]: [] for c in CONFIGS}
    last_div = {}

    for seed in SEEDS:
        for cfg in CONFIGS:
            best_S, diag = run_evolution(
                target, queries, schema,
                n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=seed,
                marginals=marginals,
                alpha_min=cfg["alpha_min"], alpha_max=cfg["alpha_max"], **COMMON,
            )
            losses[cfg["name"]].append(float(diag["best_loss"]))
            l1s[cfg["name"]].append(float(diag["normalized_l1_error"]))
            synth_df = pd.DataFrame(best_S, columns=real_df.columns)
            last_div[cfg["name"]] = diversity(synth_df, real_df)
        print(f"seed={seed} done")

    print("\n" + "=" * 70)
    print(f"test_300x10 小数据 α 多种子对比（{N_ROUNDS} 轮, seeds={SEEDS}）")
    print("真实分布: 300 条全唯一, 0% 重复, Top-1 仅 0.33%")
    print("=" * 70)
    for c in CONFIGS:
        name = c["name"]
        arr = np.array(losses[name])
        l1arr = np.array(l1s[name])
        d = last_div[name]
        print(f"\n{name} (α{c['alpha_min']}→{c['alpha_max']}):")
        print(f"  loss: {arr.mean():.3e} ± {arr.std(ddof=1):.3e}  (min {arr.min():.3e}, max {arr.max():.3e})")
        print(f"  L1  : {l1arr.mean():.4f} ± {l1arr.std(ddof=1):.4f}")
        print(f"  多样性(seed42): 唯一 {d['n_unique']}, 重复 {d['dup_rate_pct']:.1f}%, "
              f"最高频 {d['max_freq_pct']:.2f}%, Top10覆盖 {d['top10_cover']}/10")

    # 配对 t 检验：a1p5_6 vs a2_10
    a156 = np.array(losses["a1p5_6"])
    a210 = np.array(losses["a2_10"])
    t, p = stats.ttest_rel(a156, a210)
    print("\n" + "=" * 70)
    print(f"a1p5_6 vs a2_10 配对 t 检验: t={t:.3f}, p={p:.4f}")
    diff = (a156.mean() - a210.mean()) / a210.mean() * 100
    print(f"a1p5_6 相对 a2_10 loss 差异: {diff:+.2f}%")
    print("显著" if p < 0.05 else "不显著（落在种子噪声内）")


if __name__ == "__main__":
    main()
