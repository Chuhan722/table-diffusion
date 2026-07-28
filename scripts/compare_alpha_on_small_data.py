"""
小数据对比：三组 alpha 参数在 test_300x10 上的表现
- α 0.5→4.0 (旧参数)
- α 1.5→6.0 (新推荐)
- α 2.0→10.0 (高锐度)

对比指标：loss、多样性、收敛速度
"""
import numpy as np
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
import pandas as pd

SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
REAL_DATA_PATH = "data/test_300x10/test_300x10.csv"

N_RECORDS = 300
N_ROUNDS = 100
SEED = 42

COMMON = {
    "beta": 1.0,
    "h": 0.8,
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
    "device": "cuda",
    "eval_method": "vectorized",
    "batch_size": 256,
    "init_method": "marginal",
    "log_every": 0,
    "distance_mode": "geometric",
    "lambda_param": 0.5,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}

CONFIGS = [
    {"name": "α0.5→4", "alpha_min": 0.5, "alpha_max": 4.0},
    {"name": "α1.5→6", "alpha_min": 1.5, "alpha_max": 6.0},
    {"name": "α2→10", "alpha_min": 2.0, "alpha_max": 10.0},
]


def analyze_diversity(synth_df, real_df):
    """分析多样性指标"""
    n_synth = len(synth_df)
    n_real = len(real_df)

    synth_unique = len(synth_df.drop_duplicates())
    real_unique = len(real_df.drop_duplicates())

    synth_dup_rate = (n_synth - synth_unique) / n_synth * 100
    real_dup_rate = (n_real - real_unique) / n_real * 100

    synth_vc = synth_df.value_counts()
    real_vc = real_df.value_counts()

    synth_max_pct = synth_vc.iloc[0] / n_synth * 100 if len(synth_vc) > 0 else 0
    real_max_pct = real_vc.iloc[0] / n_real * 100 if len(real_vc) > 0 else 0

    # 真实 top-5 覆盖
    real_top5 = real_vc.head(5)
    covered = sum(1 for idx in real_top5.index if idx in synth_vc.index)

    return {
        "synth_unique": synth_unique,
        "synth_dup_rate": synth_dup_rate,
        "real_unique": real_unique,
        "real_dup_rate": real_dup_rate,
        "synth_max_pct": synth_max_pct,
        "real_max_pct": real_max_pct,
        "real_top5_covered": covered,
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)
    real_data = pd.read_csv(REAL_DATA_PATH)

    print("===== test_300x10 数据集 =====")
    print(f"记录数: {N_RECORDS}, 特征数: {len(schema.attributes)}, 查询数: {len(queries)}")
    print(f"真实数据唯一记录: {len(real_data.drop_duplicates())}")
    print(f"真实数据重复率: {(1 - len(real_data.drop_duplicates())/len(real_data))*100:.2f}%\n")

    results = []

    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"运行：{name}")
        print(f"{'='*60}")

        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS,
            n_rounds=N_ROUNDS,
            seed=SEED,
            marginals=marginals,
            **COMMON,
            alpha_min=config["alpha_min"],
            alpha_max=config["alpha_max"],
        )

        lh = diag["loss_history"]
        best_loss = diag["best_loss"]
        nl1 = diag["normalized_l1_error"]
        accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])

        # 多样性分析
        diversity = analyze_diversity(best_S, real_data)

        print(f"\n初始 loss: {lh[0]:.2e}")
        print(f"最优 loss: {best_loss:.2e}")
        print(f"下降比例: {(1 - best_loss / lh[0]) * 100:.1f}%")
        print(f"归一化 L1: {nl1:.4f}")
        print(f"接受率: {accept_rate * 100:.1f}%")
        print(f"\n多样性:")
        print(f"  合成唯一: {diversity['synth_unique']} ({diversity['synth_unique']/N_RECORDS*100:.1f}%)")
        print(f"  真实唯一: {diversity['real_unique']} ({diversity['real_unique']/N_RECORDS*100:.1f}%)")
        print(f"  合成重复率: {diversity['synth_dup_rate']:.2f}% (真实: {diversity['real_dup_rate']:.2f}%)")
        print(f"  合成最高频: {diversity['synth_max_pct']:.2f}% (真实: {diversity['real_max_pct']:.2f}%)")
        print(f"  真实 Top-5 覆盖: {diversity['real_top5_covered']}/5")

        results.append({
            "name": name,
            "best_loss": best_loss,
            "nl1": nl1,
            "accept_rate": accept_rate,
            "reduction_pct": (1 - best_loss / lh[0]) * 100,
            **diversity,
        })

    # 汇总对比
    print(f"\n\n{'='*85}")
    print("三方对比汇总")
    print(f"{'='*85}")

    print(f"\n{'指标':<20} {'α0.5→4':<20} {'α1.5→6':<20} {'α2→10':<20}")
    print('-' * 85)

    def row(label, key, fmt):
        vals = [fmt.format(r[key]) for r in results]
        print(f"{label:<20} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}")

    row("最优 loss", "best_loss", "{:.2e}")
    row("归一化 L1", "nl1", "{:.4f}")
    row("下降比例%", "reduction_pct", "{:.1f}")
    row("接受率", "accept_rate", "{:.3f}")

    print(f"\n{'多样性':<20}")
    row("  合成唯一", "synth_unique", "{:d}")
    row("  合成重复率%", "synth_dup_rate", "{:.2f}")
    row("  合成最高频%", "synth_max_pct", "{:.2f}")
    row("  Top-5覆盖", "real_top5_covered", "{:d}/5")

    print(f"\n参考：真实数据唯一 {results[0]['real_unique']}, 重复率 {results[0]['real_dup_rate']:.2f}%, 最高频 {results[0]['real_max_pct']:.2f}%")


if __name__ == "__main__":
    main()
