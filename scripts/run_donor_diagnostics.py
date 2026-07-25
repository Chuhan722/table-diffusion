"""
临时诊断实验：只跑 squared 和 linear 各 1 个种子

用于生成 donor 诊断数据（适应度和距离历史）
"""
import os
from datetime import datetime

import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run
from table_diffevo.marginals import load_marginals


# ========== 配置 ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
N_RECORDS = 16181
N_ROUNDS = 1000
SEED = 0  # 单种子

DISTANCE_MODES = ['squared', 'linear']  # 只跑两种
DEVICE = 'cuda'
EVAL_METHOD = 'vectorized'
BATCH_SIZE = 256
INIT_METHOD = 'marginal'
BETA = 1.0
H = 0.8
RHO = 0.01
ETA = 0.5
MU = 0.01
LOG_EVERY = 50
# ==========================


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    # 创建输出目录
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    parent_dir = os.path.join("outputs", f"donor_diagnostics_{stamp}")
    os.makedirs(parent_dir, exist_ok=True)

    print(f"诊断实验：nltcs × {N_ROUNDS} 轮 × 种子 {SEED}")
    print(f"结果将保存到: {parent_dir}/\n")

    for distance_mode in DISTANCE_MODES:
        print(f"\n{'='*60}")
        print(f"距离模式: {distance_mode.upper()}")
        print(f"{'='*60}")

        best_S, diagnostics = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS,
            n_rounds=N_ROUNDS,
            seed=SEED,
            beta=BETA, h=H, rho=RHO, eta=ETA, mu=MU,
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=LOG_EVERY,
            distance_mode=distance_mode,
        )

        # 保存
        mode_dir = os.path.join(parent_dir, distance_mode)
        run_dir = save_run(best_S, diagnostics, run_dir=mode_dir)

        lh = diagnostics["loss_history"]
        print(f"\n  初始 loss : {lh[0]:.2e}  →  最优 loss : {diagnostics['best_loss']:.2e}")
        print(f"  平均归一化L1: {diagnostics['normalized_l1_error']:.4f}")
        print(f"  耗时: {diagnostics['elapsed_sec']:.1f}s")
        print(f"  已存: {run_dir}/")

    print(f"\n{'='*60}")
    print(f"诊断实验完成！")
    print(f"{'='*60}")
    print(f"\n分析诊断数据：")
    print(f"python scripts/analyze_donor_diagnostics.py \\")
    print(f"    {parent_dir}/squared/diagnostics.json \\")
    print(f"    {parent_dir}/linear/diagnostics.json")


if __name__ == "__main__":
    main()
