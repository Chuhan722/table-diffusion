"""
扩散演化运行入口

一键跑一组多种子演化并自动落盘：
    conda run -p ./.conda python scripts/run.py

调参方式：直接修改下面"参数配置"区的常量，改完再跑。
不需要命令行传参——这个脚本就是你反复复用的实验入口。

流程：
1. 加载 schema、queries，从 queries 取 target（各查询真实计数）
2. 建日期时间父文件夹 outputs/YYYY-MM-DD_HHMM/
3. 对 SEEDS 里每个种子跑一遍 run_evolution，各存父/{顺序}-{种子}/
4. 汇总各种子的 best_loss / 归一化L1（均值±std/min/max），
   存父/summary.json 并打印
"""
import os

import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run, create_parent_dir, save_summary
from table_diffevo.marginals import load_marginals


# ========== 参数配置（调参改这里） ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"

N_RECORDS = 16181      # 合成表记录条数（nltcs train 集）
N_ROUNDS = 1000        # 最大轮数 T
SEEDS = [0, 1, 2]      # 随机种子列表（多种子跑，看结果波动；单种子写 [0] 即可）
LOG_EVERY = 50         # 逐轮进度打印频率（0=每轮 | >0=每N轮，长实验建议50）

# 计算设备（新增）
DEVICE = 'cuda'        # 'cuda'=GPU加速 | 'numpy'=原NumPy | 'cpu'=PyTorch CPU
# 注：cuda 默认用卡 0。若卡 0 被占，跑前加环境变量指定空闲卡（先 nvidia-smi 看哪块空）：
#     CUDA_VISIBLE_DEVICES=1 conda run -p ./.conda python scripts/run.py
# 代码无需改动——指定的卡在程序里自动成为 cuda:0。卡号写错会降级到 CPU（很慢）。

# 查询评价方式（性能开关，不改变结果，仅改变实现）
#   'vectorized'=向量化+分块（快，默认）| 'legacy'=旧逐查询pandas（慢，用于对拍/应急）
EVAL_METHOD = 'vectorized'
# 向量化评价的分块大小（一次算多少个查询），仅 EVAL_METHOD='vectorized' 生效
# 内存峰值 ∝ N × BATCH_SIZE；越大越快但越吃内存
BATCH_SIZE = 256

# 初始化方式（新增）
INIT_METHOD = 'marginal'  # 'random'=纯随机 | 'marginal'=按1-way边缘初始化
# 边缘测量文件（仅 INIT_METHOD='marginal' 时生效）
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

BETA = 1.0             # 选择强度（固定值）
H = 0.8                # 邻域尺度（固定值）
RHO = 0.01             # 记录参与率（固定值）
ETA = 0.5              # 块复制率（固定值）
MU = 0.01              # 变异率（固定值）
# ===========================================


def _run_params():
    """本次运行的参数快照，写入 summary.json 便于回溯。"""
    return {
        "schema_path": SCHEMA_PATH,
        "query_path": QUERY_PATH,
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "device": DEVICE,
        "eval_method": EVAL_METHOD,
        "batch_size": BATCH_SIZE,
        "init_method": INIT_METHOD,
        "beta": BETA, "h": H, "rho": RHO, "eta": ETA, "mu": MU,
    }


def _aggregate(values):
    """一组标量的均值/标准差/最小/最大。"""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])

    # 加载边缘测量（仅 init_method='marginal' 时需要）
    marginals = None
    if INIT_METHOD == 'marginal':
        marginals = load_marginals(MARGINALS_PATH)

    # 多种子：统一套一层日期时间父文件夹，各种子存 父/{顺序}-{种子}/
    parent_dir = create_parent_dir()
    per_seed = []

    for i, seed in enumerate(SEEDS):
        print(f"\n===== 种子 {seed}（{i + 1}/{len(SEEDS)}）=====")
        best_S, diagnostics = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS,
            n_rounds=N_ROUNDS,
            seed=seed,
            beta=BETA, h=H, rho=RHO, eta=ETA, mu=MU,
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=LOG_EVERY,
        )
        sub_name = f"{i}-{seed}"
        run_dir = save_run(best_S, diagnostics,
                           run_dir=os.path.join(parent_dir, sub_name))

        lh = diagnostics["loss_history"]
        print(f"  初始 loss : {lh[0]:.1f}  →  最优 loss : {diagnostics['best_loss']:.1f}")
        print(f"  平均归一化L1: {diagnostics['normalized_l1_error']:.4f}"
              f" | 中位: {diagnostics['normalized_l1_median']:.4f}"
              f" | P90: {diagnostics['normalized_l1_p90']:.4f}"
              f" | 最大: {diagnostics['normalized_l1_max']:.4f}")
        print(f"  跑了轮数  : {diagnostics['rounds_run']}"
              f"（提前停止={diagnostics['stopped_early']}）"
              f" | 耗时: {diagnostics['elapsed_sec']:.1f}s"
              f"（{diagnostics['sec_per_round'] * 1000:.0f}ms/轮） | 已存: {run_dir}/")

        per_seed.append({
            "seed": seed,
            "run_dir": sub_name,
            "best_loss": diagnostics["best_loss"],
            "normalized_l1_error": diagnostics["normalized_l1_error"],
            "elapsed_sec": diagnostics["elapsed_sec"],
        })

    # 汇总：均值±标准差/min/max，存 summary.json 并打印
    summary = {
        "params": _run_params(),
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": {
            "best_loss": _aggregate([s["best_loss"] for s in per_seed]),
            "normalized_l1_error": _aggregate(
                [s["normalized_l1_error"] for s in per_seed]),
            "elapsed_sec": _aggregate([s["elapsed_sec"] for s in per_seed]),
        },
    }
    save_summary(parent_dir, summary)

    bl = summary["aggregate"]["best_loss"]
    nl = summary["aggregate"]["normalized_l1_error"]
    print(f"\n===== 多种子汇总（{len(SEEDS)} 个种子, {N_ROUNDS} 轮, {INIT_METHOD} init）=====")
    print(f"  最优 loss    : 均值 {bl['mean']:.3e} ± {bl['std']:.2e}"
          f"  (min {bl['min']:.3e}, max {bl['max']:.3e})")
    print(f"  平均归一化L1 : 均值 {nl['mean']:.4f} ± {nl['std']:.4f}"
          f"  (min {nl['min']:.4f}, max {nl['max']:.4f})")
    el = summary["aggregate"]["elapsed_sec"]
    print(f"  单种子耗时   : 均值 {el['mean']:.1f}s"
          f"  (min {el['min']:.1f}s, max {el['max']:.1f}s)")
    print(f"  结果目录     : {parent_dir}/")


if __name__ == "__main__":
    main()
