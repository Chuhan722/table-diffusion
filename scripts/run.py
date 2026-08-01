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
N_ROUNDS = 1000        # 1-way 起点较弱，需更多轮；与旧 marginal 实验一致
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

# 初始化方式
# 'random'=纯随机 | 'marginal'=按1-way边缘 | 'pairwise_maxent'=二阶最大熵
INIT_METHOD = 'marginal'  # 用回 1-way 边缘初始化
# 边缘测量文件（仅 INIT_METHOD='marginal' 时生效）
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

# pairwise_maxent 专用参数（其他初始化方式忽略）
MAXENT_MAX_STATES = 1_000_000  # 联合状态枚举上限；nltcs 为 2^16=65,536
MAXENT_MAX_SWEEPS = 200        # IPF 最大扫描轮数
MAXENT_TOL = 1e-8              # 最大二阶单元概率误差阈值

# 抽样模式（可选：'linear', 'squared', 'multiplicative', 'none', 'geometric'）
DISTANCE_MODE = 'geometric'   # 实验最优（α2→10, λ0.5），配套参数见下方 geometric 专用段

# multiplicative 模式专用参数（其他模式忽略）
P = 1.0                # 距离陡度

# geometric 模式专用参数（其他模式忽略）
LAMBDA = 0.5           # 倾斜参数：0.5=对称，<0.5偏相似度，>0.5偏适应度
ALPHA_MIN = 2.0        # 初始锐度（前期探索）——推荐配置 α2→10
ALPHA_MAX = 10.0       # 最终锐度（后期开发）——nltcs 实验最优
DELTA = 0.05           # 底值（防 log(0)）
WINSORIZE = (0.01, 0.99)  # 稳健归一化分位点

BETA = 1.0             # 选择强度（固定值）
H = 0.8                # 邻域尺度（固定值）
RHO = 0.01             # 记录参与率（固定值）。legacy 用 0.01；
                       # single_block 推荐 0.10（nltcs 扫描 L1 最优，接受率仍健康）
ETA = 0.5              # 块复制率（固定值，仅 legacy 模式生效）
MU = 0.01              # 变异率（固定值，仅 legacy 模式生效）

# 更新机制开关
#   'legacy'=旧三参数机制（ρ 参与 + η 块复制率 + μ 变异率）
#   'single_block'=单块复制/变异互斥（参与后至多改一个合法块，η 失效，用 ε 控制变异占比）
UPDATE_MODE = 'legacy'  # 默认保持历史行为
                        # 切 single_block 时推荐配套：RHO=0.10, EPSILON=0.05
EPSILON = 0.01          # 变异占比：P(复制)=ρ(1-ε), P(变异)=ρε（仅 single_block 生效）
                        # single_block 推荐 0.05：nltcs 上 ε=0 的 L1 略优但变异归零、
                        # 理论上锁死搜索空间，取 0.05 保留探索（L1 仅差约 2%，多在噪声内）

# 残差驱动的局部扩散核（研究候选，默认关闭）
# 方向信号只连续倾斜实际单块复制概率；不做正负门控或逐候选 top-k。
RESIDUAL_DIRECTED_DIFFUSION = False
DIFFUSION_DIRECTION_STRENGTH = 1.0
DIFFUSION_DIRECTION_NORMALIZATION = 'initial_rms'

# 整代提案被拒后的缩步重试（默认关闭，保持历史行为）
MAX_RETRIES = 0        # 0=不重试；每次重试复用当轮 donor
RETRY_RHO_DECAY = 0.5  # 重试时 rho 逐次乘以该因子
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
        "maxent_max_states": MAXENT_MAX_STATES,
        "maxent_max_sweeps": MAXENT_MAX_SWEEPS,
        "maxent_tol": MAXENT_TOL,
        "distance_mode": DISTANCE_MODE,
        "p": P,
        "lambda": LAMBDA,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "delta": DELTA,
        "winsorize": WINSORIZE,
        "beta": BETA, "h": H, "rho": RHO, "eta": ETA, "mu": MU,
        "update_mode": UPDATE_MODE, "epsilon": EPSILON,
        "max_retries": MAX_RETRIES,
        "retry_rho_decay": RETRY_RHO_DECAY,
        "residual_directed_diffusion": RESIDUAL_DIRECTED_DIFFUSION,
        "diffusion_direction_strength": DIFFUSION_DIRECTION_STRENGTH,
        "diffusion_direction_normalization": (
            DIFFUSION_DIRECTION_NORMALIZATION
        ),
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
            update_mode=UPDATE_MODE, epsilon=EPSILON,
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=LOG_EVERY,
            distance_mode=DISTANCE_MODE,
            p=P,
            lambda_param=LAMBDA,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            delta=DELTA,
            winsorize_quantiles=WINSORIZE,
            max_retries=MAX_RETRIES,
            retry_rho_decay=RETRY_RHO_DECAY,
            maxent_max_states=MAXENT_MAX_STATES,
            maxent_max_sweeps=MAXENT_MAX_SWEEPS,
            maxent_tol=MAXENT_TOL,
            residual_directed_diffusion=RESIDUAL_DIRECTED_DIFFUSION,
            diffusion_direction_strength=DIFFUSION_DIRECTION_STRENGTH,
            diffusion_direction_normalization=(
                DIFFUSION_DIRECTION_NORMALIZATION
            ),
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
        init_diag = diagnostics["initialization"]
        if init_diag["method"] == "pairwise_maxent":
            print(
                f"  最大熵初始化: {init_diag['usable_pairs']} 对边缘"
                f" | {init_diag['sweeps_run']} 轮 IPF"
                f" | 最大误差 {init_diag['max_pair_error']:.2e}"
                f" | 收敛={init_diag['converged']}"
            )
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
            "initialization": diagnostics["initialization"],
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
