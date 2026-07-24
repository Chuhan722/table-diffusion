"""
扩散演化运行入口

一键跑一次演化并自动落盘：
    conda run -p ./.conda python scripts/run.py

调参方式：直接修改下面"参数配置"区的常量，改完再跑。
不需要命令行传参——这个脚本就是你反复复用的实验入口。

流程：
1. 加载 schema、queries，从 queries 取 target（各查询真实计数）
2. run_evolution 跑演化 → best_S, diagnostics
3. save_run 落盘到 outputs/<时间_编号>/
4. 打印结果摘要
"""
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run
from table_diffevo.marginals import load_marginals


# ========== 参数配置（调参改这里） ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"

N_RECORDS = 16181      # 合成表记录条数（nltcs train 集）
N_ROUNDS = 1000        # 最大轮数 T
SEED = 0               # 随机种子（复现）

# 计算设备（新增）
DEVICE = 'cuda'        # 'cuda'=GPU加速 | 'numpy'=原NumPy | 'cpu'=PyTorch CPU

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


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])

    # 加载边缘测量（仅 init_method='marginal' 时需要）
    marginals = None
    if INIT_METHOD == 'marginal':
        marginals = load_marginals(MARGINALS_PATH)

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
    )

    run_dir = save_run(best_S, diagnostics)

    # 结果摘要
    lh = diagnostics["loss_history"]
    print("演化完成")
    print(f"  初始化方式: {INIT_METHOD}")
    print(f"  计算设备  : {DEVICE}")
    print(f"  评价方式  : {EVAL_METHOD}（batch={BATCH_SIZE}）")
    print(f"  初始 loss : {lh[0]:.1f}")
    print(f"  最优 loss : {diagnostics['best_loss']:.1f}")
    print(f"  平均归一化L1: {diagnostics['normalized_l1_error']:.4f}")
    print(f"  跑了轮数  : {diagnostics['rounds_run']}"
          f"（提前停止={diagnostics['stopped_early']}）")
    print(f"  结果已保存: {run_dir}/")


if __name__ == "__main__":
    main()
