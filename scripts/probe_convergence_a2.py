"""
探测式 α 调度收敛验证实验（阶段 2）

目标：验证探测模式能否通过"经验平台期"（2 次连续探测失败）自动检测收敛，
      并测量收敛时的 L1 误差。

实验设计：
1. test_300x10 复现（新 W 语义）：seed 42/43/44，验证候选评估计数语义
2. nltcs 正式实验：W=30, P=2, stall_rel=0.02, patience=3, α∈[5,12], s=0.2, 3000 轮上限

用法：
    # 先看 GPU 占用，挑空闲卡
    nvidia-smi

    # 指定空闲卡跑（例如卡 1）
    CUDA_VISIBLE_DEVICES=1 conda run -p ./.conda python scripts/probe_convergence_a2.py

    # 或默认卡 0
    conda run -p ./.conda python scripts/probe_convergence_a2.py

调参：修改下面"参数配置"区的常量。
"""
import json
import os
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run, create_parent_dir, save_summary


# ========== 参数配置 ==========

# 工作负载选择（改这里切换 test_300x10 vs nltcs）
WORKLOAD = 'test_300x10'  # 'test_300x10' | 'nltcs'

# test_300x10 配置
# 注意：不设 n_records 常量 —— 合成表行数必须与源数据完全一致，
# 由 _load_n_records() 从查询文件的 record_count 字段自动推导（见下）。
TEST_SCHEMA_PATH = "configs/test_300x10/schema.yaml"
TEST_QUERY_PATH = "configs/test_300x10/measured_50query.json"
TEST_SEEDS = [42, 43, 44]
TEST_N_ROUNDS = 3000  # 上限，观察 α 更新行为

# nltcs 配置
NLTCS_SCHEMA_PATH = "configs/nltcs/schema.yaml"
NLTCS_QUERY_PATH = "configs/nltcs/measured_1000query.json"
NLTCS_SEEDS = [42, 43, 44]
NLTCS_N_ROUNDS = 5000  # 5000 轮上限

# 探测调度参数（两个工作负载共享）
ALPHA_SCHEDULE_MODE = 'probe'
ALPHA_MIN = 5.0        # 初始锐度
ALPHA_MAX = 16.0       # 最终锐度
PROBE_BLOCK_CANDIDATE_BUDGET = 30  # W: 块大小（候选评估数）
PROBE_P = 2            # P: 停滞阈值（连续 P 块停滞 → 触发探测）
PROBE_H_BLOCKS = 2     # H: 每支探测分支运行的块数
PROBE_H_CANDIDATE_BUDGET = PROBE_H_BLOCKS * PROBE_BLOCK_CANDIDATE_BUDGET  # H=2块 × 30评估/块 = 60次评估
PROBE_S = 0.2          # s: 探测步长（归一化 α 空间）
PROBE_C = 2            # C: 冷却块数
PROBE_STALL_REL = 0.02 # 停滞相对阈值：块降幅占当前 L1 的比例低于此值记一次停滞（量纲无关）
PROBE_PATIENCE = 3     # 耐心值：连续 N 次探测未刷新历史最好 L1 → 判定收敛

# 通用参数
LOG_EVERY = 50
DEVICE = 'cuda'
EVAL_METHOD = 'vectorized'
BATCH_SIZE = 256
# 统一使用 marginal (1-way) 初始化，确保两个工作负载起点公平
INIT_METHOD = 'marginal'
MAXENT_MAX_STATES = 1_000_000
MAXENT_MAX_SWEEPS = 200
MAXENT_TOL = 1e-8

# 扩散参数（与标准配置一致）
DISTANCE_MODE = 'geometric'
LAMBDA = 0.5
DELTA = 0.05
WINSORIZE = (0.01, 0.99)
BETA = 1.0
H = 0.8
RHO = 0.01
ETA = 0.5
MU = 0.01
MAX_RETRIES = 0
RETRY_RHO_DECAY = 0.5

# ===========================================


def _load_n_records(query_path):
    """
    从查询文件的 record_count 字段读取合成表行数。

    合成表行数必须与源数据完全一致（行数、属性都一样），所以这个值不允许手工
    设定 —— 只能来自源数据的公开元信息。查询文件顶层的 record_count 就是生成
    查询时记录的源数据行数，且属于公开统计量，运行期读取不违反铁律 6
    （不读真实私有答案/原始数据）。
    """
    with open(query_path, encoding="utf-8") as f:
        data = json.load(f)

    if "record_count" not in data:
        raise ValueError(
            f"查询文件缺少 record_count 字段，无法确定源数据行数：{query_path}"
        )

    n_records = int(data["record_count"])
    if n_records <= 0:
        raise ValueError(
            f"查询文件 record_count 非法（{n_records}）：{query_path}"
        )
    return n_records


def _workload_paths(workload):
    """返回 (schema_path, query_path, n_rounds, seeds)"""
    if workload == 'test_300x10':
        return TEST_SCHEMA_PATH, TEST_QUERY_PATH, TEST_N_ROUNDS, TEST_SEEDS
    if workload == 'nltcs':
        return NLTCS_SCHEMA_PATH, NLTCS_QUERY_PATH, NLTCS_N_ROUNDS, NLTCS_SEEDS
    raise ValueError(f"Unknown WORKLOAD: {workload}")


def _run_params(workload):
    """参数快照，写入 summary.json"""
    common = {
        "workload": workload,
        "alpha_schedule_mode": ALPHA_SCHEDULE_MODE,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "probe_block_candidate_budget": PROBE_BLOCK_CANDIDATE_BUDGET,
        "probe_P": PROBE_P,
        "probe_H_candidate_budget": PROBE_H_CANDIDATE_BUDGET,
        "probe_s": PROBE_S,
        "probe_C": PROBE_C,
        "probe_stall_rel": PROBE_STALL_REL,
        "probe_patience": PROBE_PATIENCE,
        "device": DEVICE,
        "eval_method": EVAL_METHOD,
        "batch_size": BATCH_SIZE,
        "init_method": INIT_METHOD,
        "distance_mode": DISTANCE_MODE,
        "lambda": LAMBDA,
        "delta": DELTA,
        "winsorize": WINSORIZE,
        "beta": BETA, "h": H, "rho": RHO, "eta": ETA, "mu": MU,
        "max_retries": MAX_RETRIES,
        "retry_rho_decay": RETRY_RHO_DECAY,
    }

    schema_path, query_path, n_rounds, seeds = _workload_paths(workload)
    common.update({
        "schema_path": schema_path,
        "query_path": query_path,
        # 行数自动取自源数据，不可手工设定
        "n_records": _load_n_records(query_path),
        "n_records_source": "query_file.record_count",
        "n_rounds": n_rounds,
        "seeds": seeds,
    })

    return common


def _aggregate(values):
    """统计汇总"""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    # 根据 WORKLOAD 选择配置
    workload_name = WORKLOAD
    schema_path, query_path, n_rounds, seeds = _workload_paths(WORKLOAD)

    # 合成表行数 = 源数据行数，自动推导，无常量可改
    n_records = _load_n_records(query_path)

    schema = load_schema(schema_path)
    queries = load_queries(query_path)
    target = np.array([q["result"] for q in queries])

    print(f"\n===== 探测式收敛验证实验：{workload_name} =====")
    print(f"  合成表行数: {n_records}（自动取自 {query_path} 的 record_count）")
    print(f"  α 调度模式: {ALPHA_SCHEDULE_MODE}")
    print(f"  α 范围    : [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"  块大小 W  : {PROBE_BLOCK_CANDIDATE_BUDGET} 候选评估")
    print(f"  停滞阈值 P: {PROBE_P} 块")
    print(f"  探测预算 H: {PROBE_H_CANDIDATE_BUDGET} 候选评估/支")
    print(f"  探测步长 s: {PROBE_S}")
    print(f"  停滞相对阈值: {PROBE_STALL_REL}")
    print(f"  耐心值    : {PROBE_PATIENCE}")
    print(f"  轮数上限  : {n_rounds}")
    print(f"  种子列表  : {seeds}\n")

    parent_dir = create_parent_dir()
    per_seed = []

    for i, seed in enumerate(seeds):
        print(f"\n===== 种子 {seed}（{i + 1}/{len(seeds)}）=====")
        best_S, diagnostics = run_evolution(
            target, queries, schema,
            n_records=n_records,
            n_rounds=n_rounds,
            seed=seed,
            beta=BETA, h=H, rho=RHO, eta=ETA, mu=MU,
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            log_every=LOG_EVERY,
            distance_mode=DISTANCE_MODE,
            lambda_param=LAMBDA,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            alpha_schedule_mode=ALPHA_SCHEDULE_MODE,
            alpha_value=ALPHA_MIN,  # 初始 α（探测模式会动态调整）
            delta=DELTA,
            winsorize_quantiles=WINSORIZE,
            max_retries=MAX_RETRIES,
            retry_rho_decay=RETRY_RHO_DECAY,
            maxent_max_states=MAXENT_MAX_STATES,
            maxent_max_sweeps=MAXENT_MAX_SWEEPS,
            maxent_tol=MAXENT_TOL,
            probe_block_candidate_budget=PROBE_BLOCK_CANDIDATE_BUDGET,
            probe_P=PROBE_P,
            probe_H_candidate_budget=PROBE_H_CANDIDATE_BUDGET,
            probe_s=PROBE_S,
            probe_C=PROBE_C,
            probe_stall_rel=PROBE_STALL_REL,
            probe_patience=PROBE_PATIENCE,
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

        # 探测历史摘要
        probe_history = diagnostics.get("probe_history", [])
        if probe_history:
            print(f"  探测次数  : {len(probe_history)}")
            winners = [p["winner"] for p in probe_history]
            print(f"    胜者分布: DOWN={winners.count('DOWN')}, "
                  f"HOLD={winners.count('HOLD')}, UP={winners.count('UP')}")
            max_no_improve = max((p.get("no_improve_probes", 0) for p in probe_history), default=0)
            print(f"    最长连续未刷新: {max_no_improve}")
            if diagnostics.get("stopped_early", False):
                print(f"    → 经验平台期触发（连续未刷新最好 L1 达到耐心值）")
        else:
            print(f"  探测次数  : 0（未触发探测）")

        print(f"  跑了轮数  : {diagnostics['rounds_run']}"
              f"（提前停止={diagnostics['stopped_early']}）"
              f" | 耗时: {diagnostics['elapsed_sec']:.1f}s"
              f"（{diagnostics['sec_per_round'] * 1000:.0f}ms/轮） | 已存: {run_dir}/")

        per_seed.append({
            "seed": seed,
            "run_dir": sub_name,
            "best_loss": diagnostics["best_loss"],
            "normalized_l1_error": diagnostics["normalized_l1_error"],
            "rounds_run": diagnostics["rounds_run"],
            "stopped_early": diagnostics["stopped_early"],
            "elapsed_sec": diagnostics["elapsed_sec"],
            "probe_count": len(probe_history),
            "probe_max_no_improve": max((p.get("no_improve_probes", 0) for p in probe_history), default=0),
        })

    # 汇总
    summary = {
        "params": _run_params(workload_name),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": {
            "best_loss": _aggregate([s["best_loss"] for s in per_seed]),
            "normalized_l1_error": _aggregate([s["normalized_l1_error"] for s in per_seed]),
            "rounds_run": _aggregate([s["rounds_run"] for s in per_seed]),
            "elapsed_sec": _aggregate([s["elapsed_sec"] for s in per_seed]),
            "probe_count": _aggregate([s["probe_count"] for s in per_seed]),
        },
    }
    save_summary(parent_dir, summary)

    bl = summary["aggregate"]["best_loss"]
    nl = summary["aggregate"]["normalized_l1_error"]
    rr = summary["aggregate"]["rounds_run"]
    pc = summary["aggregate"]["probe_count"]

    print(f"\n===== 多种子汇总（{len(seeds)} 个种子, {workload_name}）=====")
    print(f"  最优 loss    : 均值 {bl['mean']:.3e} ± {bl['std']:.2e}"
          f"  (min {bl['min']:.3e}, max {bl['max']:.3e})")
    print(f"  平均归一化L1 : 均值 {nl['mean']:.4f} ± {nl['std']:.4f}"
          f"  (min {nl['min']:.4f}, max {nl['max']:.4f})")
    print(f"  实际轮数    : 均值 {rr['mean']:.1f} ± {rr['std']:.1f}"
          f"  (min {rr['min']:.0f}, max {rr['max']:.0f})")
    print(f"  探测次数    : 均值 {pc['mean']:.1f} ± {pc['std']:.1f}"
          f"  (min {pc['min']:.0f}, max {pc['max']:.0f})")
    el = summary["aggregate"]["elapsed_sec"]
    print(f"  单种子耗时   : 均值 {el['mean']:.1f}s"
          f"  (min {el['min']:.1f}s, max {el['max']:.1f}s)")
    print(f"  结果目录     : {parent_dir}/")


if __name__ == "__main__":
    main()
