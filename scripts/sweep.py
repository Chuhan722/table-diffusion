"""
单参数敏感性扫描入口

固定 baseline，一次只动一个参数、其余不动，每个取值跑多种子，
汇总"参数值 → best_loss / 归一化L1 / 耗时"，用于第 3 步（衰减调度）
和第 4 步（组合微调）的依据。

一键跑：
    conda run -p ./.conda python scripts/sweep.py
若卡 0 被占，指定空闲卡（先 nvidia-smi 看哪块空）：
    CUDA_VISIBLE_DEVICES=1 conda run -p ./.conda python scripts/sweep.py

配置改下面"扫描配置"区。BASELINE 镜像 run.py 的当前默认值，
但本脚本独立持有一份，保证扫描可复现、不受 run.py 临时改动影响。

落盘：
    outputs/sweep_YYYY-MM-DD_HHMM/
        {参数}={值}/{顺序}-{种子}/     # 每次运行的 best_synthetic.csv + diagnostics.json
        sweep_summary.json             # 全部配置的汇总（含 baseline 标记）

流程：
1. 固定 1000 轮横比（公平对比，末尾看是否还在降来判断轮数够不够）
2. baseline 只跑一次，各参数的 baseline 取值点复用该结果（省时间）
3. 每个 (参数, 取值) 跑 SEEDS 个种子，聚合均值±std/min/max
"""
import json
import os
from datetime import datetime

import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run
from table_diffevo.marginals import load_marginals


# ========== 扫描配置（改这里） ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

N_RECORDS = 16181
N_ROUNDS = 1000        # 横比轮数（改这一个数即可，全脚本自动跟随；注意这不是查询条数）
SEEDS = [0, 1, 2]      # 敏感性扫描建议 ≥3 种子；单种子时 std 恒为 0，波动信息消失，
                       # 参数间差异可能只是运气，慎用来下结论。单种子只适合快速试流程。
LOG_EVERY = 100        # 扫描时打印稀疏些
DEVICE = 'cuda'
EVAL_METHOD = 'vectorized'
BATCH_SIZE = 256
INIT_METHOD = 'marginal'

# baseline：其余参数固定在哪里的锚点。
# ⚠️ 需与 run.py 的 BETA/H/RHO/ETA/MU 保持一致——改了 run.py 默认值记得同步这里，
#    否则两个脚本的"基准"对不上。故意分开是为了扫描结果不受 run.py 临时改动影响。
BASELINE = {
    "beta": 1.0,
    "h": 0.8,
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
}

# 扫描网格：参数名 → 候选值列表（含 baseline 值；顺序 h→rho→mu→eta→beta）
# 想只扫部分参数，注释掉不扫的行即可。
PARAM_GRID = {
    "h":    [0.2, 0.4, 0.8, 1.6, 3.2],
    "rho":  [0.005, 0.01, 0.02, 0.05, 0.1],
    "mu":   [0.001, 0.01, 0.05, 0.1],
    "eta":  [0.25, 0.5, 0.75, 1.0],
    "beta": [0.5, 1.0, 2.0, 4.0],
}
# ========================================


def _aggregate(values):
    """一组标量的均值/标准差/最小/最大。"""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _close(a, b, tol=1e-12):
    """浮点相等判断（识别某取值是否就是 baseline 值）。"""
    return abs(a - b) <= tol


def _run_config(target, queries, schema, marginals, params, out_dir):
    """
    对给定参数组合跑 SEEDS 个种子，落盘到 out_dir/{i}-{seed}/，
    返回逐种子记录列表。params 是含 beta/h/rho/eta/mu 的完整字典。
    """
    per_seed = []
    for i, seed in enumerate(SEEDS):
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS,
            n_rounds=N_ROUNDS,
            seed=seed,
            beta=params["beta"], h=params["h"], rho=params["rho"],
            eta=params["eta"], mu=params["mu"],
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=LOG_EVERY,
        )
        sub = f"{i}-{seed}"
        save_run(best_S, diag, run_dir=os.path.join(out_dir, sub))
        per_seed.append({
            "seed": seed,
            "best_loss": diag["best_loss"],
            "normalized_l1_error": diag["normalized_l1_error"],
            "elapsed_sec": diag["elapsed_sec"],
        })
        print(f"    种子 {seed}: best_loss={diag['best_loss']:.3e}"
              f" | nL1={diag['normalized_l1_error']:.4f}"
              f" | {diag['elapsed_sec']:.1f}s")
    return per_seed


def _agg_block(per_seed):
    """把逐种子记录聚合成 aggregate 块。"""
    return {
        "best_loss": _aggregate([s["best_loss"] for s in per_seed]),
        "normalized_l1_error": _aggregate(
            [s["normalized_l1_error"] for s in per_seed]),
        "elapsed_sec": _aggregate([s["elapsed_sec"] for s in per_seed]),
    }


def _count_configs():
    """预估非 baseline 配置数（baseline 单独只跑一次）。"""
    n = 0
    for param, values in PARAM_GRID.items():
        n += sum(1 for v in values if not _close(v, BASELINE[param]))
    return n


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH) if INIT_METHOD == 'marginal' else None

    n_configs = _count_configs()
    total_runs = (1 + n_configs) * len(SEEDS)
    print(f"扫描计划：baseline + {n_configs} 个非 baseline 配置，"
          f"每个 {len(SEEDS)} 种子，共 {total_runs} 次运行（各 {N_ROUNDS} 轮）。")

    # 父文件夹 outputs/sweep_YYYY-MM-DD_HHMM/（带 sweep_ 前缀便于和普通 run 区分）
    sweep_dir = os.path.join(
        "outputs", "sweep_" + datetime.now().strftime("%Y-%m-%d_%H%M"))
    os.makedirs(sweep_dir, exist_ok=True)

    # 1. baseline 只跑一次，各参数的 baseline 取值点复用它
    print(f"\n===== baseline {BASELINE} =====")
    baseline_seeds = _run_config(target, queries, schema, marginals,
                                 dict(BASELINE),
                                 os.path.join(sweep_dir, "baseline"))
    baseline_agg = _agg_block(baseline_seeds)

    # 2. 逐参数扫描
    results = {"baseline": {"params": dict(BASELINE),
                            "per_seed": baseline_seeds,
                            "aggregate": baseline_agg}}
    sweeps = {}
    for param, values in PARAM_GRID.items():
        print(f"\n########## 扫描 {param} ##########")
        points = []
        for v in values:
            is_base = _close(v, BASELINE[param])
            if is_base:
                print(f"  {param}={v}（=baseline，复用）")
                agg = baseline_agg
            else:
                print(f"  {param}={v}")
                params = dict(BASELINE)
                params[param] = v
                out = os.path.join(sweep_dir, f"{param}={v}")
                per_seed = _run_config(target, queries, schema, marginals,
                                       params, out)
                agg = _agg_block(per_seed)
            points.append({"value": v, "is_baseline": is_base,
                           "aggregate": agg})
        sweeps[param] = points

    # 3. 汇总落盘
    summary = {
        "baseline": results["baseline"]["params"],
        "baseline_aggregate": baseline_agg,
        "config": {
            "n_rounds": N_ROUNDS, "seeds": list(SEEDS),
            "init_method": INIT_METHOD, "device": DEVICE,
            "schema_path": SCHEMA_PATH, "query_path": QUERY_PATH,
        },
        "sweeps": sweeps,
    }
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 4. 打印每个参数的 best_loss 均值曲线（快速看敏感度）
    print(f"\n===== 扫描汇总（best_loss 均值，{N_ROUNDS}轮×{len(SEEDS)}种子）=====")
    for param, points in sweeps.items():
        cells = []
        for p in points:
            mark = "*" if p["is_baseline"] else " "
            cells.append(f"{p['value']}{mark}:{p['aggregate']['best_loss']['mean']:.2e}")
        print(f"  {param:5s} | " + "  ".join(cells))
    print(f"  (* = baseline 值)  结果目录: {sweep_dir}/")


if __name__ == "__main__":
    main()
