"""
Issue #29 快速探索实验：ρ 衰减调度 A/B/C 三臂对比（温和 s=0.005）

独立驱动，不改 run.py 默认常量。配置固定 test_300x10、精确 target、
marginal 初始化、geometric 抽样，500 轮 × 3 seeds，CPU。

三臂（中心 c=0.01，预算对齐 mean(rho_t)≈c）：
  A  rho_schedule=None         固定 ρ=c              （基线）
  B  rho_schedule='linear'     [c+s, c-s] 对称线性     mean 天然=c
  C  rho_schedule='exponential' 同端点比 r，rho_max 反解 mean=c

跑法：
    conda run -p ./.conda python scripts/exp_rho_schedule.py

产出：outputs/rho_sched_moderate_<时间>/{A,B,C}/{i-seed}/ + 各臂 summary.json
     + 顶层 compare.json 汇总三臂配对对比。
"""
import os

import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution, _compute_rho_t
from table_diffevo.io import save_run, create_parent_dir, save_summary
from table_diffevo.marginals import load_marginals


# ========== 固定基线配置 ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

N_RECORDS = 16181
N_ROUNDS = 500
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DEVICE = 'cuda'          # GPU 加速，nltcs 大表
LOG_EVERY = 50           # 每 50 轮打印一次进度

INIT_METHOD = 'marginal'
DISTANCE_MODE = 'geometric'
BETA, H, ETA, MU = 1.0, 0.8, 0.5, 0.01
LAMBDA, ALPHA_MIN, ALPHA_MAX, DELTA = 0.5, 2.0, 10.0, 0.05
WINSORIZE = (0.01, 0.99)

# ρ 调度（温和档，按 nltcs 记录数调整）
C = 0.01                 # 中心参与率（=固定基线）
S = 0.005                # 衰减跨度：线性端点 [C+S, C-S] = [0.015, 0.005]


def _discrete_mean(rho_schedule, rho_max, rho_min):
    """跑满 N_ROUNDS 时 rho_t 的离散均值（用 _compute_rho_t 逐轮求）。"""
    vals = [
        _compute_rho_t(t, N_ROUNDS, C, rho_schedule, rho_max, rho_min)
        for t in range(N_ROUNDS)
    ]
    return float(np.mean(vals))


def build_arms():
    """构造三臂参数，B/C 预算对齐到 mean(rho_t)≈C。"""
    lin_max, lin_min = C + S, C - S            # 线性对称端点，mean 天然=C
    r = lin_min / lin_max                       # 端点比，C 沿用同一 r

    # 指数：rho_max·mean(r^p) = C → rho_max = C / mean(r^p)
    unit_mean = _discrete_mean('exponential', 1.0, r)   # rho_max=1 时的均值
    exp_max = C / unit_mean
    exp_min = r * exp_max

    return [
        {"name": "fixed", "rho_schedule": None,
         "rho_max": C, "rho_min": C},
        {"name": "linear", "rho_schedule": "linear",
         "rho_max": lin_max, "rho_min": lin_min},
        {"name": "exponential", "rho_schedule": "exponential",
         "rho_max": exp_max, "rho_min": exp_min},
    ]


def _tail_mean(loss_history, tail=250):
    """末 tail 轮平均 loss（不足 tail 轮则取全程）。"""
    lh = loss_history[-tail:] if len(loss_history) >= tail else loss_history
    return float(np.mean(lh))


def run_arm(arm, schema, queries, target, marginals, parent_dir):
    """跑单臂全 seeds，落盘各 seed 结果并返回 per_seed 汇总列表。"""
    arm_dir = os.path.join(parent_dir, arm["name"])
    os.makedirs(arm_dir, exist_ok=True)
    per_seed = []
    for i, seed in enumerate(SEEDS):
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=seed,
            beta=BETA, h=H, rho=C, eta=ETA, mu=MU,
            device=DEVICE, init_method=INIT_METHOD, marginals=marginals,
            log_every=LOG_EVERY, distance_mode=DISTANCE_MODE,
            lambda_param=LAMBDA, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
            delta=DELTA, winsorize_quantiles=WINSORIZE,
            rho_schedule=arm["rho_schedule"],
            rho_max=arm["rho_max"], rho_min=arm["rho_min"],
        )
        save_run(best_S, diag, run_dir=os.path.join(arm_dir, f"{i}-{seed}"))
        rho_hist = diag["rho_t_history"]
        per_seed.append({
            "seed": seed,
            "best_loss": diag["best_loss"],
            "normalized_l1_error": diag["normalized_l1_error"],
            "tail250_mean_loss": _tail_mean(diag["loss_history"], 250),
            "traj_mean_loss": float(np.mean(diag["loss_history"])),
            "rounds_run": diag["rounds_run"],
            "rho_t_mean": float(np.mean(rho_hist)),
            "rho_t_first": rho_hist[0],
            "rho_t_last": rho_hist[-1],
        })
        print(f"  [{arm['name']}] seed {seed}: "
              f"best {diag['best_loss']:.3f} | "
              f"tail250 {per_seed[-1]['tail250_mean_loss']:.3f} | "
              f"rho_t mean {per_seed[-1]['rho_t_mean']:.5f} "
              f"({rho_hist[0]:.4f}→{rho_hist[-1]:.4f})")
    summary = {
        "arm": arm["name"],
        "rho_schedule": arm["rho_schedule"],
        "rho_max": arm["rho_max"], "rho_min": arm["rho_min"],
        "per_seed": per_seed,
    }
    save_summary(arm_dir, summary)
    return per_seed


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    arms = build_arms()
    parent_dir = create_parent_dir(prefix="rho_sched_moderate")
    print(f"结果目录: {parent_dir}/\n")
    print("三臂参数（预算对齐检查见下方 rho_t mean）:")
    for a in arms:
        print(f"  {a['name']}: schedule={a['rho_schedule']}, "
              f"max={a['rho_max']:.5f}, min={a['rho_min']:.5f}")
    print()

    results = {}
    for arm in arms:
        print(f"===== {arm['name']} 臂（schedule={arm['rho_schedule']}）=====")
        results[arm["name"]] = run_arm(
            arm, schema, queries, target, marginals, parent_dir)

    # 配对对比：fixed 为基线，linear/exponential 相对 fixed 的同 seed 改善
    baseline = "fixed"
    schedule_arms = ("linear", "exponential")

    def _mean(arm, key):
        return float(np.mean([s[key] for s in results[arm]]))

    compare = {"seeds": list(SEEDS), "baseline": baseline, "metrics": {}}
    for key in ("best_loss", "tail250_mean_loss", "traj_mean_loss",
                "normalized_l1_error"):
        row = {arm: _mean(arm, key) for arm in results}
        # 同 seed 配对：调度臂相对 fixed 改善的 seed 数
        wins = {}
        for arm in schedule_arms:
            improved = sum(
                1 for sb, sa in zip(results[arm], results[baseline])
                if sb[key] < sa[key]
            )
            wins[arm] = f"{improved}/{len(SEEDS)}"
        compare["metrics"][key] = {"mean": row, "wins_vs_fixed": wins}
    save_summary(parent_dir, {"compare": compare, "arms": [
        {"name": a["name"], "rho_schedule": a["rho_schedule"],
         "rho_max": a["rho_max"], "rho_min": a["rho_min"]} for a in arms]})

    print(f"\n===== 三臂对比（均值，配对胜 {baseline} 种子数，{len(SEEDS)} seeds）=====")
    for key, d in compare["metrics"].items():
        m = d["mean"]
        print(f"  {key}:")
        print(f"    fixed={m['fixed']:.4f}  "
              f"linear={m['linear']:.4f}  exponential={m['exponential']:.4f}")
        w = d["wins_vs_fixed"]
        print(f"    改善 vs fixed: linear {w['linear']}, "
              f"exponential {w['exponential']}")
    print(f"\n结果目录: {parent_dir}/")


if __name__ == "__main__":
    main()
