"""fitness-only 稳态单轮分段计时工具（只测量，不改主代码）。

复刻 run_fitness_only_all2way_pool_nltcs_diagnostic 的引擎配置
（nltcs，16181 行，512 池查询，device=cuda，rho 取稳态地板 0.001），
用同步计时包装器 monkeypatch evolution 模块命名空间里的关键函数，
跑 N_PROFILE_ROUNDS 轮后按环节汇总耗时。

用法：CUDA_VISIBLE_DEVICES=<空闲卡> .venv/bin/python scripts/profile_round_segments.py
     LOTTERY=1 前缀启用先抽签后选供体模式（对照提速用）。
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import table_diffevo.evolution as evo
from table_diffevo.fitness_only import FitnessOnlyConfig, run_fitness_only_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.schema import load_schema

ROOT = Path(__file__).resolve().parent.parent
N_PROFILE_ROUNDS = 60
N_RECORDS = 16181
SEED = 9908
# LOTTERY=1 时启用先抽签后选供体（换位提速）模式，其余配置完全不变
LOTTERY = os.environ.get("LOTTERY", "0") == "1"

# ---------------------------------------------------------------- 计时器
_totals = defaultdict(float)
_calls = defaultdict(int)
_first_call = defaultdict(float)


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed(name, fn):
    def wrapper(*args, **kwargs):
        _sync()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _sync()
        dt = time.perf_counter() - t0
        if _calls[name] == 0:
            _first_call[name] = dt
        _totals[name] += dt
        _calls[name] += 1
        return out
    return wrapper


# 包 evolution 模块命名空间里的调用点（闭包在调用时查模块全局，能生效）
for name in [
    "pairwise_block_distance",   # N×N 距离
    "compute_sampling_probs",    # softmax 抽样概率
    "sample_donors",             # donor 抽样
    "evolve_step",               # 提案构造（ρ抽签+复制+变异）
    "_mean_selected_distance",   # 诊断 gather
    "evaluate_vectorized",       # 向量化查询评估（当前表+提案）
    "evaluate_table",            # legacy 查询评估（应为 0 次）
    "compute_fitness",           # legacy fitness（应为 0 次）
    "compute_residual",
    "compute_loss",
]:
    setattr(evo, name, timed(name, getattr(evo, name)))

# ---------------------------------------------------------------- 输入构建
schema = load_schema(str(ROOT / "configs/nltcs/schema.yaml"))
marginals = load_marginals(str(ROOT / "configs/nltcs/init_marginals.json"))

payload = json.loads((ROOT / "configs/nltcs/all2way_issue53_v1.json").read_text())
all2way_queries = payload["queries"]
assert len(all2way_queries) == 480
all2way_targets = [float(q["result"]) for q in all2way_queries]

one_way_queries = []
one_way_targets = []
index = 0
for attribute, spec in marginals["attributes"].items():
    for value, count in zip(spec["values"], spec["counts"]):
        index += 1
        one_way_queries.append({
            "id": f"OW{index:04d}",
            "type": "single",
            "expression": f"{attribute} == {value}",
            "conditions": [
                {"attribute": attribute, "operator": "==", "value": value}
            ],
            "result": float(count),
        })
        one_way_targets.append(float(count))
assert len(one_way_queries) == 32

pool_queries = list(all2way_queries) + one_way_queries
pool_targets = np.asarray(all2way_targets + one_way_targets, dtype=float)
assert len(pool_queries) == 512

config = FitnessOnlyConfig(
    n_rounds=N_PROFILE_ROUNDS,
    seed=SEED,
    device="cuda",
    eval_method="vectorized",
    batch_size=256,
    init_method="marginal",
    log_every=100,
    rho=0.001,  # 正式跑的稳态地板值（第 1750 轮后恒定）
    eta=0.5,
    mu=0.01,
    lambda_param=0.5,
    fixed_alpha=16.0,
    delta=0.05,
    winsorize_quantiles=(0.01, 0.99),
    selection_scale_invariant_min_spread=1e-3,
    residual_geometry="relative",
    residual_geometry_floor=8.0,
    exclude_self=True,
    lottery_first_donor_selection=LOTTERY,
    record_transition_clocks=False,
    inner_early_stopping_patience_ticks=6,
)

# ---------------------------------------------------------------- 运行
wall0 = time.perf_counter()
table, diagnostics = run_fitness_only_evolution(
    pool_targets,
    pool_queries,
    schema,
    N_RECORDS,
    config=config,
    fitness_mode="residual",
    marginals=marginals,
)
wall = time.perf_counter() - wall0

# ---------------------------------------------------------------- 汇总
rounds = int(diagnostics["rounds_run"])
print(f"\n===== 分段计时汇总（{rounds} 轮，总墙钟 {wall:.2f}s，"
      f"平均 {wall/rounds*1000:.1f}ms/轮）=====")
print(f"{'环节':<28}{'总耗时s':>10}{'次数':>8}{'均值ms':>10}"
      f"{'稳态均值ms':>12}{'占比%':>8}")
accounted = 0.0
per_round = {}  # name -> 稳态每轮 ms（剔除首次调用，乘每轮调用次数）
for name in sorted(_totals, key=lambda k: -_totals[k]):
    tot = _totals[name]
    n = _calls[name]
    if n == 0:
        continue
    accounted += tot
    # 稳态均值：剔除首次调用（含 CUDA 初始化/编译）
    steady_mean = ((tot - _first_call[name]) / (n - 1) * 1000) if n > 1 else 0.0
    calls_per_round = n / rounds
    # 每轮至少调用一次的函数才计入单轮口径；init/收尾一次性调用不计
    if n >= rounds:
        per_round[name] = steady_mean * calls_per_round
    print(f"{name:<28}{tot:>10.2f}{n:>8}{tot/n*1000:>10.1f}"
          f"{steady_mean:>12.1f}{tot/wall*100:>8.1f}")
other = wall - accounted
other_per_round = other / rounds * 1000  # 含一次性 init/收尾，略高估
print(f"{'其他(Python/pandas/日志等)':<24}{other:>10.2f}{'':>8}{'':>10}"
      f"{other_per_round:>12.1f}{other/wall*100:>8.1f}")

steady_round_ms = sum(per_round.values()) + other_per_round
print(f"\n稳态单轮合计 ≈ {steady_round_ms:.1f} ms（正式跑实测 190ms/轮）")
donor_ms = sum(per_round.get(k, 0.0) for k in (
    "pairwise_block_distance", "compute_sampling_probs",
    "sample_donors", "_mean_selected_distance"))
eval_ms = per_round.get("evaluate_vectorized", 0.0)
print(f"供体机制(距离+softmax+抽样+诊断) ≈ {donor_ms:.1f} ms/轮"
      f"（{donor_ms/steady_round_ms*100:.0f}%）")
print(f"查询评估(当前表+提案两次) ≈ {eval_ms:.1f} ms/轮"
      f"（{eval_ms/steady_round_ms*100:.0f}%）")
