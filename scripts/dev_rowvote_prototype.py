#!/usr/bin/env python
"""行级多候选投票软选择原型（tab-pe 借鉴方向 A，探索性，不动 src）。

机制：每轮生成 k 个独立 proposal 表（同一 donor 分布下独立抽样的
donor/掩码/变异），对每行 i 从 k 个候选行 {T_1[i],...,T_k[i]} 中按
行级 fitness 的标准分 softmax **采样**选择（PE 式投票；无 argmax、
无接受门——变体分布上的软选择，属"改分布"）。基线 = PR #48 v3 的
no_gate_si 配置（a16 + min_spread + ds2 + rho=0.01）。

行级 fitness：F(z) = Σ_j ε_j·(a_j(z) − p_j)（当前实现 compute_fitness，
残差用选择前的当前表——所有候选行在同一残差场下投票）。

用法：CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n gsd \
  python scripts/dev_rowvote_prototype.py --arms baseline rowvote_k4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.fitness import compute_fitness
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss, compute_residual
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.update import evolve_step
from table_diffevo.generator import init_synthetic_table

SI_PARAMS = dict(
    beta=1.0, h=0.8, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    alpha=16.0, exclude_self=True,
    scale_invariant=True, scale_invariant_min_spread=1e-3,
)
RHO, ETA, MU = 0.01, 0.5, 0.01
DS = 2.0  # diffusion_direction_strength（基线走 run_evolution，原型不复现定向复制）


def run_rowvote(seed, rounds, k, vote_alpha, device="cuda"):
    """行级 k 候选投票软选择的简化主循环。

    注意（探索性原型的已知简化）：不含残差定向复制倾斜
    （copy_direction），donor 选择与基线一致；因此对照臂 baseline_plain
    也关闭定向复制，保证唯一变量是 k 候选投票。
    """
    schema = load_schema("configs/nltcs/schema.yaml")
    queries = load_queries("configs/nltcs/measured_1000query.json")
    marginals = load_marginals("configs/nltcs/init_marginals.json")
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n = 16181
    rng = np.random.default_rng(seed)
    S = init_synthetic_table(n, schema, rng, marginals=marginals)

    for t in range(rounds):
        q = evaluate_table(S, queries)
        residual = compute_residual(target, q, n_records=n)
        fitness = compute_fitness(S, queries, residual, q)
        distances = pairwise_block_distance(
            S, S, schema, device=device, return_tensor=True
        )
        probs = compute_sampling_probs(
            fitness, distances, device=device, **SI_PARAMS
        )
        candidates = []
        cand_fitness = []
        for _ in range(k):
            donor_idx = sample_donors(probs, rng, device=device)
            donors = S.iloc[donor_idx].reset_index(drop=True)
            proposal = evolve_step(
                S, donors, schema, rho=RHO, eta=ETA, mu=MU, rng=rng
            )
            candidates.append(proposal)
            # 行级投票分数：候选行在当前残差场下的 fitness
            cand_fitness.append(
                compute_fitness(proposal, queries, residual, q)
            )
        del probs
        if k == 1:
            S = candidates[0]
        else:
            score = np.stack(cand_fitness, axis=1)  # (N, k)
            mean = score.mean(axis=1, keepdims=True)
            std = score.std(axis=1, keepdims=True)
            z = (score - mean) / np.maximum(std, 1e-3)
            logits = vote_alpha * z
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            # 逐行按投票分布采样（软选择，非 argmax）
            u = rng.random(n)
            choice = (p.cumsum(axis=1) < u[:, None]).sum(axis=1)
            choice = np.clip(choice, 0, k - 1)
            frames = [c.to_numpy() for c in candidates]
            stacked = np.stack(frames, axis=1)  # (N, k, A)
            picked = stacked[np.arange(n), choice]
            S = candidates[0].copy()
            S.iloc[:, :] = picked
        if (t + 1) % 100 == 0:
            loss = compute_loss(target, evaluate_table(S, queries))
            print(f"  [seed={seed} k={k}] round {t+1}/{rounds} "
                  f"loss={loss:.4g}", flush=True)
    final_q = evaluate_table(S, queries)
    return (
        float(compute_loss(target, final_q)),
        float(compute_normalized_l1(target, final_q, n)),
    )


def run_baseline_si(seed, rounds):
    """基线：v3 no_gate_si 配置直接走 run_evolution（含定向复制）。"""
    schema = load_schema("configs/nltcs/schema.yaml")
    queries = load_queries("configs/nltcs/measured_1000query.json")
    marginals = load_marginals("configs/nltcs/init_marginals.json")
    target = np.asarray([q["result"] for q in queries], dtype=float)
    _, diag = run_evolution(
        target=target, queries=queries, schema=schema, marginals=marginals,
        n_records=16181, n_rounds=rounds, seed=seed, device="cuda",
        log_every=0, return_final_table=True,
        rho=RHO, beta=1.0, h=0.8, eta=ETA, mu=MU, lambda_param=0.5,
        delta=0.05, winsorize_quantiles=(0.01, 0.99),
        distance_mode="geometric", init_method="marginal",
        residual_directed_diffusion=True, diffusion_direction_strength=DS,
        diffusion_direction_normalization="initial_rms",
        alpha_min=16.0, alpha_max=16.0, exclude_self=True, tol=float("inf"),
        selection_scale_invariant=True,
        selection_scale_invariant_min_spread=1e-3,
    )
    ft = diag.pop("final_table")
    fq = evaluate_table(ft, queries)
    return (
        float(compute_loss(target, fq)),
        float(compute_normalized_l1(target, fq, 16181)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True, choices=[
        "baseline_si", "baseline_plain", "rowvote_k2", "rowvote_k4",
        "rowvote_k4_a8",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument(
        "--out", default="outputs/gate_free_self_cooling/dev_rowvote.json"
    )
    args = parser.parse_args()

    results = {}
    for arm in args.arms:
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            if arm == "baseline_si":
                loss, l1 = run_baseline_si(seed, args.rounds)
            elif arm == "baseline_plain":
                loss, l1 = run_rowvote(seed, args.rounds, k=1, vote_alpha=0)
            elif arm == "rowvote_k2":
                loss, l1 = run_rowvote(seed, args.rounds, k=2, vote_alpha=4.0)
            elif arm == "rowvote_k4":
                loss, l1 = run_rowvote(seed, args.rounds, k=4, vote_alpha=4.0)
            elif arm == "rowvote_k4_a8":
                loss, l1 = run_rowvote(seed, args.rounds, k=4, vote_alpha=8.0)
            per_seed.append({
                "seed": seed, "final_loss": loss, "final_l1": l1,
                "elapsed_sec": round(time.time() - t0, 1),
            })
            print(f"[{arm} seed={seed}] loss={loss:.4g} L1={l1:.6f} "
                  f"({per_seed[-1]['elapsed_sec']}s)", flush=True)
        results[arm] = {
            "per_seed": per_seed,
            "mean_l1": float(np.mean([r["final_l1"] for r in per_seed])),
        }
        print(f"== {arm}: mean_L1={results[arm]['mean_l1']:.6f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=1, ensure_ascii=False))
    print("saved ->", out)


if __name__ == "__main__":
    main()
