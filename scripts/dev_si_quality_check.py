#!/usr/bin/env python
"""dev 质量检查：高锐度尺度不变选择是否牺牲未测量高阶质量（探索性）。

风险：恒定标准分锐度 alpha 很高时，所有行从少数 donor 复制，可能导致
多样性坍缩——workload L1 变好但未测量 3/4-way、联合 TVD、支持集变差。
只看 workload L1 冻结正式 alpha 会重蹈"挑有利指标"。

隐私边界：生成阶段只读公开输入；参考表在该 seed 全部臂生成完成后才
读取（离线评价，与生成隔离）。

用法：CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd \
  python scripts/dev_si_quality_check.py --arms si_a16_ds2 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from scripts import compare_factorized_gibbs_closed_loop as offline_helpers
else:
    import compare_factorized_gibbs_closed_loop as offline_helpers
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

SHARED = dict(
    eta=0.5, mu=0.01, beta=1.0, h=0.8, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    exclude_self=True,
)

ARMS = {
    "si_a12_ds2": dict(
        rho=0.01, tol=float("inf"), selection_scale_invariant=True,
        alpha_min=12.0, alpha_max=12.0,
    ),
    "si_a16_ds2": dict(
        rho=0.01, tol=float("inf"), selection_scale_invariant=True,
        alpha_min=16.0, alpha_max=16.0,
    ),
    "si_a24_ds2": dict(
        rho=0.01, tol=float("inf"), selection_scale_invariant=True,
        alpha_min=24.0, alpha_max=24.0,
    ),
    # 有门 legacy 基线（与正式协议同配置），提供同种子质量对照。
    "historical_gate": dict(
        rho=0.01, tol=1e-9, alpha_min=2.0, alpha_max=10.0,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument(
        "--out",
        default="outputs/gate_free_self_cooling/dev_si_quality_check.json",
    )
    args = parser.parse_args()

    schema = load_schema("configs/nltcs/schema.yaml")
    queries = load_queries("configs/nltcs/measured_1000query.json")
    marginals = load_marginals("configs/nltcs/init_marginals.json")
    target = np.asarray([q["result"] for q in queries], dtype=float)
    columns = schema.attribute_names()

    tables = {}
    results = {}
    for arm in args.arms:
        t0 = time.time()
        _best, diag = run_evolution(
            target=target, schema=schema, queries=queries,
            marginals=marginals, n_records=16181, n_rounds=args.rounds,
            seed=args.seed, device="cuda", log_every=0,
            return_final_table=True, **{**SHARED, **ARMS[arm]},
        )
        table = diag.pop("final_table")
        tables[arm] = table
        answers = evaluate_table(table, queries)
        results[arm] = {
            "params": ARMS[arm],
            "seed": args.seed,
            "rounds": args.rounds,
            "final_loss": float(compute_loss(answers, target)),
            "final_l1": float(
                compute_normalized_l1(answers, target, n_records=16181)
            ),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(f"[{arm}] L1={results[arm]['final_l1']:.6f}", flush=True)

    # 生成全部完成后才读取参考表（离线评价，与生成隔离）。
    domains = offline_helpers._discretization_domains(marginals)
    measured_triples = offline_helpers._measured_cell_keys(
        queries, marginals, order=3
    )
    # 一次实验一份源数据（第三轮审查）：本脚本源数据为 train，离线参考
    # 只用 train；test 须以其为源数据独立建实验。
    references = {
        "train": pd.read_csv(
            "data/nltcs/nltcs.train.data", header=None, names=columns
        )[columns],
    }
    for arm in args.arms:
        results[arm]["offline"] = {}
        for ref_name, reference in references.items():
            m = offline_helpers._offline_metrics(
                reference, tables[arm], marginals, domains, measured_triples
            )
            results[arm]["offline"][ref_name] = {
                "unmeasured_3way_l1": float(m["unmeasured_3way"]["mean"]),
                "unmeasured_4way_l1": float(m["unmeasured_4way"]["mean"]),
                "raw_joint_tvd": float(m["raw_joint"]["tvd"]),
                "binned_joint_tvd": float(m["binned_joint"]["tvd"]),
                "raw_unique_states": int(m["raw_joint"]["n_unique"]),
                "raw_support_overlap": int(m["raw_joint"]["support_overlap"]),
            }
        o = results[arm]["offline"]
        print(
            f"[{arm}] un3way(train)={o['train']['unmeasured_3way_l1']:.6f} "
            f"un4way={o['train']['unmeasured_4way_l1']:.6f} "
            f"tvd(train/test)={o['train']['binned_joint_tvd']:.4f}/"
            f"{o['test']['binned_joint_tvd']:.4f} "
            f"unique={o['train']['raw_unique_states']}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=1, ensure_ascii=False))
    print("saved ->", out)


if __name__ == "__main__":
    main()
