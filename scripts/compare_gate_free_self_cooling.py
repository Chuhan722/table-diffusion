"""无门控残差自冷却与门控的配对对照（Issue #43/#44 探索性协议）。

在 test_300x10 上以配对种子比较三臂：

- ``no_gate``：关闭整代接受门（tol=inf）、恒定扰动；
- ``no_gate_self_cooling``：关闭接受门 + 残差自冷却（默认 linear，p=1）；
- ``historical_gate``：主循环历史贪心判据（现默认）。

本脚本用于机制方向验证与冒烟复现，属探索性协议：不做正式分类，输出仅供
Issue #43 消融协议与 Issue #44 正式预注册实验设计参考。真实训练/测试表不参与
任何环节。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries, evaluate_table
from table_diffevo.schema import load_schema

SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
QUERY_PATH = Path("configs/test_300x10/measured_50query.json")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
OUTPUT_DIR = Path("outputs/gate_free_self_cooling")

# 与 Issue #43 三臂探索一致的固定参数（残差定向扩散开启）。
SHARED_PARAMS = dict(
    rho=0.01, beta=1.0, h=0.8, eta=0.5, mu=0.01, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    alpha_min=2.0, alpha_max=10.0, exclude_self=True,
)


def build_arms(cooling_exponent, monotone):
    return {
        "no_gate": dict(tol=float("inf")),
        "no_gate_self_cooling": dict(
            tol=float("inf"), residual_self_cooling=cooling_exponent,
            self_cooling_monotone=monotone,
        ),
        "historical_gate": {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument(
        "--cooling-exponent", type=float, default=1.0,
        help="残差自冷却指数 p（1=linear，2=quadratic，0.5=sqrt）",
    )
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument(
        "--non-monotone", action="store_true",
        help="机制消融：冷却跟随当前残差比（默认用历史最低残差比，温度只降不升）",
    )
    args = parser.parse_args()

    schema = load_schema(str(SCHEMA_PATH))
    queries = load_queries(str(QUERY_PATH))
    marginals = load_marginals(str(MARGINALS_PATH))
    target = np.asarray([q["result"] for q in queries], dtype=float)

    arms = build_arms(args.cooling_exponent, not args.non_monotone)
    rows = []
    for seed in args.seeds:
        for arm, extra in arms.items():
            _, diag = run_evolution(
                target=target, queries=queries, schema=schema, n_records=300,
                n_rounds=args.rounds, seed=seed, marginals=marginals,
                log_every=0, return_final_table=True,
                **SHARED_PARAMS, **extra,
            )
            losses = diag["loss_history"]
            # 终态口径（第四轮审查）：loss_history[-1] 是最后一次 proposal
            # 之前的状态；final_loss 必须从真实最终表重算。
            final_table = diag.pop("final_table")
            final_q = evaluate_table(final_table, queries)
            rows.append({
                "seed": int(seed),
                "arm": arm,
                "final_loss": float(compute_loss(target, final_q)),
                "pre_final_proposal_loss": float(losses[-1]),
                "best_loss": float(diag["best_loss"]),
                "tail_mean_loss": float(np.mean(losses[-args.tail:])),
                "min_cooling_factor": float(
                    min(diag["self_cooling_history"])
                ),
                "rounds_run": int(diag["rounds_run"]),
            })
            row = rows[-1]
            print(
                f"seed={seed} {arm:22s} final={row['final_loss']:9.1f} "
                f"best={row['best_loss']:9.1f} "
                f"tail{args.tail}={row['tail_mean_loss']:9.1f} "
                f"min_cool={row['min_cooling_factor']:.4f}",
                flush=True,
            )

    print("\n=== 按臂汇总（配对种子均值）===")
    summary = {}
    for arm in arms:
        subset = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            key: float(np.mean([row[key] for row in subset]))
            for key in ("final_loss", "best_loss", "tail_mean_loss")
        }
        print(
            f"{arm:22s} final={summary[arm]['final_loss']:9.1f} "
            f"best={summary[arm]['best_loss']:9.1f} "
            f"tail{args.tail}={summary[arm]['tail_mean_loss']:9.1f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = OUTPUT_DIR / f"gate_ablation_{stamp}.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "exploratory_gate_free_self_cooling",
                "seeds": args.seeds,
                "rounds": args.rounds,
                "cooling_exponent": args.cooling_exponent,
                "monotone": not args.non_monotone,
                "shared_params": {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in SHARED_PARAMS.items()
                },
                "rows": rows,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=1,
        )
    print(f"\noutput={output}")


if __name__ == "__main__":
    main()
