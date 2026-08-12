#!/usr/bin/env python
"""尺度不变选择正式实验（预注册协议，Issue #44 机制迭代阶段二）。

假设：行内标准化的尺度不变选择（恒定标准分锐度）解决无门控扩散演化的
低温选择退化；配置对齐后无门在主指标上打平或超过有门（门冗余）。

唯一变量：2×2 = {无门 tol=inf, 有门 tol=1e-9} × {si, legacy 选择}。
- si：selection_scale_invariant=True, alpha_min=alpha_max=16（dev 冻结，
  seed 42..44 只用于定标与质量检查，不进正式种子）；
- legacy：历史默认谱系 alpha 2..10，scale_invariant=False。
其余参数全部共享（rho=0.01、ds=2.0 默认方向强度、eta/mu/beta/h 同
PR #45 正式协议）。

判定（运行前冻结，主判定数据集 nltcs，最终表 measured L1 五种子均值）：
1. 机制改进：no_gate_si / no_gate_legacy ≤ 0.60；
2. 门冗余（公平对照）：no_gate_si / gate_si ≤ 1.10；
3. 质量风险带：train/test 未测量 3/4-way 与分箱 TVD 相对 gate_legacy
   劣化 >5% 报警（any_quality_risk）；支持集唯一状态数如实报告不设门槛
   （dev 已知高锐度收窄支持集，纳入观察指标）。
分类：两判定均过 = supports_scale_invariant_selection；仅 1 过 =
mechanism_gain_gate_not_redundant；仅 2 过 = gate_redundant_no_gain；
均不过 = not_supported。test_300x10 辅助数据集独立判定，不合并结论。

种子 100..104；2000 轮；生成只读公开输入（schema/queries/marginals/
n_records/seed），参考表仅在全部生成完成后离线读取。
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
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

FORMAL_SEEDS = [100, 101, 102, 103, 104]
FORMAL_ROUNDS = 2000
FROZEN_SI_ALPHA = 16.0  # dev 定标冻结（seed 42..44）
MECHANISM_MAX_RATIO = 0.60
REDUNDANCY_MAX_RATIO = 1.10
QUALITY_RISK_REL = 0.05

DATASETS = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "device": "numpy",
        "references": {"reference": Path("data/test_300x10/test_300x10.csv")},
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "device": "cuda",
        "references": {
            "train": Path("data/nltcs/nltcs.train.data"),
            "test": Path("data/nltcs/nltcs.test.data"),
        },
    },
}

SHARED_PARAMS = dict(
    rho=0.01, beta=1.0, h=0.8, eta=0.5, mu=0.01, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    exclude_self=True,
)

ARMS = {
    "no_gate_si": dict(
        tol=float("inf"), selection_scale_invariant=True,
        alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
    ),
    "gate_si": dict(
        tol=1e-9, selection_scale_invariant=True,
        alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
    ),
    "no_gate_legacy": dict(
        tol=float("inf"), alpha_min=2.0, alpha_max=10.0,
    ),
    "gate_legacy": dict(
        tol=1e-9, alpha_min=2.0, alpha_max=10.0,
    ),
}

OUTPUT_PATH = Path(
    "outputs/gate_free_self_cooling/"
    f"formal_scale_invariant_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round.json"
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame):
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _environment():
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": status == "",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "argv": sys.argv,
        "started_at": datetime.now().astimezone().isoformat(),
    }


def _load_reference(path, columns):
    if str(path).endswith(".csv"):
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, header=None, names=columns)
    return frame[columns]


def _run_dataset(name, spec, seeds, rounds):
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n_records = spec["n_records"]
    runs = []
    tables = {}
    for seed in seeds:
        for arm, extra in ARMS.items():
            start = time.perf_counter()
            best_S, diag = run_evolution(
                target=target, queries=queries, schema=schema,
                n_records=n_records, n_rounds=rounds, seed=seed,
                marginals=marginals, log_every=0, device=spec["device"],
                return_final_table=True,
                **SHARED_PARAMS, **extra,
            )
            elapsed = time.perf_counter() - start
            losses = diag["loss_history"]
            final_table = diag.pop("final_table")
            final_q = evaluate_table(final_table, queries)
            final_loss = float(compute_loss(final_q, target))
            final_l1 = float(
                compute_normalized_l1(final_q, target, n_records=n_records)
            )
            best_q = evaluate_table(best_S, queries)
            best_l1 = float(
                compute_normalized_l1(best_q, target, n_records=n_records)
            )
            tables[(seed, arm)] = final_table
            runs.append({
                "dataset": name,
                "arm": arm,
                "seed": seed,
                "rounds_run": diag["rounds_run"],
                "candidate_evaluations": diag["candidate_evaluation_count"],
                "pre_final_proposal_loss": float(losses[-1]),
                "final_loss": final_loss,
                "final_table_measured_l1": final_l1,
                "best_loss": float(min(losses)),
                "best_table_measured_l1": best_l1,
                "tail_mean_loss": float(np.mean(losses[-100:])),
                "final_table_sha256": _frame_sha256(final_table),
                "elapsed_sec": round(elapsed, 1),
            })
            print(
                f"[{name} seed={seed} {arm}] final={final_loss:.4g} "
                f"L1={final_l1:.6f} ({elapsed:.0f}s)",
                flush=True,
            )
    # 隐私边界：该数据集全部生成完成后才读取真实参考表做离线评价。
    columns = schema.attribute_names()
    domains = offline_helpers._discretization_domains(marginals)
    measured_triples = offline_helpers._measured_cell_keys(
        queries, marginals, order=3
    )
    references = {
        ref_name: _load_reference(path, columns)
        for ref_name, path in spec["references"].items()
    }
    for run in runs:
        table = tables[(run["seed"], run["arm"])]
        run["offline"] = {}
        for ref_name, reference in references.items():
            metrics = offline_helpers._offline_metrics(
                reference, table, marginals, domains, measured_triples
            )
            run["offline"][ref_name] = {
                "unmeasured_3way_l1": float(
                    metrics["unmeasured_3way"]["mean"]
                ),
                "unmeasured_4way_l1": float(
                    metrics["unmeasured_4way"]["mean"]
                ),
                "raw_joint_tvd": float(metrics["raw_joint"]["tvd"]),
                "binned_joint_tvd": float(metrics["binned_joint"]["tvd"]),
                "raw_unique_states": int(metrics["raw_joint"]["n_unique"]),
                "raw_support_overlap": int(
                    metrics["raw_joint"]["support_overlap"]
                ),
            }
    return runs


def _arm_mean(runs, arm, metric):
    values = [run[metric] for run in runs if run["arm"] == arm]
    return float(np.mean(values))


def _offline_mean(runs, arm, ref_name, metric):
    values = [
        run["offline"][ref_name][metric]
        for run in runs if run["arm"] == arm and ref_name in run["offline"]
    ]
    return float(np.mean(values)) if values else None


def _judge(runs):
    metric = "final_table_measured_l1"
    no_gate_si = _arm_mean(runs, "no_gate_si", metric)
    gate_si = _arm_mean(runs, "gate_si", metric)
    no_gate_legacy = _arm_mean(runs, "no_gate_legacy", metric)
    gate_legacy = _arm_mean(runs, "gate_legacy", metric)

    mechanism_ratio = (
        no_gate_si / no_gate_legacy if no_gate_legacy > 0 else float("inf")
    )
    redundancy_ratio = (
        no_gate_si / gate_si if gate_si > 0 else float("inf")
    )
    mechanism_passed = mechanism_ratio <= MECHANISM_MAX_RATIO
    redundancy_passed = redundancy_ratio <= REDUNDANCY_MAX_RATIO

    per_seed_wins = 0
    seeds = sorted({run["seed"] for run in runs})
    for seed in seeds:
        si = [r[metric] for r in runs
              if r["arm"] == "no_gate_si" and r["seed"] == seed][0]
        legacy = [r[metric] for r in runs
                  if r["arm"] == "no_gate_legacy" and r["seed"] == seed][0]
        if si < legacy:
            per_seed_wins += 1

    quality = {}
    any_quality_risk = False
    ref_names = sorted({name for run in runs for name in run["offline"]})
    for ref_name in ref_names:
        for q_metric in (
            "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
        ):
            candidate_value = _offline_mean(
                runs, "no_gate_si", ref_name, q_metric
            )
            baseline_value = _offline_mean(
                runs, "gate_legacy", ref_name, q_metric
            )
            if candidate_value is None or baseline_value is None:
                continue
            rel = (
                (candidate_value - baseline_value) / baseline_value
                if baseline_value > 0 else 0.0
            )
            flagged = rel > QUALITY_RISK_REL
            any_quality_risk = any_quality_risk or flagged
            quality[f"{ref_name}.{q_metric}"] = {
                "no_gate_si": candidate_value,
                "gate_legacy": baseline_value,
                "relative_change": rel,
                "flagged": flagged,
            }
        # 支持集为观察指标（dev 已知高锐度收窄支持集），报告不设门槛。
        quality[f"{ref_name}.raw_unique_states"] = {
            "no_gate_si": _offline_mean(
                runs, "no_gate_si", ref_name, "raw_unique_states"
            ),
            "gate_legacy": _offline_mean(
                runs, "gate_legacy", ref_name, "raw_unique_states"
            ),
            "observational": True,
        }

    if mechanism_passed and redundancy_passed:
        classification = "supports_scale_invariant_selection"
    elif mechanism_passed:
        classification = "mechanism_gain_gate_not_redundant"
    elif redundancy_passed:
        classification = "gate_redundant_no_gain"
    else:
        classification = "not_supported"

    return {
        "arm_means": {
            "no_gate_si": no_gate_si,
            "gate_si": gate_si,
            "no_gate_legacy": no_gate_legacy,
            "gate_legacy": gate_legacy,
        },
        "mechanism": {
            "ratio": mechanism_ratio,
            "threshold": MECHANISM_MAX_RATIO,
            "passed": mechanism_passed,
            "no_gate_si_wins": per_seed_wins,
        },
        "gate_redundancy": {
            "ratio": redundancy_ratio,
            "threshold": REDUNDANCY_MAX_RATIO,
            "passed": redundancy_passed,
        },
        "quality": quality,
        "any_quality_risk": any_quality_risk,
        "classification": classification,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), 
        default=sorted(DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="容忍脏工作树（强制标记 formal=false，产物不得用作正式结论）",
    )
    args = parser.parse_args()

    environment = _environment()
    if not environment["git_worktree_clean_including_untracked"]:
        if not args.allow_dirty:
            print("工作树不干净（含未跟踪文件）。正式运行要求干净树；"
                  "调试可加 --allow-dirty（产物标记 formal=false）。")
            sys.exit(1)

    payload = {
        "protocol": {
            "hypothesis": (
                "尺度不变选择解决无门低温选择退化；配置对齐后门在主指标上"
                "冗余"
            ),
            "seeds": list(args.seeds),
            "rounds": args.rounds,
            "frozen_si_alpha": FROZEN_SI_ALPHA,
            "arms": {
                arm: {
                    k: (None if isinstance(v, float) and np.isinf(v) else v)
                    for k, v in extra.items()
                }
                for arm, extra in ARMS.items()
            },
            "shared_params": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in SHARED_PARAMS.items()
            },
            "primary_metric": "final_table_measured_l1",
            "thresholds": {
                "mechanism_max_ratio": MECHANISM_MAX_RATIO,
                "redundancy_max_ratio": REDUNDANCY_MAX_RATIO,
                "quality_risk_rel": QUALITY_RISK_REL,
            },
            "dev_calibration_note": (
                "si alpha=16 由 dev seed 42..44 定标冻结；dev 种子不进正式"
                "种子集；支持集唯一状态数为观察指标（dev 已知收窄）"
            ),
        },
        "environment": environment,
        "formal": environment["git_worktree_clean_including_untracked"],
        "datasets": {},
    }

    for name in args.datasets:
        spec = DATASETS[name]
        runs = _run_dataset(name, spec, args.seeds, args.rounds)
        judgment = _judge(runs)
        payload["datasets"][name] = {
            "runs": runs,
            "judgment": judgment,
        }
        print(f"== {name}: {json.dumps(judgment, ensure_ascii=False, indent=1)}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("output=" + str(out))
    print("sha256=" + _sha256_file(out))


if __name__ == "__main__":
    main()
