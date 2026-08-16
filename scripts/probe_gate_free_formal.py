"""无门控残差自冷却的正式配对实验（Issue #44 预注册协议）。

预注册（运行前冻结，dev 种子 42..44 只用于定标，不进入正式判定）：

- 数据集与轮数：test_300x10（CPU，2000 轮）、nltcs（CUDA，2000 轮）；
- 正式种子：100..104（全新，配对）；
- 共同参数显式固定 rho=0.01（项目标准配置；2026-08-12 审查后由继承默认
  rho=0.1 更正并重跑，见 PR #45 记录）；
- 四臂（gate × cooling 完整对照，审查后由三臂扩展）：
  * ``no_gate_self_cooling``：tol=inf 关闭接受门 + 残差自冷却（p 由 dev 定标
    冻结，写入本文件常量）；
  * ``no_gate``：tol=inf 关闭接受门、恒定扰动（机制消融参考）；
  * ``historical_gate``：主循环历史贪心判据（现默认，baseline）；
  * ``gate_self_cooling``：历史贪心门 + 同一自冷却（隔离门与冷却的贡献）；
- 共同参数与 Issue #43 三臂探索一致（残差定向扩散开启），不使用
  self_cooling_stop_ratio、best 选择、重试或早停干预；
- 主指标：最终状态（非 best）measured workload L1；
- 次级：final/best loss 回漂比、末窗均值、未测量 3-way/4-way L1、联合 TVD
  （原始+分箱）、支持集、墙钟；
- 预注册判定（配对种子均值；dev 定标决定的调整全部记录于此）：
  1. 主判定：candidate（无门+冷却）最终表 measured L1 ≤ 1.10 × 历史门；
  2. 质量判定：candidate 相对历史门的未测量 3-way、4-way、分箱 TVD 任一
     恶化超过 5% 即触发质量风险；
  两条均过 → ``supports_gate_free_self_cooling``；主判定不过 →
  ``not_supported``；主判定过但质量风险触发 → ``inconclusive``。
  回漂比（final/best loss）降级为诊断项、不参与判定：dev 定标显示单调锁温
  （棘轮式回漂消除）反而恶化终点质量（final 96.8 vs 89.2），而主判定已直接
  度量终点质量本身；该调整发生在正式种子运行前，属 dev 定标决策。
- dev 定标记录（seed 42..44，只用于定标）：p∈{0.5,1,1.5,2} 中 p=1 终点最优；
  monotone=False 优于 True；dev 预跑已观察到无门家族未测量 3/4-way 相对
  有门存在 +11%~17% 恶化与唯一状态数下降（复制搅动），正式运行将按上述
  质量判定如实分类，不因此调整门槛。
- 隐私边界：生成阶段只读取公开 schema、查询、marginals 与公开记录数；真实
  参考表（test_300x10.csv；nltcs 只用 train——一次实验一份源数据）只在
  全部生成结束后离线评价。

正式运行要求工作树（含未跟踪文件）干净、输出拒绝覆盖，记录 commit、公开输入
哈希与环境。
"""

import argparse
import hashlib
import json
import os
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
TAIL_WINDOW = 100
# dev 种子 42..44 定标结果写入此常量后冻结（见 PR #45 记录）。
FORMAL_COOLING_EXPONENT = 1.0
PRIMARY_L1_MAX_RATIO = 1.10
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
        # 一次实验一份源数据（第三轮审查）：queries/marginals/target/
        # n_records 全部来自 train（nltcs.csv 与 train 一致），离线参考
        # 因此只用 train。test 若要评价必须以其为源数据独立建实验
        # （queries/marginals/target/n_records=3236 重新构造），不能用
        # train 源合成表事后对比或抽样修补。
        "references": {
            "train": Path("data/nltcs/nltcs.train.data"),
        },
    },
}

SHARED_PARAMS = dict(
    rho=0.01, beta=1.0, h=0.8, eta=0.5, mu=0.01, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    alpha_min=2.0, alpha_max=10.0, exclude_self=True,
)

ARMS = {
    "no_gate_self_cooling": dict(
        tol=float("inf"), residual_self_cooling=FORMAL_COOLING_EXPONENT,
        self_cooling_monotone=False,
    ),
    "no_gate": dict(tol=float("inf")),
    "historical_gate": {},
    # gate×cooling 完整对照（审查意见）：隔离"门"与"冷却"两个因素。
    "gate_self_cooling": dict(
        residual_self_cooling=FORMAL_COOLING_EXPONENT,
        self_cooling_monotone=False,
    ),
}

OUTPUT_PATH = Path(
    "outputs/gate_free_self_cooling/"
    f"formal_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round.json"
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
            # 主指标基于最终状态表（非 best 追踪）；best 只作次级诊断。
            final_table = diag.pop("final_table")
            final_q = evaluate_table(final_table, queries)
            best_q = evaluate_table(best_S, queries)
            # 终态平方 loss 从最终表重算（审查意见 3）：loss_history[-1] 是
            # 最后一次 proposal 之前的状态，最后一轮接受后不再入历史。
            final_loss = float(compute_loss(target, final_q))
            runs.append({
                "dataset": name,
                "seed": int(seed),
                "arm": arm,
                "final_loss": final_loss,
                "pre_final_proposal_loss": float(losses[-1]),
                "best_loss": float(diag["best_loss"]),
                "drift_ratio": float(
                    final_loss / diag["best_loss"]
                ) if diag["best_loss"] > 0 else 1.0,
                # 口径注记：loss_history 是每轮 round-start（最后一次 proposal 之前）
                # 的状态，窗口不含末轮接受后的真实终态，故命名 pre_proposal。
                "tail_mean_pre_proposal_loss": float(
                    np.mean(losses[-TAIL_WINDOW:])
                ),
                "final_table_measured_l1": float(
                    compute_normalized_l1(target, final_q, n_records)
                ),
                "best_table_measured_l1": float(
                    compute_normalized_l1(target, best_q, n_records)
                ),
                "min_cooling_factor": float(
                    min(diag["self_cooling_history"])
                ),
                "rounds_run": int(diag["rounds_run"]),
                "candidate_evaluations": int(
                    diag["candidate_evaluation_count"]
                ),
                "elapsed_sec": float(elapsed),
                "final_table_sha256": _frame_sha256(final_table),
                "best_table_sha256": _frame_sha256(best_S),
            })
            tables[(seed, arm)] = final_table
            print(
                f"[{name}] seed={seed} {arm:22s} "
                f"final={final_loss:11.1f} best={diag['best_loss']:11.1f} "
                f"drift={runs[-1]['drift_ratio']:.3f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return schema, queries, marginals, target, runs, tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="非正式试跑允许脏工作树（输出不作正式判定）")
    args = parser.parse_args()

    formal = (
        args.seeds == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and set(args.datasets) == set(DATASETS)
        and not args.allow_dirty
    )
    environment = _environment()
    if formal and not args.allow_dirty and not environment[
        "git_worktree_clean_including_untracked"
    ]:
        raise RuntimeError("正式协议要求工作树（含未跟踪文件）完全干净")
    if args.output.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{args.output}")

    payload = {
        "experiment": "gate_free_self_cooling_formal",
        "formal_protocol": bool(formal),
        "protocol": {
            "seeds": list(args.seeds),
            "rounds": int(args.rounds),
            "tail_window": TAIL_WINDOW,
            "cooling_exponent": FORMAL_COOLING_EXPONENT,
            "arms": {
                k: {
                    kk: ("inf" if isinstance(vv, float) and np.isinf(vv)
                         else vv)
                    for kk, vv in v.items()
                }
                for k, v in ARMS.items()
            },
            "shared_params": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in SHARED_PARAMS.items()
            },
            "primary_metric": "final_table_measured_l1",
            "thresholds": {
                "primary_l1_max_ratio": PRIMARY_L1_MAX_RATIO,
                "quality_risk_rel": QUALITY_RISK_REL,
                "drift_ratio_role": "diagnostic_only",
            },
        },
        "environment": environment,
        "public_input_sha256": {},
        "datasets": {},
    }

    for name in args.datasets:
        spec = DATASETS[name]
        payload["public_input_sha256"][name] = {
            key: _sha256_file(spec[key])
            for key in ("schema", "queries", "marginals")
        }
        schema, queries, marginals, target, runs, tables = _run_dataset(
            name, spec, args.seeds, args.rounds
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
                    "binned_joint_tvd": float(
                        metrics["binned_joint"]["tvd"]
                    ),
                    "raw_unique_states": int(
                        metrics["raw_joint"]["n_unique"]
                    ),
                    "raw_support_overlap": int(
                        metrics["raw_joint"]["support_overlap"]
                    ),
                }
        payload["datasets"][name] = {
            "reference_sha256": {
                ref_name: _sha256_file(path)
                for ref_name, path in spec["references"].items()
            },
            "runs": runs,
            "judgment": _judge(runs),
        }
        print(json.dumps(
            payload["datasets"][name]["judgment"],
            ensure_ascii=False, indent=1,
        ))

    payload["finished_at"] = datetime.now().astimezone().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(f"output={args.output}")
    print(f"sha256={_sha256_file(args.output)}")


def _arm_mean(runs, arm, metric):
    values = [run[metric] for run in runs if run["arm"] == arm]
    return float(np.mean(values))


def _offline_mean(runs, arm, ref_name, metric):
    values = [
        run["offline"][ref_name][metric]
        for run in runs if run["arm"] == arm
    ]
    return float(np.mean(values))


def _judge(runs):
    candidate = "no_gate_self_cooling"
    baseline = "historical_gate"
    l1_candidate = _arm_mean(runs, candidate, "final_table_measured_l1")
    l1_baseline = _arm_mean(runs, baseline, "final_table_measured_l1")
    drift = _arm_mean(runs, candidate, "drift_ratio")
    primary_pass = l1_candidate <= PRIMARY_L1_MAX_RATIO * l1_baseline
    ref_names = sorted({name for run in runs for name in run["offline"]})
    risks = {}
    for ref_name in ref_names:
        for metric in (
            "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
        ):
            candidate_value = _offline_mean(runs, candidate, ref_name, metric)
            baseline_value = _offline_mean(runs, baseline, ref_name, metric)
            if baseline_value > 0:
                relative = (candidate_value - baseline_value) / baseline_value
            elif candidate_value > 0:
                relative = float("inf")  # baseline 为 0 而 candidate 恶化
            else:
                relative = 0.0
            risks[f"{ref_name}:{metric}"] = {
                "candidate": candidate_value,
                "baseline": baseline_value,
                "relative": (
                    "inf" if np.isinf(relative) else float(relative)
                ),
                "flagged": bool(relative > QUALITY_RISK_REL),
            }
    any_risk = any(item["flagged"] for item in risks.values())
    if primary_pass and not any_risk:
        classification = "supports_gate_free_self_cooling"
    elif not primary_pass:
        classification = "gate_free_self_cooling_not_supported"
    else:
        classification = "gate_free_self_cooling_inconclusive"
    return {
        "primary_l1": {
            "candidate": l1_candidate,
            "baseline": l1_baseline,
            "ratio": float(l1_candidate / l1_baseline)
            if l1_baseline > 0 else None,
            "passed": bool(primary_pass),
        },
        "drift_ratio_diagnostic": {"candidate": drift},
        "quality_risks": risks,
        "any_quality_risk": bool(any_risk),
        "classification": classification,
    }


if __name__ == "__main__":
    main()
