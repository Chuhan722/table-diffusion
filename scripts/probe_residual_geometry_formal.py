#!/usr/bin/env python
"""残差信号几何正式实验（Issue #57 预注册协议）。

诊断背景（dev，见 docs/设计/残差信号几何_绝对与相对适应度口径.md）：
绝对残差口径把大计数查询压到抽样涨落之下（bootstrap iid 参考线
0.001598±0.000174，平台 0.000864-0.0011 已低于该线），而对稀有模式
查询系统性无力（p<0.05 查询占平台残差 L1 44.8%，bootstrap 对照
22.3%）。相对残差近似 KL 梯度口径，dev 三种子（42..44）配对改善
-70.5%。

五臂（唯一变量 = residual_geometry 及 floor；其余共享 #48 v3 冻结的
no_gate_si 配置：tol=inf、si α≡16、min_spread=1e-3、ds=2.0、rho=0.01、
marginal 初始化）：
- absolute：现状口径（baseline）；
- relative_f8：相对残差 floor=8（主 candidate，dev 定标）；
- relative_f1 / relative_f4 / relative_f16：floor 敏感性次要臂。

判定（运行前冻结，主判定数据集 nltcs，最终表 measured L1 五种子）：
1. 主判定：relative_f8 配对 5/5 种子低于 absolute 且均值改善 ≥30%
   → supports_relative_geometry；3-4/5 或均值改善 <30% → mixed；
   否则 not_supported。
2. floor 稳健性（观察，不进主判定）：若 f8 不是四个 floor 臂中均值
   最优，结果标注 floor_suboptimal 提示（供后续定标，不改变主分类）。
3. 质量风险带：train 未测量 3/4-way L1 与分箱 TVD 相对 absolute 劣化
   >5% 报警；报警时 supports 降级为 supports_with_quality_risk。
4. test_300x10 辅助独立判定（冒烟一致性），不合并主结论。

种子 200..204（与 dev 42..44 不重叠）；2000 轮固定预算（同轮数口径，
不含墙钟等价声明）；生成只读公开输入，参考表仅在全部生成完成后离线
读取。失败结果全部保留。
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

FORMAL_SEEDS = [200, 201, 202, 203, 204]
FORMAL_ROUNDS = 2000
FROZEN_SI_ALPHA = 16.0   # #48 v3 冻结
FROZEN_MIN_SPREAD = 1e-3  # #48 v3 冻结
PRIMARY_CANDIDATE = "relative_f8"
PRIMARY_BASELINE = "absolute"
PRIMARY_MIN_WINS = 5
PRIMARY_MIN_IMPROVEMENT = 0.30  # 均值改善 ≥30%
MIXED_MIN_WINS = 3
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
        # 一次实验一份源数据：queries/marginals/target/n_records 全部
        # 来自 train，离线参考因此只用 train。
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
    exclude_self=True, tol=float("inf"),
    selection_scale_invariant=True,
    selection_scale_invariant_min_spread=FROZEN_MIN_SPREAD,
    alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
)

ARMS = {
    "absolute": dict(residual_geometry="absolute"),
    "relative_f8": dict(
        residual_geometry="relative", residual_geometry_floor=8.0,
    ),
    "relative_f1": dict(
        residual_geometry="relative", residual_geometry_floor=1.0,
    ),
    "relative_f4": dict(
        residual_geometry="relative", residual_geometry_floor=4.0,
    ),
    "relative_f16": dict(
        residual_geometry="relative", residual_geometry_floor=16.0,
    ),
}

OUTPUT_PATH = Path(
    "outputs/residual_geometry/"
    f"formal_residual_geometry_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round"
    ".json"
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
        ["git", *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _load_reference(path, columns):
    frame = pd.read_csv(path)
    if list(frame.columns) != columns:
        frame.columns = columns
    return frame


def _run_dataset(name, spec, seeds, rounds):
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n_records = spec["n_records"]
    columns_check = schema.attribute_names()
    # 源数据一致性校验：生成前只允许用公开元信息。
    with open(spec["queries"], "r", encoding="utf-8") as handle:
        query_meta = json.load(handle)
    meta_count = query_meta.get("record_count")
    if meta_count is not None and int(meta_count) != n_records:
        raise RuntimeError(
            f"[{name}] queries record_count {meta_count} 与 n_records "
            f"{n_records} 不一致，违反源数据规则"
        )
    # initial_state 原生输出（n_rounds=0，只依赖公开输入；
    # scripts/audit_formal_json.py 可独立复验）。
    init_l1_by_seed = {}
    init_loss_by_seed = {}
    for seed in seeds:
        _, diag0 = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=n_records, n_rounds=0, seed=seed,
            marginals=marginals, log_every=-1, device=spec["device"],
            return_final_table=True,
            **SHARED_PARAMS, **ARMS[PRIMARY_BASELINE],
        )
        table0 = diag0.pop("final_table")
        q0 = evaluate_table(table0, queries)
        init_loss_by_seed[str(seed)] = float(compute_loss(q0, target))
        init_l1_by_seed[str(seed)] = float(
            compute_normalized_l1(q0, target, n_records=n_records)
        )
    initial_state = {
        "measured_l1_mean": float(np.mean(list(init_l1_by_seed.values()))),
        "measured_l1_by_seed": init_l1_by_seed,
        "loss_by_seed": init_loss_by_seed,
        "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
    }
    runs = []
    tables = {}
    for seed in seeds:
        for arm, extra in ARMS.items():
            start = time.perf_counter()
            _, diag = run_evolution(
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
            tables[(seed, arm)] = final_table
            if len(final_table) != n_records or (
                list(final_table.columns) != columns_check
            ):
                raise RuntimeError(
                    f"[{name} seed={seed} {arm}] 合成表行数/列名与源数据"
                    "不一致"
                )
            raw_abs = np.abs(target - final_q)
            p_rate = target / n_records
            rare_mask = p_rate < 0.05
            runs.append({
                "dataset": name,
                "arm": arm,
                "seed": seed,
                "rounds_run": diag["rounds_run"],
                "candidate_evaluations": diag["candidate_evaluation_count"],
                "pre_final_proposal_loss": float(losses[-1]),
                "final_loss": final_loss,
                "final_table_measured_l1": final_l1,
                "best_loss": float(diag["best_loss"]),
                # 稀有查询残差诊断（本实验的机制指标）：p<0.05 与其余
                # 查询的平均绝对计数残差。
                "rare_query_mean_abs_residual": (
                    float(raw_abs[rare_mask].mean())
                    if rare_mask.any() else None
                ),
                "common_query_mean_abs_residual": (
                    float(raw_abs[~rare_mask].mean())
                    if (~rare_mask).any() else None
                ),
                "exact_match_queries": int((raw_abs == 0).sum()),
                "row_max_prob_mean_final": (
                    float(diag["row_max_prob_mean_history"][-1])
                    if diag.get("row_max_prob_mean_history") else None
                ),
                "effective_donors_mean_final": (
                    float(diag["effective_donors_mean_history"][-1])
                    if diag.get("effective_donors_mean_history") else None
                ),
                # 口径注记：loss_history 为 round-start/pre-proposal 状态窗口。
                "tail_mean_pre_proposal_loss": float(np.mean(losses[-100:])),
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
    for ref_name, reference in references.items():
        if len(reference) != n_records:
            raise RuntimeError(
                f"[{name}] 参考 {ref_name} 行数 {len(reference)} 与"
                f" n_records {n_records} 不一致，违反源数据规则"
            )
        if list(reference.columns) != columns:
            raise RuntimeError(f"[{name}] 参考 {ref_name} 列名不一致")
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
    reference_sha256 = {
        ref_name: _sha256_file(path)
        for ref_name, path in spec["references"].items()
    }
    return runs, initial_state, reference_sha256


def _arm_metric_by_seed(runs, arm, metric):
    return {
        run["seed"]: run[metric]
        for run in runs if run["arm"] == arm
    }


def _arm_mean(runs, arm, metric):
    values = [run[metric] for run in runs if run["arm"] == arm]
    return float(np.mean(values))


def _offline_mean(runs, arm, ref_name, metric):
    values = [
        run["offline"][ref_name][metric]
        for run in runs if run["arm"] == arm and ref_name in run["offline"]
    ]
    return float(np.mean(values)) if values else None


def _judge(runs, ref_name):
    metric = "final_table_measured_l1"
    base_by_seed = _arm_metric_by_seed(runs, PRIMARY_BASELINE, metric)
    cand_by_seed = _arm_metric_by_seed(runs, PRIMARY_CANDIDATE, metric)
    seeds = sorted(base_by_seed)
    wins = sum(
        1 for seed in seeds if cand_by_seed[seed] < base_by_seed[seed]
    )
    base_mean = float(np.mean([base_by_seed[s] for s in seeds]))
    cand_mean = float(np.mean([cand_by_seed[s] for s in seeds]))
    improvement = (
        (base_mean - cand_mean) / base_mean if base_mean > 0 else 0.0
    )
    primary_pass = (
        wins >= PRIMARY_MIN_WINS and improvement >= PRIMARY_MIN_IMPROVEMENT
    )
    mixed = (not primary_pass) and (
        wins >= MIXED_MIN_WINS and cand_mean < base_mean
    )
    # floor 稳健性（观察项，不进主判定）
    floor_means = {
        arm: _arm_mean(runs, arm, metric)
        for arm in ARMS if arm != PRIMARY_BASELINE
    }
    floor_best = min(floor_means, key=floor_means.get)
    floor_suboptimal = floor_best != PRIMARY_CANDIDATE
    # 质量风险带：train 侧未测量 3/4-way 与分箱 TVD 相对 absolute
    # 劣化 >QUALITY_RISK_REL 报警。
    risks = {}
    for quality_metric in (
        "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
    ):
        base_q = _offline_mean(runs, PRIMARY_BASELINE, ref_name, quality_metric)
        cand_q = _offline_mean(runs, PRIMARY_CANDIDATE, ref_name, quality_metric)
        if base_q is None or cand_q is None or base_q <= 0:
            risks[quality_metric] = None
            continue
        rel = (cand_q - base_q) / base_q
        risks[quality_metric] = {
            "baseline_mean": base_q,
            "candidate_mean": cand_q,
            "relative_change": float(rel),
            "flagged": bool(rel > QUALITY_RISK_REL),
        }
    any_quality_risk = any(
        item is not None and item["flagged"] for item in risks.values()
    )
    if primary_pass:
        classification = "supports_relative_geometry"
        if any_quality_risk:
            classification += "_with_quality_risk"
    elif mixed:
        classification = "mixed"
    else:
        classification = "not_supported"
    return {
        "metric": metric,
        "primary_baseline": PRIMARY_BASELINE,
        "primary_candidate": PRIMARY_CANDIDATE,
        "baseline_l1_by_seed": {str(s): base_by_seed[s] for s in seeds},
        "candidate_l1_by_seed": {str(s): cand_by_seed[s] for s in seeds},
        "paired_wins": int(wins),
        "n_seeds": len(seeds),
        "baseline_mean": base_mean,
        "candidate_mean": cand_mean,
        "relative_improvement": float(improvement),
        "primary_pass": bool(primary_pass),
        "floor_arm_means": floor_means,
        "floor_best_arm": floor_best,
        "floor_suboptimal_flag": bool(floor_suboptimal),
        "quality_risks": risks,
        "any_quality_risk": bool(any_quality_risk),
        "classification": classification,
        "thresholds": {
            "primary_min_wins": PRIMARY_MIN_WINS,
            "primary_min_improvement": PRIMARY_MIN_IMPROVEMENT,
            "mixed_min_wins": MIXED_MIN_WINS,
            "quality_risk_rel": QUALITY_RISK_REL,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="允许脏工作树运行（输出无条件标记 formal=false）",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=FORMAL_SEEDS,
        help="种子列表（偏离预注册值时输出标记 formal=false）",
    )
    parser.add_argument(
        "--rounds", type=int, default=FORMAL_ROUNDS,
        help="每个 run 的轮数（偏离预注册值时输出标记 formal=false）",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS),
        help="要运行的数据集（偏离预注册全集时输出标记 formal=false）",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="输出 JSON 路径（已存在时拒绝覆盖）",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"输出 {args.output} 已存在，拒绝覆盖；请换路径或手动移除"
        )

    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit("工作树不干净；正式运行要求干净树（或 --allow-dirty）")

    matches_prereg = (
        not dirty
        and list(args.seeds) == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and sorted(args.datasets) == sorted(DATASETS)
    )
    formal = bool(matches_prereg and not args.allow_dirty)

    input_sha256 = {}
    for name in args.datasets:
        spec = DATASETS[name]
        input_sha256[name] = {
            "schema": _sha256_file(spec["schema"]),
            "queries": _sha256_file(spec["queries"]),
            "marginals": _sha256_file(spec["marginals"]),
        }

    result = {
        "protocol": {
            "issue": 57,
            "seeds": list(args.seeds),
            "rounds": args.rounds,
            "datasets": list(args.datasets),
            "arms": {
                arm: {
                    key: (str(value) if value == float("inf") else value)
                    for key, value in extra.items()
                }
                for arm, extra in ARMS.items()
            },
            "shared_params": {
                key: (str(value) if value == float("inf") else value)
                for key, value in SHARED_PARAMS.items()
            },
            "primary_dataset": "nltcs",
            "frozen_si_alpha": FROZEN_SI_ALPHA,
            "frozen_min_spread": FROZEN_MIN_SPREAD,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": dirty,
            "formal": formal,
            "started_at": datetime.now().astimezone().isoformat(),
            "environment": _environment(),
            "input_sha256": input_sha256,
            "command": " ".join(sys.argv),
        },
        "datasets": {},
    }

    for name in args.datasets:
        spec = DATASETS[name]
        print(f"=== 数据集 {name} ===", flush=True)
        runs, initial_state, reference_sha256 = _run_dataset(
            name, spec, args.seeds, args.rounds
        )
        ref_name = next(iter(spec["references"]))
        result["datasets"][name] = {
            "initial_state": initial_state,
            "reference_sha256": reference_sha256,
            "runs": runs,
            "judgement": _judge(runs, ref_name),
        }
        print(
            f"[{name}] 判定: "
            f"{result['datasets'][name]['judgement']['classification']}",
            flush=True,
        )

    result["provenance"]["finished_at"] = (
        datetime.now().astimezone().isoformat()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.output} (formal={formal})", flush=True)


if __name__ == "__main__":
    main()
