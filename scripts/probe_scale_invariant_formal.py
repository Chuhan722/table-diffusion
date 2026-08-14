#!/usr/bin/env python
"""尺度不变选择正式实验 v3（预注册协议修订版，Issue #44 机制迭代阶段二）。

v3 修订（第二轮审查后重新预注册；v1/v2 输出均归档为非正式历史）：
- NaN 漏洞修复：scale_invariant+exclude_self 时对角 logit 在 softmax 前
  置 -inf（自身占优时其余 donor 下溢 → 0/0 → NaN，双路径回归测试）；
- 证据链对齐 #45/#47：--allow-dirty 无条件 formal=false、输出存在拒绝
  覆盖、参考表只在生成结束后读取（生成前只用 queries record_count 元
  信息校验）、记录 schema/queries/marginals/参考表 SHA-256、原生输出
  initial_state、tail_mean_pre_proposal_loss 口径改名、tol=inf 记录为
  "inf" 字符串；
- 标准化归因参与最终分类：supports_scale_invariant_selection 要求
  attribution 通过，否则 supports_combined_config_only；质量风险改为
  分类后缀 _with_quality_risk；
- 逐行集中度诊断：row_max_prob（均值/最大）、有效 donor 数
  exp(行熵)——全局 top share 不能判断单行确定性选择。

v2 修订（第一轮审查后；v1 输出归档 *.prefix_legacy.json）：
- 代码修复：exclude_self 行统计顺序（标准化统计只在非自身候选上计算）、
  低信号保护 min_spread（放大倍数有界、低离散度平滑退化均匀）；
- 单变量归因臂 no_gate_legacy_a16（不标准化 + alpha≡16）：把"标准化
  本身"与"alpha 数值/调度"的贡献拆开；
- best_loss 改用主循环 diag["best_loss"]（含末轮接受后状态）；
- nltcs 离线参考限定 train（一次实验一份源数据）；
- formal 标志同时校验干净树 + seeds/rounds/datasets/冻结参数与预注册
  一致；
- any_quality_risk 纳入分类（风险时 supports 降级为
  supports_with_quality_risk）。

五臂：{无门 tol=inf, 有门 tol=1e-9} × {si, legacy} + no_gate_legacy_a16。
- si：selection_scale_invariant=True, min_spread=1e-3（dev 冻结），
  alpha_min=alpha_max=16（dev 冻结，seed 42..44 只用于定标）；
- legacy：历史默认谱系 alpha 2..10；
- legacy_a16：不标准化 + alpha≡16（归因对照）。
其余参数全部共享（rho=0.01、ds=2.0、eta/mu/beta/h 同 PR #45 正式协议）。

判定（运行前冻结，主判定数据集 nltcs，最终表 measured L1 五种子均值）：
1. 机制改进：no_gate_si / no_gate_legacy ≤ 0.60；
1b. 标准化归因（新增）：no_gate_si / no_gate_legacy_a16 ≤ 0.90 且
    ≥4/5 种子更低（隔离标准化本身的贡献；不过则改进归因于 alpha 数值）；
2. 门冗余（公平对照）：no_gate_si / gate_si ≤ 1.10；
3. 质量风险带：train 未测量 3/4-way 与分箱 TVD 相对 gate_legacy 劣化
   >5% 报警；报警时 supports 降级为 supports_with_quality_risk；支持集
   唯一状态数与 donor 集中度为观察指标。
分类：判定 1+1b+2 均过 = supports_scale_invariant_selection；判定 1+2
过但 1b 不过 = supports_combined_config_only；仅 1 过 =
mechanism_gain_gate_not_redundant；仅 2 过 = gate_redundant_no_gain；均不
过 = not_supported。supports 分类遇质量风险时追加 _with_quality_risk；
test_300x10 辅助独立判定，不合并结论。

种子 100..104；2000 轮；生成只读公开输入，参考表仅在全部生成完成后
离线读取。
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
FROZEN_MIN_SPREAD = 1e-3  # 低信号保护下限（dev 定标冻结）
MECHANISM_MAX_RATIO = 0.60
ATTRIBUTION_MAX_RATIO = 0.90  # 标准化归因：si vs 同 alpha 不标准化
ATTRIBUTION_MIN_WINS = 4
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
        # 一次实验一份源数据（v2）：queries/marginals/target/n_records
        # 全部来自 train，离线参考因此只用 train；test 若要评价须以其为
        # 源数据独立建实验。
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
    exclude_self=True,
)

ARMS = {
    "no_gate_si": dict(
        tol=float("inf"), selection_scale_invariant=True,
        selection_scale_invariant_min_spread=FROZEN_MIN_SPREAD,
        alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
    ),
    "gate_si": dict(
        tol=1e-9, selection_scale_invariant=True,
        selection_scale_invariant_min_spread=FROZEN_MIN_SPREAD,
        alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
    ),
    "no_gate_legacy": dict(
        tol=float("inf"), alpha_min=2.0, alpha_max=10.0,
    ),
    "gate_legacy": dict(
        tol=1e-9, alpha_min=2.0, alpha_max=10.0,
    ),
    # 单变量归因对照（v2 新增）：与 no_gate_si 只差 scale_invariant 一个
    # 变量（同 alpha≡16 恒定），隔离"标准化本身"的贡献。
    "no_gate_legacy_a16": dict(
        tol=float("inf"), alpha_min=FROZEN_SI_ALPHA,
        alpha_max=FROZEN_SI_ALPHA,
    ),
}

OUTPUT_PATH = Path(
    "outputs/gate_free_self_cooling/"
    f"formal_scale_invariant_v3_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round"
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
    columns_check = schema.attribute_names()
    # 源数据一致性校验（v3）：生成前只允许用公开元信息——查询文件顶层
    # record_count 必须与 n_records 一致（真实参考表在全部生成结束后才
    # 读取并二次校验，不得提前打开，见第二轮审查意见 2）。
    with open(spec["queries"], "r", encoding="utf-8") as handle:
        query_meta = json.load(handle)
    meta_count = query_meta.get("record_count")
    if meta_count is not None and int(meta_count) != n_records:
        raise RuntimeError(
            f"[{name}] queries record_count {meta_count} 与 n_records "
            f"{n_records} 不一致，违反源数据规则"
        )
    # initial_state 原生输出（只依赖公开输入的 n_rounds=0 状态；
    # scripts/audit_formal_json.py 可独立复验）。
    init_l1_by_seed = {}
    init_loss_by_seed = {}
    for seed in seeds:
        _, diag0 = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=n_records, n_rounds=0, seed=seed,
            marginals=marginals, log_every=-1, device=spec["device"],
            return_final_table=True,
            **SHARED_PARAMS, **next(iter(ARMS.values())),
        )
        table0 = diag0.pop("final_table")
        q0 = evaluate_table(table0, queries)
        init_loss_by_seed[str(seed)] = float(compute_loss(q0, target))
        init_l1_by_seed[str(seed)] = float(
            compute_normalized_l1(q0, target, n_records=n_records)
        )
    initial_state = {
        "measured_l1_mean": float(
            np.mean(list(init_l1_by_seed.values()))
        ),
        "measured_l1_by_seed": init_l1_by_seed,
        "loss_by_seed": init_loss_by_seed,
        "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
    }
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
            # 源数据一致性校验（v2）：合成表行数/列名必须与源数据一致。
            if len(final_table) != n_records or (
                list(final_table.columns) != columns_check
            ):
                raise RuntimeError(
                    f"[{name} seed={seed} {arm}] 合成表行数/列名与源数据"
                    "不一致"
                )
            runs.append({
                "dataset": name,
                "arm": arm,
                "seed": seed,
                "rounds_run": diag["rounds_run"],
                "candidate_evaluations": diag["candidate_evaluation_count"],
                "pre_final_proposal_loss": float(losses[-1]),
                "final_loss": final_loss,
                "final_table_measured_l1": final_l1,
                # v2：用主循环的 best_loss（含末轮接受后状态），修正
                # min(loss_history) 遗漏最终状态导致 best>final 的问题。
                "best_loss": float(diag["best_loss"]),
                "best_table_measured_l1": best_l1,
                "donor_top_share_max": (
                    float(max(diag["donor_top_share_history"]))
                    if diag.get("donor_top_share_history") else None
                ),
                # 逐行集中度诊断（第二轮审查意见 4）：全局 top share 不能
                # 判断单行 softmax 是否接近确定性。
                "row_max_prob_mean_final": (
                    float(diag["row_max_prob_mean_history"][-1])
                    if diag.get("row_max_prob_mean_history") else None
                ),
                "row_max_prob_max_overall": (
                    float(max(diag["row_max_prob_max_history"]))
                    if diag.get("row_max_prob_max_history") else None
                ),
                "effective_donors_mean_final": (
                    float(diag["effective_donors_mean_history"][-1])
                    if diag.get("effective_donors_mean_history") else None
                ),
                # 口径注记：loss_history 为 round-start/pre-proposal 状态窗口。
                "tail_mean_pre_proposal_loss": float(
                    np.mean(losses[-100:])
                ),
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
    # 参考表形状二次校验（离线阶段，生成已全部结束）。
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
    no_gate_legacy_a16 = _arm_mean(runs, "no_gate_legacy_a16", metric)

    mechanism_ratio = (
        no_gate_si / no_gate_legacy if no_gate_legacy > 0 else float("inf")
    )
    redundancy_ratio = (
        no_gate_si / gate_si if gate_si > 0 else float("inf")
    )
    attribution_ratio = (
        no_gate_si / no_gate_legacy_a16
        if no_gate_legacy_a16 > 0 else float("inf")
    )
    mechanism_passed = mechanism_ratio <= MECHANISM_MAX_RATIO
    redundancy_passed = redundancy_ratio <= REDUNDANCY_MAX_RATIO

    per_seed_wins = 0
    attribution_wins = 0
    seeds = sorted({run["seed"] for run in runs})
    for seed in seeds:
        si = [r[metric] for r in runs
              if r["arm"] == "no_gate_si" and r["seed"] == seed][0]
        legacy = [r[metric] for r in runs
                  if r["arm"] == "no_gate_legacy" and r["seed"] == seed][0]
        a16 = [r[metric] for r in runs
               if r["arm"] == "no_gate_legacy_a16" and r["seed"] == seed][0]
        if si < legacy:
            per_seed_wins += 1
        if si < a16:
            attribution_wins += 1
    attribution_passed = (
        attribution_ratio <= ATTRIBUTION_MAX_RATIO
        and attribution_wins >= ATTRIBUTION_MIN_WINS
    )

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

    # v3：标准化归因参与最终分类（第二轮审查意见 3）——分类名要支持
    # "尺度不变机制"，归因判定必须通过；机制+冗余过但归因不过时只能
    # 宣称"组合配置有效"。质量风险仍触发降级后缀。
    if mechanism_passed and redundancy_passed and attribution_passed:
        classification = "supports_scale_invariant_selection"
    elif mechanism_passed and redundancy_passed:
        classification = "supports_combined_config_only"
    elif mechanism_passed:
        classification = "mechanism_gain_gate_not_redundant"
    elif redundancy_passed:
        classification = "gate_redundant_no_gain"
    else:
        classification = "not_supported"
    if any_quality_risk and classification.startswith("supports"):
        classification += "_with_quality_risk"

    return {
        "arm_means": {
            "no_gate_si": no_gate_si,
            "gate_si": gate_si,
            "no_gate_legacy": no_gate_legacy,
            "gate_legacy": gate_legacy,
            "no_gate_legacy_a16": no_gate_legacy_a16,
        },
        "mechanism": {
            "ratio": mechanism_ratio,
            "threshold": MECHANISM_MAX_RATIO,
            "passed": mechanism_passed,
            "no_gate_si_wins": per_seed_wins,
        },
        # v3 起参与最终分类（supports_scale_invariant_selection 要求
        # 归因通过）：隔离"标准化本身"vs"alpha 数值"的贡献。
        "standardization_attribution": {
            "ratio": attribution_ratio,
            "threshold": ATTRIBUTION_MAX_RATIO,
            "min_wins": ATTRIBUTION_MIN_WINS,
            "wins": attribution_wins,
            "passed": attribution_passed,
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
    out_path = Path(args.out)
    if out_path.exists():
        print(f"输出已存在，不覆盖：{out_path}")
        sys.exit(1)

    # v2：formal 标志同时校验协议参数与预注册一致——种子、轮数、数据集
    # 任何偏离都强制 formal=false，防止非正式运行被误标记。
    protocol_deviations = []
    if list(args.seeds) != FORMAL_SEEDS:
        protocol_deviations.append(f"seeds={args.seeds} != {FORMAL_SEEDS}")
    if args.rounds != FORMAL_ROUNDS:
        protocol_deviations.append(f"rounds={args.rounds} != {FORMAL_ROUNDS}")
    if sorted(args.datasets) != sorted(DATASETS):
        protocol_deviations.append(
            f"datasets={sorted(args.datasets)} != {sorted(DATASETS)}"
        )

    # --allow-dirty 无条件强制非正式（与 #45/#47 规范对齐）：脏树试跑
    # 即使协议参数与预注册一致也不得标记为正式产物。
    formal_flag = (
        environment["git_worktree_clean_including_untracked"]
        and not protocol_deviations
        and not args.allow_dirty
    )
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
                    # tol=inf 显式记录为 "inf" 字符串（None 含义不清）。
                    k: ("inf" if isinstance(v, float) and np.isinf(v) else v)
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
                "attribution_max_ratio": ATTRIBUTION_MAX_RATIO,
                "attribution_min_wins": ATTRIBUTION_MIN_WINS,
                "redundancy_max_ratio": REDUNDANCY_MAX_RATIO,
                "quality_risk_rel": QUALITY_RISK_REL,
            },
            "dev_calibration_note": (
                "si alpha=16 与 min_spread=1e-3 由 dev seed 42..44 定标"
                "冻结；dev 种子不进正式种子集；支持集唯一状态数与 donor"
                "集中度为观察指标（dev 已知高锐度收窄支持集）"
            ),
            "frozen_min_spread": FROZEN_MIN_SPREAD,
        },
        "environment": environment,
        "protocol_deviations": protocol_deviations,
        "formal": formal_flag,
        "public_input_sha256": {
            name: {
                key: _sha256_file(DATASETS[name][key])
                for key in ("schema", "queries", "marginals")
            }
            for name in args.datasets
        },
        "datasets": {},
    }
    if protocol_deviations:
        print("协议偏离（formal=false）: " + "; ".join(protocol_deviations))

    for name in args.datasets:
        spec = DATASETS[name]
        runs, initial_state, reference_sha256 = _run_dataset(
            name, spec, args.seeds, args.rounds
        )
        judgment = _judge(runs)
        payload["datasets"][name] = {
            "reference_sha256": reference_sha256,
            "initial_state": initial_state,
            "runs": runs,
            "judgment": judgment,
        }
        print(f"== {name}: {json.dumps(judgment, ensure_ascii=False, indent=1)}",
              flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("output=" + str(out_path))
    print("sha256=" + _sha256_file(out_path))


if __name__ == "__main__":
    main()
