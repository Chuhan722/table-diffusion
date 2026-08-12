"""扩散必要性 2×2 消融（Issue #43 预注册协议）。

回答两个归因问题：

1. **扩散方向是否必要**：无门条件下，扩散核（fitness/距离加权 donor + 残差
   定向复制）是否显著优于随机核（均匀 donor + 无定向复制，其余扰动结构与
   冷却完全相同）；
2. **门是否遮蔽归因**：有门条件下，随机核是否接近扩散核——若接近，说明贪心
   门本身就能驱动收敛，门内比较无法体现扩散的贡献。

四臂（2×2，配对种子，等轮数等候选预算）：

- ``diffusion_no_gate_cooling``：扩散核 + 无门 + 残差自冷却（p=1，非单调）；
- ``random_no_gate_cooling``：随机核（alpha=0 → geometric 抽样退化为均匀
  donor；residual_directed_diffusion=False）+ 无门 + 同一冷却；
- ``diffusion_gate``：扩散核 + 历史贪心门（现默认）；
- ``random_gate``：随机核 + 历史贪心门。

随机核只移除方向来源（donor 加权与定向复制），保留 rho/eta/mu 扰动结构与
冷却调度（冷却只缩放幅度、不提供方向），使唯一变量是"方向是否来自分布"。

预注册判定（配对种子均值，最终表 measured L1）：

- ``diffusion_necessary_no_gate``：无门条件 扩散/随机 比值 ≤ 0.90 且 ≥4/5
  种子扩散更低；
- ``gate_confounds_attribution``：有门条件 随机/扩散 比值 ≤ 1.10（随机进入
  扩散的 10% 以内即判门遮蔽归因）；
- 两个标志独立报告，不合成单一分类。质量指标（未测量 3/4-way、分箱 TVD）
  全臂报告，不设门槛，只作解释。

协议与 ``probe_gate_free_formal.py`` 共享数据集、种子（100..104）、轮数
（2000）与共同参数（含 2026-08-12 审查后显式固定的 rho=0.01；此前首轮运行
继承默认 rho=0.1，保留为历史记录）；扩散臂与正式运行完全同配置，其结果可与
正式 JSON 交叉核对。隐私边界一致：生成只读公开输入，参考表在数据集全部生成后离线评价。
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
TAIL_WINDOW = 100
COOLING_EXPONENT = 1.0
NECESSITY_MAX_RATIO = 0.90
NECESSITY_MIN_WINS = 4
CONFOUND_MAX_RATIO = 1.10

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

BASE_PARAMS = dict(
    rho=0.01, beta=1.0, h=0.8, eta=0.5, mu=0.01, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", exclude_self=True,
)

DIFFUSION_KERNEL = dict(
    residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    alpha_min=2.0, alpha_max=10.0,
)

# alpha=0 时 geometric 抽样 softmax(0·log A) 退化为逐行均匀分布（exclude_self
# 下排除自身），fitness 与距离都不再影响 donor；再关闭残差定向复制即为随机核。
RANDOM_KERNEL = dict(
    residual_directed_diffusion=False,
    alpha_min=0.0, alpha_max=0.0,
)

NO_GATE_COOLING = dict(
    tol=float("inf"),
    residual_self_cooling=COOLING_EXPONENT,
    self_cooling_monotone=False,
)

ARMS = {
    "diffusion_no_gate_cooling": {**DIFFUSION_KERNEL, **NO_GATE_COOLING},
    "random_no_gate_cooling": {**RANDOM_KERNEL, **NO_GATE_COOLING},
    "diffusion_gate": {**DIFFUSION_KERNEL},
    "random_gate": {**RANDOM_KERNEL},
}

OUTPUT_PATH = Path(
    "outputs/gate_free_self_cooling/"
    f"necessity_ablation_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round.json"
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
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": (
            _git("status", "--porcelain") == ""
        ),
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


def _serializable(value):
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    if isinstance(value, tuple):
        return list(value)
    return value


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
                **BASE_PARAMS, **extra,
            )
            elapsed = time.perf_counter() - start
            losses = diag["loss_history"]
            final_table = diag.pop("final_table")
            final_q = evaluate_table(final_table, queries)
            # 终态平方 loss 从最终表重算（与 formal 脚本同口径修正）。
            final_loss = float(compute_loss(target, final_q))
            runs.append({
                "dataset": name,
                "seed": int(seed),
                "arm": arm,
                "final_loss": final_loss,
                "pre_final_proposal_loss": float(losses[-1]),
                "best_loss": float(diag["best_loss"]),
                "tail_mean_loss": float(np.mean(losses[-TAIL_WINDOW:])),
                "final_table_measured_l1": float(
                    compute_normalized_l1(target, final_q, n_records)
                ),
                "rounds_run": int(diag["rounds_run"]),
                "candidate_evaluations": int(
                    diag["candidate_evaluation_count"]
                ),
                "elapsed_sec": float(elapsed),
                "final_table_sha256": _frame_sha256(final_table),
            })
            tables[(seed, arm)] = final_table
            print(
                f"[{name}] seed={seed} {arm:26s} "
                f"final={final_loss:12.1f} "
                f"L1={runs[-1]['final_table_measured_l1']:.6f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return schema, queries, marginals, runs, tables


def _arm_values(runs, arm, metric):
    return [run[metric] for run in runs if run["arm"] == arm]


def _judge(runs):
    metric = "final_table_measured_l1"
    diffusion_ng = _arm_values(runs, "diffusion_no_gate_cooling", metric)
    random_ng = _arm_values(runs, "random_no_gate_cooling", metric)
    diffusion_g = _arm_values(runs, "diffusion_gate", metric)
    random_g = _arm_values(runs, "random_gate", metric)
    ng_ratio = float(np.mean(diffusion_ng) / np.mean(random_ng))
    ng_wins = int(sum(
        d < r for d, r in zip(diffusion_ng, random_ng)
    ))
    gate_ratio = float(np.mean(random_g) / np.mean(diffusion_g))
    return {
        "no_gate": {
            "diffusion_mean_l1": float(np.mean(diffusion_ng)),
            "random_mean_l1": float(np.mean(random_ng)),
            "diffusion_over_random_ratio": ng_ratio,
            "diffusion_wins": ng_wins,
            "diffusion_necessary_no_gate": bool(
                ng_ratio <= NECESSITY_MAX_RATIO
                and ng_wins >= NECESSITY_MIN_WINS
            ),
        },
        "gate": {
            "diffusion_mean_l1": float(np.mean(diffusion_g)),
            "random_mean_l1": float(np.mean(random_g)),
            "random_over_diffusion_ratio": gate_ratio,
            "gate_confounds_attribution": bool(
                gate_ratio <= CONFOUND_MAX_RATIO
            ),
        },
        "thresholds": {
            "necessity_max_ratio": NECESSITY_MAX_RATIO,
            "necessity_min_wins": NECESSITY_MIN_WINS,
            "confound_max_ratio": CONFOUND_MAX_RATIO,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    formal = (
        args.seeds == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and set(args.datasets) == set(DATASETS)
    )
    environment = _environment()
    if formal and not args.allow_dirty and not environment[
        "git_worktree_clean_including_untracked"
    ]:
        raise RuntimeError("正式协议要求工作树（含未跟踪文件）完全干净")
    if args.output.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{args.output}")

    payload = {
        "experiment": "diffusion_necessity_2x2_ablation",
        "formal_protocol": bool(formal),
        "protocol": {
            "seeds": list(args.seeds),
            "rounds": int(args.rounds),
            "tail_window": TAIL_WINDOW,
            "arms": {
                arm: {key: _serializable(value)
                      for key, value in extra.items()}
                for arm, extra in ARMS.items()
            },
            "base_params": {
                key: _serializable(value)
                for key, value in BASE_PARAMS.items()
            },
            "primary_metric": "final_table_measured_l1",
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
        schema, queries, marginals, runs, tables = _run_dataset(
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
                    "binned_joint_tvd": float(
                        metrics["binned_joint"]["tvd"]
                    ),
                    "raw_unique_states": int(
                        metrics["raw_joint"]["n_unique"]
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


if __name__ == "__main__":
    main()
