"""运行并离线评价 nltcs 残差驱动扩散算子。

生成阶段固定使用原始 marginal、精确 measured workload 和 geometric donor
配置。真实 train/test 只在全部 run_evolution 调用完成后读取，不进入方向势能、
原始提案、接受、早停或强度选择。
"""

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.io import create_parent_dir, save_run, save_summary
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
TRAIN_PATH = "data/nltcs/nltcs.csv"
TEST_PATH = "data/nltcs/nltcs.test.data"
N_RECORDS = 16_181


def _joint_distribution(frame, columns):
    values = frame[columns].to_numpy(dtype=np.int64)
    weights = 1 << np.arange(len(columns), dtype=np.int64)
    codes = values @ weights
    return np.bincount(codes, minlength=2 ** len(columns)) / len(values)


def _joint_metrics(synthetic, train_probability, test_probability, columns):
    synthetic_probability = _joint_distribution(synthetic, columns)
    return {
        "train_tvd": float(
            0.5 * np.abs(synthetic_probability - train_probability).sum()
        ),
        "test_tvd": float(
            0.5 * np.abs(synthetic_probability - test_probability).sum()
        ),
        "n_unique": int(np.count_nonzero(synthetic_probability)),
        "missing_train_mass": float(
            train_probability[synthetic_probability == 0].sum()
        ),
        "novel_synthetic_mass": float(
            synthetic_probability[train_probability == 0].sum()
        ),
    }


def _aggregate(records, key):
    values = np.asarray([record[key] for record in records], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def _mean_or_zero(values):
    return float(np.mean(values)) if len(values) else 0.0


def _mean_without_none(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else 0.0


def _first_attempts(nested):
    return [float(values[0]) for values in nested if values]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*args):
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def _environment_snapshot(device):
    commit_code, commit = _git_text("rev-parse", "HEAD")
    status_code, status = _git_text("status", "--porcelain")
    snapshot = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_worktree_clean": status_code == 0 and status == "",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device": device,
    }
    try:
        import torch
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False
        snapshot["gpu"] = None
    else:
        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["gpu"] = (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available() else None
        )
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument(
        "--direction-strength", type=float, default=1.0
    )
    parser.add_argument(
        "--direction-normalization",
        choices=["none", "initial_rms"],
        default="initial_rms",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="关闭方向机制，运行同配置历史扩散核对照",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds 必须为正数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if (
        not np.isfinite(args.direction_strength)
        or args.direction_strength < 0.0
    ):
        parser.error("--direction-strength 必须是非负有限数值")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries], dtype=float)
    marginals = load_marginals(MARGINALS_PATH)
    columns = schema.attribute_names()
    environment = _environment_snapshot(args.device)
    direction_enabled = not args.baseline

    if args.output_dir:
        parent = Path(args.output_dir)
        if parent.exists() and any(parent.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空，不覆盖：{parent}")
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path(create_parent_dir(prefix="residual_directed_nltcs"))

    runs = []
    generated_tables = []
    for position, seed in enumerate(args.seeds):
        print(
            f"\n===== 种子 {seed}（{position + 1}/{len(args.seeds)}）=====",
            flush=True,
        )
        best, diagnostics = run_evolution(
            target,
            queries,
            schema,
            n_records=N_RECORDS,
            n_rounds=args.rounds,
            seed=seed,
            beta=1.0,
            h=0.8,
            rho=0.01,
            eta=0.5,
            mu=0.01,
            device=args.device,
            eval_method="vectorized",
            batch_size=256,
            init_method="marginal",
            marginals=marginals,
            log_every=args.log_every,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha_min=2.0,
            alpha_max=10.0,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
            max_retries=0,
            residual_directed_diffusion=direction_enabled,
            diffusion_direction_strength=args.direction_strength,
            diffusion_direction_normalization=args.direction_normalization,
        )

        run_dir = parent / f"{position}-{seed}"
        save_run(best, diagnostics, run_dir=str(run_dir))
        accept_history = diagnostics["accept_history"]
        loss_history = diagnostics["loss_history"]
        raw_gains = _first_attempts(
            diagnostics["raw_proposal_gain_history"]
        )
        raw_linear = _first_attempts(
            diagnostics["raw_proposal_linear_gain_history"]
        )
        raw_quadratic = _first_attempts(
            diagnostics["raw_proposal_quadratic_penalty_history"]
        )
        raw_gain_array = np.asarray(raw_gains, dtype=float)
        raw_late_start = 3 * len(raw_gains) // 4
        late_raw_gains = raw_gain_array[raw_late_start:]
        accept_late_start = 3 * len(accept_history) // 4
        loss_late_start = 3 * len(loss_history) // 4
        record = {
            "seed": int(seed),
            "run_dir": run_dir.name,
            "best_loss": float(diagnostics["best_loss"]),
            "training_l1": float(diagnostics["normalized_l1_error"]),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
            "direction_evaluation_elapsed_sec": float(
                diagnostics["direction_evaluation_elapsed_sec"]
            ),
            "rounds_run": int(diagnostics["rounds_run"]),
            "stopped_early": bool(diagnostics["stopped_early"]),
            "accept_rate": _mean_or_zero(accept_history),
            "late_accept_rate": _mean_or_zero(
                accept_history[accept_late_start:]
            ),
            "late_loss_improvement": float(
                loss_history[loss_late_start] - diagnostics["best_loss"]
            ),
            "raw_proposal_gain_mean": _mean_or_zero(raw_gains),
            "raw_proposal_positive_rate": _mean_or_zero(
                raw_gain_array > 0.0
            ),
            "raw_positive_gain_mean": _mean_or_zero(
                raw_gain_array[raw_gain_array > 0.0]
            ),
            "raw_negative_gain_mean": _mean_or_zero(
                raw_gain_array[raw_gain_array < 0.0]
            ),
            "late_raw_proposal_gain_mean": _mean_or_zero(late_raw_gains),
            "late_raw_proposal_positive_rate": _mean_or_zero(
                late_raw_gains > 0.0
            ),
            "raw_proposal_linear_gain_mean": _mean_or_zero(raw_linear),
            "raw_proposal_quadratic_penalty_mean": _mean_or_zero(
                raw_quadratic
            ),
            "copy_direction_mean": _mean_without_none(
                diagnostics["copy_direction_mean_history"]
            ),
            "copy_direction_positive_rate": _mean_without_none(
                diagnostics["copy_direction_positive_rate_history"]
            ),
            "copy_direction_negative_rate": _mean_without_none(
                diagnostics["copy_direction_negative_rate_history"]
            ),
            "negative_direction_copy_probability": _mean_without_none(
                diagnostics["negative_direction_copy_probability_history"]
            ),
            "positive_direction_copy_probability": _mean_without_none(
                diagnostics["positive_direction_copy_probability_history"]
            ),
            "copy_probability_entropy": _mean_without_none(
                diagnostics["copy_probability_entropy_history"]
            ),
            "direction_reference_scale": (
                float(diagnostics["direction_reference_scale"])
                if diagnostics["direction_reference_scale"] is not None
                else 0.0
            ),
            "initialization": diagnostics["initialization"],
        }
        runs.append(record)
        generated_tables.append(best)
        reverse_probability = (
            f"{record['negative_direction_copy_probability']:.1%}"
            if direction_enabled else "不适用"
        )
        print(
            f"loss={record['best_loss']:.1f} | L1={record['training_l1']:.6f} "
            f"| raw gain={record['raw_proposal_gain_mean']:+.1f} "
            f"| raw 正收益={record['raw_proposal_positive_rate']:.1%} "
            f"| 反向复制概率={reverse_probability}",
            flush=True,
        )

    # 所有生成调用结束后才读取真实训练/测试表，严格隔离离线评价。
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH, header=None, names=columns)
    train_probability = _joint_distribution(train, columns)
    test_probability = _joint_distribution(test, columns)
    for record, best in zip(runs, generated_tables):
        record.update(
            _joint_metrics(best, train_probability, test_probability, columns)
        )
        run_dir = parent / record["run_dir"]
        with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print(
            f"seed={record['seed']} 离线评价 | "
            f"训练 TVD={record['train_tvd']:.6f} | "
            f"测试 TVD={record['test_tvd']:.6f}",
            flush=True,
        )

    metric_names = [
        "best_loss",
        "training_l1",
        "train_tvd",
        "test_tvd",
        "n_unique",
        "missing_train_mass",
        "novel_synthetic_mass",
        "rounds_run",
        "accept_rate",
        "late_accept_rate",
        "late_loss_improvement",
        "raw_proposal_gain_mean",
        "raw_proposal_positive_rate",
        "raw_positive_gain_mean",
        "raw_negative_gain_mean",
        "late_raw_proposal_gain_mean",
        "late_raw_proposal_positive_rate",
        "raw_proposal_linear_gain_mean",
        "raw_proposal_quadratic_penalty_mean",
        "copy_direction_mean",
        "copy_direction_positive_rate",
        "copy_direction_negative_rate",
        "negative_direction_copy_probability",
        "positive_direction_copy_probability",
        "copy_probability_entropy",
        "direction_reference_scale",
        "elapsed_sec",
        "direction_evaluation_elapsed_sec",
    ]
    summary = {
        "experiment": (
            "residual_directed_diffusion_nltcs"
            if direction_enabled else "baseline_diffusion_nltcs"
        ),
        "scope": "fixed_workload_exact_target_no_noise",
        "primary_evidence": "raw_proposal_before_generation_acceptance",
        "hypothesis": (
            "在原始无噪声配置和相同计算预算下，连续残差倾斜应改善接受检查前的"
            "原始提案，同时保留可观测的反向复制概率。"
        ),
        "baseline": "同配置固定 eta=0.5 的历史扩散核（方向机制关闭）",
        "params": {
            "seeds": args.seeds,
            "n_rounds": args.rounds,
            "device": args.device,
            "init_method": "marginal",
            "residual_directed_diffusion": direction_enabled,
            "diffusion_direction_strength": (
                args.direction_strength if direction_enabled else None
            ),
            "diffusion_direction_normalization": (
                args.direction_normalization if direction_enabled else None
            ),
        },
        "privacy_note": (
            "run_evolution 不读取原始表；train/test 仅在全部生成结束后离线评价。"
        ),
        "input_sha256": {
            "schema": _sha256_file(SCHEMA_PATH),
            "queries": _sha256_file(QUERY_PATH),
            "marginals": _sha256_file(MARGINALS_PATH),
        },
        "environment": environment,
        "runs": runs,
        "aggregate": {
            name: _aggregate(runs, name) for name in metric_names
        },
    }
    save_summary(str(parent), summary)
    print(f"\n结果目录: {parent}", flush=True)


if __name__ == "__main__":
    main()
