"""运行并离线评价 nltcs 的适应度支配软门控实验。

生成阶段固定使用仓库原始的 marginal、精确 measured workload 和 geometric 配置；
训练/测试原始表只在 run_evolution 返回后用于离线联合 TVD 与支持集评价。

示例：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python \
        scripts/run_fitness_dominance_nltcs.py \
        --seeds 0 --rounds 1500 --exploration-rate 0.02 \
        --output-dir outputs/fitness_soft_dominance_nltcs_seed0
"""
import argparse
import json
from pathlib import Path

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
    """初始即达标、没有提案历史时按零次事件记录为 0。"""
    return float(np.mean(values)) if len(values) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--device", choices=["cuda", "cpu", "numpy"], default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--exploration-rate", type=float, default=0.02)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds 必须为正数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if not 0.0 <= args.exploration_rate <= 1.0:
        parser.error("--exploration-rate 必须在 [0, 1] 内")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries], dtype=float)
    marginals = load_marginals(MARGINALS_PATH)
    columns = schema.attribute_names()

    if args.output_dir:
        parent = Path(args.output_dir)
        if parent.exists() and any(parent.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空，不覆盖：{parent}")
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path(create_parent_dir(prefix="fitness_soft_dominance_nltcs"))

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
            fitness_dominance_gate=True,
            fitness_dominance_exploration_rate=args.exploration_rate,
            max_retries=0,
        )

        run_dir = parent / f"{position}-{seed}"
        save_run(best, diagnostics, run_dir=str(run_dir))
        dominance_rates = np.asarray(
            diagnostics["fitness_dominance_rate_history"], dtype=float
        )
        copy_scales = np.asarray(
            diagnostics["fitness_copy_participation_scale_history"],
            dtype=float,
        )
        accept_history = diagnostics["accept_history"]
        loss_history = diagnostics["loss_history"]
        accept_late_start = 3 * len(accept_history) // 4
        loss_late_start = 3 * len(loss_history) // 4
        record = {
            "seed": seed,
            "run_dir": run_dir.name,
            "best_loss": float(diagnostics["best_loss"]),
            "training_l1": float(diagnostics["normalized_l1_error"]),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
            "rounds_run": int(diagnostics["rounds_run"]),
            "stopped_early": bool(diagnostics["stopped_early"]),
            "accept_rate": _mean_or_zero(accept_history),
            "late_accept_rate": _mean_or_zero(
                accept_history[accept_late_start:]
            ),
            "late_loss_improvement": float(
                loss_history[loss_late_start] - diagnostics["best_loss"]
            ),
            "mean_dominance_rate": _mean_or_zero(dominance_rates),
            "mean_copy_participation_scale": _mean_or_zero(copy_scales),
            "initialization": diagnostics["initialization"],
        }
        runs.append(record)
        generated_tables.append(best)
        print(
            f"loss={record['best_loss']:.1f} | L1={record['training_l1']:.6f} | "
            f"支配率={record['mean_dominance_rate']:.2%} | "
            f"复制缩放={record['mean_copy_participation_scale']:.2%}",
            flush=True,
        )

    # 所有生成调用结束后才读取真实训练/测试表，保证二者只存在于离线评价阶段。
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
        "best_loss", "training_l1", "train_tvd", "test_tvd",
        "n_unique", "missing_train_mass", "novel_synthetic_mass",
        "rounds_run", "accept_rate", "late_accept_rate",
        "late_loss_improvement", "mean_dominance_rate",
        "mean_copy_participation_scale", "elapsed_sec",
    ]
    summary = {
        "experiment": "fitness_soft_dominance_nltcs",
        "scope": "fixed_workload_exact_target_no_noise",
        "params": {
            "seeds": args.seeds,
            "n_rounds": args.rounds,
            "device": args.device,
            "init_method": "marginal",
            "fitness_dominance_gate": True,
            "fitness_dominance_exploration_rate": args.exploration_rate,
        },
        "privacy_note": (
            "run_evolution 不读取原始表；train/test 仅在结束后计算离线评价指标。"
        ),
        "runs": runs,
        "aggregate": {
            name: _aggregate(runs, name) for name in metric_names
        },
    }
    save_summary(str(parent), summary)
    print(f"\n结果目录: {parent}", flush=True)


if __name__ == "__main__":
    main()
