"""
运行并离线评价 nltcs 的二阶最大熵初始化实验。

生成器运行时仍只使用 schema 与 measured_1000query.json。训练/测试原始表只在
演化结束后用于计算联合 TVD，不能把这里的评价函数搬进生成主循环。

示例：
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pairwise_maxent_nltcs.py \
        --seeds 0 1 2 --rounds 600
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.io import create_parent_dir, save_run, save_summary
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
TRAIN_PATH = "data/nltcs/nltcs.csv"
TEST_PATH = "data/nltcs/nltcs.test.data"
N_RECORDS = 16_181


def _joint_distribution(frame, columns):
    """nltcs 二值行编码后的经验联合分布。"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--device", choices=["cuda", "cpu", "numpy"], default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries], dtype=float)
    columns = schema.attribute_names()

    # 以下两次原始数据读取只服务离线评价，不传给 run_evolution。
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH, header=None, names=columns)
    train_probability = _joint_distribution(train, columns)
    test_probability = _joint_distribution(test, columns)

    if args.output_dir:
        parent = Path(args.output_dir)
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path(create_parent_dir(prefix="pairwise_maxent_nltcs"))

    runs = []
    for position, seed in enumerate(args.seeds):
        print(f"\n===== 种子 {seed}（{position + 1}/{len(args.seeds)}）=====", flush=True)
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
            init_method="pairwise_maxent",
            log_every=args.log_every,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha_min=2.0,
            alpha_max=10.0,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
            max_retries=0,
        )

        run_dir = parent / f"{position}-{seed}"
        save_run(best, diagnostics, run_dir=str(run_dir))
        joint = _joint_metrics(
            best, train_probability, test_probability, columns
        )
        record = {
            "seed": seed,
            "run_dir": run_dir.name,
            "best_loss": float(diagnostics["best_loss"]),
            "training_l1": float(diagnostics["normalized_l1_error"]),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
            "accept_rate": float(np.mean(diagnostics["accept_history"])),
            "initialization": diagnostics["initialization"],
            **joint,
        }
        runs.append(record)
        with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print(
            f"loss={record['best_loss']:.1f} | L1={record['training_l1']:.6f} | "
            f"训练 TVD={record['train_tvd']:.6f} | "
            f"测试 TVD={record['test_tvd']:.6f}",
            flush=True,
        )

    metric_names = [
        "best_loss", "training_l1", "train_tvd", "test_tvd",
        "n_unique", "missing_train_mass", "novel_synthetic_mass",
        "accept_rate", "elapsed_sec",
    ]
    summary = {
        "experiment": "pairwise_maxent_nltcs",
        "params": {
            "seeds": args.seeds,
            "n_rounds": args.rounds,
            "device": args.device,
            "init_method": "pairwise_maxent",
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
