"""GSD 无噪声生成脚本（在官方 private_gsd 的独立 venv 里执行）。

角色边界：
- 本脚本只负责“喂 9522 格频率统计（69 个一维 + 2346 个二维 2x2）→
  官方 GeneticSD 默认参数搜索 → 输出合成 0/1 表 + 生成 manifest”。
- 全部质量评估（measured/heldout/one-way/家族内）由主仓库 runner 用
  冻结考卷和主评估器完成；本脚本内的 l1 数字仅作 sanity 记录。
- 零噪声路径：private_measure_all_statistics(rho=inf) => sigma=0，
  本脚本对 true vs noised 统计做逐格断言（fail-closed）。

官方默认对照（genetic_sd.py GSDSynthesizer.fit / GeneticSD.__init__）：
- num_generations=50000000、stop_early=True、stop_eary_threshold=0.0001
- genetic_operators 空时替换为 AVAILABLE_GENETIC_OPERATORS（全量四操作符）
- sparse_statistics=True、N_prime=len(data)、stop_early_min_generation=N_prime
唯一与官方 fit 不同的是喂食内容：官方只加 k=2 模块；本协议为与引擎
plants 9522 池（9384 二维 + 138 一维）同餐（PGM 同餐不可行，有收据），
链式加 k=1 与 k=2 两个 Marginals 模块（k=1 对 69 二值列 = 138 格）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics
from genetic_sd.adaptive_statistics.marginals import Marginals
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.generator.mutation_strategies import (
    AVAILABLE_GENETIC_OPERATORS,
)
from genetic_sd.utils import Dataset, Domain

EXPECTED_ONE_WAY_WORKLOADS = 69
EXPECTED_TWO_WAY_WORKLOADS = 2346
EXPECTED_STAT_COUNT = 9522  # 69*2 + 2346*4


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_binary_table(path: Path, expected_sha256: str) -> pd.DataFrame:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"输入数据 SHA 漂移: expected={expected_sha256}, "
            f"observed={observed}"
        )
    frame = pd.read_csv(path)
    values = frame.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise RuntimeError("输入数据存在非 0/1 取值，协议只支持二值表")
    return frame


def _build_statistics(
    dataset: Dataset, seed: int
) -> tuple[AdaptiveChainedStatistics, dict]:
    stats = AdaptiveChainedStatistics(dataset)
    one_way = Marginals.get_all_kway_combinations(dataset.domain, k=1)
    two_way = Marginals.get_all_kway_combinations(dataset.domain, k=2)
    if len(one_way.kway_combinations) != EXPECTED_ONE_WAY_WORKLOADS:
        raise RuntimeError("一维 workload 数量漂移")
    if len(two_way.kway_combinations) != EXPECTED_TWO_WAY_WORKLOADS:
        raise RuntimeError("二维 workload 数量漂移")
    stats.add_stat_module_and_fit(one_way)
    stats.add_stat_module_and_fit(two_way)

    key = jax.random.PRNGKey(seed)
    key, key_dp = jax.random.split(key)
    stats.private_measure_all_statistics(key=key_dp, rho=np.inf)

    true_stats = np.asarray(stats.get_all_true_statistics())
    noised_stats = np.asarray(stats.get_selected_noised_statistics())
    if true_stats.shape[0] != EXPECTED_STAT_COUNT:
        raise RuntimeError(
            f"统计格数漂移: expected={EXPECTED_STAT_COUNT}, "
            f"observed={true_stats.shape[0]}"
        )
    zero_noise_max_abs_diff = float(
        np.max(np.abs(true_stats - noised_stats))
    )
    if zero_noise_max_abs_diff != 0.0:
        raise RuntimeError(
            "零噪声断言失败：rho=inf 下 true 与 noised 统计不一致，"
            f"max_abs_diff={zero_noise_max_abs_diff}"
        )
    audit = {
        "one_way_workloads": EXPECTED_ONE_WAY_WORKLOADS,
        "two_way_workloads": EXPECTED_TWO_WAY_WORKLOADS,
        "statistic_count": int(true_stats.shape[0]),
        "rho": "inf",
        "zero_noise_max_abs_diff": zero_noise_max_abs_diff,
    }
    return stats, audit


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_csv)
    output_path = Path(args.output_csv)
    manifest_path = Path(args.manifest_json)
    for target in (output_path, manifest_path):
        if target.exists():
            raise FileExistsError(f"输出已存在，不覆盖：{target}")

    frame = _load_binary_table(data_path, args.expected_data_sha256)
    columns = list(frame.columns)
    n_records = int(len(frame))

    config = {col: {"type": "string", "size": 2} for col in columns}
    domain = Domain(config)
    dataset = Dataset(frame.astype(float), domain)

    stats, stats_audit = _build_statistics(dataset, args.seed)

    generator = GeneticSD(
        domain=domain,
        data_size=n_records,
        num_generations=args.num_generations,
        genetic_operators=AVAILABLE_GENETIC_OPERATORS,
        print_progress=False,
        stop_early=True,
        stop_eary_threshold=args.early_stop_threshold,
        sparse_statistics=True,
    )
    key_fit = jax.random.PRNGKey(args.seed + 1)
    started = time.time()
    sync_data = generator.fit(key_fit, stats)
    wall_seconds = time.time() - started

    true_stats = np.asarray(stats.get_all_true_statistics())
    stat_fn = stats.get_dataset_statistics_fn()
    sync_stats = np.asarray(stat_fn(sync_data))
    family_l1_mean = float(np.mean(np.abs(true_stats - sync_stats)))
    family_l1_max = float(np.max(np.abs(true_stats - sync_stats)))

    out_frame = sync_data.df[columns].astype(float).round().astype(int)
    if not np.isin(out_frame.to_numpy(), (0, 1)).all():
        raise RuntimeError("GSD 输出存在非 0/1 取值")
    if len(out_frame) != n_records:
        raise RuntimeError(
            f"GSD 输出行数漂移: expected={n_records}, "
            f"observed={len(out_frame)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_path, index=False)

    manifest = {
        "generator": "private_gsd_genetic_sd_kernel",
        "noise_model": "none_rho_inf_exact_statistics",
        "input": {
            "path": str(data_path),
            "sha256": args.expected_data_sha256,
            "columns": columns,
            "record_count": n_records,
        },
        "statistics": stats_audit,
        "parameters": {
            "seed": int(args.seed),
            "key_dp_seed": int(args.seed),
            "key_fit_seed": int(args.seed + 1),
            "num_generations_cap": int(args.num_generations),
            "stop_early": True,
            "stop_early_threshold": float(args.early_stop_threshold),
            "stop_early_min_generation": n_records,
            "genetic_operators": list(AVAILABLE_GENETIC_OPERATORS),
            "sparse_statistics": True,
            "n_prime": n_records,
            "official_defaults_note": (
                "all GeneticSD parameters mirror GSDSynthesizer.fit "
                "defaults; only the fed statistics add k=1 marginals "
                "for same-meal parity with the engine 9522-cell pool"
            ),
        },
        "runtime": {
            "wall_seconds_fit": wall_seconds,
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
        },
        "sanity_family_fit_frequency_l1": {
            "mean": family_l1_mean,
            "max": family_l1_max,
            "note": (
                "frequency-scale sanity only; canonical metrics are "
                "computed by the main-repo runner on frozen exams"
            ),
        },
        "output": {
            "path": str(output_path),
            "row_count": int(len(out_frame)),
            "sha256": _sha256_file(output_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--num-generations", type=int, default=50000000
    )
    parser.add_argument(
        "--early-stop-threshold", type=float, default=0.0001
    )
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
