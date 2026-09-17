"""GSD 文献基线，官方 GeneticSD 零噪声吃一阶加全二阶统计量搜索出表。

与其分类主实验同口径，all-2way 一次性喂料，官方默认参数，
遗传搜索直接在合成表上迭代，rho 取无穷即零噪声，逐格断言无噪声漂移。

运行环境，借官方 private_gsd 仓的虚拟环境，只用其安装的官方库，
/home/chuhan/projects/private_gsd/.venv/bin/python \
    scripts/run_baseline_gsd.py --seed 1 --out results/gsd_seed1.csv

同餐断言，官方统计模块从分箱码表算出的全二阶计数，
必须逐格等于考卷 548 条答案，任何一格不等立即失败退出，
保证它吃的统计量与我们考卷完全同一份。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics
from genetic_sd.adaptive_statistics.marginals import Marginals
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.generator.mutation_strategies import AVAILABLE_GENETIC_OPERATORS
from genetic_sd.utils import Dataset, Domain

import baseline_common as bc


def _encode_binned(fields, cats, bins) -> pd.DataFrame:
    """真实表编码为分箱整数码表，只作官方统计模块的计算载体。"""
    with open(bc.DATA_CSV, encoding="utf-8-sig") as fh:
        raw = list(csv.DictReader(fh))
    data = {}
    for f in fields:
        if f == "age":
            col = []
            for r in raw:
                v = int(r[f])
                idx = next(
                    i for i, (lo, hi) in enumerate(bins) if lo <= v <= hi
                )
                col.append(idx)
        else:
            index = {v: i for i, v in enumerate(cats[f])}
            col = [index[r[f]] for r in raw]
        data[f] = col
    return pd.DataFrame(data, columns=fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-generations", type=int, default=50000000)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0001)
    args = parser.parse_args()

    fields, cats, bins, pair_tables, total = bc.load_exam_structure()
    _, domains, _ = bc.load_real_domains()
    frame = _encode_binned(fields, cats, bins)

    for (fa, fb), target in pair_tables.items():
        ka, kb = len(cats[fa]), len(cats[fb])
        flat = frame[fa].to_numpy() * kb + frame[fb].to_numpy()
        counts = np.bincount(flat, minlength=ka * kb).reshape(ka, kb)
        if not np.array_equal(counts, target):
            raise RuntimeError(f"同餐断言失败，字段对 ({fa},{fb}) 与考卷不一致")
    print("同餐断言通过，码表全二阶逐格等于考卷 548 条答案")

    config = {f: {"type": "string", "size": len(cats[f])} for f in fields}
    domain = Domain(config)
    dataset = Dataset(frame.astype(float), domain)

    stats = AdaptiveChainedStatistics(dataset)
    one_way = Marginals.get_all_kway_combinations(domain, k=1)
    two_way = Marginals.get_all_kway_combinations(domain, k=2)
    stats.add_stat_module_and_fit(one_way)
    stats.add_stat_module_and_fit(two_way)
    key = jax.random.PRNGKey(args.seed)
    key, key_dp = jax.random.split(key)
    stats.private_measure_all_statistics(key=key_dp, rho=np.inf)

    true_stats = np.asarray(stats.get_all_true_statistics())
    noised = np.asarray(stats.get_selected_noised_statistics())
    drift = float(np.max(np.abs(true_stats - noised)))
    if drift != 0.0:
        raise RuntimeError(f"零噪声断言失败，最大漂移 {drift}")
    expected = sum(len(cats[f]) for f in fields) + bc.EXPECTED_CELLS
    if true_stats.shape[0] != expected:
        raise RuntimeError(
            f"统计格数漂移 {true_stats.shape[0]} != {expected}"
        )
    print(f"统计量 {true_stats.shape[0]} 格，零噪声逐格一致")

    generator = GeneticSD(
        domain=domain,
        data_size=total,
        num_generations=args.num_generations,
        genetic_operators=AVAILABLE_GENETIC_OPERATORS,
        print_progress=False,
        stop_early=True,
        stop_eary_threshold=args.early_stop_threshold,
        sparse_statistics=True,
    )
    started = time.time()
    sync_data = generator.fit(jax.random.PRNGKey(args.seed + 1), stats)
    fit_seconds = time.time() - started

    stat_fn = stats.get_dataset_statistics_fn()
    sync_stats = np.asarray(stat_fn(sync_data))
    l1 = np.abs(true_stats - sync_stats)
    print(
        f"搜索 {fit_seconds:.1f} 秒，频率尺度残差 平均 {l1.mean():.6f} "
        f"最大 {l1.max():.6f}"
    )

    out_codes = sync_data.df[fields].astype(float).round().astype(int)
    if len(out_codes) != total:
        raise RuntimeError(f"输出行数漂移 {len(out_codes)} != {total}")
    codes = {f: out_codes[f].to_numpy() for f in fields}
    mean_d, max_d = bc.binned_counts(codes, pair_tables, cats)
    print(f"输出码表二阶计数偏差 平均 {mean_d:.3f} 最大 {max_d:.0f}")

    rows = bc.decode_rows(codes, fields, cats, bins, domains["age"], args.seed)
    bc.write_table(Path(args.out), rows, fields)
    print(f"已写出 {args.out}，{len(rows)} 行")


if __name__ == "__main__":
    main()
