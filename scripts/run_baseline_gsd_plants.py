"""GSD 文献基线 plants 版，官方 GeneticSD 零噪声吃一阶加全二阶统计量搜索出表。

与其分类主实验同口径，all-2way 一次性喂料，官方默认参数，
遗传搜索直接在合成表上迭代，rho 取无穷即零噪声，逐格断言无噪声漂移。
plants 全部 01 字段无需分箱与解码，码即值，常值字段值域按实际大小为一。

运行环境，借官方 private_gsd 仓的虚拟环境，只用其安装的官方库，
机器共享，先用 nvidia-smi 挑空闲卡，
CUDA_VISIBLE_DEVICES=1 /home/chuhan/projects/private_gsd/.venv/bin/python \
    scripts/run_baseline_gsd_plants.py --seed 1 --out results/plants_gsd_seed1.csv

同餐断言，官方统计模块从码表算出的全二阶计数，
必须逐格等于考卷 9248 条答案，任何一格不等立即失败退出。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics
from genetic_sd.adaptive_statistics.marginals import Marginals
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.generator.mutation_strategies import AVAILABLE_GENETIC_OPERATORS
from genetic_sd.utils import Dataset, Domain

REPO = Path(__file__).resolve().parent.parent


def load_exam(data_csv: Path, exam_json: Path):
    """考卷重建每字段值域与全部字段对列联表。"""
    with open(data_csv, encoding="utf-8-sig") as fh:
        raw = list(csv.DictReader(fh))
    fields = list(raw[0].keys())
    cats = {f: sorted({r[f] for r in raw}) for f in fields}
    with open(exam_json, encoding="utf-8") as fh:
        exam = json.load(fh)
    total = exam["record_count"]
    if total != len(raw):
        raise RuntimeError("考卷行数与真实表不一致")
    tables = {
        (fa, fb): np.zeros((len(cats[fa]), len(cats[fb])), dtype=np.int64)
        for fa, fb in combinations(fields, 2)
    }
    for q in exam["queries"]:
        (ca, cb) = q["conditions"]
        fa, fb = ca["attribute"], cb["attribute"]
        ia = cats[fa].index(ca["value"])
        ib = cats[fb].index(cb["value"])
        tables[(fa, fb)][ia, ib] = q["result"]
    if sum(t.size for t in tables.values()) != exam["query_count"]:
        raise RuntimeError("考卷格数与字段对列联表规模不一致")
    frame = pd.DataFrame(
        {f: [cats[f].index(r[f]) for r in raw] for f in fields}, columns=fields
    )
    return fields, cats, tables, total, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", type=str, default="plants", help="数据目录名，表名须同名")
    parser.add_argument("--num-generations", type=int, default=50000000)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0001)
    args = parser.parse_args()

    data_dir = REPO / "data" / args.data
    exam_found = sorted(data_dir.glob("measured_*query.json"))
    if len(exam_found) != 1:
        raise RuntimeError(f"数据目录 {data_dir} 下 measured_*query.json 须恰有一个")
    fields, cats, pair_tables, total, frame = load_exam(
        data_dir / f"{args.data}.csv", exam_found[0]
    )

    for (fa, fb), target in pair_tables.items():
        ka, kb = len(cats[fa]), len(cats[fb])
        flat = frame[fa].to_numpy() * kb + frame[fb].to_numpy()
        counts = np.bincount(flat, minlength=ka * kb).reshape(ka, kb)
        if not np.array_equal(counts, target):
            raise RuntimeError(f"同餐断言失败，字段对 ({fa},{fb}) 与考卷不一致")
    print(f"同餐断言通过，码表全二阶逐格等于考卷 {sum(t.size for t in pair_tables.values())} 条答案")

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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(total):
            writer.writerow([cats[f][int(out_codes[f].iloc[i])] for f in fields])
    print(f"已写出 {args.out}，{total} 行")


if __name__ == "__main__":
    main()
