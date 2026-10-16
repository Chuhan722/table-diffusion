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

加噪对照给 --noisy-exam 指噪声考卷 json，拟合目标整卷替换成噪声答案，
此时按官方 oneshot 口径只挂全二阶模块，评价与残差打印仍对真实统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics
from genetic_sd.adaptive_statistics.marginals import Marginals
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.generator.mutation_strategies import AVAILABLE_GENETIC_OPERATORS
from genetic_sd.utils import Dataset, Domain

REPO = Path(__file__).resolve().parent.parent


def load_exam(data_csv: Path, exam_json: Path, dtype=np.int64):
    """考卷重建每字段值域与全部字段对列联表，噪声考卷传浮点 dtype。"""
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
        (fa, fb): np.zeros((len(cats[fa]), len(cats[fb])), dtype=dtype)
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
    parser.add_argument(
        "--noisy-exam", type=str, default="",
        help="外部噪声考卷 json 路径，给了就按其答案当拟合目标，"
        "官方 oneshot 口径只挂全二阶，评价照旧对真实统计",
    )
    parser.add_argument(
        "--generation-log", type=str, default="",
        help="第六十步记账：每 --log-every 代记 (代数, 最优 fitness, 累计秒, 累计候选评估数) 到该 CSV，"
        "只读不碰计算，需 gsd_patched 带 generation_log 钩子",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--snapshot-fitness", type=str, default="",
        help="第六十二步快照：逗号分隔的 fitness 阈值（官方口径 ‖目标频率−频率‖²，= ℒ/N²），"
        "在 --log-every 的记账点上最优 fitness 首次不高于某阈值时，把当时的最优表写成 "
        "{prefix}_snap{i}.csv 并在 {prefix}_snaps.csv 记一行，只读不碰计算，需 gsd_patched 带 snapshot 钩子",
    )
    parser.add_argument("--snapshot-prefix", type=str, default="")
    args = parser.parse_args()
    if bool(args.snapshot_fitness) != bool(args.snapshot_prefix):
        raise SystemExit("--snapshot-fitness 与 --snapshot-prefix 须同时给出")
    snapshot_fitness = (
        [float(x) for x in args.snapshot_fitness.split(",") if x.strip()]
        if args.snapshot_fitness else []
    )
    if any(not (x > 0.0) for x in snapshot_fitness):
        raise SystemExit("--snapshot-fitness 须为正数")

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
    key = jax.random.PRNGKey(args.seed)
    key, key_dp = jax.random.split(key)
    if args.noisy_exam:
        # 官方 oneshot 口径只挂全二阶，外部噪声答案整卷替换拟合目标
        _, _, noisy_tables, _, _ = load_exam(
            data_dir / f"{args.data}.csv", Path(args.noisy_exam), dtype=np.float64
        )
        two_way = Marginals.get_all_kway_combinations(domain, k=2)
        stats.add_stat_module_and_fit(two_way)
        stats.private_measure_all_statistics(key=key_dp, rho=np.inf)
        pairs = [tuple(c) for c in stats.stat_modules[0].kway_combinations]
        rebuilt = []
        for workload_id, wfn, _noised, true in stats.selected_workloads[0]:
            fa, fb = pairs[workload_id]
            true_freq = pair_tables[(fa, fb)].astype(np.float64).ravel() / total
            if not np.allclose(np.asarray(true), true_freq, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"噪声注入顺序断言失败，字段对 ({fa},{fb})")
            noisy_freq = jnp.asarray(noisy_tables[(fa, fb)].ravel() / total)
            rebuilt.append((workload_id, wfn, noisy_freq, true))
        stats.selected_workloads[0] = rebuilt
        want = np.concatenate(
            [noisy_tables[p].ravel() / total for p in pairs]
        )
        got = np.asarray(stats.get_selected_noised_statistics())
        drift = float(np.max(np.abs(want - got)))
        if drift > 1e-6:
            raise RuntimeError(f"噪声目标整卷断言失败，最大漂移 {drift}")
        print(f"噪声目标注入 {got.shape[0]} 格，与噪声考卷逐格一致")
        true_stats = np.asarray(stats.get_all_true_statistics())
    else:
        one_way = Marginals.get_all_kway_combinations(domain, k=1)
        two_way = Marginals.get_all_kway_combinations(domain, k=2)
        stats.add_stat_module_and_fit(one_way)
        stats.add_stat_module_and_fit(two_way)
        stats.private_measure_all_statistics(key=key_dp, rho=np.inf)

        true_stats = np.asarray(stats.get_all_true_statistics())
        noised = np.asarray(stats.get_selected_noised_statistics())
        drift = float(np.max(np.abs(true_stats - noised)))
        if drift != 0.0:
            raise RuntimeError(f"零噪声断言失败，最大漂移 {drift}")
        print(f"统计量 {true_stats.shape[0]} 格，零噪声逐格一致")

    def write_table(path: Path, data: Dataset) -> None:
        """码表 → 类别字符串 CSV，终表与快照同一写法。"""
        codes = data.df[fields].astype(float).round().astype(int)
        if len(codes) != total:
            raise RuntimeError(f"输出行数漂移 {len(codes)} != {total}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for i in range(total):
                writer.writerow([cats[f][int(codes[f].iloc[i])] for f in fields])

    def snapshot_hook(index, threshold, gen, best_fitness, elapsed, cum_candidates, member):
        prefix = Path(args.snapshot_prefix)
        write_table(Path(f"{prefix}_snap{index}.csv"), Dataset.from_numpy_to_dataset(domain, member))
        log_path = Path(f"{prefix}_snaps.csv")
        new = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as fh:
            if new:
                fh.write("index,threshold,gen,best_fitness,elapsed_s,cum_candidates\n")
            fh.write(f"{index},{threshold!r},{gen},{best_fitness!r},{elapsed:.3f},{cum_candidates}\n")
        print(f"快照{index} 阈值 {threshold!r} 代 {gen} fitness {best_fitness!r} 秒 {elapsed:.1f}", flush=True)

    generator = GeneticSD(
        domain=domain,
        data_size=total,
        num_generations=args.num_generations,
        genetic_operators=AVAILABLE_GENETIC_OPERATORS,
        print_progress=False,
        stop_early=True,
        stop_eary_threshold=args.early_stop_threshold,
        sparse_statistics=True,
        **({"log_every": args.log_every} if (args.generation_log or snapshot_fitness) else {}),
        **({"generation_log": args.generation_log} if args.generation_log else {}),
        **({"snapshot_fitness": snapshot_fitness, "snapshot_hook": snapshot_hook}
           if snapshot_fitness else {}),
    )
    started = time.time()
    sync_data = generator.fit(jax.random.PRNGKey(args.seed + 1), stats)
    fit_seconds = time.time() - started
    if args.generation_log:
        print(f"停止代数 {generator.stop_generation}，记账已写入 {args.generation_log}")

    stat_fn = stats.get_dataset_statistics_fn()
    sync_stats = np.asarray(stat_fn(sync_data))
    l1 = np.abs(true_stats - sync_stats)
    print(
        f"搜索 {fit_seconds:.1f} 秒，频率尺度残差 平均 {l1.mean():.6f} "
        f"最大 {l1.max():.6f}"
    )
    if args.noisy_exam:
        l1n = np.abs(np.asarray(stats.get_selected_noised_statistics()) - sync_stats)
        print(f"对噪声目标残差 平均 {l1n.mean():.6f} 最大 {l1n.max():.6f}")

    write_table(Path(args.out), sync_data)
    print(f"已写出 {args.out}，{total} 行")


if __name__ == "__main__":
    main()
