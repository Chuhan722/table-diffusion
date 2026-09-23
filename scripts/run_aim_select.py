"""AIM 选择层移植，select-measure-generate 全套循环，生成段换残差演化引擎。

照搬 AIM 官方算法结构，见 private-pgm/mechanisms/aim.py 与论文 arXiv 2201.12677，
选择用指数机制挑当前表答得最差的边际，测量用高斯机制，逐轮入卷引擎热启动续跑。
与官方的三处差异，一是树宽过滤整个拆除，引擎无模型内存约束候选永远全量入围，
二是预算原子换 bounded 口径与本仓考卷一致，高斯 L2 敏感度根号二即花费 1/sigma 方，
指数机制效用 L1 敏感度 2 wgt，三是生成段由 PGM 拟合换成引擎续跑，当前估计即当前表。

隐私账全在本外壳，引擎只吃带噪考卷属后处理。zCDP 总预算 rho=cdp_rho(eps,delta)，
初始一阶全测与每轮测量花 1/sigma 方，每轮选择花 eps_t 方除 8，
剩余不足两轮时末轮吃光全部剩余，annealing 判据照搬，
测量后表投影变化不及噪声期望则 sigma 减半 epsilon 翻倍。

用法示例，
  .venv/bin/python scripts/run_aim_select.py --data nltcs --epsilon 1.0 --seed 1 \
      --rounds 64 --engine-rounds 2000 --final-rounds 30000 \
      --engine-extra "--batched --pairing --work-frac 0.10 --damping 0.5 --stay 0.99 ..." \
      --workdir runs/aimsel_nltcs_s1 --out results/nltcs_eps1_aimsel_seed1.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from make_noisy_exam import cdp_rho  # noqa: E402
from resevo.dataset import load_table  # noqa: E402
from resevo.initialization import sample_initial_rows  # noqa: E402


def powerset_nonempty(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def compile_workload(workload: list[tuple[tuple[str, ...], float]]) -> dict:
    """workload 的 downward closure 带相关性权重，照官方 compile_workload。"""
    weights = {cl: wt for cl, wt in workload}
    closure = set()
    for cl in weights:
        closure.update(powerset_nonempty(cl))

    def score(cl):
        return sum(w * len(set(cl) & set(wcl)) for wcl, w in weights.items())

    return {cl: score(cl) for cl in sorted(closure, key=lambda c: (len(c), c))}


class TableProjector:
    """行表到任意边际计数向量的投影器，列先编码成整数索引后按混合基数聚合。"""

    def __init__(self, schema):
        self.fields = list(schema.fields)
        self.findex = {f: j for j, f in enumerate(self.fields)}
        self.value_order = {
            f: list(schema.domains[j]) for j, f in enumerate(self.fields)
        }
        self.value_index = {
            f: {v: i for i, v in enumerate(vs)} for f, vs in self.value_order.items()
        }
        self.sizes = {f: len(vs) for f, vs in self.value_order.items()}
        self._codes = None

    def encode(self, rows: list[tuple[str, ...]]) -> None:
        n, d = len(rows), len(self.fields)
        codes = np.empty((n, d), dtype=np.int64)
        for j, f in enumerate(self.fields):
            vi = self.value_index[f]
            codes[:, j] = [vi[r[j]] for r in rows]
        self._codes = codes

    def project(self, clique: tuple[str, ...]) -> np.ndarray:
        idx = [self.findex[f] for f in clique]
        sizes = [self.sizes[f] for f in clique]
        flat = np.zeros(len(self._codes), dtype=np.int64)
        for j, sz in zip(idx, sizes):
            flat = flat * sz + self._codes[:, j]
        return np.bincount(flat, minlength=int(np.prod(sizes))).astype(np.float64)

    def domain_size(self, clique: tuple[str, ...]) -> int:
        return int(np.prod([self.sizes[f] for f in clique]))


def exponential_mechanism(
    qualities: dict, epsilon: float, sensitivity: float, rng: np.random.Generator
):
    keys = list(qualities)
    q = np.array([qualities[k] for k in keys])
    q = q - q.max()
    logits = 0.5 * epsilon / sensitivity * q
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return keys[rng.choice(len(keys), p=p)]


def write_exam(
    path: Path,
    measurements: list[dict],
    projector: TableProjector,
    record_count: int,
    desc: str,
) -> None:
    """累计测量写成本仓考卷 JSON，逐格一条查询，同边际重复测量各自成题。"""
    queries = []
    for mi, m in enumerate(measurements):
        cl, y = m["clique"], m["y"]
        orders = [projector.value_order[f] for f in cl]
        shapes = [len(o) for o in orders]
        for gi, combo_idx in enumerate(itertools.product(*[range(s) for s in shapes])):
            conds = [
                {"attribute": f, "operator": "==", "value": orders[k][ci]}
                for k, (f, ci) in enumerate(zip(cl, combo_idx))
            ]
            expr = " AND ".join(f"{c['attribute']} == {c['value']}" for c in conds)
            queries.append(
                {
                    "id": f"M{mi:04d}_{gi:05d}",
                    "type": f"eq{len(cl)}way",
                    "expression": expr,
                    "conditions": conds,
                    "result": float(y[gi]),
                }
            )
    payload = {
        "dataset": "aim_select_dynamic",
        "record_count": record_count,
        "query_count": len(queries),
        "result_unit": "records",
        "description": desc,
        "queries": queries,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def noise_floor(measurements: list[dict]) -> float:
    return 0.5 * sum(m["sigma"] ** 2 * m["y"].size for m in measurements)


def run_engine(
    data_name: str,
    exam_path: Path,
    init_table: Path,
    rounds: int,
    floor: float,
    seed_base: int,
    extra: str,
    table_out: Path,
    log_path: Path,
) -> None:
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "run_plants.py"),
        "--data", data_name,
        "--exam", str(exam_path),
        "--init-table", str(init_table),
        "--rounds", str(rounds),
        "--menu-seed", str(seed_base),
        "--sample-seed", str(seed_base % 100000),
        "--noise-floor", f"{floor:.6g}",
        "--save-table", str(table_out),
    ] + shlex.split(extra)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        lf.flush()
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser(description="AIM 选择层移植外壳")
    ap.add_argument("--data", required=True, help="干净数据目录名，如 nltcs")
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, required=True, help="机制随机种子，噪声与指数机制与引擎种子同源")
    ap.add_argument("--rounds", type=int, default=0, help="选择轮数 T，0 用官方默认 16 倍字段数")
    ap.add_argument("--engine-rounds", type=int, default=2000, help="每轮引擎续跑轮数上限")
    ap.add_argument("--final-rounds", type=int, default=30000, help="末轮引擎长跑轮数上限")
    ap.add_argument("--engine-extra", type=str, default="", help="直通引擎的其余参数字符串")
    ap.add_argument("--workdir", type=str, required=True, help="循环工作目录")
    ap.add_argument("--out", type=str, required=True, help="最终合成表 CSV")
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    data_dir = REPO / "data" / args.data
    schema, real_rows = load_table(str(data_dir / f"{args.data}.csv"))
    n_total = len(real_rows)
    d = schema.num_fields
    projector = TableProjector(schema)
    projector.encode(real_rows)
    real_answers = {}

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "engine_log.txt"
    T = args.rounds or 16 * d

    rho = float(cdp_rho(args.epsilon, args.delta))
    # bounded 口径，高斯 L2 敏感度根号二，单次测量花 1/sigma 方
    sigma = float(np.sqrt(2.0 * T / (2 * 0.9 * rho)))
    epsilon_t = float(np.sqrt(8 * 0.1 * rho / T))

    workload = [(cl, 1.0) for cl in itertools.combinations(schema.fields, 2)]
    candidates = compile_workload(workload)
    for cl in candidates:
        real_answers[cl] = projector.project(cl)

    aim_log = {
        "data": args.data, "epsilon": args.epsilon, "delta": args.delta,
        "rho": rho, "T": T, "sigma0": sigma, "epsilon_t0": epsilon_t,
        "seed": args.seed, "bounded": True, "rounds": [],
    }

    # 初始，一阶全测抽初始表，不调引擎
    measurements = []
    oneway = [cl for cl in candidates if len(cl) == 1]
    rho_used = 0.0
    for cl in oneway:
        y = real_answers[cl] + rng.normal(0, sigma, real_answers[cl].size)
        measurements.append({"clique": cl, "y": y, "sigma": sigma})
        rho_used += 1.0 / sigma**2
    marginals = []
    for j, f in enumerate(schema.fields):
        m = next(m for m in measurements if m["clique"] == (f,))
        counts = np.clip(m["y"], 0.0, None)
        atoms = [
            ({"attribute": f, "operator": "==", "value": v}, float(c))
            for v, c in zip(projector.value_order[f], counts)
        ]
        marginals.append(atoms)
    init_rows = sample_initial_rows(marginals, schema, n_total, rng)
    if rho_used > 0.5 * rho:
        raise SystemExit(
            f"初始一阶已花 {rho_used:.6f} 超总预算 {rho:.6f} 一半，T={T} 太小，"
            "官方默认 16 倍字段数，请加大 --rounds"
        )
    cur_table = workdir / "table_round0.csv"
    import csv as _csv
    with open(cur_table, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(schema.fields)
        w.writerows(init_rows)
    print(f"初始一阶 {len(oneway)} 条已测，sigma {sigma:.2f}，rho 已用 {rho_used:.6f}/{rho:.6f}")

    cur_rows = init_rows
    t = 0
    terminate = False
    while not terminate:
        t += 1
        per_round = 1.0 / sigma**2 + epsilon_t**2 / 8
        if rho - rho_used < 2 * per_round:
            remaining = rho - rho_used
            sigma = float(np.sqrt(2.0 / (2 * 0.9 * remaining)))
            epsilon_t = float(np.sqrt(8 * 0.1 * remaining))
            terminate = True
        rho_used += 1.0 / sigma**2 + epsilon_t**2 / 8

        # 选择，效用为当前表答卷 L1 误差减噪声偏置，敏感度 2 wgt
        projector.encode(cur_rows)
        errors = {}
        for cl, wgt in candidates.items():
            xest = projector.project(cl)
            bias = np.sqrt(2 / np.pi) * sigma * projector.domain_size(cl)
            errors[cl] = wgt * (
                np.linalg.norm(real_answers[cl] - xest, 1) - bias
            )
        max_sens = 2 * max(abs(w) for w in candidates.values())
        chosen = exponential_mechanism(errors, epsilon_t, max_sens, rng)

        # 测量入卷
        y = real_answers[chosen] + rng.normal(0, sigma, real_answers[chosen].size)
        measurements.append({"clique": chosen, "y": y, "sigma": sigma})
        z_before = projector.project(chosen)

        exam_path = workdir / f"exam_round{t}.json"
        write_exam(
            exam_path, measurements, projector, n_total,
            f"AIM 选择循环第 {t} 轮累计考卷，eps {args.epsilon} seed {args.seed}",
        )
        eng_rounds = args.final_rounds if terminate else args.engine_rounds
        next_table = workdir / f"table_round{t}.csv"
        run_engine(
            args.data, exam_path, cur_table, eng_rounds,
            noise_floor(measurements), args.seed * 100000 + t,
            args.engine_extra, next_table, log_path,
        )
        _, cur_rows = load_table(str(next_table))
        cur_table = next_table

        # annealing 判据，表在选中边际上的变化不及噪声期望则加码
        projector.encode(cur_rows)
        w_after = projector.project(chosen)
        moved = float(np.linalg.norm(w_after - z_before, 1))
        thresh = float(np.sqrt(2 / np.pi) * sigma * projector.domain_size(chosen))
        annealed = moved <= thresh
        aim_log["rounds"].append(
            {
                "t": t, "clique": list(chosen), "sigma": sigma,
                "epsilon_t": epsilon_t, "rho_used": rho_used,
                "moved_l1": moved, "anneal_thresh": thresh,
                "annealed": annealed, "terminate": terminate,
                "n_measurements": len(measurements),
            }
        )
        print(
            f"轮 {t} 选 {chosen} sigma {sigma:.2f} 动量 {moved:.0f}/{thresh:.0f}"
            f" rho {rho_used:.6f}/{rho:.6f}{' 加码' if annealed else ''}{' 末轮' if terminate else ''}"
        )
        if annealed and not terminate:
            sigma /= 2
            epsilon_t *= 2

    import shutil
    shutil.copyfile(cur_table, args.out)
    aim_log["wall_seconds"] = time.time() - t_start
    aim_log["final_table"] = str(args.out)
    with open(workdir / "aim_log.json", "w", encoding="utf-8") as fh:
        json.dump(aim_log, fh, ensure_ascii=False, indent=1)
    print(f"收官，共 {t} 轮，墙钟 {aim_log['wall_seconds']:.0f} 秒,终表 {args.out}")


if __name__ == "__main__":
    main()
