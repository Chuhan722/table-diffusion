"""AIM 选择层 + 残差演化引擎（第二步，自适应选测），记账与 AIM 官方逐项对齐。

相对第二十一步旧版 run_aim_select.py 的改动，
1. 记账全部换成 AIM 官方的 unbounded zCDP 口径（邻接表差一条记录，Δ₂=1）：
   高斯测量每个边际花 1/(2σ²)，指数机制选择花 ε_t²/8，效用敏感度 = wgt；
   σ₀=√(T/(2·0.9ρ))，ε_t=√(8·0.1ρ/T)，T=16d；末轮吃光剩余 rem，σ=√(1/(2·0.9rem))，
   ε=√(8·0.1rem)。每笔开销记入账本，收官断言 Σ 账本 = ρ。
2. 考卷同团重复测量逆方差合并（y=Σy_i/σ_i² / Σ1/σ_i²，σ=1/√Σ1/σ_i²），每题带 sigma，
   引擎用 --exam-weights sigma 逆方差加权（与第 0 步回放完全一致）。
3. 中间轮引擎热启动短跑（--init-table 上一轮表，粗平台阈值），只为给选择器
   提供"当前估计"；末轮不续跑，而是在最终累计考卷上用第 0 步混合跑的原生配置
   （--init-oneway，一次性长跑）重新拟合，终表与第 0 步混合跑唯一的区别是考卷来源。
4. 不做 AIM 的模型大小过滤（引擎没有联结树内存约束），候选永远全量入围。

与 AIM 官方相同之处：候选 = 二阶 workload 的 downward closure 及其相关性权重，
选择效用 wgt·(‖x−x̂‖₁ − √(2/π)σ·n)，测量入卷后若选中边际的变化不及噪声期望
（‖w−z‖₁ ≤ √(2/π)σ·n）则 σ 减半 ε_t 翻倍，剩余不足两轮则末轮吃光。

用法示例（A6000，GPU 版），
  .venv/bin/python scripts/run_aim_select_v2.py --data adult --epsilon 0.1 --seed 1 \
      --engine-rounds 1000 --workdir runs/aimsel2_adult0p1_s1 \
      --out results/aimsel/adult_eps0p1_aimsel_seed1.csv \
      --engine-extra "--gpu --batched --pairing --work-rows 4096 ... --gpu-entry-budget 40000000"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from make_noisy_exam import cdp_rho  # noqa: E402
from run_aim_select import TableProjector, compile_workload, exponential_mechanism  # noqa: E402
from resevo.dataset import load_table  # noqa: E402
from resevo.initialization import sample_initial_rows  # noqa: E402


def merge_measurements(measurements: list[dict]) -> dict:
    """同团重复测量逆方差合并，返回 clique -> {y, sigma, n_repeat}。"""
    merged: dict = {}
    for m in measurements:
        cl = m["clique"]
        prec = 1.0 / m["sigma"] ** 2
        slot = merged.setdefault(cl, {"num": np.zeros_like(m["y"]), "prec": 0.0, "n": 0})
        slot["num"] = slot["num"] + prec * m["y"]
        slot["prec"] += prec
        slot["n"] += 1
    return {
        cl: {"y": s["num"] / s["prec"], "sigma": float(1.0 / np.sqrt(s["prec"])), "n_repeat": s["n"]}
        for cl, s in merged.items()
    }


def write_exam(path: Path, measurements: list[dict], projector: TableProjector,
               record_count: int, desc: str) -> dict:
    """累计测量合并后写成带 sigma 的考卷 JSON，返回合并统计。"""
    merged = merge_measurements(measurements)
    queries = []
    for mi, (cl, m) in enumerate(sorted(merged.items(), key=lambda kv: (len(kv[0]), kv[0]))):
        orders = [projector.value_order[f] for f in cl]
        for gi, combo_idx in enumerate(itertools.product(*[range(len(o)) for o in orders])):
            conds = [
                {"attribute": f, "operator": "==", "value": orders[k][ci]}
                for k, (f, ci) in enumerate(zip(cl, combo_idx))
            ]
            queries.append({
                "id": f"M{mi:04d}_{gi:05d}",
                "type": f"eq{len(cl)}way",
                "expression": " AND ".join(f"{c['attribute']} == {c['value']}" for c in conds),
                "conditions": conds,
                "result": float(m["y"][gi]),
                "sigma": m["sigma"],
            })
    sigmas = [m["sigma"] for m in merged.values()]
    payload = {
        "dataset": "aim_select_v2_dynamic",
        "record_count": record_count,
        "query_count": len(queries),
        "result_unit": "records",
        "description": desc + "；同团重复测量逆方差合并，每题带 sigma",
        "queries": queries,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return {
        "cliques": len(merged),
        "pairs": sum(1 for cl in merged if len(cl) == 2),
        "repeats": sum(1 for m in merged.values() if m["n_repeat"] > 1),
        "sigma_min": float(min(sigmas)), "sigma_max": float(max(sigmas)),
        "queries": len(queries),
    }


def run_engine(data_name: str, exam_path: Path, rounds: int, seeds: dict, flags: list[str],
               extra: str, table_out: Path, curve_out: Path | None, log_path: Path) -> float:
    cmd = [
        str(REPO / ".venv" / "bin" / "python"), "-u", "-B",
        str(REPO / "scripts" / "run_plants.py"),
        "--data", data_name, "--exam", str(exam_path),
        "--rounds", str(rounds), "--exam-weights", "sigma",
        "--menu-seed", str(seeds["menu"]), "--sample-seed", str(seeds["sample"]),
        "--init-seed", str(seeds["init"]),
        "--save-table", str(table_out),
    ] + flags + shlex.split(extra)
    if curve_out is not None:
        cmd += ["--out", str(curve_out)]
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        lf.flush()
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="AIM 选择层 + 残差演化引擎（unbounded 记账）")
    ap.add_argument("--data", required=True, help="干净数据目录名，如 adult")
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, required=True, help="机制随机种子（噪声、指数机制），引擎种子同源")
    ap.add_argument("--rounds", type=int, default=0, help="名义轮数 T，0 用官方默认 16·d")
    ap.add_argument("--engine-rounds", type=int, default=1000, help="中间轮引擎热启动续跑轮数上限")
    ap.add_argument("--engine-stop", type=str, default="1e-3,50",
                    help="中间轮引擎平台停止 阈值,滞后，默认 1e-3,50")
    ap.add_argument("--final-rounds", type=int, default=30000, help="末轮一次性拟合轮数上限")
    ap.add_argument("--final-stop", type=str, default="1e-4,100",
                    help="末轮引擎平台停止 阈值,滞后，默认 1e-4,100（与第 0 步混合跑一致）")
    ap.add_argument("--engine-extra", type=str, default="",
                    help="直通引擎的公共参数字符串（gpu、batched、pairing、rescue、struct 等）")
    ap.add_argument("--workdir", type=str, required=True, help="循环工作目录（考卷、中间表、日志）")
    ap.add_argument("--out", type=str, required=True, help="最终合成表 CSV")
    ap.add_argument("--keep-tables", action="store_true", help="保留所有中间表（默认只留最后一张）")
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    data_dir = REPO / "data" / args.data
    schema, real_rows = load_table(str(data_dir / f"{args.data}.csv"))
    n_total, d = len(real_rows), schema.num_fields
    projector = TableProjector(schema)
    projector.encode(real_rows)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "engine_log.txt"
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = args.rounds or 16 * d
    e_thr, e_lag = args.engine_stop.split(",")
    f_thr, f_lag = args.final_stop.split(",")

    rho = float(cdp_rho(args.epsilon, args.delta))
    # unbounded 口径（AIM 官方）：高斯 Δ₂=1，每个边际花 1/(2σ²)；选择花 ε_t²/8
    sigma = float(np.sqrt(T / (2 * 0.9 * rho)))
    epsilon_t = float(np.sqrt(8 * 0.1 * rho / T))

    workload = [(cl, 1.0) for cl in itertools.combinations(schema.fields, 2)]
    candidates = compile_workload(workload)
    real_answers = {cl: projector.project(cl) for cl in candidates}
    max_sens = max(abs(w) for w in candidates.values())

    ledger: list[dict] = []
    measurements: list[dict] = []

    # 初始一阶全测，σ₀，抽初始表（不调引擎，与 AIM 一致）
    oneway = [cl for cl in candidates if len(cl) == 1]
    for cl in oneway:
        y = real_answers[cl] + rng.normal(0, sigma, real_answers[cl].size)
        measurements.append({"clique": cl, "y": y, "sigma": sigma})
        ledger.append({"t": 0, "kind": "measure", "clique": list(cl), "sigma": sigma,
                       "cost": 0.5 / sigma**2})
    rho_used = sum(e["cost"] for e in ledger)
    marginals = []
    for j, f in enumerate(schema.fields):
        m = next(m for m in measurements if m["clique"] == (f,))
        counts = np.clip(m["y"], 0.0, None)
        marginals.append([
            ({"attribute": f, "operator": "==", "value": v}, float(c))
            for v, c in zip(projector.value_order[f], counts)
        ])
    cur_rows = sample_initial_rows(marginals, schema, n_total, rng)
    cur_table = workdir / "table_round0.csv"
    with open(cur_table, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(schema.fields)
        w.writerows(cur_rows)
    print(f"ρ={rho:.6g} T={T} σ₀={sigma:.2f} ε_t={epsilon_t:.5f}；"
          f"初始一阶 {len(oneway)} 团已测，ρ 已用 {rho_used:.4g} ({rho_used / rho:.1%})")

    aim_log = {
        "data": args.data, "epsilon": args.epsilon, "delta": args.delta, "rho": rho,
        "T": T, "sigma0": sigma, "epsilon_t0": epsilon_t, "seed": args.seed,
        "accounting": "unbounded zCDP (AIM 官方): 高斯 1/(2σ²)/边际, 指数机制 ε²/8, 敏感度 wgt",
        "engine_rounds": args.engine_rounds, "engine_stop": args.engine_stop,
        "final_rounds": args.final_rounds, "final_stop": args.final_stop,
        "engine_extra": args.engine_extra, "rounds": [],
    }

    common_flags = ["--stop-threshold", e_thr, "--stop-lag", e_lag]
    t, terminate = 0, False
    while not terminate:
        t += 1
        per_round = 0.5 / sigma**2 + epsilon_t**2 / 8
        if rho - rho_used < 2 * per_round:
            remaining = rho - rho_used
            sigma = float(np.sqrt(1.0 / (2 * 0.9 * remaining)))
            epsilon_t = float(np.sqrt(8 * 0.1 * remaining))
            terminate = True

        # 选择：当前表答卷 L1 误差减噪声偏置，敏感度 wgt（unbounded）
        projector.encode(cur_rows)
        errors = {}
        for cl, wgt in candidates.items():
            xest = projector.project(cl)
            bias = np.sqrt(2 / np.pi) * sigma * projector.domain_size(cl)
            errors[cl] = wgt * (np.linalg.norm(real_answers[cl] - xest, 1) - bias)
        chosen = exponential_mechanism(errors, epsilon_t, max_sens, rng)
        ledger.append({"t": t, "kind": "select", "clique": list(chosen),
                       "epsilon_t": epsilon_t, "cost": epsilon_t**2 / 8})

        # 测量入卷
        n_cells = real_answers[chosen].size
        y = real_answers[chosen] + rng.normal(0, sigma, n_cells)
        measurements.append({"clique": chosen, "y": y, "sigma": sigma})
        ledger.append({"t": t, "kind": "measure", "clique": list(chosen), "sigma": sigma,
                       "cost": 0.5 / sigma**2})
        rho_used = sum(e["cost"] for e in ledger)
        z_before = projector.project(chosen)

        exam_path = workdir / f"exam_round{t}.json"
        stats = write_exam(exam_path, measurements, projector, n_total,
                           f"AIM 选择循环第 {t} 轮累计考卷，{args.data} eps {args.epsilon} seed {args.seed}")
        next_table = workdir / f"table_round{t}.csv"
        secs = run_engine(
            args.data, exam_path, args.engine_rounds,
            {"menu": args.seed * 100000 + t, "sample": args.seed * 1000 + t, "init": args.seed},
            ["--init-table", str(cur_table)] + common_flags, args.engine_extra,
            next_table, None, log_path,
        )
        _, cur_rows = load_table(str(next_table))
        if not args.keep_tables and t >= 2:
            (workdir / f"table_round{t - 1}.csv").unlink(missing_ok=True)
        cur_table = next_table

        # annealing 判据：表在选中边际上的变化不及噪声期望则 σ 减半 ε 翻倍
        projector.encode(cur_rows)
        moved = float(np.linalg.norm(projector.project(chosen) - z_before, 1))
        thresh = float(np.sqrt(2 / np.pi) * sigma * n_cells)
        annealed = moved <= thresh
        aim_log["rounds"].append({
            "t": t, "clique": list(chosen), "cells": int(n_cells), "sigma": sigma,
            "epsilon_t": epsilon_t, "rho_used": rho_used, "moved_l1": moved,
            "anneal_thresh": thresh, "annealed": annealed, "terminate": terminate,
            "exam": stats, "engine_seconds": secs,
        })
        print(f"轮 {t} 选 {chosen} 格 {n_cells} σ {sigma:.2f} 动量 {moved:.0f}/{thresh:.0f}"
              f" ρ {rho_used / rho:.1%} 卷 {stats['cliques']}团/{stats['pairs']}对"
              f" 引擎 {secs:.0f}s{' 加码' if annealed else ''}{' 末轮' if terminate else ''}")
        if annealed and not terminate:
            sigma /= 2
            epsilon_t *= 2

    # 账本核对：全部开销之和必须等于 ρ
    total = sum(e["cost"] for e in ledger)
    if not np.isclose(total, rho, rtol=1e-9, atol=0.0):
        raise SystemExit(f"账本不平：Σ开销 {total:.12g} ≠ ρ {rho:.12g}")
    n_sel = sum(1 for e in ledger if e["kind"] == "select")
    n_meas = sum(1 for e in ledger if e["kind"] == "measure")
    sel_cost = sum(e["cost"] for e in ledger if e["kind"] == "select")
    print(f"账本：选择 {n_sel} 笔 {sel_cost:.4g} ({sel_cost / rho:.1%})，"
          f"测量 {n_meas} 笔 {total - sel_cost:.4g} ({(total - sel_cost) / rho:.1%})，Σ={total:.6g}=ρ ✓")

    # 末轮：在最终累计考卷上用第 0 步混合跑的原生配置一次性重新拟合
    final_exam = workdir / "exam_final.json"
    shutil.copyfile(exam_path, final_exam)
    print(f"末轮重拟合：{stats['cliques']} 团（{stats['pairs']} 对），σ 范围 "
          f"{stats['sigma_min']:.1f}~{stats['sigma_max']:.1f}，{stats['queries']} 题")
    secs = run_engine(
        args.data, final_exam, args.final_rounds,
        {"menu": 20260920 + args.seed, "sample": args.seed, "init": args.seed},
        ["--init-oneway", "--stop-threshold", f_thr, "--stop-lag", f_lag], args.engine_extra,
        out_path, out_path.with_name(out_path.stem + "_curve.csv"), log_path,
    )
    aim_log.update({
        "ledger": ledger, "ledger_total": total, "n_select": n_sel, "n_measure": n_meas,
        "final_exam": str(final_exam), "final_engine_seconds": secs,
        "wall_seconds": time.time() - t_start, "final_table": str(out_path),
    })
    with open(workdir / "aim_log.json", "w", encoding="utf-8") as fh:
        json.dump(aim_log, fh, ensure_ascii=False, indent=1)
    print(f"收官，共 {t} 轮，末轮引擎 {secs:.0f}s，墙钟 {aim_log['wall_seconds']:.0f} 秒，终表 {out_path}")


if __name__ == "__main__":
    main()
