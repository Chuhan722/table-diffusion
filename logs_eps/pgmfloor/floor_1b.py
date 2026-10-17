"""1B（AIM 选测录像）两臂地板停阈值：我们臂与 PGM 臂各算各的损失地板，并核对两臂读的是同一份测量。

我们臂：run_plants.py --exam-weights sigma 下记账 = ½ Σ w (y−q)²，w = (σ_min/σ)²，
        地板 = ½ Σ w σ² = ½ · 题数 · σ_min²（题数为合并重复团后的格数）。
PGM 臂：mbi 损失 = ½ Σ ((Qμ−y)/σ)²，录像里重复团分开计，地板 = ½ · 录像格数。
两边最小点相同（逆方差合并是充分统计量），地板只差一个常数，各停各的 ×1.0 即同口径。

用法：
  .venv/bin/python logs_eps/pgmfloor/floor_1b.py            # 打印表并写 floor_1b.json / scores/floor_1b.md
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EPS = ["0p1", "0p3", "1", "3", "10"]
SEEDS = [1, 2, 3]


def our_exam(eps: str, seed: int) -> dict:
    files = glob.glob(str(REPO / "data" / f"adult_aimrep{eps}_s{seed}" / "measured_*query.json"))
    if len(files) != 1:
        raise SystemExit(f"adult_aimrep{eps}_s{seed} 考卷文件数 {len(files)} != 1")
    with open(files[0], encoding="utf-8") as fh:
        exam = json.load(fh)
    qs = exam["queries"]
    sig = np.array([q["sigma"] for q in qs], dtype=np.float64)
    w = (sig.min() / sig) ** 2
    cells = {}
    for q in qs:
        key = tuple((c["attribute"], c["value"]) for c in q["conditions"])
        cells[key] = (q["result"], q["sigma"])
    return {"n": len(qs), "sigma_min": float(sig.min()), "floor": float(0.5 * np.sum(w * sig**2)),
            "source_dump": Path(exam["source_dump"]).name, "cells": cells, "file": files[0]}


def pgm_dump(eps: str, seed: int) -> dict:
    path = REPO / "results" / "aimreplay" / f"adult_eps{eps}_aimfull_seed{seed}_measurements.json"
    with open(path, encoding="utf-8") as fh:
        dump = json.load(fh)
    columns = list(dump["columns"])
    sizes = {c: int(s) for c, s in zip(columns, dump["domain_sizes"])}
    col_pos = {c: i for i, c in enumerate(columns)}
    n_cells = 0
    acc: dict[tuple[str, ...], dict] = defaultdict(lambda: {"num": None, "prec": 0.0})
    for m in dump["measurements"]:
        clique = tuple(m["clique"])
        y = np.asarray(m["y"], dtype=np.float64).reshape([sizes[a] for a in clique])
        n_cells += y.size
        order = sorted(range(len(clique)), key=lambda i: col_pos[clique[i]])
        key = tuple(clique[i] for i in order)
        y = np.transpose(y, order)
        prec = 1.0 / float(m["sigma"]) ** 2
        slot = acc[key]
        slot["num"] = y * prec if slot["num"] is None else slot["num"] + y * prec
        slot["prec"] += prec
    merged = {}
    for key, slot in acc.items():
        y = slot["num"] / slot["prec"]
        sigma = 1.0 / np.sqrt(slot["prec"])
        for idx in np.ndindex(*y.shape):
            merged[tuple((a, str(v)) for a, v in zip(key, idx))] = (float(y[idx]), float(sigma))
    return {"n": n_cells, "n_meas": len(dump["measurements"]), "floor": 0.5 * n_cells, "merged": merged,
            "file": path.name}


def main() -> None:
    rows, out = [], {}
    for eps in EPS:
        for seed in SEEDS:
            ours, pgm = our_exam(eps, seed), pgm_dump(eps, seed)
            same_src = ours["source_dump"] == pgm["file"]
            if set(ours["cells"]) != set(pgm["merged"]):
                raise SystemExit(f"ε={eps} s{seed} 两臂格子集合不同")
            dy = max(abs(ours["cells"][k][0] - pgm["merged"][k][0]) for k in ours["cells"])
            ds = max(abs(ours["cells"][k][1] - pgm["merged"][k][1]) for k in ours["cells"])
            rows.append((eps, seed, ours["n"], ours["sigma_min"], ours["floor"], pgm["n_meas"], pgm["n"], pgm["floor"],
                         same_src, dy, ds))
            out[f"{eps}_s{seed}"] = {"ours_cells": ours["n"], "ours_sigma_min": ours["sigma_min"],
                                     "ours_floor": ours["floor"], "pgm_measurements": pgm["n_meas"],
                                     "pgm_cells": pgm["n"], "pgm_floor": pgm["floor"],
                                     "same_source": same_src, "max_abs_dy": dy, "max_abs_dsigma": ds}
    lines = ["# 1B 两臂地板停阈值（adult，AIM 选测录像）", "",
             "我们臂地板 = ½·题数·σ_min²（--exam-weights sigma 记账，合并重复团）；PGM 臂地板 = ½·录像格数（重复团分开）。",
             "「同源」= 我们考卷的来源录像就是 PGM 读的那份；「Δy/Δσ」= 录像按逆方差合并后与我们考卷逐格最大差。", "",
             "| ε | 种子 | 我们 题数 | σ_min | 我们 地板 | PGM 测量条数 | PGM 格数 | PGM 地板 | 同源 | Δy | Δσ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for eps, seed, n, smin, fl, nm, nc, fp, same, dy, ds in rows:
        lines.append(f"| {eps.replace('p', '.')} | {seed} | {n} | {smin:.4g} | {fl:.6g} | {nm} | {nc} | {fp:.1f} | "
                     f"{'是' if same else '否'} | {dy:.1e} | {ds:.1e} |")
    md = "\n".join(lines) + "\n"
    print(md)
    (REPO / "logs_eps" / "scores" / "floor_1b.md").write_text(md, encoding="utf-8")
    with open(Path(__file__).with_name("floor_1b.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
