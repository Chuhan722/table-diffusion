"""第一阶段 1B 分组阅卷：按 AIM 录像"测过的对 / 没测的对 / 一阶"分组，3 种子均，出 markdown。

对每个 ε、每个种子调 scripts/score_pair_groups.py（--dump 给录像），把各臂的四列取 3 种子均；
单位 ×10⁻⁶（L1/N）。默认臂：hybrid、pgmrep10k（都跑到收敛，只比这两臂；参考臂可用 --arms 加回）。

用法: .venv/bin/python logs_eps/stage1/pairgroups_stage1.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3]
      [--arms hybrid,pgmrep10k,pgmrep,aimfull] [--out logs_eps/scores/stage1_pairgroups.md]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps" / "stage1"))
from grade_stage1 import arm_path  # noqa: E402

COLS = ["二阶总体", "测过的对", "没测的对", "一阶"]


def run_one(tag: str, seed: int, arms: list[str]) -> tuple[dict[str, list[float]], tuple[int, int]]:
    dump = ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{seed}_measurements.json"
    labels = [(a, arm_path(a, tag, seed)) for a in arms]
    labels = [(a, p) for a, p in labels if p.exists()]
    cmd = [sys.executable, str(ROOT / "scripts/score_pair_groups.py"), "--real", str(ROOT / "data/adult/adult.csv"),
           "--dump", str(dump)] + [f"{a}={p}" for a, p in labels]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    m = re.search(r"测过 (\d+) 对.*?没测 (\d+) 对", out)
    counts = (int(m.group(1)), int(m.group(2)))
    rows: dict[str, list[float]] = {}
    for line in out.splitlines():
        mm = re.match(r"(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$", line)
        if mm and mm.group(1) != "表":
            rows[mm.group(1)] = [float(x) if x != "-" else float("nan") for x in mm.groups()[1:]]
    return rows, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arms", default="hybrid,pgmrep10k")
    ap.add_argument("--out", default="logs_eps/scores/stage1_pairgroups.md")
    args = ap.parse_args()
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lines = ["## 1B 分组阅卷：AIM 测过的对 / 没测的对 / 一阶（×10⁻⁶ L1/N，3 种子均）", "",
             "独立参照 = 真一阶乘积当合成边际（不花预算只知道一阶）。Δ 行 = (hybrid − pgmrep10k)/pgmrep10k 逐种子配对均。", "",
             "| ε | 测过/没测 对数 | 臂 | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 3) + "|"]
    for tag in args.eps.split(","):
        per: dict[str, list[list[float]]] = {}
        counts = []
        for s in seeds:
            rows, c = run_one(tag, s, arms)
            counts.append(c)
            for a, v in rows.items():
                per.setdefault(a, []).append(v)
        cnt = "/".join(str(c[0]) for c in counts) + " · " + "/".join(str(c[1]) for c in counts)
        for a, vs in per.items():
            mean = np.mean(vs, axis=0)
            lines.append(f"| {tag.replace('p', '.')} | {cnt} | {a} | " + " | ".join(
                "—" if np.isnan(v) else f"{v:,.0f}" for v in mean) + " |")
        if "hybrid" in per and "pgmrep10k" in per and len(per["hybrid"]) == len(per["pgmrep10k"]):
            d = (np.array(per["hybrid"]) - np.array(per["pgmrep10k"])) / np.array(per["pgmrep10k"])
            lines.append(f"| {tag.replace('p', '.')} | | Δ hybrid vs pgmrep10k | " + " | ".join(f"{v:+.1%}" for v in d.mean(axis=0)) + " |")
    text = "\n".join(lines)
    print(text)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
