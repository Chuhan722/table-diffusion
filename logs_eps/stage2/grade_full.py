"""第二阶段 B 核对阅卷：一条命令版完整配方（hybridfull 臂，run_plants.py --fiber-passes 100 + 定稿清洗）对 2B-4 的三步手工版（hybridfiberp1u）。

两臂差别只有第二段起点：hybridfull 从第一段终表热启动，hybridfiberp1u 从一阶初始化重来（且第一段直接用 2A 的 hybridp1u 表）。
同时列 hybridfull 的第一段表（_stage1，应与 2A hybridp1u 同臂同种子、但引擎版本不同，分数相近即可）。
Δ：hybridfull vs pgm模型（主判定）、vs hybridfiberp1u（一条命令版的净差）、vs pgm表；按已有的种子阅。

用法: .venv/bin/python logs_eps/stage2/grade_full.py [--tags 0p1,0p3] [--out logs_eps/scores/stage2b_full.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_pgm_exact import AXES, ROOT, TAGS, Real, load_int_table  # noqa: E402


def pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="0p1,0p3,1,3,10")
    ap.add_argument("--out", default="logs_eps/scores/stage2b_full.md")
    args = ap.parse_args()
    R = Real()
    lines = ["# 第二阶段 B 核对：一条命令版完整配方（hybridfull）阅卷", "",
             "单位 ×10⁻⁶（L1/N），多种子取均值；hybridfull = `run_plants.py --fiber-passes 100` + 定稿一阶清洗，一条命令内做完清洗→收敛→洗牌→热启动再收敛；"
             "hybridfull·第一段 = 该命令第一段终表（只清洗、没管没测对）；hybridfiberp1u = 2B-4 三步手工版（第二段从一阶初始化重来）；"
             "hybridp1u = 2A 只清洗臂；pgm模型 = 同录像同配置独立拟合的精确边际（不出表），pgm表 = 该拟合出的表。", ""]
    summary = []
    for tag in args.tags.split(","):
        rows = {"hybridfull": [], "hybridfull·第一段": [], "hybridfiberp1u": [], "hybridp1u": [], "pgm表": [], "pgm模型": []}
        seeds = []
        for s in [1, 2, 3]:
            ours = ROOT / f"results/stage2/adult_eps{tag}_hybridfull_seed{s}.csv"
            if not ours.exists():
                continue
            z = np.load(ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz", allow_pickle=True)
            mp = {tuple(int(v) for v in p) for p in z["measured_pairs"]}
            rows["hybridfull"].append(R.axes_table(load_int_table(ours, R.columns), mp))
            rows["hybridfull·第一段"].append(R.axes_table(load_int_table(ours.with_name(ours.stem + "_stage1.csv"), R.columns), mp))
            ref = ROOT / f"results/stage2/adult_eps{tag}_hybridfiberp1u_seed{s}.csv"
            if not ref.exists():  # ε≥1 清洗恒等，三步版就是 2B-3 的 hybridfiber
                ref = ROOT / f"results/stage2/adult_eps{tag}_hybridfiber_seed{s}.csv"
            rows["hybridfiberp1u"].append(R.axes_table(load_int_table(ref, R.columns), mp))
            p1u = ROOT / f"results/stage2/adult_eps{tag}_hybridp1u_seed{s}.csv"
            if not p1u.exists():
                p1u = ROOT / f"results/stage1/adult_eps{tag}_hybrid_seed{s}.csv"
            rows["hybridp1u"].append(R.axes_table(load_int_table(p1u, R.columns), mp))
            rows["pgm表"].append(R.axes_table(load_int_table(ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}.csv", R.columns), mp))
            rows["pgm模型"].append(R.axes_npz(z, "", "hs_big", "eq_exact", mp))
            seeds.append(s)
        if not seeds:
            continue
        mean = {k: {a: float(np.mean([r[a] for r in v])) for a in AXES} for k, v in rows.items()}
        lines.append(f"### ε = {TAGS[tag]}（种子 {','.join(map(str, seeds))}）")
        lines.append("")
        lines.append("| 臂 | " + " | ".join(AXES) + " |")
        lines.append("|---" * (len(AXES) + 1) + "|")
        for k in rows:
            lines.append(f"| {k} | " + " | ".join(f"{mean[k][a]:,.0f}" for a in AXES) + " |")
        for lab, a, b in [("Δ hybridfull vs pgm模型", "hybridfull", "pgm模型"),
                          ("Δ hybridfull vs hybridfiberp1u（一条命令版 vs 三步手工版）", "hybridfull", "hybridfiberp1u"),
                          ("Δ hybridfull·第一段 vs hybridp1u（第一段复现 2A）", "hybridfull·第一段", "hybridp1u"),
                          ("Δ hybridfull vs hybridfull·第一段（第二段的净效果）", "hybridfull", "hybridfull·第一段"),
                          ("Δ hybridfull vs pgm表", "hybridfull", "pgm表")]:
            lines.append(f"| {lab} | " + " | ".join(pct(mean[a][x], mean[b][x]) for x in AXES) + " |")
        per = " ".join(f"s{s}:{pct(rows['hybridfull'][i]['二阶'], rows['pgm模型'][i]['二阶'])}" for i, s in enumerate(seeds))
        per_ref = " ".join(f"s{s}:{pct(rows['hybridfull'][i]['二阶'], rows['hybridfiberp1u'][i]['二阶'])}" for i, s in enumerate(seeds))
        lines.append("")
        lines.append(f"逐种子 hybridfull vs pgm模型 二阶：{per}；vs hybridfiberp1u 二阶：{per_ref}。")
        lines.append("")
        d = (mean["hybridfull"]["二阶"] / mean["pgm模型"]["二阶"] - 1) * 100
        verdict = "平" if abs(d) <= 2 else ("胜" if d < 0 else "负")
        summary.append((TAGS[tag], ",".join(map(str, seeds)), pct(mean["hybridfull"]["二阶"], mean["pgm模型"]["二阶"]), verdict,
                        pct(mean["hybridfull"]["一阶"], mean["pgm模型"]["一阶"]),
                        pct(mean["hybridfull"]["二阶·测过"], mean["pgm模型"]["二阶·测过"]),
                        pct(mean["hybridfull"]["二阶·没测"], mean["pgm模型"]["二阶·没测"]),
                        pct(mean["hybridfull"]["三阶均"], mean["pgm模型"]["三阶均"]),
                        pct(mean["hybridfull"]["二阶"], mean["hybridfiberp1u"]["二阶"]),
                        pct(mean["hybridfull"]["二阶"], mean["pgm表"]["二阶"])))
    lines.append("### 主轴二阶汇总")
    lines.append("")
    lines.append("| ε | 种子 | hybridfull vs pgm模型（二阶） | 判定 | 一阶 | 测过 | 没测 | 三阶均 | vs hybridfiberp1u | vs pgm表 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        lines.append("| " + " | ".join(r) + " |")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(summary) + 3):]))


if __name__ == "__main__":
    main()
