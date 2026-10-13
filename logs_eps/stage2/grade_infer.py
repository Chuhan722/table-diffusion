"""第二阶段 B 阅卷：没测的对补 PGM 推断目标（hybridinfer 臂）vs PGM 的分布本身 / 第一阶段我们的表 / PGM 出的表。

hybridinfer = hybrid 臂命令逐字 + 考卷多了 AIM 没测的对的 PGM 推断二阶边际（scripts/add_infer_targets.py，
推断来自 pgm_exact_marginals.py 同录像同配置的拟合，不出表；sigma = 原二阶 sigma 中位数）。
pgm模型 / pgm表 与 grade_pgm_exact.py 同（同一个拟合）。轴同口径。
三条 Δ：hybridinfer vs pgm模型（主判定）、hybridinfer vs hybrid（补目标的净效果）、hybridinfer vs pgm表（对 PGM 实际产物）。
只阅已有表的种子，表头注明。

用法: .venv/bin/python logs_eps/stage2/grade_infer.py [--out logs_eps/scores/stage2b_infer.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_pgm_exact import AXES, ROOT, TAGS, Real, load_int_table  # noqa: E402

MAIN = ["一阶", "二阶", "二阶·测过", "二阶·没测", "三阶均", "等值", "半空间"]


def pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs_eps/scores/stage2b_infer.md")
    args = ap.parse_args()
    R = Real()
    lines = ["# 第二阶段 B：没测的对补 PGM 推断目标（hybridinfer）阅卷", "",
             "单位 ×10⁻⁶（L1/N），多种子取均值；hybridinfer = hybrid 臂 + 没测对的 PGM 推断二阶边际为伪测量；"
             "pgm模型 = 同录像同配置独立拟合的精确边际（不出表），pgm表 = 该拟合出的表，hybrid = 第一阶段我们的表。", ""]
    summary = []
    for tag in ["0p1", "0p3", "1", "3", "10"]:
        rows = {"hybridinfer": [], "hybrid": [], "pgm表": [], "pgm模型": []}
        seeds = []
        for s in [1, 2, 3]:
            ours = ROOT / f"results/stage2/adult_eps{tag}_hybridinfer_seed{s}.csv"
            if not ours.exists():
                continue
            z = np.load(ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz", allow_pickle=True)
            mp = {tuple(int(v) for v in p) for p in z["measured_pairs"]}
            rows["hybridinfer"].append(R.axes_table(load_int_table(ours, R.columns), mp))
            rows["hybrid"].append(R.axes_table(load_int_table(ROOT / f"results/stage1/adult_eps{tag}_hybrid_seed{s}.csv", R.columns), mp))
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
        for lab, a, b in [("Δ hybridinfer vs pgm模型", "hybridinfer", "pgm模型"),
                          ("Δ hybridinfer vs hybrid（补目标净效果）", "hybridinfer", "hybrid"),
                          ("Δ hybridinfer vs pgm表", "hybridinfer", "pgm表")]:
            lines.append(f"| {lab} | " + " | ".join(pct(mean[a][x], mean[b][x]) for x in AXES) + " |")
        per = " ".join(f"s{s}:{pct(rows['hybridinfer'][i]['二阶'], rows['pgm模型'][i]['二阶'])}" for i, s in enumerate(seeds))
        lines.append("")
        lines.append(f"逐种子主轴二阶 hybridinfer vs pgm模型：{per}。")
        lines.append("")
        d = (mean["hybridinfer"]["二阶"] / mean["pgm模型"]["二阶"] - 1) * 100
        verdict = "平" if abs(d) <= 2 else ("胜" if d < 0 else "负")
        summary.append((TAGS[tag], ",".join(map(str, seeds)), pct(mean["hybridinfer"]["二阶"], mean["pgm模型"]["二阶"]), verdict,
                        pct(mean["hybridinfer"]["二阶·测过"], mean["pgm模型"]["二阶·测过"]),
                        pct(mean["hybridinfer"]["二阶·没测"], mean["pgm模型"]["二阶·没测"]),
                        pct(mean["hybridinfer"]["一阶"], mean["pgm模型"]["一阶"]),
                        pct(mean["hybridinfer"]["二阶"], mean["hybrid"]["二阶"]),
                        pct(mean["hybridinfer"]["二阶"], mean["pgm表"]["二阶"]),
                        pct(mean["hybrid"]["二阶"], mean["pgm模型"]["二阶"])))
    lines.append("### 主轴二阶汇总")
    lines.append("")
    lines.append("| ε | 种子 | hybridinfer vs pgm模型 | 判定 | 测过 | 没测 | 一阶 | vs hybrid | vs pgm表 | 对照：hybrid vs pgm模型 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        lines.append("| " + " | ".join(r) + " |")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(summary) + 3):]))


if __name__ == "__main__":
    main()
