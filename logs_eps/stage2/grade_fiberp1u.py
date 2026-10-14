"""第二阶段 B-4 阅卷：p1u 一阶清洗 + 纤维平均目标叠加（hybridfiberp1u 臂），只有 ε=0.1/0.3（ε≥1 清洗恒等于不清洗）。

hybridfiberp1u = hybridfiber 臂命令 + 定稿清洗三参（--oneway-pool 2 --oneway-pool-scope uncovered --oneway-pool-cover-ignore-prefix P2_），
纤维目标由 2A 的 p1u 收敛表洗出（清洗后的一阶随洗牌原样进没测对的目标）。
四条 Δ：hybridfiberp1u vs pgm模型（主判定）、vs hybridfiber（清洗的净效果，同卷同引擎）、vs hybridp1u（管没测对的净效果，同清洗）、vs pgm表。

用法: .venv/bin/python logs_eps/stage2/grade_fiberp1u.py [--out logs_eps/scores/stage2b_fiberp1u.md]
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
    ap.add_argument("--out", default="logs_eps/scores/stage2b_fiberp1u.md")
    args = ap.parse_args()
    R = Real()
    lines = ["# 第二阶段 B-4：一阶清洗 + 纤维平均目标叠加（hybridfiberp1u）阅卷", "",
             "单位 ×10⁻⁶（L1/N），多种子取均值；hybridfiberp1u = hybridfiber 臂 + 定稿一阶清洗（只洗没被真测量覆盖的字段，补进的伪目标不算覆盖），纤维目标由 p1u 表洗出；"
             "hybridfiber = 2B-3 不清洗的纤维臂；hybridp1u = 2A 只清洗不管没测对的臂；pgm模型 = 同录像同配置独立拟合的精确边际（不出表），pgm表 = 该拟合出的表。", ""]
    summary = []
    for tag in ["0p1", "0p3"]:
        rows = {"hybridfiberp1u": [], "hybridfiber": [], "hybridp1u": [], "pgm表": [], "pgm模型": []}
        seeds = []
        for s in [1, 2, 3]:
            ours = ROOT / f"results/stage2/adult_eps{tag}_hybridfiberp1u_seed{s}.csv"
            if not ours.exists():
                continue
            z = np.load(ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz", allow_pickle=True)
            mp = {tuple(int(v) for v in p) for p in z["measured_pairs"]}
            rows["hybridfiberp1u"].append(R.axes_table(load_int_table(ours, R.columns), mp))
            rows["hybridfiber"].append(R.axes_table(load_int_table(ROOT / f"results/stage2/adult_eps{tag}_hybridfiber_seed{s}.csv", R.columns), mp))
            rows["hybridp1u"].append(R.axes_table(load_int_table(ROOT / f"results/stage2/adult_eps{tag}_hybridp1u_seed{s}.csv", R.columns), mp))
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
        for lab, a, b in [("Δ hybridfiberp1u vs pgm模型", "hybridfiberp1u", "pgm模型"),
                          ("Δ hybridfiberp1u vs hybridfiber（清洗的净效果）", "hybridfiberp1u", "hybridfiber"),
                          ("Δ hybridfiberp1u vs hybridp1u（管没测对的净效果）", "hybridfiberp1u", "hybridp1u"),
                          ("Δ hybridfiberp1u vs pgm表", "hybridfiberp1u", "pgm表")]:
            lines.append(f"| {lab} | " + " | ".join(pct(mean[a][x], mean[b][x]) for x in AXES) + " |")
        per = " ".join(f"s{s}:{pct(rows['hybridfiberp1u'][i]['二阶'], rows['pgm模型'][i]['二阶'])}" for i, s in enumerate(seeds))
        per_un = " ".join(f"s{s}:{pct(rows['hybridfiberp1u'][i]['二阶·没测'], rows['pgm模型'][i]['二阶·没测'])}" for i, s in enumerate(seeds))
        per_one = " ".join(f"s{s}:{pct(rows['hybridfiberp1u'][i]['一阶'], rows['pgm模型'][i]['一阶'])}" for i, s in enumerate(seeds))
        lines.append("")
        lines.append(f"逐种子 hybridfiberp1u vs pgm模型：二阶 {per}；没测的对 {per_un}；一阶 {per_one}。")
        lines.append("")
        d = (mean["hybridfiberp1u"]["二阶"] / mean["pgm模型"]["二阶"] - 1) * 100
        verdict = "平" if abs(d) <= 2 else ("胜" if d < 0 else "负")
        summary.append((TAGS[tag], ",".join(map(str, seeds)), pct(mean["hybridfiberp1u"]["二阶"], mean["pgm模型"]["二阶"]), verdict,
                        pct(mean["hybridfiberp1u"]["一阶"], mean["pgm模型"]["一阶"]),
                        pct(mean["hybridfiberp1u"]["二阶·测过"], mean["pgm模型"]["二阶·测过"]),
                        pct(mean["hybridfiberp1u"]["二阶·没测"], mean["pgm模型"]["二阶·没测"]),
                        pct(mean["hybridfiberp1u"]["三阶均"], mean["pgm模型"]["三阶均"]),
                        pct(mean["hybridfiberp1u"]["二阶"], mean["hybridfiber"]["二阶"]),
                        pct(mean["hybridfiberp1u"]["二阶"], mean["hybridp1u"]["二阶"]),
                        pct(mean["hybridfiberp1u"]["二阶"], mean["pgm表"]["二阶"])))
    lines.append("### 主轴二阶汇总")
    lines.append("")
    lines.append("| ε | 种子 | hybridfiberp1u vs pgm模型（二阶） | 判定 | 一阶 | 测过 | 没测 | 三阶均 | vs hybridfiber | vs hybridp1u | vs pgm表 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        lines.append("| " + " | ".join(r) + " |")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(summary) + 3):]))


if __name__ == "__main__":
    main()
