"""第五十四步图：adult ε=0.1 / ε=1 八种子，第 2 步 aimsel（AIM 选择器 × 我们引擎）相对 AIM 官方（=1）的逐种子比值与均值。

用法: .venv/bin/python logs_eps/aimsel/plot_step54.py
读 logs_eps/scores/step54_perseed.txt（/tmp/score_any.py 对 32 张表的输出），输出 logs_eps/scores/aimsel_step54.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "logs_eps/scores/step54_perseed.txt"
EPS = ["0p1", "1"]
SEEDS = range(1, 9)
COLS = {"measured（考卷题）": 0, "二阶 L1/N": 1, "三阶均": 2, "半空间": 6}

scores = {}
for line in CACHE.read_text(encoding="utf-8").splitlines():
    p = [x.strip() for x in line.split("|")]
    if len(p) >= 9 and p[0].startswith("adult_eps"):
        scores[p[0]] = [float(x) for x in p[1:8]]

fig, axes = plt.subplots(2, len(COLS), figsize=(4.4 * len(COLS), 7))
for r, e in enumerate(EPS):
    for c, (name, idx) in enumerate(COLS.items()):
        ax = axes[r][c]
        a = np.array([scores[f"adult_eps{e}_aimsel_seed{s}"][idx] for s in SEEDS])
        b = np.array([scores[f"adult_eps{e}_aimfull_seed{s}"][idx] for s in SEEDS])
        ratio = a / b
        mean_ratio = a.mean() / b.mean()
        # 均值比的标准误：两组独立，δ 法
        se = mean_ratio * np.sqrt((a.std(ddof=1) / np.sqrt(len(a)) / a.mean()) ** 2
                                  + (b.std(ddof=1) / np.sqrt(len(b)) / b.mean()) ** 2)
        colors = ["#2b7bba" if s <= 3 else "#5aa9e6" for s in SEEDS]
        bars = ax.bar(list(SEEDS), ratio, width=0.7, color=colors)
        for bar, v in zip(bars, ratio):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=7)
        mb = ax.bar([9.3], [mean_ratio], width=0.7, color="#d9534f", yerr=[se], capsize=4)
        ax.text(9.3, mean_ratio + se + 0.01, f"{mean_ratio:.3f}\n({(mean_ratio - 1) * 100:+.1f}%)",
                ha="center", fontsize=8, color="#d9534f")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(list(SEEDS) + [9.3])
        ax.set_xticklabels([f"s{s}" for s in SEEDS] + ["8种子均"], fontsize=8)
        ax.set_ylim(0.7, 1.15)
        ax.set_title(f"adult ε={e.replace('p', '.')}  {name}（AIM 官方=1）", fontsize=10)
        if r == 0 and c == 0:
            ax.text(0.02, 0.96, "深蓝 s1–3（第五十三步）  浅蓝 s4–8（本步新增）\n红 = 均值比 ±1SE",
                    transform=ax.transAxes, fontsize=8, va="top")
plt.suptitle("第 2 步（AIM 选择器 × 我们生成器，Δ2=1）相对 AIM 官方的误差比，<1 为我们更好", fontsize=12)
plt.tight_layout()
out = ROOT / "logs_eps/scores/aimsel_step54.png"
plt.savefig(out, dpi=130)
print(out)
