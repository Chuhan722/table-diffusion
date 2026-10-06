"""第五十二步汇总图：adult ε=0.1/1 各臂 measured 与二阶（3 种子均，×1e-6），AIM 官方为 1.0 基线。

用法: .venv/bin/python logs_eps/alloc/plot_step12.py  (读 logs_eps/scores/alloc_eps{0p1,1}_summary.md，输出 logs_eps/scores/alloc_step12.png)
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
ARMS = [("原生", "原生(全测,bounded)"), ("k105f05", "全测 Δ2=1 两档σ"), ("best", "固定规则最好"),
        ("hybrid", "第0步混合(AIM卷)"), ("aimsel", "第2步 AIM选择器+我们"), ("AIM官方", "AIM 官方")]
BEST = {"0p1": "k33f50", "1": "auto_K83"}


def load(eps):
    rows = {}
    for line in (ROOT / f"logs_eps/scores/alloc_eps{eps}_summary.md").read_text(encoding="utf-8").splitlines():
        p = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(p) >= 5 and re.fullmatch(r"[\d.]+", p[0]):
            rows[p[1]] = (float(p[3]), float(p[4]), float(p[6]))
    return rows


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
titles = ["measured", "二阶 L1/N", "三阶 top5%"]
for k, ax in enumerate(axes):
    for i, eps in enumerate(("0p1", "1")):
        rows = load(eps)
        base = rows["AIM官方"][k]
        vals, x = [], []
        for j, (key, lab) in enumerate(ARMS):
            key = BEST[eps] if key == "best" else key
            if key in rows:
                vals.append(rows[key][k] / base)
                x.append(j + i * 0.4)
        bars = ax.bar(x, vals, width=0.38, label=f"ε={eps.replace('p', '.')}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks([j + 0.2 for j in range(len(ARMS))])
    ax.set_xticklabels([lab for _, lab in ARMS], rotation=25, ha="right", fontsize=8)
    ax.set_title(f"{titles[k]}（AIM 官方 = 1）")
    ax.legend(fontsize=8)
plt.tight_layout()
out = ROOT / "logs_eps/scores/alloc_step12.png"
plt.savefig(out, dpi=130)
print(out)
