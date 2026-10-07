"""第五十三步图：adult ε=0.1 / ε=1 逐种子，第 2 步 aimsel 与第 0 步混合相对 AIM 官方（=1）的比值。

用法: .venv/bin/python logs_eps/aimsel/plot_step53.py
读 logs_eps/scores/step53_perseed.txt（没有则调 /tmp/score_any.py 生成），输出 logs_eps/scores/aimsel_step53.png
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "logs_eps/scores/step53_perseed.txt"
EPS = ["0p1", "1"]
ARMS = {"AIM": "results/adult_eps{e}_aimfull_seed{s}.csv",
        "hybrid": "results/aimreplay/adult_eps{e}_hybrid_seed{s}.csv",
        "aimsel": "results/aimsel/adult_eps{e}_aimsel_seed{s}.csv"}
COLS = {"measured": 1, "二阶 L1/N": 2, "三阶均": 3, "半空间": 7}

if not CACHE.exists():
    tables = [str(ROOT / p.format(e=e, s=s)) for e in EPS for s in (1, 2, 3) for p in ARMS.values()]
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "/tmp/score_any.py", "adult",
                          str(ROOT / "data/adult/adult.csv")] + tables,
                         capture_output=True, text=True, check=True).stdout
    CACHE.write_text(out, encoding="utf-8")
scores = {}
for line in CACHE.read_text(encoding="utf-8").splitlines():
    p = [x.strip() for x in line.split("|")]
    if len(p) >= 9 and p[0].startswith("adult_eps"):
        scores[p[0]] = [float(x) for x in p[1:8]]

fig, axes = plt.subplots(2, len(COLS), figsize=(4.2 * len(COLS), 7))
for r, e in enumerate(EPS):
    for c, (name, idx) in enumerate(COLS.items()):
        ax = axes[r][c]
        for k, arm in enumerate(("hybrid", "aimsel")):
            vals = []
            for s in (1, 2, 3):
                base = scores[f"adult_eps{e}_aimfull_seed{s}"][idx - 1]
                vals.append(scores[f"adult_eps{e}_{arm}_seed{s}"][idx - 1] / base)
            mean = sum(vals) / 3
            x = [s + (k - 0.5) * 0.36 for s in (1, 2, 3)] + [4.3 + (k - 0.5) * 0.36]
            bars = ax.bar(x, vals + [mean], width=0.34,
                          label={"hybrid": "第0步混合(AIM卷+我们)", "aimsel": "第2步(AIM选择器+我们)"}[arm])
            for b, v in zip(bars, vals + [mean]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks([1, 2, 3, 4.3])
        ax.set_xticklabels(["s1", "s2", "s3", "均"])
        ax.set_ylim(0.7, 1.25)
        ax.set_title(f"adult ε={e.replace('p', '.')}  {name}（AIM 官方=1）", fontsize=10)
        if r == 0 and c == 0:
            ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = ROOT / "logs_eps/scores/aimsel_step53.png"
plt.savefig(out, dpi=130)
print(out)
