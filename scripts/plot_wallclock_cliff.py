# -*- coding: utf-8 -*-
"""墙钟悬崖图:加噪 ε=1 场,单场墙钟(三种子均),log 刻度。

口径说明(2026-10-05 统一配方场定稿):
- 全部为加噪 ε=1 δ=1e-5 场(AIM/PGM/Tab-PE 本就只有加噪场;我们/GSD 取加噪主线)。
- 我们: 统一配方 12 场 A6000 实测(logs_night/*_eps1_rule_s{1,2,3}.log 耗时行:
        nltcs 421.8/392.2/429.2s 均 414.4s; adult 8784.3/8364.8/6776.3s 均 7975.1s;
        acsmob 12956.4/12788.7/13161.9s 均 12969.0s; plants 516.0/492.7/153.8s 均 387.5s)。
- GSD:  nltcs/acsmob = A6000(logs_seed/geps*, logs_acs/gsd_s*); adult/plants = 4090(笔记:
        搜索 13-17min / 511-541s)。
- AIM:  全 A6000 CPU(logs_aim/*: nltcs 复测 1554/1793/1748s(带清池参数,与其余科同口径);
        adult 2521/2309/2756s; acsmob 13964/13048/11999s;
        plants 77869/253544/141287s=21.6/70.4/39.2h)。
- PGM:  nltcs numpy 版全场 35min(笔记第四十三步); 其余三科 OOM(估算内存 1.5e9/7.4e9/2.2e15 MB)。
- Tab-PE: 等行数场分钟级(第四十三步),快但误差数十倍。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datasets = ["nltcs", "adult", "acsmob", "plants"]
xlabels = [
    "nltcs\n(16 cols, 16K rows)",
    "adult\n(15 cols, 33K rows)",
    "acsmob\n(21 cols, 80K rows)",
    "plants\n(69 cols, 17K rows)",
]

# 小时,三种子均;None=OOM
H = 3600.0
hours = {
    "Ours (GPU)":   [414.4 / H, 7975.1 / H, 12969.0 / H, 387.5 / H],
    "GSD (GPU)":    [153.4 / H, 15.0 / 60, 4907.9 / H, 526.0 / H],
    "AIM (CPU)":    [1698.3 / H, 2528.7 / H, 13003.7 / H, 157566.7 / H],
    "Tab-PE (CPU)": [0.017, 0.033, 0.075, 0.033],
    "PGM (CPU)":    [35.0 / 60, None, None, None],
}
colors = {
    "Ours (GPU)": "#1a7a3a",
    "GSD (GPU)": "#6ea8dc",
    "AIM (CPU)": "#e07b2a",
    "Tab-PE (CPU)": "#9467bd",
    "PGM (CPU)": "#8c564b",
}
# AIM plants 三种子 21.6/39.2/70.4h,误差棒=min~max
aim_plants = (77869 / H, 253544 / H)
pgm_oom_mb = ["1.5e9 MB", "7.4e9 MB", "2.2e15 MB"]  # adult/acsmob/plants

fig, ax = plt.subplots(figsize=(13.5, 6.2))
n_m = len(hours)
width = 0.16
x = np.arange(len(datasets))

for i, (name, vals) in enumerate(hours.items()):
    pos = x + (i - (n_m - 1) / 2) * width
    for j, v in enumerate(vals):
        if v is None:
            continue
        kw = {}
        if name == "AIM (CPU)" and j == 3:
            lo, hi = aim_plants
            kw = dict(yerr=[[v - lo], [hi - v]], capsize=5,
                      error_kw=dict(lw=1.6, ecolor="black"))
        ax.bar(pos[j], v, width, color=colors[name],
               label=name if j == 0 else None, **kw)

# PGM OOM 标注
for j, mb in zip([1, 2, 3], pgm_oom_mb):
    pos_pgm = x[j] + (4 - (n_m - 1) / 2) * width
    ax.text(pos_pgm, 0.013, f"PGM: OOM\n({mb})", ha="center", va="bottom",
            fontsize=8.5, color=colors["PGM (CPU)"], rotation=0)

# AIM plants 方差标注
ax.text(x[3] + (2 - (n_m - 1) / 2) * width, 78,
        "21.6h ~ 70.4h\n(3 seeds)", ha="center", fontsize=9.5)

ax.set_yscale("log")
ax.set_ylim(0.01, 200)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("Wall-clock per run (hours, log scale)")
ax.set_title(r"Wall-clock comparison at $\varepsilon=1$ (noisy runs, mean of 3 seeds)")
ax.legend(loc="upper left", ncol=5, fontsize=9.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figs/wallclock_cliff.png", dpi=150)
fig.savefig("figs/wallclock_cliff.pdf")
print("saved figs/wallclock_cliff.{png,pdf}")
