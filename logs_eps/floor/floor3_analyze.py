"""第六十三步：第六十二步同法复跑种子 2、3，三种子汇总（adult 1A，五档 ε，两臂，沿噪声地板倍数取快照）。

复用 floor_analyze 的读取/核对/阅卷/分解（seed 参数），种子 1 的点来自第六十二步（含终×快照与起点表），
种子 2、3 只有本场的地板×快照与终表。输出 logs_eps/scores/floor_step63.md 与三种子图。
不在此定最终配置：只把三种子的停点形状、地板停收益、两臂同停法比值摆出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_analyze import (  # noqa: E402
    AXES, EPS, FLOOR, MAIN, ROOT, TAGS, best_label, decomp_table, decompose, find, gsd_name, gsd_points,
    gsd_table, load_cache, ours_name, ours_points, ours_table, pct,
)

SEEDS = [1, 2, 3]
OUT = ROOT / "logs_eps" / "scores" / "floor_step63.md"
PNG = ROOT / "logs_eps" / "floor" / "adult_地板停点_三种子_2026-10-16.png"

CONCLUSION = """\
## 结论（三种子，五档 ε；数字见下表）

1. **收齐与核对。** 种子 2、3 两臂五档共 20 场全部正常结束（rc=0），加上第六十二步的种子 1 共 30 场。每场的快照轮号与曲线首穿越轮一致，
   每张快照表按噪声考卷重算的 ℒ 与记账一致（我们 2·记账损失，GSD N²·fitness），分解中 ‖e‖²/Σσ² 三种子均在 0.99～1.00。
2. **谷形在三个种子上完全一样。** 我们的二阶最优层级：ε=0.1 ×0.95、0.3 ×0.95、1 ×0.9、3 ×0.8/0.7/0.8、10 ×0.7（种子 1 的 ε=0.1、10 落在
   第六十一步的终×1.2 快照：ε=0.1 处与 ×0.95 只差 0.1 个百分点；ε=10 处它位于 ×0.56，比 ×0.7 好 1.8 个百分点，说明 ε=10 的谷底在 ×0.7 网格
   下限之下，种子 2、3 无此快照）；GSD 三种子全部 ×0.9/×0.9/×0.9/×0.7/终表。ε 越大谷底越深，ε=10 的谷底几乎与终表齐平，均与种子 1 相同。
3. **地板停（×1.0）相对终表（我们，二阶，三种子均值，括号为三种子极差）：**
   −33.6%（0.4 点）、−32.1%（0.3）、−21.7%（0.8）、−9.3%（0.9）、+3.5%（1.0）；三阶均 −34.9/−32.2/−17.7/−2.8/+3.5%。
   与各自最优层级只差 +1.4/+1.3/+1.0/+2.5/+6.1%。种子间的差异比"停法之间"的差异小一个数量级。
4. **GSD 的地板停相对终表：** −4.7%（极差 6.5 点）、−15.4%（4.1）、−14.8%（0.5）、−4.2%（2.0）、+5.9%（1.4）；
   但距其自身最优 +24.9/+11.1/+1.4/+1.5/+5.9%——低 ε 时 GSD 在 ×1.0 处离谷底还很远（谷底在 ×0.9 或更低，网格到 ×0.9 为止），
   即"触地板即停"对 GSD 在低 ε 并不是好停点，这一点三种子一致。
5. **两臂同停法对照（×1.0，我们 ÷ GSD，三种子均值）：** 二阶 0.707/0.801/0.911/0.943/0.974（极差 0.041/0.032/0.011/0.026/0.018），
   三阶均 0.781/0.838/0.932/0.971/0.998，半空间 0.743/0.847/0.962/0.978/1.044，三阶top5% 0.974/0.918/0.904/0.929/1.002。
   二阶在 15 个（ε，种子）格上全部 <1，无一例外；三阶均 14 格 <1（ε=10 种子 1 为 1.003）；半空间 ε≤0.3 全部 <1，ε=1、3 各有一个种子
   在 1.01～1.02，ε=10 三种子均 >1。
6. **各自最优层级对照：** 二阶 0.870/0.879/0.915/0.935/0.973，三阶均 0.887/0.898/0.936/0.968/1.008；15 格二阶全部 <1。
7. **跑到底则两臂持平：** 终表二阶 1.015/0.998/0.991/0.996/0.996，三阶均 1.021/0.992/0.982/0.996/1.003，均在 ±2.1% 内；
   半空间 ε=0.1 为 1.051、ε=3 为 0.956、ε=10 为 1.041，其余 ±2% 内。
8. **分解：** 地板处真值平方误差 T 我们 ÷ GSD 三种子均值 0.711/0.780/0.810/0.815/0.841（15 格全部 <0.86）；终表处 0.98～1.02。
   追噪项 2⟨d,e⟩ 在地板处我们恒低于 GSD。结论与第六十二步相同：两臂的终点一致，差别在到达地板前走的路——同一损失下我们离真值更近。
9. **到地板的代价：** 我们 ×1.0 处的轮占比 1.0/1.2/1.9/4.1/16.2%（搜索循环秒 5～10/10～12/22～46/44～93/110～191，受他人负载影响），
   GSD 代占比 19.9/19.1/19.1/14.4/13.4%（净秒 102～140/111～152/130～163/139～179/200～213）。
10. **ε=10 的例外在三种子上稳定：** 地板停比终表差 3.0～4.0%（我们）、5.2～6.6%（GSD）；地板以上 ×5～×2 处 GSD 更好，×1.5 持平（差 ≤1%），
    ×1.2 以下我们略好 1～4%（三种子同）。
11. **限制：** GSD 每 1000 代才查一次阈值，×1.0 实际落在 0.991～0.998；GSD 的 GPU 搜索非逐位确定；种子 2、3 没有第六十一步的终×快照与
    起点表，其"最优"受 ×0.7 网格下限约束（ε=3、10）；GSD ε=0.1 的谷底仍在 ×0.9 以下未探明。
12. 本步只做复核，不定最终停法。
"""


def mean_str(vals: list[float], fmt: str) -> str:
    return fmt.format(float(np.mean(vals))) if vals else "—"


def summary_arm(arm: str, pts_by: dict[int, dict[str, list[dict] | None]]) -> list[str]:
    share_key, share_name = ("round_share", "轮占比") if arm == "我们" else ("gen_share", "代占比")
    lines = [
        f"## 汇总（{arm}，三种子）：二阶最优层级、地板停（×1.0）与终表之差、×1.0 与最优之差",
        "",
        f"| ε | 种子 | 二阶最优层级 | 三阶均最优 | ×1.0 {share_name} | ×1.0 候选占比 | ×1.0 二阶 Δ终 | ×1.0 三阶均 Δ终 | ×1.0 三阶top5% Δ终 | ×1.0 半空间 Δ终 | 二阶最优 Δ终 | ×1.0 二阶 Δ最优 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        acc: dict[str, list[float]] = {k: [] for k in ["d2", "d3", "d5", "dh", "db", "dm", "sh", "cs"]}
        for seed in SEEDS:
            pts = pts_by.get(seed, {}).get(tag)
            if not pts:
                continue
            final = pts[-1]["scores"]
            f1 = find(pts, "地板×1")
            bl = best_label(pts, "二阶")
            bp = find(pts, bl)["scores"]
            d = {ax: (f1["scores"][ax] - final[ax]) / final[ax] for ax in AXES}
            db = (bp["二阶"] - final["二阶"]) / final["二阶"]
            dm = (f1["scores"]["二阶"] - bp["二阶"]) / bp["二阶"]
            for k, v in zip(["d2", "d3", "d5", "dh", "db", "dm", "sh", "cs"],
                            [d["二阶"], d["三阶均"], d["三阶top5%"], d["半空间"], db, dm, f1[share_key], f1["cand_share"]]):
                acc[k].append(v)
            lines.append(
                f"| {EPS[tag]} | {seed} | {bl} | {best_label(pts, '三阶均')} | {f1[share_key]:.1%} | {f1['cand_share']:.1%} | "
                f"{pct(d['二阶'])} | {pct(d['三阶均'])} | {pct(d['三阶top5%'])} | {pct(d['半空间'])} | {pct(db)} | {pct(dm)} |"
            )
        if len(acc["d2"]) > 1:
            lines.append(
                f"| {EPS[tag]} | **均** | | | {mean_str(acc['sh'], '{:.1%}')} | {mean_str(acc['cs'], '{:.1%}')} | "
                f"**{mean_str(acc['d2'], '{:+.1%}')}** | {mean_str(acc['d3'], '{:+.1%}')} | {mean_str(acc['d5'], '{:+.1%}')} | "
                f"{mean_str(acc['dh'], '{:+.1%}')} | {mean_str(acc['db'], '{:+.1%}')} | **{mean_str(acc['dm'], '{:+.1%}')}** |"
            )
    lines.append("")
    return lines


def arms_three(ours: dict[int, dict], gsd: dict[int, dict]) -> list[str]:
    lines = [
        "## 两臂同停法对照（三种子）：我们 ÷ GSD（<1 我们好），每格 s1 / s2 / s3，均 = 三种子比值的算术平均",
        "",
        "| ε | 层级 | " + " | ".join(f"{ax} s1/s2/s3 | {ax} 均" for ax in MAIN) + " |",
        "|---|---|" + "---|---|" * len(MAIN),
    ]
    for tag in TAGS:
        for level in ["地板×1.2", "地板×1", "各自最优", "终表"]:
            cells = []
            for ax in MAIN:
                vals = []
                for seed in SEEDS:
                    po_all, pg_all = ours.get(seed, {}).get(tag), gsd.get(seed, {}).get(tag)
                    if not po_all or not pg_all:
                        vals.append(None)
                        continue
                    if level == "各自最优":
                        po, pg = find(po_all, best_label(po_all, "二阶")), find(pg_all, best_label(pg_all, "二阶"))
                    else:
                        po, pg = find(po_all, level), find(pg_all, level)
                    vals.append(None if po is None or pg is None else po["scores"][ax] / pg["scores"][ax])
                shown = [f"{v:.3f}" if v is not None else "—" for v in vals]
                got = [v for v in vals if v is not None]
                cells.append(" / ".join(shown))
                cells.append(f"**{np.mean(got):.3f}**" if len(got) > 1 else (f"{got[0]:.3f}" if got else "—"))
            if all(c == "—" for c in cells[::2]):
                continue
            name = {"地板×1": "地板停 ×1.0", "各自最优": "各自二阶最优层级"}.get(level, level)
            lines.append(f"| {EPS[tag]} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def decomp_three(ours: dict[int, dict], gsd: dict[int, dict]) -> list[str]:
    lines = [
        "## 分解（三种子）：地板停处与终表处的真值平方误差 T（除以 Σσ²），我们 ÷ GSD",
        "",
        "| ε | 种子 | T 我们 ×1.0 | T GSD ×1.0 | 比 | T 我们 终表 | T GSD 终表 | 比 | 追噪项 我们 ×1.0 | 追噪项 GSD ×1.0 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        for seed in SEEDS:
            po_all, pg_all = ours.get(seed, {}).get(tag), gsd.get(seed, {}).get(tag)
            if not po_all or not pg_all:
                continue
            po, pg = find(po_all, "地板×1"), find(pg_all, "地板×1")
            do, dg = decompose(po["path"], tag, seed), decompose(pg["path"], tag, seed)
            fo, fg = decompose(po_all[-1]["path"], tag, seed), decompose(pg_all[-1]["path"], tag, seed)
            lines.append(
                f"| {EPS[tag]} | {seed} | {do['T']:.3f} | {dg['T']:.3f} | {do['T'] / dg['T']:.3f} | "
                f"{fo['T']:.3f} | {fg['T']:.3f} | {fo['T'] / fg['T']:.3f} | {do['cross']:.3f} | {dg['cross']:.3f} |"
            )
    lines.append("")
    return lines


def cost_three(ours, gsd, tot_o, tot_g) -> list[str]:
    lines = [
        "## 到地板的代价（三种子）：×1.0 快照处各自花了多少（我们秒 = 搜索循环内；GSD 净秒 = 循环内扣记账突发）",
        "",
        "| ε | 种子 | 我们 轮 | 我们 候选 | 我们 秒 | 我们 全程轮 | GSD 代 | GSD 候选 | GSD 净秒 | GSD 全程代 | 我们终损失/地板 | GSD 终 ℒ/地板 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        for seed in SEEDS:
            po_all = ours.get(seed, {}).get(tag)
            if not po_all:
                continue
            po, to = find(po_all, "地板×1"), tot_o[seed][tag]
            pg_all = gsd.get(seed, {}).get(tag)
            if pg_all:
                pg, tg = find(pg_all, "地板×1"), tot_g[seed][tag]
                g = f"{pg['gen']:,} | {pg['cand_share'] * tg['cand']:.2g} | {pg['tnet']:.0f} | {tg['gens']:,} | {po_all[-1]['mult']:.3f} | {pg_all[-1]['mult']:.3f}"
            else:
                g = f"— | — | — | — | {po_all[-1]['mult']:.3f} | —"
            lines.append(
                f"| {EPS[tag]} | {seed} | {po['round']} | {po['cand_share'] * to['cand']:.2g} | {po['t_share'] * to['t']:.0f} | "
                f"{to['rounds']} | {g} |"
            )
    lines.append("")
    return lines


def plot(ours: dict[int, dict], gsd: dict[int, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = ["二阶", "三阶均"]
    styles = {1: "-", 2: "--", 3: ":"}
    fig, axes = plt.subplots(len(rows), len(TAGS), figsize=(3.4 * len(TAGS), 3.1 * len(rows)), squeeze=False)
    for j, tag in enumerate(TAGS):
        xs_all: list[float] = []
        for seed in SEEDS:
            pts = ours.get(seed, {}).get(tag)
            if pts:
                xs_all += [p["mult"] for p in pts if p["label"] != "起点"]
        if not xs_all:
            continue
        lo, hi = min(xs_all) * 0.93, max(xs_all) * 1.08
        for i, ax_name in enumerate(rows):
            ax = axes[i][j]
            ys: list[float] = []
            for arm, by, color in [("我们", ours, "C0"), ("GSD", gsd, "C3")]:
                for seed in SEEDS:
                    pts = by.get(seed, {}).get(tag)
                    if not pts:
                        continue
                    sel = [p for p in pts if p["label"] != "起点"]
                    x = [p["mult"] for p in sel]
                    y = [p["scores"][ax_name] * 1e6 for p in sel]
                    ax.plot(x, y, styles[seed], marker="o", ms=2, lw=1, color=color, label=f"{arm} s{seed}")
                    ax.plot([pts[-1]["mult"]], [pts[-1]["scores"][ax_name] * 1e6], "s", ms=5, color=color, mfc="none")
                    ys += [yy for xx, yy in zip(x, y) if lo <= xx <= hi]
            ax.axvline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_xscale("log")
            ticks = [t for t in (5, 3, 2, 1.5, 1.2, 1, 0.9, 0.8, 0.7, 0.6, 0.5) if min(xs_all) * 0.98 <= t <= max(xs_all) * 1.02]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(hi, lo)
            if ys:
                pad = 0.05 * (max(ys) - min(ys))
                ax.set_ylim(min(ys) - pad, max(ys) + pad)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"ε = {EPS[tag]}")
            if i == len(rows) - 1:
                ax.set_xlabel("损失 / 噪声地板（→ 搜索方向）")
            if j == 0:
                ax.set_ylabel(f"{ax_name} 误差 ×10⁻⁶")
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle("adult 三种子：真值误差随损失下降的形状（实线 s1、虚线 s2、点线 s3；空心方块 = 终表；竖虚线 = 地板）", fontsize=10)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    ours: dict[int, dict] = {}
    gsd: dict[int, dict] = {}
    tot_o: dict[int, dict] = {}
    tot_g: dict[int, dict] = {}
    body: list[str] = []
    missing = []
    for seed in SEEDS:
        ours[seed], gsd[seed], tot_o[seed], tot_g[seed] = {}, {}, {}, {}
        for tag in TAGS:
            if not (FLOOR / f"{ours_name(tag, seed)}.csv").exists():
                missing.append(f"我们 s{seed} ε={EPS[tag]}")
                continue
            pts, notes, tot = ours_points(tag, cache, seed)
            ours[seed][tag], tot_o[seed][tag] = pts, tot
            if seed != 1:
                t = ours_table(tag, pts, notes, tot)
                t[0] = t[0].replace(f"## 我们 ε = {EPS[tag]}", f"## 我们 种子 {seed} ε = {EPS[tag]}")
                body += t + decomp_table(f"我们 种子 {seed}", tag, pts)
        for tag in TAGS:
            if not (FLOOR / f"{gsd_name(tag, seed)}.csv").exists():
                missing.append(f"GSD s{seed} ε={EPS[tag]}")
                continue
            pts, notes, tot = gsd_points(tag, cache, seed)
            gsd[seed][tag], tot_g[seed][tag] = pts, tot
            if seed != 1:
                t = gsd_table(tag, pts, notes, tot)
                t[0] = t[0].replace(f"## GSD ε = {EPS[tag]}", f"## GSD 种子 {seed} ε = {EPS[tag]}")
                body += t + decomp_table(f"GSD 种子 {seed}", tag, pts)
    head = [
        "# 第六十三步：第六十二步同法复跑种子 2、3，三种子汇总（adult 1A，五档 ε，两臂）",
        "",
        CONCLUSION,
        "设定：命令与第六十二步逐字相同、只换种子（数据目录 s2/s3 的噪声考卷、菜单/抽样/初始化种子、GSD 种子）；"
        "噪声地板 ½Σσ² 三种子逐位相同，阈值网格原样复用。种子 1 的点取自第六十二步（含第六十一步的终×快照与起点表），"
        "种子 2、3 只有本场的地板×快照与终表。口径、核对项、阅卷同第六十二步；各层级表只列种子 2、3（种子 1 见第六十二步记录）。",
        "",
        ("未收：" + "、".join(missing)) if missing else "三种子两臂五档全部收齐。",
        "",
    ]
    tail = (summary_arm("我们", ours) + summary_arm("GSD", gsd) + arms_three(ours, gsd)
            + decomp_three(ours, gsd) + cost_three(ours, gsd, tot_o, tot_g))
    OUT.write_text("\n".join(head + tail + body), encoding="utf-8")
    plot(ours, gsd)
    print(f"written {OUT} and {PNG}")


if __name__ == "__main__":
    main()
