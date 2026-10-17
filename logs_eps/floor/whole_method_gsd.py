"""第六十四步：我们 vs GSD 的"整体方法"比较（adult 1A，三种子，五档 ε）——不新跑，只换配对。

两种比较：
  比生成器：两臂同一停止规则（到噪声地板即停），只换搜索引擎 → 第六十三步的 地板×1 对 地板×1。
  比整体方法：各用各的停法。我们 = 引擎 + 地板停（兜底原早停；1A 上清洗与洗牌平均自动恒等），
             GSD = 原生早停跑到底 → 本步的 地板×1（我们）对 终表（GSD）。
我们的地板×1 快照与硬地板停交出的表逐位相同（tests/test_snapshot.py::test_snapshot_equals_stop_floor_table），
故可直接用第六十二/六十三步的快照表，不必复跑。
时间口径：我们 = 搜索循环内到地板的秒（含首轮 GPU 预热）；GSD = 搜索循环到其原生停点的原始秒与净秒
（净秒扣除其每 32,561 代一次的记账突发）。两者都不含进程启动（我们数秒，GSD 约一分钟编译）。
输出 logs_eps/scores/whole_method_gsd.md。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_analyze import EPS, ROOT, TAGS, find, gsd_points, load_cache, ours_points  # noqa: E402

SEEDS = [1, 2, 3]
AX = ["二阶", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]
OUT = ROOT / "logs_eps" / "scores" / "whole_method_gsd.md"


def collect(cache: dict) -> dict:
    data = {}
    for tag in TAGS:
        rows = []
        for s in SEEDS:
            po, _, to = ours_points(tag, cache, s)
            pg, _, tg = gsd_points(tag, cache, s)
            a, g_end, g_floor = find(po, "地板×1"), pg[-1], find(pg, "地板×1")
            rows.append(dict(
                seed=s, ours=a["scores"], gsd_end=g_end["scores"], gsd_floor=g_floor["scores"],
                ours_round=a["round"], ours_cand=a["cand_share"] * to["cand"], ours_t=a["t_share"] * to["t"],
                gsd_gens=tg["gens"], gsd_cand=tg["cand"], gsd_t=tg["t"], gsd_tnet=tg["t_net"],
                gsd_floor_tnet=g_floor["tnet"],
            ))
        data[tag] = rows
    return data


def mean(rows, key, ax=None):
    return float(np.mean([r[key][ax] if ax else r[key] for r in rows]))


def ratio_rows(rows, ax, opp):
    return [r["ours"][ax] / r[opp][ax] for r in rows]


def joined(rows, key, fmt):
    return " / ".join(format(r[key], fmt) for r in rows)


def main() -> None:
    cache = load_cache()
    data = collect(cache)
    L = []
    L.append("# 第六十四步：我们 vs GSD 整体方法比较（adult 1A，三种子，五档 ε；不新跑，只换配对）\n")
    L.append("我们 = 地板停交出的表（地板×1 快照，与硬地板停逐位相同）；GSD = 原生早停跑到底的终表。"
             "比值 = 我们 ÷ GSD，三种子均值，括号内逐种子。误差 ×10⁻⁶。\n")

    L.append("## 一、主轴与各轴（我们 地板停 ÷ GSD 终表）\n")
    L.append("| ε | " + " | ".join(AX) + " |")
    L.append("|---|" + "---|" * len(AX))
    for tag in TAGS:
        rows = data[tag]
        cells = []
        for ax in AX:
            rr = ratio_rows(rows, ax, "gsd_end")
            cells.append(f"{np.mean(rr):.3f}（{' / '.join(f'{x:.3f}' for x in rr)}）")
        L.append(f"| {EPS[tag]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("对照——比生成器（两臂同地板停，第六十三步）：\n")
    L.append("| ε | " + " | ".join(AX) + " |")
    L.append("|---|" + "---|" * len(AX))
    for tag in TAGS:
        rows = data[tag]
        L.append(f"| {EPS[tag]} | " + " | ".join(f"{np.mean(ratio_rows(rows, ax, 'gsd_floor')):.3f}" for ax in AX) + " |")
    L.append("")

    L.append("## 二、绝对误差（三种子均）\n")
    L.append("| ε | 臂 | " + " | ".join(AX) + " |")
    L.append("|---|---|" + "---|" * len(AX))
    for tag in TAGS:
        rows = data[tag]
        L.append(f"| {EPS[tag]} | 我们 地板停 | " + " | ".join(f"{mean(rows, 'ours', ax) * 1e6:,.0f}" for ax in AX) + " |")
        L.append(f"| {EPS[tag]} | GSD 终表 | " + " | ".join(f"{mean(rows, 'gsd_end', ax) * 1e6:,.0f}" for ax in AX) + " |")
    L.append("")

    L.append("## 三、代价（我们 到地板；GSD 到其原生停点）\n")
    L.append("| ε | 我们 轮（逐种子） | 我们 候选 | 我们 秒（逐种子） | GSD 代（逐种子） | GSD 候选 | GSD 原始秒（逐种子） | GSD 净秒 | 秒之比 原 / 净 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        rows = data[tag]
        ot, gt, gn = mean(rows, "ours_t"), mean(rows, "gsd_t"), mean(rows, "gsd_tnet")
        L.append(
            f"| {EPS[tag]} | {mean(rows, 'ours_round'):.0f}（{joined(rows, 'ours_round', 'd')}） | "
            f"{mean(rows, 'ours_cand'):.2g} | {ot:.0f}（{joined(rows, 'ours_t', '.0f')}） | "
            f"{mean(rows, 'gsd_gens'):,.0f}（{joined(rows, 'gsd_gens', ',d')}） | {mean(rows, 'gsd_cand'):.2g} | "
            f"{gt:.0f}（{joined(rows, 'gsd_t', '.0f')}） | {gn:.0f} | "
            f"1/{gt / ot:.0f} / 1/{gn / ot:.0f} |"
        )
    L.append("")
    L.append("GSD 到地板的净秒（比生成器那行的时间）供对照：" + "、".join(
        f"ε={EPS[tag]} {mean(data[tag], 'gsd_floor_tnet'):.0f}" for tag in TAGS) + "。\n")

    L.append("## 四、读法\n")
    two = [np.mean(ratio_rows(data[t], "二阶", "gsd_end")) for t in TAGS]
    gen = [np.mean(ratio_rows(data[t], "二阶", "gsd_floor")) for t in TAGS]
    L.append(
        f"1. 整体方法（各用各的停法）：二阶 {' / '.join(f'{x:.2f}' for x in two)}；"
        f"比生成器（同停法）：{' / '.join(f'{x:.2f}' for x in gen)}。"
        "两行之差 = GSD 跑到底比它自己地板停多学（或少学）的噪声：ε = 0.3 / 1 差 12～14 个百分点，ε = 0.1 / 3 差 3～4 个百分点，"
        "ε = 10 反向 6 个百分点（GSD 跑到底比其地板停好 5.9%）。"
    )
    L.append("2. ε = 10 整体方法我们略输（1.03）：地板停在 ε=10 比跑到底差 3.5%，GSD 跑到底则拿到了这一截。这是统一 1.0 倍规则的代价，如实写。")
    L.append("3. 时间：我们到地板的秒 对 GSD 整场搜索秒，原始口径 1/170～1/27，净口径 1/65～1/11。GSD 的原始秒含其实现的记账开销（原版行为），净秒已扣。")
    L.append("4. 本比较不新跑任何东西：我们臂的表就是第六十二/六十三步的地板×1 快照（与硬地板停逐位相同），GSD 臂的表就是其终表。")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {OUT.relative_to(ROOT)}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
