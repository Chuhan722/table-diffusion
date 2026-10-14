"""1A 墙钟副轴：我们 ours2way vs GSD（adult 五档 ε × 3 种子，同一台 A6000、同 GPU 道 24 核，两家各自原生早停）。

整场墙钟取发射脚本的 START/END 时间戳（含读卷、建表、落表）；内计时取各自日志（我们 = 演化循环"耗时"，GSD = 其 fit 的"搜索"秒数）。
GSD 跑于 10-07/08（logs_eps/baseline2/*_gpu[01].log），我们跑于 10-11/12（logs_eps/stage1/*.log），主机他人负载不同，CPU 端（我们出菜单 50~90 ms/轮）受影响、GPU 端不受。

用法: .venv/bin/python logs_eps/stage1/wallclock_1a.py [--out logs_eps/scores/stage1_wallclock.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAGS = [("0p1", "0.1"), ("0p3", "0.3"), ("1", "1"), ("3", "3"), ("10", "10")]


def stamps(files: list[str], arm: str) -> dict[str, float]:
    d: dict[str, dict[str, dt.datetime]] = {}
    for f in files:
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.match(r"(\S+)Z (START|END) (adult_eps\S+_%s_seed\d)" % arm, line)
            if m:
                d.setdefault(m.group(3), {})[m.group(2)] = dt.datetime.fromisoformat(m.group(1))
    return {k: (v["END"] - v["START"]).total_seconds() for k, v in d.items() if "START" in v and "END" in v}


def inner(pattern: str, rx: str) -> dict[str, float]:
    d = {}
    for f in glob.glob(str(ROOT / pattern)):
        m = re.search(rx, open(f, encoding="utf-8", errors="ignore").read())
        if m:
            d[Path(f).stem] = float(m.group(1))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs_eps/scores/stage1_wallclock.md")
    args = ap.parse_args()
    gsd = stamps(glob.glob(str(ROOT / "logs_eps/baseline2/*_gpu[01].log")), "gsd")
    ours = stamps(glob.glob(str(ROOT / "logs_eps/stage1/*.log")), "ours2way")
    gsd_in = inner("logs_eps/baseline2/adult_eps*_gsd_seed*.log", r"搜索 ([\d.]+) 秒")
    ours_in = inner("logs_eps/stage1/adult_eps*_ours2way_seed*.log", r"耗时 ([\d.]+) 秒")
    rounds = inner("logs_eps/stage1/adult_eps*_ours2way_seed*.log", r"实际轮数 (\d+)")
    lines = ["# 1A 墙钟副轴：我们 vs GSD（adult，单位分钟，3 种子均（逐种子））", "",
             "> 本表是 10-11/12 的 1A 旧场（稠密通路，第五十九步 乙/甲 提速前）。提速后同命令复跑的五档墙钟见 `speed_wallclock_step59.md`（终表逐位同）。", "",
             "同一台 A6000、同 GPU 道 24 核；两家原生早停（我们 --stop-threshold 1e-4 --stop-lag 100，上限 30000 轮；GSD stop_early 阈 1e-4，代数上限 5e7 即不设限）。"
             "整场 = 发射脚本 START→END；内计时 = 我们演化循环 / GSD fit。", "",
             "| ε | GSD 整场 | GSD fit | 我们整场 | 我们演化 | 我们轮数 | 我们每轮 ms | GSD÷我们 |",
             "|---|---|---|---|---|---|---|---|"]
    tg = to = 0.0
    for tag, e in TAGS:
        g = [gsd[f"adult_eps{tag}_gsd_seed{s}"] / 60 for s in (1, 2, 3)]
        gi = [gsd_in[f"adult_eps{tag}_gsd_seed{s}"] / 60 for s in (1, 2, 3)]
        o = [ours[f"adult_eps{tag}_ours2way_seed{s}"] / 60 for s in (1, 2, 3)]
        oi = [ours_in[f"adult_eps{tag}_ours2way_seed{s}"] for s in (1, 2, 3)]
        r = [rounds[f"adult_eps{tag}_ours2way_seed{s}"] for s in (1, 2, 3)]
        ms = [1000 * a / b for a, b in zip(oi, r)]
        tg += sum(g); to += sum(o)
        lines.append(
            f"| {e} | {st.mean(g):.1f}（{'/'.join(f'{x:.0f}' for x in g)}） | {st.mean(gi):.1f} | "
            f"{st.mean(o):.1f}（{'/'.join(f'{x:.0f}' for x in o)}） | {st.mean(oi) / 60:.1f} | {st.mean(r):.0f}（{'/'.join(f'{x:.0f}' for x in r)}） | "
            f"{st.mean(ms):.0f}（{'/'.join(f'{x:.0f}' for x in ms)}） | {st.mean(g) / st.mean(o):.1f}× |"
        )
    lines.append("")
    lines.append(f"15 场合计：GSD {tg / 60:.1f} h，我们 {to / 60:.1f} h。")
    lines.append("")
    lines.append("读法：ε=0.1 我们快 4.8×、0.3 快 1.6×；ε=1 慢 1.4×、ε=3 慢 1.7×；ε=10 持平。我们的墙钟 = 轮数 × 每轮：轮数随 ε 从 2.3k 涨到 14k（噪声越小、平台判据越晚触发），每轮 85~270 ms；"
                 "GSD 的代数随 ε 涨得慢（25→60 min）。所以“比 GSD 快”只在低 ε 成立，不是通盘结论。")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
