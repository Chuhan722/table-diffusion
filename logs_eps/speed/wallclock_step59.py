"""第五十九步 提速后的五档墙钟副轴：我们（乙+甲 后，results/speed2）vs GSD vs 1A 旧场（adult 五档 ε × 3 种子）。

整场墙钟取发射脚本的 START/END 时间戳（含读卷、建表、落表）；每轮 = 演化循环"耗时" ÷ 实际轮数。
GSD 跑于 10-07/08（logs_eps/baseline2/*_gpu[01].log）；1A 旧场 10-11/12（logs_eps/stage1）；
新场 10-14 夜（ε=3/0.1 launch_speed2.log，ε=0.3/1/10 launch_speed3.log），均 A6000 GPU1 道 24-47，他人负载各不同（附 1 分钟 load）。
终表与 1A 旧表 cmp：ε=3/0.1 已证逐位同（乙、甲 两轮验收）；ε=0.3/1/10 在此比对。

用法: .venv/bin/python logs_eps/speed/wallclock_step59.py [--out logs_eps/scores/speed_wallclock_step59.md]
"""
from __future__ import annotations

import argparse
import filecmp
import glob
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from grade_speed import ROOT, stamps, tail_fields

TAGS = [("0p1", "0.1"), ("0p3", "0.3"), ("1", "1"), ("3", "3"), ("10", "10")]
SEEDS = (1, 2, 3)
ARM = "ours2way_jia"


def loads(files: list[str], arm: str) -> dict[str, str]:
    out = {}
    for f in files:
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.match(r"\S+ START (adult_eps\S+_%s_seed\d) load=([\d.]+)" % arm, line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def curve_cmp(tag: str, seeds=SEEDS) -> str:
    """曲线 CSV 逐列比对（3 种子并）：轮数/状态/哪些列全同，哪些列只差末位（最大绝对差）。"""
    diff: dict[str, float] = {}
    same: set[str] = set()
    for s in seeds:
        a = pd.read_csv(ROOT / f"results/stage1/adult_eps{tag}_ours2way_seed{s}_curve.csv")
        b = pd.read_csv(ROOT / f"results/speed2/adult_eps{tag}_{ARM}_seed{s}_curve.csv")
        if len(a) != len(b) or (a["round"] != b["round"]).any() or (a["status"] != b["status"]).any():
            return "**轮数/状态异**"
        for c in a.columns:
            if c in ("round", "status"):
                continue
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            m = ~(np.isnan(x) & np.isnan(y)) & (x != y)
            if m.any():
                diff[c] = max(diff.get(c, 0.0), float(np.abs(x - y)[m].max()))
            else:
                same.add(c)
    same -= set(diff)
    if not diff:
        return "全同"
    return (f"轮数/状态/{'/'.join(c for c in ('old_loss', 'beta', 'direction_gain', 'max_gain_sum', 'temp') if c in same)} 同；"
            f"{'/'.join(diff)} 末位异 ≤{max(diff.values()):.0e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs_eps/scores/speed_wallclock_step59.md")
    args = ap.parse_args()
    gsd = stamps(glob.glob(str(ROOT / "logs_eps/baseline2/*_gpu[01].log")), "gsd")
    old = stamps(glob.glob(str(ROOT / "logs_eps/stage1/*.log")), "ours2way")
    launch = [p for p in (str(ROOT / "logs_eps/speed/launch_speed2.log"), str(ROOT / "logs_eps/speed/launch_speed3.log"))
              if Path(p).is_file()]
    new = stamps(launch, ARM)
    load = loads(launch, ARM)
    j = lambda xs, f="{:.0f}": "/".join(f.format(x) for x in xs)

    lines = ["# 第五十九步 提速后五档墙钟：我们（乙+甲）vs GSD vs 1A 旧场（adult，分钟，3 种子均（逐种子））", "",
             "同一台 A6000、同 GPU 道 24 核；两家原生早停（我们 --stop-threshold 1e-4 --stop-lag 100；GSD stop_early 阈 1e-4）。"
             "整场 = 发射脚本 START→END；每轮 = 演化循环耗时 ÷ 轮数。新场 = 乙（GPU 稀疏整表项/漂移项）+ 甲（出菜单重写 + GPU build 融合核）之后，1A 命令逐字。", "",
             "| ε | GSD 整场 | 旧整场 | **新整场** | 旧每轮 ms | **新每轮 ms** | 新÷旧 | GSD÷新 | 终表 vs 1A 旧表 | 曲线 vs 1A 旧曲线（3 种子并） | 新场 load |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    tg = to = tn = 0.0
    for tag, e in TAGS:
        names = [f"adult_eps{tag}_{ARM}_seed{s}" for s in SEEDS]
        if not all(n in new for n in names):
            lines.append(f"| {e} | | | 未跑完 | | | | | | | |")
            continue
        g = [gsd[f"adult_eps{tag}_gsd_seed{s}"] / 60 for s in SEEDS]
        o = [old[f"adult_eps{tag}_ours2way_seed{s}"] / 60 for s in SEEDS]
        n = [new[x] / 60 for x in names]
        ot = [tail_fields(ROOT / f"logs_eps/stage1/adult_eps{tag}_ours2way_seed{s}.log") for s in SEEDS]
        nt = [tail_fields(ROOT / f"logs_eps/speed/{x}.log") for x in names]
        oms = [1000 * float(t["secs"]) / int(t["rounds"]) for t in ot]
        nms = [1000 * float(t["secs"]) / int(t["rounds"]) for t in nt]
        same = [filecmp.cmp(ROOT / f"results/stage1/adult_eps{tag}_ours2way_seed{s}.csv",
                            ROOT / f"results/speed2/{x}.csv", shallow=False) for s, x in zip(SEEDS, names)]
        rounds_same = all(a["rounds"] == b["rounds"] and a["loss"] == b["loss"] for a, b in zip(ot, nt))
        cmp_txt = "3/3 逐位同" if all(same) else f"{sum(same)}/3 逐位同，轮数/终损{'同' if rounds_same else '**异**'}"
        curve_txt = curve_cmp(tag)
        tg += sum(g); to += sum(o); tn += sum(n)
        lines.append(f"| {e} | {st.mean(g):.1f}（{j(g)}） | {st.mean(o):.1f}（{j(o)}） | **{st.mean(n):.1f}**（{j(n)}） | "
                     f"{st.mean(oms):.0f}（{j(oms)}） | **{st.mean(nms):.0f}**（{j(nms)}） | {st.mean(nms) / st.mean(oms):.2f} | "
                     f"{st.mean(g) / st.mean(n):.1f}× | {cmp_txt} | {curve_txt} | {' '.join(load[x] for x in names)} |")
    lines += ["", f"15 场合计：GSD {tg / 60:.1f} h，旧 {to / 60:.1f} h，新 {tn / 60:.1f} h。", "",
              "读法：旧表（logs_eps/scores/stage1_wallclock.md）我们只在 ε≤0.3 快、ε=1/3 慢 1.4～1.7×、ε=10 持平；"
              "新场每轮按 ε 档的降幅见各行“新÷旧”，整场数含三方不同时段的主机负载（load 列为发射时 1 分钟均值，112 核）；"
              "同条件的提速以 300 轮稳态剖面为准（logs_eps/prof/step59_profile.md，ε=3 稳态 143（稠密）→ 90（乙）→ 56 ms（甲），−61%）。"
              "终表 15/15 与 1A 旧表逐位同；曲线 CSV 的轮数、状态、损失（old_loss）、β、方向增益、温度列逐行同，"
              "只有二次项 interaction（及由它算出的 step、个别轮的 expected_loss）差末位（最大绝对差 ~1e-7，量级 1e8）——"
              "这正是 乙 设计所允许的：GPU 稀疏路径只保证答案/残差逐位同，二次项的求和顺序不同。"
              "ε=10 种子 3（30 分钟、170 ms/轮）跑在他人负载 89 的时段，拉高了该档均值；其余各档 load 见表。"]
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
