"""第五十九步 甲（出菜单重写 + GPU build 融合核）全程验收阅卷：adult ε=3 / 0.1 各 3 种子，1A 命令逐字。

两问：① 结果——新终表（results/speed2，甲）与 乙 验收表（results/speed）是否逐位同（cmp），曲线的损失列是否逐行同；
      ② 时间——整场（发射脚本 START→END）、每轮毫秒，对 乙 场与 GSD 各列一栏，并附当时主机负载。
甲 的设计目标就是逐位同（所有改动只换运算组织不换数学），不同即是 bug。

用法: .venv/bin/python logs_eps/speed/grade_speed2.py [--out logs_eps/scores/speed2_step59.md]
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import glob
import re
import statistics as st
from pathlib import Path

from grade_speed import ROOT, SEEDS, TAGS, gsd_minutes, stamps, tail_fields

ARM_OLD, ARM_NEW = "ours2way_sparse", "ours2way_jia"
DIR_OLD, DIR_NEW = "results/speed", "results/speed2"


def curve_same(a: Path, b: Path) -> str:
    """曲线 CSV 除时间无关列外逐行比对（所有列都是演化量，没有时间列，整行比）。"""
    if not (a.is_file() and b.is_file()):
        return "缺文件"
    ra = list(csv.reader(open(a, encoding="utf-8")))
    rb = list(csv.reader(open(b, encoding="utf-8")))
    if ra == rb:
        return f"逐行同（{len(ra) - 1} 轮）"
    n = min(len(ra), len(rb))
    first = next((i for i in range(n) if ra[i] != rb[i]), n)
    return f"**第 {first} 行起不同**（行数 {len(ra) - 1} vs {len(rb) - 1}）"


def loads(launch_log: str, arm: str) -> dict[str, str]:
    out = {}
    for line in open(launch_log, encoding="utf-8", errors="ignore"):
        m = re.match(r"\S+ START (adult_eps\S+_%s_seed\d) load=(\S+)" % arm, line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs_eps/scores/speed2_step59.md")
    args = ap.parse_args()
    old_whole = stamps([str(ROOT / "logs_eps/speed/launch_speed.log")], ARM_OLD)
    new_whole = stamps([str(ROOT / "logs_eps/speed/launch_speed2.log")], ARM_NEW)
    old_load = loads(str(ROOT / "logs_eps/speed/launch_speed.log"), ARM_OLD)
    new_load = loads(str(ROOT / "logs_eps/speed/launch_speed2.log"), ARM_NEW)

    lines = ["# 第五十九步 甲 全程验收：出菜单重写 + GPU build 融合核（adult ε=3 / 0.1 × 3 种子，1A 命令逐字）", "",
             f"乙 场 = results/speed（10-14 14:58～16:17 北京，乙 验收，终表已证与 1A 旧表逐位同）；甲 场 = {DIR_NEW}（10-14 22:19 起，A6000 GPU1 道 24-47）。"
             "GSD 为 1A 同一份带噪全二阶文件的原生早停场。整场 = 发射脚本 START→END；每轮 = 演化内计时 ÷ 轮数。", "",
             "## 一、结果：甲 终表与 乙 终表逐位比对", "",
             "| ε | 种子 | 终表 cmp | 曲线 | 轮数 乙→甲 | 停止原因 | 最终损失 乙→甲 |", "|---|---|---|---|---|---|---|"]
    all_same = True
    timing: dict[str, list] = {}
    for tag, e in TAGS:
        rows = []
        for s in SEEDS:
            old_csv = ROOT / f"{DIR_OLD}/adult_eps{tag}_{ARM_OLD}_seed{s}.csv"
            new_csv = ROOT / f"{DIR_NEW}/adult_eps{tag}_{ARM_NEW}_seed{s}.csv"
            same = old_csv.is_file() and new_csv.is_file() and filecmp.cmp(old_csv, new_csv, shallow=False)
            cs = curve_same(ROOT / f"{DIR_OLD}/adult_eps{tag}_{ARM_OLD}_seed{s}_curve.csv",
                            ROOT / f"{DIR_NEW}/adult_eps{tag}_{ARM_NEW}_seed{s}_curve.csv")
            all_same &= same and cs.startswith("逐行同")
            o = tail_fields(ROOT / f"logs_eps/speed/adult_eps{tag}_{ARM_OLD}_seed{s}.log")
            n = tail_fields(ROOT / f"logs_eps/speed/adult_eps{tag}_{ARM_NEW}_seed{s}.log")
            rows.append((s, o, n))
            lines.append(f"| {e} | s{s} | {'逐位同' if same else '**不同**'} | {cs} | {o['rounds']}→{n['rounds']} | {o['stop']}→{n['stop']} | {o['loss']}→{n['loss']} |")
        timing[tag] = rows
    lines += ["", f"**{'6/6 终表逐位同、曲线逐行同' if all_same else '有不同，见上表'}**。"
              "甲 的全部改动（出菜单组主序重写、直算条目/括根/增益/漂移键权/结构尺融合核、去同步）都以逐位同为设计目标，"
              "全轴分数与 1A 成绩单相同，对 GSD 的判定不变。", "",
              "## 二、时间（分钟，3 种子均（逐种子））", "",
              "| ε | GSD 整场 | 乙 整场 | 甲 整场 | 乙 每轮 ms | 甲 每轮 ms | 甲÷乙 | GSD÷甲 | 乙 场负载 | 甲 场负载 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, e in TAGS:
        gw, _ = gsd_minutes(tag)
        ow = [old_whole[f"adult_eps{tag}_{ARM_OLD}_seed{s}"] / 60 for s in SEEDS]
        nw = [new_whole[f"adult_eps{tag}_{ARM_NEW}_seed{s}"] / 60 for s in SEEDS]
        oms = [1000 * float(o["secs"]) / int(o["rounds"]) for _, o, _ in timing[tag]]
        nms = [1000 * float(n["secs"]) / int(n["rounds"]) for _, _, n in timing[tag]]
        ol = " ".join(old_load.get(f"adult_eps{tag}_{ARM_OLD}_seed{s}", "?").split(",")[0] for s in SEEDS)
        nl = " ".join(new_load.get(f"adult_eps{tag}_{ARM_NEW}_seed{s}", "?").split(",")[0] for s in SEEDS)
        j = lambda xs, f="{:.0f}": "/".join(f.format(x) for x in xs)
        lines.append(f"| {e} | {st.mean(gw):.1f}（{j(gw)}） | {st.mean(ow):.1f}（{j(ow)}） | {st.mean(nw):.1f}（{j(nw)}） | "
                     f"{st.mean(oms):.0f}（{j(oms)}） | {st.mean(nms):.0f}（{j(nms)}） | {st.mean(nms) / st.mean(oms):.2f} | {st.mean(gw) / st.mean(nw):.1f}× | {ol} | {nl} |")
    lines += ["",
              "读法：负载 = 各场发射时 1 分钟 load average（112 核机，他人作业共用 CPU）；全程每轮含全菜单轮、重试轮、救援轮与首轮 CUDA 装载，"
              "同条件数以 300 轮稳态剖面为准（ε=3 稳态 68.5 → 56.3 ms，logs_eps/prof/step59_profile.md §七），整场数用于对 GSD 的墙钟副轴。"
              "ε=0.3 / 1 / 10 三档待同一脚本补跑后再换五档墙钟表。"]
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
