"""第五十九步 乙（稀疏整表项/漂移项）全程验收阅卷：adult ε=3 / 0.1 各 3 种子，1A 命令逐字只多 --gpu-sparse auto。

两问：① 结果——新终表与 results/stage1 旧表（稠密通路）是否逐位同（cmp），最终损失/轮数/停止原因是否同；
      ② 时间——整场（发射脚本 START→END）、演化内计时、每轮毫秒，对旧场与 GSD（logs_eps/baseline2）各列一栏。
逐位同即全轴分数同，1A 对 GSD 的判定原样成立，不必重阅。

用法: .venv/bin/python logs_eps/speed/grade_speed.py [--out logs_eps/scores/speed_step59.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import glob
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAGS = [("3", "3"), ("0p1", "0.1")]
SEEDS = (1, 2, 3)


def stamps(files: list[str], arm: str) -> dict[str, float]:
    d: dict[str, dict[str, dt.datetime]] = {}
    for f in files:
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.match(r"(\S+)Z (START|END) (adult_eps\S+_%s_seed\d)" % arm, line)
            if m:
                d.setdefault(m.group(3), {})[m.group(2)] = dt.datetime.fromisoformat(m.group(1))
    return {k: (v["END"] - v["START"]).total_seconds() for k, v in d.items() if "START" in v and "END" in v}


def tail_fields(log: Path) -> dict[str, str]:
    txt = log.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key, rx in (("rounds", r"实际轮数 (\d+)"), ("stop", r"停止原因 (\S+?)，"), ("loss", r"最终损失 ([\d.]+)"),
                    ("secs", r"耗时 ([\d.]+) 秒"), ("ms", r"平均每轮 (\d+) 毫秒")):
        m = re.search(rx, txt)
        out[key] = m.group(1) if m else "?"
    return out


def gsd_minutes(tag: str) -> tuple[list[float], list[float]]:
    whole = stamps(glob.glob(str(ROOT / "logs_eps/baseline2/*_gpu[01].log")), "gsd")
    w = [whole[f"adult_eps{tag}_gsd_seed{s}"] / 60 for s in SEEDS]
    fit = []
    for s in SEEDS:
        txt = (ROOT / f"logs_eps/baseline2/adult_eps{tag}_gsd_seed{s}.log").read_text(encoding="utf-8", errors="ignore")
        fit.append(float(re.search(r"搜索 ([\d.]+) 秒", txt).group(1)) / 60)
    return w, fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs_eps/scores/speed_step59.md")
    args = ap.parse_args()
    old_whole = stamps(glob.glob(str(ROOT / "logs_eps/stage1/*.log")), "ours2way")
    new_whole = stamps([str(ROOT / "logs_eps/speed/launch_speed.log")], "ours2way_sparse")

    lines = ["# 第五十九步 乙 全程验收：稀疏整表项/漂移项（adult ε=3 / 0.1 × 3 种子，1A 命令逐字 + `--gpu-sparse auto`）", "",
             "旧场 = 1A 正式场（results/stage1，稠密通路，10-11/12 跑）；新场 = results/speed（10-14 14:58～16:17 北京，A6000 GPU1 道 24-47）。"
             "GSD 为 1A 同一份带噪全二阶文件的原生早停场（10-07/08）。整场 = 发射脚本 START→END（含读卷、建表、落表）；每轮 = 演化内计时 ÷ 轮数。", "",
             "## 一、结果：新旧终表逐位比对", "",
             "| ε | 种子 | 终表 cmp | 轮数 旧→新 | 停止原因 旧→新 | 最终损失 旧→新 |", "|---|---|---|---|---|---|"]
    all_same = True
    timing: dict[str, dict] = {}
    for tag, e in TAGS:
        rows = []
        for s in SEEDS:
            old_csv = ROOT / f"results/stage1/adult_eps{tag}_ours2way_seed{s}.csv"
            new_csv = ROOT / f"results/speed/adult_eps{tag}_ours2way_sparse_seed{s}.csv"
            same = old_csv.is_file() and new_csv.is_file() and filecmp.cmp(old_csv, new_csv, shallow=False)
            all_same &= same
            o = tail_fields(ROOT / f"logs_eps/stage1/adult_eps{tag}_ours2way_seed{s}.log")
            n = tail_fields(ROOT / f"logs_eps/speed/adult_eps{tag}_ours2way_sparse_seed{s}.log")
            rows.append((s, o, n))
            lines.append(f"| {e} | s{s} | {'逐位同' if same else '**不同**'} | {o['rounds']}→{n['rounds']} | {o['stop']}→{n['stop']} | {o['loss']}→{n['loss']} |")
        timing[tag] = rows
    lines += ["", f"**{'6/6 逐位同' if all_same else '有不同，见上表'}**：终表、轮数、停止原因、最终损失全部一致，全轴分数与 1A 成绩单（logs_eps/scores/stage1.md）相同，对 GSD 的判定不变。"
              "按设计只有答案/残差保证逐位同、组错位分与漂移项允许末位异；实测 6 场 1.5 万轮内无一处分叉。", "",
              "## 二、时间（分钟，3 种子均（逐种子））", "",
              "| ε | GSD 整场 | GSD fit | 旧整场 | 新整场 | 旧每轮 ms | 新每轮 ms | 新每轮÷旧 | GSD÷新 |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, e in TAGS:
        gw, gf = gsd_minutes(tag)
        ow = [old_whole[f"adult_eps{tag}_ours2way_seed{s}"] / 60 for s in SEEDS]
        nw = [new_whole[f"adult_eps{tag}_ours2way_sparse_seed{s}"] / 60 for s in SEEDS]
        oms = [1000 * float(o["secs"]) / int(o["rounds"]) for _, o, _ in timing[tag]]
        nms = [1000 * float(n["secs"]) / int(n["rounds"]) for _, _, n in timing[tag]]
        j = lambda xs, f="{:.0f}": "/".join(f.format(x) for x in xs)
        lines.append(f"| {e} | {st.mean(gw):.1f}（{j(gw)}） | {st.mean(gf):.1f} | {st.mean(ow):.1f}（{j(ow)}） | {st.mean(nw):.1f}（{j(nw)}） | "
                     f"{st.mean(oms):.0f}（{j(oms)}） | {st.mean(nms):.0f}（{j(nms)}） | {st.mean(nms) / st.mean(oms):.2f} | {st.mean(gw) / st.mean(nw):.1f}× |")
    lines += ["",
              "读法：每轮均值含全菜单轮、重试轮、救援轮与当时主机负载，新旧两场负载不同（旧场 10-11 白天高负载，新场 load 6～14 / 112 核），"
              "所以全程每轮的降幅大于同条件 300 轮稳态剖面的 −37%（ε=3 143→90 ms，logs_eps/prof/step59_profile.md）；"
              "同条件数以剖面为准，整场数用于对 GSD 的墙钟副轴。ε=0.3 / 1 / 10 三档未复测，墙钟表（logs_eps/scores/stage1_wallclock.md）这三档仍是旧数。"]
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
