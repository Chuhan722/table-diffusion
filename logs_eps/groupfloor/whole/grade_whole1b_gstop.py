"""第六十九步阅卷：1B 整体方法 第一段分档地板停（results/groupfloor/whole1b，A6000）÷ PGM 原生（第五十一步 pgmrep10k），
并与同命令只去掉分档停开关、第一段跑到收敛的版本（results/groupfloor/whole1b_conv，带清洗）、第六十七步地板停 ×1.0 版（results/whole1b）并排。
分档停 ÷ 收敛 两臂只差第一段的停法这一个开关，是停法代价的干净消融。

同一 AIM 选测录像，两臂各按作者原样跑完。每张表用 scripts/score_any.py 阅七轴（×10⁻⁶，与真表的 L1/N），
按种子配对算比值；分组阅卷（scripts/score_pair_groups.py：测过的对 / 没测的对 / 一阶）同样配对；
我们的停法与核对比从 *_lock.json 读出，分档停点各档÷档地板从 *_curve_groups.json 读出。只阅已出表的场次。
输出 logs_eps/scores/whole_method_gstop.md。

用法: .venv/bin/python logs_eps/groupfloor/whole/grade_whole1b_gstop.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "logs_eps" / "pgmfloor"))
from grade_pgmfloor import AXES, GROUPS, fmt_ratio, pair_groups, score_any  # noqa: E402

OUT = ROOT / "logs_eps" / "scores" / "whole_method_gstop.md"
ARMS = {
    "gstop": "results/groupfloor/whole1b/adult_eps{tag}_oursgstop_seed{s}.csv",
    "conv": "results/groupfloor/whole1b_conv/adult_eps{tag}_oursconv_seed{s}.csv",
    "floor": "results/whole1b/adult_eps{tag}_ourswhole_seed{s}.csv",
    "pgm": "results/stage1/adult_eps{tag}_pgmrep10k_seed{s}.csv",
}
LABEL = {"gstop": "我们 分档停", "conv": "我们 第一段收敛（同命令去掉分档停）", "floor": "我们 地板停×1.0（第六十七步）", "pgm": "PGM 原生 10000 轮"}
LOGDIR = {"gstop": "logs_eps/groupfloor/whole", "conv": "logs_eps/groupfloor/whole_conv"}
PAIRS = [("gstop", "pgm"), ("conv", "pgm"), ("gstop", "conv"), ("gstop", "floor")]


def path(arm: str, tag: str, s: int) -> Path:
    return ROOT / ARMS[arm].format(tag=tag, s=s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    lines = ["# 1B 比整体方法：第一段分档地板停 ÷ PGM 原生（并排 同命令第一段收敛版、第六十七步地板停）", "",
             "两臂吃同一份 AIM 录像测量：我们 = 一阶清洗 z=2（未被二阶覆盖的字段）→ 第一段分档地板停（每个 σ 档都 ≤ 自己的 ½Σwσ²，"
             "<30 格的档并入邻档）→ 纤维洗牌 100 趟 → 第二段真题锁定、每题平均差一个人即停；PGM = mirror descent 跑满 10000 轮。"
             "收敛版 = 同一命令只去掉分档停开关，第一段由平台早停（1e-4/100 轮）收工，清洗、洗牌、第二段全同；"
             "第六十七步版只差第一段是总和地板停 ×1.0。"
             "七轴为与真表的 L1/N（×10⁻⁶），比值 <1 前者误差更小。", ""]
    ratio_tabs = {p: ["| ε | 场次 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"] for p in PAIRS}
    group_tabs = {p: ["| ε | 场次 | " + " | ".join(GROUPS) + " |", "|---|---|---|---|---|"] for p in PAIRS}
    abs_tab = ["| ε | 臂 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"]
    abs_group_tab = ["| ε | 臂 | " + " | ".join(GROUPS) + " |", "|---|---|---|---|---|"]
    stop_tab = ["| ε | 种子 | 臂 | 第一段停法/轮/秒 | 第一段总和÷地板 | 停点各档÷档地板 | 洗牌秒 | 第二段停法/轮/秒 | 第二段÷地板 "
                "| 终表真题÷第一段 | 合计秒 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    speed = {}
    for tag in args.eps.split(","):
        done = [s for s in seeds if all(path(a, tag, s).exists() for a in ARMS)]
        if not done:
            continue
        e = tag.replace("p", ".")
        speed[tag] = []
        r_axes = {p: {ax: [] for ax in AXES[1:]} for p in PAIRS}
        r_groups = {p: {g: [] for g in GROUPS} for p in PAIRS}
        absm = {a: [] for a in ARMS}
        absg = {a: [] for a in ARMS}
        for s in done:
            tabs = {a: path(a, tag, s) for a in ARMS}
            sc = score_any(list(tabs.values()))
            got = {a: sc[p.stem] for a, p in tabs.items()}
            pg = pair_groups(tag, s, tabs)
            for p in PAIRS:
                for i, ax in enumerate(AXES[1:], start=1):
                    r_axes[p][ax].append(got[p[0]][i] / got[p[1]][i])
                for j, g in enumerate(GROUPS):
                    r_groups[p][g].append(pg[p[0]][j] / pg[p[1]][j])
            for a in ARMS:
                absm[a].append(got[a][1:])
                absg[a].append(pg[a])
            gp = tabs["gstop"]
            gcsv = gp.with_name(gp.stem + "_curve_groups.csv")
            gj = gp.with_name(gp.stem + "_curve_groups.json")
            per_group, total_floor = "—", None
            if gcsv.exists():
                rows = list(csv.reader(gcsv.open(encoding="utf-8")))
                floors = [float(x) for x in rows[-1][1:]] if rows and rows[-1][0] == "floor" else []
                data = [r for r in rows[1:] if r and r[0] != "floor"]
                if floors and data:
                    total_floor = sum(floors)
                    last = data[-1]  # 最后一条数据行 = 分档停交出的起点表那轮
                    per_group = " / ".join(f"{float(v) / f:.2f}" for v, f in zip(last[1:], floors))
                    if gj.exists():
                        meta = json.load(open(gj, encoding="utf-8"))
                        sig = [f"{min(x['sigmas']):.1f}" + ("+" if len(x["sigmas"]) > 1 else "") for x in meta["groups"]]
                        per_group = "σ " + "/".join(sig) + "：" + per_group
                    per_group += f"（轮 {last[0]}）"
            tot = {}
            for arm in ("gstop", "conv"):
                ap_ = tabs[arm]
                lock = ap_.with_name(ap_.stem + "_lock.json")
                o = json.load(open(lock, encoding="utf-8")) if lock.exists() else {}
                log = ROOT / LOGDIR[arm] / f"{ap_.stem}.log"
                shuffle_s = "—"
                if log.exists():
                    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                        if line.startswith("纤维洗牌") and "耗时 " in line:
                            shuffle_s = line.split("耗时 ")[1].split(" 秒")[0]
                if not o:
                    continue
                over = o.get("stage1_over_floor")
                if over is None and total_floor:
                    over = o["stage1_final_loss"] / total_floor  # 收敛版没给地板，用同格分档曲线的各档地板之和
                tot[arm] = o["stage1_elapsed_s"] + o["stage2_elapsed_s"] + (float(shuffle_s) if shuffle_s != "—" else 0.0)
                stop_tab.append(
                    f"| {e} | {s} | {LABEL[arm].split('（')[0]} | {o['stage1_stop_reason']} / {o['stage1_rounds']} / {o['stage1_elapsed_s']:.0f} "
                    f"| {over:.3f} | {per_group if arm == 'gstop' else '—'} | {shuffle_s} | {o['stage2_stop_reason']} / {o['stage2_rounds']} / {o['stage2_elapsed_s']:.0f} "
                    f"| {o['stage2_over_floor']:.3f} | {o['final_over_stage1']:.3f} | {tot[arm]:.0f} |")
            if len(tot) == 2:
                speed[tag].append(tot["conv"] / tot["gstop"])
        for p in PAIRS:
            ratio_tabs[p].append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_axes[p][ax]) for ax in AXES[1:]) + " |")
            group_tabs[p].append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_groups[p][g]) for g in GROUPS) + " |")
        for a in ARMS:
            m = np.mean(np.array(absm[a]), axis=0)
            abs_tab.append(f"| {e} | {LABEL[a]} | " + " | ".join(f"{v:,.0f}" for v in m) + " |")
            mg = np.mean(np.array(absg[a]), axis=0)
            abs_group_tab.append(f"| {e} | {LABEL[a]} | " + " | ".join(f"{v:,.0f}" for v in mg) + " |")
    sec = 1
    for p in PAIRS:
        lines += [f"## {sec}. {LABEL[p[0]]} ÷ {LABEL[p[1]]}（种子配对，均值与逐种子）", ""] + ratio_tabs[p] + [""]
        lines += [f"### {sec}a. 分组：AIM 测过的对 / 没测的对 / 一阶", ""] + group_tabs[p] + [""]
        sec += 1
    lines += [f"## {sec}. 各臂绝对分（×10⁻⁶，种子均）", ""] + abs_tab + [""] + abs_group_tab + [""]
    sec += 1
    lines += [f"## {sec}. 我们两臂的停法、核对比与耗时", "", "第一段总和÷地板 = 第一段交出的表的总损失 ÷ 总地板（分档停 <1 因快的档已过头；收敛版更低）；"
              "停点各档÷档地板 应全 ≤1、最慢档 ≈1；第二段÷地板 应 ≈1；终表真题÷第一段 ≈1 表示第二段守住了测过的边缘。"
              "秒为 GPU（A6000）上各段耗时，不含载入。", ""] + stop_tab + [""]
    lines += ["收敛版合计秒 ÷ 分档停合计秒（种子均）：" + "；".join(
        f"ε={t.replace('p', '.')} {np.mean(v):.2f}×" for t, v in speed.items() if v), ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
