"""第六十七步阅卷：1B 比整体方法 我们（清洗 + 地板停 + 锁定洗牌）÷ PGM 原生（同录像跑满 10000 轮，第五十一步 pgmrep10k）。

同一 AIM 选测录像，两臂各按作者原样跑完（怎么停是生成器自带的零件，不替对手改）。每张表用
scripts/score_any.py 阅七轴（×10⁻⁶），按种子配对算 我们÷PGM；分组阅卷（scripts/score_pair_groups.py：
测过的对 / 没测的对 / 一阶）同样配对；我们的停法与核对比从 *_lock.json 读出。
只阅我们已出表的场次。输出 logs_eps/scores/whole_method_pgm.md。

用法: .venv/bin/python logs_eps/whole1b/grade_whole1b.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps" / "pgmfloor"))
from grade_pgmfloor import AXES, GROUPS, fmt_ratio, pair_groups, score_any  # noqa: E402

OUT = ROOT / "logs_eps" / "scores" / "whole_method_pgm.md"
ARMS = {
    "ours": "results/whole1b/adult_eps{tag}_ourswhole_seed{s}.csv",
    "pgm": "results/stage1/adult_eps{tag}_pgmrep10k_seed{s}.csv",
}
LABEL = {"ours": "我们 整体方法", "pgm": "PGM 原生 10000 轮"}


def path(arm: str, tag: str, s: int) -> Path:
    return ROOT / ARMS[arm].format(tag=tag, s=s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    lines = ["# 1B 比整体方法：我们（清洗 + 地板停 + 锁定洗牌）÷ PGM 原生，同一 AIM 选测录像", "",
             "两臂吃同一份 AIM 录像测量，各按作者原样跑完：我们 = 一阶清洗 z=2（未被二阶覆盖的字段）→ 第一段地板停 ×1.0 "
             "→ 纤维洗牌 100 趟 → 第二段真题锁定、每题平均差一个人即停；PGM = mirror descent 跑满 10000 轮（第五十一步 pgmrep10k）。"
             "七轴为与真表的 L1/N（×10⁻⁶），比值 <1 我们误差更小。", ""]
    ratio_tab = ["| ε | 场次 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"]
    group_tab = ["| ε | 场次 | " + " | ".join(GROUPS) + " |", "|---|---|---|---|---|"]
    abs_tab = ["| ε | 臂 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"]
    stop_tab = ["| ε | 种子 | 清洗改动格子 | 第一段停法/轮/秒 | 第一段÷地板 | 洗牌秒 | 第二段停法/轮/秒 | 第二段÷地板 "
                "| 终表真题÷第一段 | 合计秒 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in args.eps.split(","):
        done = [s for s in seeds if path("ours", tag, s).exists() and path("pgm", tag, s).exists()]
        if not done:
            continue
        e = tag.replace("p", ".")
        r_axes = {ax: [] for ax in AXES[1:]}
        r_groups = {g: [] for g in GROUPS}
        absm = {a: [] for a in ARMS}
        for s in done:
            tabs = {a: path(a, tag, s) for a in ARMS}
            sc = score_any(list(tabs.values()))
            got = {a: sc[p.stem] for a, p in tabs.items()}
            for i, ax in enumerate(AXES[1:], start=1):
                r_axes[ax].append(got["ours"][i] / got["pgm"][i])
            for a in absm:
                absm[a].append(got[a][1:])
            pg = pair_groups(tag, s, tabs)
            for j, g in enumerate(GROUPS):
                r_groups[g].append(pg["ours"][j] / pg["pgm"][j])
            lock = path("ours", tag, s).with_name(path("ours", tag, s).stem + "_lock.json")
            o = json.load(open(lock, encoding="utf-8")) if lock.exists() else {}
            log = ROOT / f"logs_eps/whole1b/adult_eps{tag}_ourswhole_seed{s}.log"
            pooled, shuffle_s = "—", "—"
            if log.exists():
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("一阶平分清洗"):
                        pooled = line.split("未测出格子 ")[1].split("（")[0] + "/" + line.split("格子 ")[1].split("，")[0]
                    elif line.startswith("纤维洗牌"):
                        shuffle_s = line.split("耗时 ")[1].split(" 秒")[0]
            if o:
                total = o["stage1_elapsed_s"] + o["stage2_elapsed_s"] + (float(shuffle_s) if shuffle_s != "—" else 0.0)
                stop_tab.append(
                    f"| {e} | {s} | {pooled} | {o['stage1_stop_reason']} / {o['stage1_rounds']} / {o['stage1_elapsed_s']:.0f} "
                    f"| {o['stage1_over_floor']:.3f} | {shuffle_s} | {o['stage2_stop_reason']} / {o['stage2_rounds']} / {o['stage2_elapsed_s']:.0f} "
                    f"| {o['stage2_over_floor']:.3f} | {o['final_over_stage1']:.3f} | {total:.0f} |")
        ratio_tab.append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_axes[ax]) for ax in AXES[1:]) + " |")
        group_tab.append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_groups[g]) for g in GROUPS) + " |")
        for a in ARMS:
            m = np.mean(np.array(absm[a]), axis=0)
            abs_tab.append(f"| {e} | {LABEL[a]} | " + " | ".join(f"{v:,.0f}" for v in m) + " |")
    lines += ["## 1. 我们 ÷ PGM 原生（种子配对，均值与逐种子）", ""] + ratio_tab + [""]
    lines += ["## 2. 分组：AIM 测过的对 / 没测的对 / 一阶（我们 ÷ PGM 原生）", ""] + group_tab + [""]
    lines += ["## 3. 各臂绝对分（×10⁻⁶，种子均）", ""] + abs_tab + [""]
    lines += ["## 4. 我们的停法与核对比", "", "清洗改动格子 = 一阶清洗判为未测出的格子数 / 清洗范围内格子数（0 即恒等）；"
              "第一段÷地板 与 第二段÷地板 应 ≈1（略低是一轮过冲）；终表真题÷第一段 ≈1 表示第二段守住了测过的边缘。"
              "秒为 GPU（A6000）上各段耗时，不含载入。", ""] + stop_tab + [""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
