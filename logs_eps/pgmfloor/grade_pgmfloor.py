"""第六十六步阅卷：1B 比生成器 我们(地板停+锁定洗牌) vs PGM(地板停)，同一 AIM 选测录像、同一停止规则。

每张表用 scripts/score_any.py 阅七轴（×10⁻⁶），按种子配对算 我们÷PGM；分组阅卷
（scripts/score_pair_groups.py：测过的对 / 没测的对 / 一阶）同样配对；两臂停法从
*_lock.json（我们）与 *_stop.json（PGM）读出。参考列：PGM 原生跑满 10000 轮（pgmrep10k）。
只阅两臂都已出表的场次。输出 logs_eps/scores/pgmfloor_1b.md。

用法: .venv/bin/python logs_eps/pgmfloor/grade_pgmfloor.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AXES = ["measured", "二阶", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]
GROUPS = ["测过的对", "没测的对", "一阶"]
OUT = ROOT / "logs_eps" / "scores" / "pgmfloor_1b.md"
ARMS = {
    "ours": "results/pgmfloor/adult_eps{tag}_oursfloor_seed{s}.csv",
    "pgm": "results/pgmfloor/adult_eps{tag}_pgmfloor_seed{s}.csv",
    "pgm10k": "results/stage1/adult_eps{tag}_pgmrep10k_seed{s}.csv",
    "oursdeep": "results/pgmfloor/adult_eps{tag}_oursdeep_seed{s}.csv",
}


def path(arm: str, tag: str, s: int) -> Path:
    return ROOT / ARMS[arm].format(tag=tag, s=s)


def score_any(paths: list[Path]) -> dict[str, list[float]]:
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/score_any.py"), "adult",
         str(ROOT / "data/adult/adult.csv")] + [str(p) for p in paths],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout
    res = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 9 or parts[0] in ("表", "adult"):
            continue
        try:
            res[parts[0]] = [float(x) * 1e6 for x in parts[1:8]]
        except ValueError:
            continue
    return res


def pair_groups(tag: str, s: int, arms: dict[str, Path]) -> dict[str, list[float]]:
    dump = ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{s}_measurements.json"
    cmd = [sys.executable, str(ROOT / "scripts/score_pair_groups.py"), "--real", str(ROOT / "data/adult/adult.csv"),
           "--dump", str(dump)] + [f"{a}={p}" for a, p in arms.items()]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    rows = {}
    for line in out.splitlines():
        mm = re.match(r"(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$", line)
        if mm and mm.group(1) != "表":
            rows[mm.group(1)] = [float(mm.group(i)) for i in (2, 3, 4)]
    return rows


def stops(tag: str, s: int) -> dict:
    lock = path("ours", tag, s).with_name(path("ours", tag, s).stem + "_lock.json")
    stop = path("pgm", tag, s).with_name(path("pgm", tag, s).stem + "_stop.json")
    o = json.load(open(lock, encoding="utf-8")) if lock.exists() else {}
    p = json.load(open(stop, encoding="utf-8")) if stop.exists() else {}
    return {"ours": o, "pgm": p}


def fmt_ratio(xs: list[float]) -> str:
    return f"**{np.mean(xs):.3f}**（" + "/".join(f"{x:.3f}" for x in xs) + "）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    lines = ["# 1B 比生成器：我们（地板停 + 锁定洗牌）÷ PGM（地板停），同一 AIM 选测录像", "",
             "两臂同一停止规则：各自损失首次到噪声地板 ×1.0 即停（我们逐轮查，PGM 逐步查）。我们的第二段只补没测的对、"
             "真题锁定在第一段终表计数上，不再看带噪测量。七轴为与真表的 L1/N（×10⁻⁶），比值 <1 我们误差更小。"
             "参考列 PGM 原生 = 同录像跑满 10000 轮（第五十一步 pgmrep10k）。", ""]
    ratio_tab = ["| ε | 场次 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"]
    group_tab = ["| ε | 场次 | " + " | ".join(GROUPS) + " |", "|---|---|---|---|---|"]
    abs_tab = ["| ε | 臂 | " + " | ".join(AXES[1:]) + " |", "|---" * (len(AXES) + 1) + "|"]
    stop_tab = ["| ε | 种子 | 我们 第一段停法/轮/秒 | 我们 第一段÷地板 | 我们 第二段停法/轮/秒 | 终表真题÷地板 | 终表÷第一段 "
                "| PGM 停法/步/秒 | PGM ÷地板 |", "|---|---|---|---|---|---|---|---|---|"]
    for tag in args.eps.split(","):
        done = [s for s in seeds if path("ours", tag, s).exists() and path("pgm", tag, s).exists()]
        if not done:
            continue
        r_axes = {ax: [] for ax in AXES[1:]}
        r_groups = {g: [] for g in GROUPS}
        r_ref = {ax: [] for ax in AXES[1:]}
        r_deep = {ax: [] for ax in AXES[1:]}
        absm = {"ours": [], "pgm": [], "pgm10k": [], "oursdeep": []}
        for s in done:
            tabs = {a: path(a, tag, s) for a in ARMS if path(a, tag, s).exists()}
            sc = score_any(list(tabs.values()))
            got = {a: sc[p.stem] for a, p in tabs.items()}
            for i, ax in enumerate(AXES[1:], start=1):
                r_axes[ax].append(got["ours"][i] / got["pgm"][i])
                if "pgm10k" in got:
                    r_ref[ax].append(got["pgm"][i] / got["pgm10k"][i])
                if "oursdeep" in got:
                    r_deep[ax].append(got["oursdeep"][i] / got["ours"][i])
            for a in absm:
                if a in got:
                    absm[a].append(got[a][1:])
            pg = pair_groups(tag, s, {"ours": tabs["ours"], "pgm": tabs["pgm"]})
            for j, g in enumerate(GROUPS):
                r_groups[g].append(pg["ours"][j] / pg["pgm"][j])
            st = stops(tag, s)
            o, p = st["ours"], st["pgm"]
            if o and p:
                stop_tab.append(
                    f"| {tag.replace('p', '.')} | {s} | {o['stage1_stop_reason']} / {o['stage1_rounds']} / {o['stage1_elapsed_s']:.0f} "
                    f"| {o['stage1_over_floor']:.3f} | {o['stage2_stop_reason']} / {o['stage2_rounds']} / {o['stage2_elapsed_s']:.0f} "
                    f"| {o['final_over_floor']:.3f} | {o['final_over_stage1']:.3f} "
                    f"| {p['stop_reason']} / {p['iters_done']} / {p['fit_s']:.0f} | {p['final_over_floor']:.3f} |")
        e = tag.replace("p", ".")
        ratio_tab.append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_axes[ax]) for ax in AXES[1:]) + " |")
        group_tab.append(f"| {e} | {len(done)} | " + " | ".join(fmt_ratio(r_groups[g]) for g in GROUPS) + " |")
        for a, label in (("ours", "我们 地板停"), ("pgm", "PGM 地板停"), ("pgm10k", "PGM 原生 10000 轮"),
                         ("oursdeep", "我们 第二段深跑到救援耗尽（先导场）")):
            if absm[a]:
                m = np.mean(np.array(absm[a]), axis=0)
                abs_tab.append(f"| {e} | {label} | " + " | ".join(f"{v:,.0f}" for v in m) + " |")
        if r_ref["二阶"]:
            abs_tab.append(f"| {e} | PGM 地板停 ÷ PGM 原生 | " + " | ".join(fmt_ratio(r_ref[ax]) for ax in AXES[1:]) + " |")
        if r_deep["二阶"]:
            abs_tab.append(f"| {e} | 我们 深跑 ÷ 我们 地板停 | " + " | ".join(fmt_ratio(r_deep[ax]) for ax in AXES[1:]) + " |")
    lines += ["## 1. 我们 ÷ PGM（同停法，种子配对，均值与逐种子）", ""] + ratio_tab + [""]
    lines += ["## 2. 分组：AIM 测过的对 / 没测的对 / 一阶（我们 ÷ PGM）", ""] + group_tab + [""]
    lines += ["## 3. 各臂绝对分（×10⁻⁶，种子均）与 PGM 地板停相对其原生跑法", ""] + abs_tab + [""]
    lines += ["## 4. 两臂停法与核对比", "", "我们 第一段÷地板 与 PGM ÷地板 都应 ≈1（略低是一步/一轮的过冲）；"
              "终表真题÷地板 是第二段洗牌加锁定后终表对带噪测量的损失相对地板，终表÷第一段 ≈1 表示测过的边缘守住了。", ""] + stop_tab + [""]
    lines += ["时间脚注：我们的秒是 GPU（A6000）搜索循环内秒，PGM 的秒是本机 CPU 拟合秒，设备不同，只作量级参考。"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
