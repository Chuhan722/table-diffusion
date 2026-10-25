"""第七十四步 acsmob 1A 全网格阅卷：探针并行自动停（FORCE=conv 版，F 跑到底）15 格。

每格五张表过 scripts/score_any.py acsmob：F 的 G 快照（地板停会交的表）、F 终表（收敛停会交的表）、P 终表、GSD 收敛终表、AIM 官方表。
  判停法真值 = <格>_decision.json（探针判的 floor/conv）；方法交表 = floor→F_G，conv→F 终；方法墙钟 = floor→tD−t0，conv→tE−t0。
  判停对错 = 方法交表的二阶 是否等于 min(F_G, F 终) 的二阶（上帝视角）。
  GSD 墙钟 = logs_eps/baseline2/acsmob_eps<tag>_gsd_seed<s>.log 的"搜索 X 秒"；AIM 墙钟 = 同目录 aimfull 日志"完成 Xs"（ε=1 无日志）。
用法：cd residual-evolution && .venv/bin/python logs_eps/acsmob/grade_acsmob.py [变体名=acsmob]
输出 logs_eps/scores/acsmob_1a.md 与 results/autostop_acsmob/grade.json。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv/bin/python"
SCORE = ROOT / "scripts/score_any.py"
AXES = ["measured", "二阶", "三阶均", "三阶top5%", "三阶最大", "等值", "半空间"]
TAGS = ["0p1", "0p3", "1", "3", "10"]
EPS = {"0p1": "0.1", "0p3": "0.3", "1": "1", "3": "3", "10": "10"}
SEEDS = (1, 2, 3)


def score(tables: list[Path]) -> dict[str, dict[str, float]]:
    out = subprocess.run([str(PY), str(SCORE), "acsmob", *map(str, tables)], check=True,
                         capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    res: dict[str, dict[str, float]] = {}
    for line in out[1:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        res[parts[0]] = dict(zip(AXES, map(float, parts[1:8])))
    return res


def secs(path: Path, pat: str) -> float | None:
    if not path.exists():
        return None
    m = re.findall(pat, path.read_text(encoding="utf-8", errors="replace"))
    return float(m[-1]) if m else None


def grade_cell(variant: str, tag: str, seed: int) -> dict:
    out = ROOT / f"results/autostop_{variant}"
    cell = f"1A_eps{tag}_s{seed}"
    dec = json.load(open(out / f"{cell}_decision.json", encoding="utf-8"))
    tm = json.load(open(out / f"{cell}_times.json", encoding="utf-8"))
    fg = json.load(open(out / f"F_{cell}_curve_groups.json", encoding="utf-8"))
    tables = {"F_G": out / f"F_{cell}_snapG.csv", "F_终": out / f"F_{cell}.csv", "P_终": out / f"P_{cell}.csv",
              "GSD": ROOT / f"results/baseline2/acsmob_eps{tag}_gsd_seed{seed}.csv",
              "AIM": ROOT / f"results/acsmob_eps{tag}_aimfull_seed{seed}.csv"}
    sc = score(list(tables.values()))
    byname = {k: sc[v.stem] for k, v in tables.items()}
    d = dec["decision"]
    deliver = "F_G" if d == "floor" else "F_终"
    best = min(("F_G", "F_终"), key=lambda k: byname[k]["二阶"])
    wall_method = (tm["tD"] - tm["t0"]) if d == "floor" else (tm["tE"] - tm["t0"])
    f_snaps = (out / f"F_{cell}_snaps.csv").read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
    return dict(cell=cell, tag=tag, eps=EPS[tag], seed=seed, decision=d, reason_P=dec["stop_reason_P"],
                G_round_P=dec["G_round"], P_last_round=dec["P_last_round"], P_loop_s=dec["P_last_elapsed_s"],
                z_final=dec["z_final"], F_G_round=int(f_snaps[2]), F_G_loop_s=float(f_snaps[4]),
                F_rounds=fg["rounds"], F_reason=fg["stop_reason"], F_loop_s=last_elapsed(out / f"F_{cell}_curve.csv"),
                wall_method=wall_method, wall_all=tm["wall"], F_G_mtime_s=tm["F_snapG_mtime"] - tm["t0"],
                deliver=deliver, best=best, correct=(deliver == best),
                gsd_s=secs(ROOT / f"logs_eps/baseline2/acsmob_eps{tag}_gsd_seed{seed}.log", r"搜索 ([0-9.]+) 秒"),
                aim_s=secs(ROOT / f"logs_eps/baseline2/acsmob_eps{tag}_aimfull_seed{seed}.log", r"完成 ([0-9.]+)s"),
                scores=byname)


def last_elapsed(curve: Path) -> float | None:
    rows = curve.read_text(encoding="utf-8").strip().splitlines()
    head = rows[0].split(",")
    i = head.index("elapsed_s")
    for line in reversed(rows[1:]):
        p = line.split(",")
        if p[0] != "final" and p[i]:
            return float(p[i])
    return None


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "acsmob"
    jobs = [(variant, t, s) for t in TAGS for s in SEEDS]
    with ThreadPoolExecutor(max_workers=5) as ex:
        cells = list(ex.map(lambda j: grade_cell(*j), jobs))
    json.dump(cells, open(ROOT / f"results/autostop_{variant}/grade.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    L = ["# 第七十四步：acsmob 1A 全网格阅卷（探针并行自动停 + 显著回升停，F 跑到底核对判停；10-24）", ""]
    L.append("方法交表：探针判地板停 → F 的 G 快照；判收敛停 → F 终表。方法墙钟：地板停格 = 判停时刻 − 起跑（F 的 G 快照早已落盘），收敛停格 = 交表时刻 − 起跑。"
             "阅卷尺 `scripts/score_any.py acsmob`（真表二阶 = 全字段对 TVD 均值）。GSD = 同一噪声考卷跑到收敛的终表；AIM = 官方全流程表（只作参考）。")
    L.append("")
    L.append("## 一、逐格：判停、对错、墙钟")
    L.append("")
    L.append("| 格 | 探针判 | P 停因 | P 过线轮 | P 终轮 | z 终 | F_G 轮 | F 终轮 | F 停因 | 交表 | 上帝视角最优 | 判对 | 方法墙钟 s | F_G 落盘 s | 全程墙钟 s | GSD 搜索 s | AIM s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        L.append(f"| {c['cell']} | {c['decision']} | {c['reason_P']} | {c['G_round_P']} | {c['P_last_round']} | {c['z_final']:+.2f} | {c['F_G_round']} | {c['F_rounds']} | {c['F_reason']} | "
                 f"{c['deliver']} | {c['best']} | {'✓' if c['correct'] else '✗'} | {c['wall_method']:.0f} | {c['F_G_mtime_s']:.0f} | {c['wall_all']:.0f} | {fmt(c['gsd_s'], 0)} | {fmt(c['aim_s'], 0)} |")
    L.append("")
    L.append("## 二、逐格：真值误差（二阶 TVD ×1e3；括号为 ÷GSD）")
    L.append("")
    L.append("| 格 | F_G | F 终 | P 终 | GSD | AIM | 方法交表 ÷GSD | 方法交表 ÷AIM | F 终 ÷GSD |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        s = c["scores"]
        g = s["GSD"]["二阶"]
        dv = s[c["deliver"]]["二阶"]
        L.append(f"| {c['cell']} | {s['F_G']['二阶']*1e3:.2f} ({s['F_G']['二阶']/g:.3f}) | {s['F_终']['二阶']*1e3:.2f} ({s['F_终']['二阶']/g:.3f}) | {s['P_终']['二阶']*1e3:.2f} | "
                 f"{g*1e3:.2f} | {s['AIM']['二阶']*1e3:.2f} | **{dv/g:.3f}** | {dv/s['AIM']['二阶']:.3f} | {s['F_终']['二阶']/g:.3f} |")
    L.append("")
    L.append("## 三、按 ε 汇总（三种子均值）")
    L.append("")
    L.append("| ε | 判停 | 判对 | 方法墙钟 s | GSD 搜索 s | 快几倍 | F_G 落盘 s | 交表 ÷GSD 二阶 | 三阶均 | 三阶top5% | 半空间 | 交表 ÷AIM 二阶 | F 终 ÷GSD 二阶 | F_G ÷GSD 二阶 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        cs = [c for c in cells if c["tag"] == t]
        n = len(cs)
        mean = lambda f: sum(f(c) for c in cs) / n  # noqa: E731
        ratio = lambda k, ax: mean(lambda c: c["scores"][c["deliver"] if k == "deliver" else k][ax] / c["scores"]["GSD"][ax])  # noqa: E731
        decs = "/".join(c["decision"] for c in cs)
        wm, wg = mean(lambda c: c["wall_method"]), mean(lambda c: c["gsd_s"] or float("nan"))
        L.append(f"| {EPS[t]} | {decs} | {sum(c['correct'] for c in cs)}/{n} | {wm:.0f} | {wg:.0f} | {wg/wm:.1f} | {mean(lambda c: c['F_G_mtime_s']):.0f} | "
                 f"**{ratio('deliver','二阶'):.3f}** | {ratio('deliver','三阶均'):.3f} | {ratio('deliver','三阶top5%'):.3f} | {ratio('deliver','半空间'):.3f} | "
                 f"{mean(lambda c: c['scores'][c['deliver']]['二阶'] / c['scores']['AIM']['二阶']):.3f} | {ratio('F_终','二阶'):.3f} | {ratio('F_G','二阶'):.3f} |")
    L.append("")
    L.append("## 四、全轴（方法交表 ÷GSD，三种子均值）")
    L.append("")
    L.append("| ε | " + " | ".join(AXES) + " |")
    L.append("|---|" + "---|" * len(AXES))
    for t in TAGS:
        cs = [c for c in cells if c["tag"] == t]
        L.append(f"| {EPS[t]} | " + " | ".join(f"{sum(c['scores'][c['deliver']][ax] / c['scores']['GSD'][ax] for c in cs) / len(cs):.3f}" for ax in AXES) + " |")
    L.append("")
    (ROOT / "logs_eps/scores/acsmob_1a.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
