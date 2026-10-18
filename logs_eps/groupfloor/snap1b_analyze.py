"""第六十八步 对照场阅卷：1B 第一段跑到收敛，沿地板倍数 / 分档过线点取的快照 对真表 阅分，找谷底、看分档停点落在哪。

每格（ε × 种子）的点：×1.0（= 第六十七步地板停表，md5 须同）、×0.9…×0.4、分档过线点 G、收敛终表；
参照：旧收敛第一段（results/stage1/*_hybrid_seed*.csv，10-15 版）、PGM 原生（results/stage1/*_pgmrep10k_seed*.csv）。
每张表阅：七轴（scripts/score_any.py，×10⁻⁶）、分组（scripts/score_pair_groups.py：测过的对 / 没测的对 / 一阶）、
按 σ 档的 对带噪÷档地板 与 对真值÷档地板（档地板 = ½Σ_{i∈g} w σ²），并给出"最慢档"（×1.0 处对带噪最高的档）。
输出 logs_eps/scores/groupfloor_snap1b.md 与图 logs_eps/groupfloor/adult_1b_谷底_<日期>.png。只读已有表，不跑实验。

用法: .venv/bin/python logs_eps/groupfloor/snap1b_analyze.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps" / "pgmfloor"))
from grade_pgmfloor import AXES, GROUPS, pair_groups, score_any  # noqa: E402

SNAP = ROOT / "results/groupfloor/snap1b"
OUT = ROOT / "logs_eps/scores/groupfloor_snap1b.md"
MULT = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
REAL = pd.read_csv(ROOT / "data/adult/adult.csv")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def counts(df: pd.DataFrame, qs: list) -> np.ndarray:
    cache: dict = {}
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        attrs = tuple(c["attribute"] for c in q["conditions"])
        vals = tuple(int(c["value"]) for c in q["conditions"])
        if attrs not in cache:
            g = df.groupby(list(attrs)).size()
            cache[attrs] = {(k if isinstance(k, tuple) else (k,)): v for k, v in g.items()}
        out[i] = cache[attrs].get(vals, 0)
    return out


def load_exam(tag: str, seed: int) -> list:
    path = glob.glob(str(ROOT / f"data/adult_aimrep{tag}_s{seed}/measured_*query.json"))[0]
    return json.load(open(path))["queries"]


def read_curve_final(path: Path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    body = [r for r in rows[1:] if r[0] != "final"]
    fin = [r for r in rows[1:] if r[0] == "final"][0]
    return len(body), float(fin[1]), fin[7]


def config_points(tag: str, seed: int, floor: float):
    """该格全部要阅的表：(标签, 路径, 轮, 损失÷地板)。"""
    name = f"adult_eps{tag}_snap_seed{seed}"
    final = SNAP / f"{name}.csv"
    if not final.exists():
        return None
    snaps = {}
    with open(SNAP / f"{name}_snaps.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            snaps[r["index"]] = (int(r["round"]), float(r["old_loss"]))
    pts = []
    for i, m in enumerate(MULT):
        p = SNAP / f"{name}_snap{i}.csv"
        if p.exists() and str(i) in snaps:
            pts.append((f"×{m:.1f}", p, snaps[str(i)][0], snaps[str(i)][1] / floor))
    pG = SNAP / f"{name}_snapG.csv"
    if pG.exists() and "G" in snaps:
        pts.append(("分档G", pG, snaps["G"][0], snaps["G"][1] / floor))
    n_rounds, fin_loss, reason = read_curve_final(SNAP / f"{name}_curve.csv")
    pts.append((f"收敛({reason})", final, n_rounds, fin_loss / floor))
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    tags = args.eps.split(",")

    lines = ["# 1B 第一段谷底：沿地板倍数与分档过线点的快照 对真表（第六十八步对照场）", "",
             "第六十七步整体方法第一段命令去掉地板停跑到收敛，快照纯记账（×1.0 快照与第六十七步地板停表 md5 逐字节同为核对项）。"
             "误差为与真表的 L1/N（×10⁻⁶）；\"最慢档\" = ×1.0 处对带噪÷档地板最高的 σ 档；对真值÷档地板 <1 表示比噪声水平更接近真表。"
             "参照：旧收敛第一段 = 10-15 版第一段（无清洗、跑到收敛）；PGM 原生 = 同录像跑满 10000 轮。", ""]
    per_cfg = {}  # (tag, seed) -> dict(label -> dict(metrics))
    checks = []
    for tag in tags:
        for seed in seeds:
            qs = load_exam(tag, seed)
            y = np.array([q["result"] for q in qs])
            sig = np.array([q["sigma"] for q in qs])
            w = (sig.min() / sig) ** 2
            floor = 0.5 * float(np.sum(w * sig**2))
            pts = config_points(tag, seed, floor)
            if pts is None:
                continue
            ref_old = ROOT / f"results/stage1/adult_eps{tag}_hybrid_seed{seed}.csv"
            ref_pgm = ROOT / f"results/stage1/adult_eps{tag}_pgmrep10k_seed{seed}.csv"
            pts_all = pts + [("旧收敛第一段", ref_old, None, None), ("PGM原生", ref_pgm, None, None)]
            # 核对：×1.0 快照 = 第六十七步第一段表
            ref67 = ROOT / f"results/whole1b/adult_eps{tag}_ourswhole_seed{seed}_stage1.csv"
            same = md5(pts[0][1]) == md5(ref67) if pts[0][0] == "×1.0" else None
            checks.append((tag, seed, same))
            tr = counts(REAL, qs)
            keys = np.round(sig, 1)
            uniq = sorted(set(keys))
            Fg = {u: 0.5 * float(np.sum((w * sig**2)[keys == u])) for u in uniq}
            sc = score_any([p for _, p, _, _ in pts_all])
            pg = pair_groups(tag, seed, {lab: p for lab, p, _, _ in pts_all})
            rows = {}
            for lab, p, rnd, ratio in pts_all:
                q_tab = counts(pd.read_csv(p), qs)
                noisy = {u: 0.5 * float(np.sum((w * (y - q_tab) ** 2)[keys == u])) / Fg[u] for u in uniq}
                truth = {u: 0.5 * float(np.sum((w * (tr - q_tab) ** 2)[keys == u])) / Fg[u] for u in uniq}
                rows[lab] = dict(
                    round=rnd, ratio=ratio if ratio is not None else 0.5 * float(np.sum(w * (y - q_tab) ** 2)) / floor,
                    truth_total=0.5 * float(np.sum(w * (tr - q_tab) ** 2)) / floor,
                    axes=sc[p.stem], groups=pg[lab], noisy=noisy, truth=truth,
                )
            slow = max(uniq, key=lambda u: rows["×1.0"]["noisy"][u]) if "×1.0" in rows else uniq[-1]
            per_cfg[(tag, seed)] = dict(rows=rows, slow=slow, floor=floor, uniq=uniq, Fg=Fg)
            lines += [f"## ε={tag} 种子 {seed}：题 {len(qs)}，σ 档 {len(uniq)}，最慢档 σ={slow:.1f}（地板份额 {Fg[slow] / floor:.1%}），"
                      f"×1.0 快照 md5 {'同' if same else '不同'} 第六十七步地板停表", ""]
            lines += ["| 点 | 轮 | 损失÷地板 | 对真值÷地板 | 最慢档 对带噪÷档地板 | 最慢档 对真值÷档地板 | measured | 二阶 | 三阶均 | 三阶top5% | 等值 | 半空间 | 测过的对 | 没测的对 | 一阶 |",
                      "|---" * 15 + "|"]
            for lab, r in rows.items():
                ax = r["axes"]; g = r["groups"]
                lines.append(
                    f"| {lab} | {r['round'] if r['round'] is not None else '-'} | {r['ratio']:.3f} | {r['truth_total']:.3f} | "
                    f"{r['noisy'][slow]:.2f} | {r['truth'][slow]:.2f} | {ax[0]:,.0f} | {ax[1]:,.0f} | {ax[2]:,.0f} | {ax[3]:,.0f} | "
                    f"{ax[5]:,.0f} | {ax[6]:,.0f} | {g[0]:,.0f} | {g[1]:,.0f} | {g[2]:,.0f} |")
            lines.append("")
            # 各档对真值明细（×1.0 / G / 收敛）
            lines += ["各 σ 档 对真值÷档地板（对带噪÷档地板）：", "",
                      "| σ | 题数 | 地板份额 | ×1.0 | 分档G | 收敛 | 旧收敛 |", "|---|---|---|---|---|---|---|"]
            conv_lab = [k for k in rows if k.startswith("收敛")][0]
            for u in uniq:
                cells = []
                for lab in ("×1.0", "分档G", conv_lab, "旧收敛第一段"):
                    if lab in rows:
                        cells.append(f"{rows[lab]['truth'][u]:.2f}（{rows[lab]['noisy'][u]:.2f}）")
                    else:
                        cells.append("-")
                lines.append(f"| {u:.1f} | {int(np.sum(keys == u))} | {Fg[u] / floor:.1%} | " + " | ".join(cells) + " |")
            lines.append("")

    # 汇总
    lines += ["## 汇总（三种子均；比值 = 该点 ÷ 旧收敛第一段，<1 比旧收敛好）", ""]
    labels_order = [f"×{m:.1f}" for m in MULT] + ["分档G", "收敛"]
    for metric, idx, src in (("二阶", 1, "axes"), ("测过的对", 0, "groups"), ("没测的对", 1, "groups"), ("三阶均", 2, "axes")):
        lines += [f"### {metric}", "", "| ε | " + " | ".join(labels_order) + " | PGM原生 | 旧收敛绝对值 |", "|---" * (len(labels_order) + 3) + "|"]
        for tag in tags:
            cfgs = [per_cfg[(tag, s)] for s in seeds if (tag, s) in per_cfg]
            if not cfgs:
                continue
            cells = []
            for lab in labels_order + ["PGM原生"]:
                vals = []
                for c in cfgs:
                    key = lab if lab != "收敛" else [k for k in c["rows"] if k.startswith("收敛")][0]
                    if key in c["rows"]:
                        vals.append(c["rows"][key][src][idx] / c["rows"]["旧收敛第一段"][src][idx])
                cells.append(f"{np.mean(vals):.3f}" if len(vals) == len(cfgs) else (f"{np.mean(vals):.3f}*" if vals else "-"))
            base = np.mean([c["rows"]["旧收敛第一段"][src][idx] for c in cfgs])
            lines.append(f"| {tag} | " + " | ".join(cells) + f" | {base:,.0f} |")
        lines += ["", "（* = 部分种子缺该点，只平均有的）", ""]
    # 谷底与分档点
    lines += ["### 每格谷底（按二阶 / 测过的对）与分档过线点", "",
              "| ε | 种子 | 二阶最优点 | 分档G 二阶÷最优 | 测过的对最优点 | 分档G 测过÷最优 | 分档G 损失÷地板 | 分档G 轮 / 收敛轮 | 最慢档 G 对真值 / 收敛 对真值 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (tag, seed), c in per_cfg.items():
        rows = c["rows"]
        cand = {k: v for k, v in rows.items() if k not in ("旧收敛第一段", "PGM原生")}
        best2 = min(cand, key=lambda k: cand[k]["axes"][1])
        bestm = min(cand, key=lambda k: cand[k]["groups"][0])
        conv = [k for k in rows if k.startswith("收敛")][0]
        g = rows.get("分档G")
        if g is None:
            lines.append(f"| {tag} | {seed} | {best2} | - | {bestm} | - | - | - / {rows[conv]['round']} | - |")
            continue
        lines.append(
            f"| {tag} | {seed} | {best2} | {g['axes'][1] / cand[best2]['axes'][1]:.3f} | {bestm} | "
            f"{g['groups'][0] / cand[bestm]['groups'][0]:.3f} | {g['ratio']:.3f} | {g['round']} / {rows[conv]['round']} | "
            f"{g['truth'][c['slow']]:.2f} / {rows[conv]['truth'][c['slow']]:.2f} |")
    lines += ["", "核对：×1.0 快照 md5 与第六十七步地板停表 " + "，".join(f"ε={t} s{s} {'同' if ok else '不同'}" for t, s, ok in checks), ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-40:]))
    print("->", OUT)

    if args.no_plot or not per_cfg:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import date

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, len(tags), figsize=(4.2 * len(tags), 7.5), squeeze=False)
    styles = {1: "-", 2: "--", 3: ":"}
    for j, tag in enumerate(tags):
        for row_i, (metric, idx, src) in enumerate((("二阶", 1, "axes"), ("测过的对", 0, "groups"))):
            ax = axes[row_i][j]
            for s in seeds:
                c = per_cfg.get((tag, s))
                if not c:
                    continue
                rows = c["rows"]
                base = rows["旧收敛第一段"][src][idx]
                pts = [(v["ratio"], v[src][idx] / base) for k, v in rows.items() if k.startswith("×") or k.startswith("收敛")]
                pts.sort(key=lambda t: -t[0])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], styles.get(s, "-"), marker="o", ms=3, color="C0", label=f"s{s}" if row_i == 0 else None)
                if "分档G" in rows:
                    ax.plot([rows["分档G"]["ratio"]], [rows["分档G"][src][idx] / base], marker="*", ms=12, color="C3", ls="none")
                ax.axhline(rows["PGM原生"][src][idx] / base, color="C2", lw=0.8, ls="-." if s == 1 else ":")
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xscale("log"); ax.invert_xaxis()
            ax.set_title(f"ε={tag.replace('p', '.')} {metric}")
            ax.set_xlabel("第一段损失 ÷ 地板（向右越深）")
            if j == 0:
                ax.set_ylabel("÷ 旧收敛第一段（1 = 持平）")
            if row_i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("adult 1B 第一段：沿地板倍数的快照 vs 真表（★ 分档过线点，绿线 PGM 原生，黑线 旧收敛）")
    fig.tight_layout()
    png = ROOT / f"logs_eps/groupfloor/adult_1b_谷底_{date.today().isoformat()}.png"
    fig.savefig(png, dpi=130)
    print("->", png)


if __name__ == "__main__":
    main()
