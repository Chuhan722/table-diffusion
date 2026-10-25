"""离线回放：换一种"显著回升停"的触发规则，各格几时触发、会不会误触发（只读已有探针记录，不跑任何实验）。

数据：adult 第七十三步 sig 跑 30 格（results/autostop_sig/P_<格>_probe.csv；1A ε≤3 为地板停真值，其余收敛停）
      acsmob 第七十四步 15 格（results/autostop_acsmob/P_<格>_probe.csv；ε≤1 地板停，ε≥3 收敛停）。
  地板停格的记录止于 z=2×3 触发处（更早触发的规则在记录内就能看到）；收敛停格记录到收敛（可查全程有无误触发）。
规则：
  R(z,k)  连续 k 次记账 diff_mean ≥ z·diff_se（现行 R(2,3)）
  C(k)    连续 k 次记账 diff_mean > 0 且 L1_B 比上一次记账高（"爬过过线水平并在升"）
  U(m)    连续 m 次记账 L1_B 比上一次高（纯拐头，不看水平）
用法：cd residual-evolution && python3 logs_eps/acsmob/replay_rule.py  → logs_eps/scores/fission_rule_replay.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = [("adult", ROOT / "results/autostop_sig", ["1A", "1B"]), ("acsmob", ROOT / "results/autostop_acsmob", ["1A"])]
TAGS = ["0p1", "0p3", "1", "3", "10"]
RULES = ["R(2,3)", "R(1.5,3)", "R(1,3)", "R(1,1)", "R(0.5,3)", "C(3)", "C(5)", "U(3)", "U(5)", "U(10)"]


def load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["round"] == "final":
            continue
        rows.append(dict(round=int(r["round"]), t=float(r["elapsed_s"]), l1=float(r["l1_b"]), g=r["at_group_floor"] == "1",
                         dm=float(r["diff_mean"]) if r.get("diff_mean") else None, se=float(r["diff_se"]) if r.get("diff_se") else None))
    return rows


def after_g(rows: list[dict]) -> list[dict]:
    i = next((k for k, r in enumerate(rows) if r["g"]), None)
    return rows[i:] if i is not None else []


def trigger(rows: list[dict], rule: str) -> dict | None:
    kind, arg = rule[0], rule[2:-1]
    seq = after_g(rows)
    if not seq:
        return None
    run = 0
    for i, r in enumerate(seq):
        if kind == "R":
            z, k = map(float, arg.split(","))
            hit = r["dm"] is not None and r["se"] and r["dm"] >= z * r["se"]
        elif kind == "C":
            k = float(arg)
            hit = i > 0 and r["dm"] is not None and r["dm"] > 0 and r["l1"] > seq[i - 1]["l1"]
        else:
            k = float(arg)
            hit = i > 0 and r["l1"] > seq[i - 1]["l1"]
        run = run + 1 if hit else 0
        if run >= k:
            return r
    return None


def main() -> None:
    cells = []
    for ds, out, exams in SETS:
        for exam in exams:
            for tag in TAGS:
                for seed in (1, 2, 3):
                    cell = f"{exam}_eps{tag}_s{seed}"
                    rows = load(out / f"P_{cell}_probe.csv")
                    dec = json.load(open(out / f"{cell}_decision.json", encoding="utf-8"))
                    seq = after_g(rows)
                    zs = [r["dm"] / r["se"] for r in seq if r["dm"] is not None and r["se"]]
                    cells.append(dict(ds=ds, cell=cell, truth=dec["decision"], G=seq[0], last=rows[-1], zmax=max(zs) if zs else None,
                                      zmin=min(zs) if zs else None, zfin=zs[-1] if zs else None, n_after=len(seq),
                                      trig={ru: trigger(rows, ru) for ru in RULES}))
    L = ["# 显著回升停 触发规则离线回放（adult sig 30 格 + acsmob 15 格；只读记录）", ""]
    L.append("规则：R(z,k) = 连续 k 次 diff_mean ≥ z·diff_se（现行 R(2,3)）；C(k) = 连续 k 次 diff_mean > 0 且 L1_B 较上次升；U(m) = 连续 m 次 L1_B 较上次升（不看水平）。"
             "地板停格记录止于现行规则触发处，收敛停格记录到收敛。表中数字 = 触发时探针循环内秒数（括号为轮），— = 全程不触发。地板停格越早越好；收敛停格任何触发都是误判。")
    L.append("")
    L.append("## 一、收敛停格（真值 conv）：过线后 z 的最高点，以及各规则是否误触发")
    L.append("")
    L.append("| 格 | z 最高 | z 终 | 过线后记账次数 | " + " | ".join(RULES) + " |")
    L.append("|---|---|---|---|" + "---|" * len(RULES))
    bad = {ru: 0 for ru in RULES}
    for c in cells:
        if c["truth"] != "conv":
            continue
        cols = []
        for ru in RULES:
            t = c["trig"][ru]
            if t:
                bad[ru] += 1
            cols.append(f"误 {t['t']:.0f} ({t['round']})" if t else "—")
        L.append(f"| {c['ds']} {c['cell']} | {c['zmax']:+.2f} | {c['zfin']:+.2f} | {c['n_after']} | " + " | ".join(cols) + " |")
    nconv = sum(c["truth"] == "conv" for c in cells)
    L.append(f"| **误触发格数 / {nconv}** | | | | " + " | ".join(f"**{bad[ru]}**" for ru in RULES) + " |")
    L.append("")
    L.append("## 二、地板停格（真值 floor）：各规则触发时刻（探针循环内秒，括号为轮）")
    L.append("")
    L.append("| 格 | 过线 | " + " | ".join(RULES) + " |")
    L.append("|---|---|" + "---|" * len(RULES))
    for c in cells:
        if c["truth"] != "floor":
            continue
        cols = []
        for ru in RULES:
            t = c["trig"][ru]
            cols.append(f"{t['t']:.0f} ({t['round']})" if t else "—")
        L.append(f"| {c['ds']} {c['cell']} | {c['G']['t']:.0f} ({c['G']['round']}) | " + " | ".join(cols) + " |")
    L.append("")
    L.append("## 三、地板停格按数据集×ε 的触发秒均值（三种子；现行 R(2,3) 为基准）")
    L.append("")
    L.append("| 数据集 ε | 过线秒 | " + " | ".join(RULES) + " |")
    L.append("|---|---|" + "---|" * len(RULES))
    for ds, _, _ in SETS:
        for tag in TAGS:
            cs = [c for c in cells if c["ds"] == ds and c["truth"] == "floor" and f"_eps{tag}_" in c["cell"]]
            if not cs:
                continue
            cols = []
            for ru in RULES:
                ts = [c["trig"][ru]["t"] for c in cs if c["trig"][ru]]
                cols.append(f"{sum(ts) / len(ts):.0f}" + ("" if len(ts) == len(cs) else f" ({len(ts)}/{len(cs)} 触发)") if ts else "—")
            L.append(f"| {ds} {tag} | {sum(c['G']['t'] for c in cs) / len(cs):.0f} | " + " | ".join(cols) + " |")
    L.append("")
    (ROOT / "logs_eps/scores/fission_rule_replay.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
