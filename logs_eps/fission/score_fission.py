"""第七十一步 噪声分裂探针阅卷：学习卷 y+e 上跑出的各张表，对 模拟卷 y−e 的损失 L_B 能否替代对真表的损失 L_T 判"有谷/没谷"。

每格（1A/1B × ε）要阅的表：快照 0..12（地板_A × 2.0/1.5/1.3/1.2/1.1/1.0/0.95/0.9/0.8/0.7/0.6/0.5/0.4）、
分档全过线快照 G（= 地板停会交出的表）、终表（= 收敛停会交出的表）。每张表算：
  L_A = ½Σw(y_A−q)²（学习卷，应≈引擎 old_loss）、L_B = ½Σw(y_B−q)²（模拟卷）、L_T = ½Σw(y_真−q)²（真表）、
  L_y = ½Σw(y−q)²（原卷），以及七轴阅卷的 二阶 L1/N（scripts/score_any.py）与 1B 的 测过的对（scripts/score_pair_groups.py）。
原理：q 只依赖 y_A，y_B 的噪声与 y_A 独立 → E[L_B] = L_T + 地板_A（常数），故 L_B 与 L_T 同形。
判据（平方版）：L_B(G) < L_B(终) → 有谷 → 地板停；否则 → 收敛停。
判据（绝对值版）：每题 L1_B = mean|y_B−q|，同样比 G 与终表。L1_B 是真 L1 误差经 N(0,σ'²) 平滑后的量（E|a+Z|），
  幅度被压缩但次序保留；汇报用的二阶 L1/N 是 L1 口径，故以绝对值版为准。
与真表判据 L_T(G) ? L_T(终)、L1_T、二阶(G) ? 二阶(终)、测过的对 对照。

用法：cd residual-evolution && .venv/bin/python logs_eps/fission/score_fission.py
输出：logs_eps/scores/fission_probe.md、results/fission/fission_curves.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps" / "pgmfloor"))
from grade_pgmfloor import pair_groups, score_any  # noqa: E402

RES = ROOT / "results/fission"
OUT_MD = ROOT / "logs_eps/scores/fission_probe.md"
OUT_PNG = RES / "fission_curves.png"
MULT = [2.0, 1.5, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
REAL = pd.read_csv(ROOT / "data/adult/adult.csv")
SEED = 1


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


def half_wsq(w: np.ndarray, y: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(w * (y - q) ** 2))


def read_final(path: Path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    body = [r for r in rows[1:] if r[0] != "final"]
    fin = [r for r in rows[1:] if r[0] == "final"][0]
    return len(body), float(fin[1]), fin[7], float(body[-1][-1])


def points(name: str) -> list[tuple[str, Path, int | None, float | None, float | None]]:
    """(标签, 路径, 轮, 引擎损失, 秒)。"""
    out = []
    snaps = RES / f"{name}_snaps.csv"
    rec = {}
    if snaps.exists():
        for r in csv.DictReader(open(snaps, encoding="utf-8")):
            rec[r["index"]] = (int(r["round"]), float(r["old_loss"]), float(r["elapsed_s"]))
    for i, m in enumerate(MULT):
        p = RES / f"{name}_snap{i}.csv"
        if p.exists():
            rd, ol, el = rec.get(str(i), (None, None, None))
            out.append((f"×{m:g}", p, rd, ol, el))
    pg = RES / f"{name}_snapG.csv"
    if pg.exists():
        rd, ol, el = rec.get("G", (None, None, None))
        out.append(("G(地板停)", pg, rd, ol, el))
    fin = RES / f"{name}.csv"
    if fin.exists():
        n_rounds, loss, reason, el = read_final(RES / f"{name}_curve.csv")
        out.append((f"终(收敛停,{reason})", fin, n_rounds, loss, el))
    return out


def grade(cfg: dict) -> dict | None:
    exam, tag, src = cfg["exam"], cfg["tag"], cfg["src"]
    name = f"fis_{exam}_eps{tag}_s{SEED}"
    pts = points(name)
    if not pts or not (RES / f"{name}.csv").exists():
        return None
    qa = json.load(open(ROOT / "data" / cfg["data"] / cfg["exam_file"], encoding="utf-8"))["queries"]
    qb = json.load(open(RES / f"{src}_fisB.json", encoding="utf-8"))["queries"]
    qy = json.load(open(ROOT / "data" / src / cfg["exam_file"], encoding="utf-8"))["queries"]
    sig = np.array([q["sigma"] for q in qy], dtype=np.float64)
    w = np.ones_like(sig) if cfg["mode"] == "none" else (sig.min() / sig) ** 2
    ya = np.array([q["result"] for q in qa]); yb = np.array([q["result"] for q in qb]); yy = np.array([q["result"] for q in qy])
    yt = counts(REAL, qy)
    floor_a = cfg["floor_split"]
    seven = score_any([p for _, p, *_ in pts])
    keys = [f"k{i}" for i in range(len(pts))]
    groups = pair_groups(tag, SEED, {k: p for k, (_, p, *_) in zip(keys, pts)}) if exam == "1B" else {}
    rows = []
    for k, (lab, p, rd, ol, el) in zip(keys, pts):
        q = counts(pd.read_csv(p), qy)
        la, lb, lt, ly = half_wsq(w, ya, q), half_wsq(w, yb, q), half_wsq(w, yt, q), half_wsq(w, yy, q)
        sv = seven.get(p.stem)
        gr = groups.get(k) if groups else None
        l1b = float(np.mean(np.abs(yb - q))); l1t = float(np.mean(np.abs(yt - q)))
        rows.append(dict(label=lab, path=p, round=rd, engine=ol, sec=el, L_A=la, L_B=lb, L_T=lt, L_y=ly,
                         L1_B=l1b, L1_T=l1t, opt=(lb - la) / floor_a,
                         L_B_adj=lb - floor_a, two=(sv[1] if sv else float("nan")),
                         measured=(gr[0] if gr else (sv[1] if sv else float("nan")))))
    byl = {r["label"]: r for r in rows}
    g = next((r for r in rows if r["label"].startswith("G")), None)
    fin = rows[-1]
    dec = None
    if g is not None:
        dec = dict(
            probe="地板停" if g["L_B"] < fin["L_B"] else "收敛停",
            probe_L1="地板停" if g["L1_B"] < fin["L1_B"] else "收敛停",
            dL1B=(g["L1_B"] - fin["L1_B"]) / fin["L1_B"],
            truth_LT="地板停" if g["L_T"] < fin["L_T"] else "收敛停",
            truth_two="地板停" if g["two"] < fin["two"] else "收敛停",
            truth_measured="地板停" if g["measured"] < fin["measured"] else "收敛停",
            dLB=(g["L_B"] - fin["L_B"]) / fin["L_B"], dLT=(g["L_T"] - fin["L_T"]) / fin["L_T"],
            d_two=(g["two"] - fin["two"]) / fin["two"], d_meas=(g["measured"] - fin["measured"]) / fin["measured"],
        )
    adj = np.array([r["L_B_adj"] for r in rows]); lt = np.array([r["L_T"] for r in rows])
    corr = float(np.corrcoef(adj, lt)[0, 1]) if len(rows) > 2 else float("nan")
    argmin_b = min(rows, key=lambda r: r["L_B"])["label"]; argmin_t = min(rows, key=lambda r: r["L_T"])["label"]
    argmin_two = min(rows, key=lambda r: r["two"])["label"]
    argmin_l1b = min(rows, key=lambda r: r["L1_B"])["label"]
    return dict(exam=exam, tag=tag, name=name, floor_a=floor_a, rows=rows, decision=dec, corr=corr,
                argmin_b=argmin_b, argmin_t=argmin_t, argmin_two=argmin_two, argmin_l1b=argmin_l1b, byl=byl)


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{nd}f}"


def main() -> None:
    cfgs = json.load(open(RES / "floors.json", encoding="utf-8"))
    results = [r for r in (grade(c) for c in cfgs) if r]
    if not results:
        raise SystemExit("无结果可阅")
    lines = ["# 第七十一步 噪声分裂探针试点（adult 种子 1，1A/1B × ε=0.1/1）", "",
             "学习卷 y+e 上跑第一段到收敛；每张表对 模拟卷 y−e 的平方损失 L_B（减去地板_A 后应≈对真表的 L_T）与每题绝对误差 L1_B = mean|y_B−q|。",
             "判据：地板停表 G 的 L_B（平方版）或 L1_B（绝对值版）低于终表 → 有谷 → 地板停；否则收敛停。",
             "真表判据同形：L_T（平方）、L1_T（每题绝对）、二阶 L1/N×10⁻⁶（汇报口径）、测过的对（1B）。", ""]
    lines += ["## 判定汇总", "", "| 考卷 | ε | 探针判(L_B) | 探针判(L1_B) | 真表 L_T 判 | 二阶判 | 测过的对判 | G 对终 L_B 差 | G 对终 L1_B 差 | G 对终 L_T 差 | G 对终 二阶差 | corr(L_B−地板_A, L_T) | argmin L_B | argmin L1_B | argmin L_T | argmin 二阶 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = ok2 = 0
    for r in results:
        d = r["decision"]
        if d is None:
            lines.append(f"| {r['exam']} | {r['tag']} | 无 G 快照 | | | | | | | | | {r['corr']:.3f} | {r['argmin_b']} | {r['argmin_l1b']} | {r['argmin_t']} | {r['argmin_two']} |")
            continue
        ok += d["probe_L1"] == d["truth_two"]
        ok2 += d["probe"] == d["truth_two"]
        lines.append(f"| {r['exam']} | {r['tag']} | **{d['probe']}** | {d['probe_L1']} | {d['truth_LT']} | {d['truth_two']} | {d['truth_measured']} | "
                     f"{d['dLB']:+.1%} | {d['dL1B']:+.1%} | {d['dLT']:+.1%} | {d['d_two']:+.1%} | {r['corr']:.3f} | {r['argmin_b']} | {r['argmin_l1b']} | {r['argmin_t']} | {r['argmin_two']} |")
    n_dec = sum(1 for r in results if r['decision'])
    lines += ["", f"绝对值版探针判与二阶真判一致：{ok}/{n_dec}；平方版：{ok2}/{n_dec}。", ""]
    for r in results:
        lines += [f"## {r['exam']} ε={r['tag']}（地板_A = {r['floor_a']:,.0f}）", "",
                  "| 表 | 轮 | 秒 | 引擎损失÷地板_A | L_A÷地板_A | L_B÷地板_A | (L_B−地板_A)÷地板_A | L_T÷地板_A | L_y÷地板_A | (L_B−L_A)÷地板_A | 每题 L1_B | 每题 L1_T | 二阶 | 测过的对 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        fa = r["floor_a"]
        for x in r["rows"]:
            lines.append(f"| {x['label']} | {fmt(x['round'])} | {fmt(x['sec'], 1)} | {fmt(x['engine'] / fa if x['engine'] else None, 3)} | "
                         f"{x['L_A'] / fa:.3f} | {x['L_B'] / fa:.3f} | {x['L_B_adj'] / fa:+.3f} | {x['L_T'] / fa:.3f} | {x['L_y'] / fa:.3f} | "
                         f"{x['opt']:+.3f} | {x['L1_B']:.2f} | {x['L1_T']:.2f} | {x['two']:.1f} | {x['measured']:.1f} |")
        lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12 + len(results)]))
    print(f"写入 {OUT_MD}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, len(results), figsize=(4.2 * len(results), 7), squeeze=False)
    for j, r in enumerate(results):
        fa = r["floor_a"]
        rows = [x for x in r["rows"] if x["round"] is not None]
        rows.sort(key=lambda x: x["round"])
        rd = [x["round"] + 1 for x in rows]  # 对数轴，轮 0 画在 1
        ax = axes[0][j]
        ax.plot(rd, [x["L_B_adj"] / fa for x in rows], "o-", label="(L_B−地板_A)÷地板_A 模拟卷")
        ax.plot(rd, [x["L_T"] / fa for x in rows], "s--", label="L_T÷地板_A 真表")
        for x in rows:
            if x["label"].startswith("G"):
                ax.axvline(x["round"] + 1, color="gray", ls=":", lw=1)
        ax.set_title(f"{r['exam']} ε={r['tag']}"); ax.set_xlabel("轮+1"); ax.set_xscale("log"); ax.legend(fontsize=7)
        ax = axes[1][j]
        ax.plot(rd, [x["L1_B"] for x in rows], "o-", color="C0", label="每题 L1_B（模拟卷）")
        ax2 = ax.twinx()
        ax2.plot(rd, [x["L1_T"] for x in rows], "s--", color="C1", label="每题 L1_T（真表）")
        ax2.plot(rd, [x["two"] / 1000.0 for x in rows], "^:", color="C2", label="二阶 L1/N ×1e-3（真表）")
        for x in rows:
            if x["label"].startswith("G"):
                ax.axvline(x["round"] + 1, color="gray", ls=":", lw=1)
        ax.set_xlabel("轮+1"); ax.set_xscale("log"); ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"图 {OUT_PNG}")


if __name__ == "__main__":
    main()
