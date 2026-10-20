"""第七十三步 探针提前停：分析密集探针曲线，评估"回升即停"规则。

读取 results/fission_dense/<cell>_probe.csv（每 5 轮一行：轮次、秒、训练损失、模拟卷 L1_B、是否分档全过线轮），
对照 results/autostop/<cell>_decision.json 的离线判定（floor / conv）。

规则族 R(δ, M)：分档全过线之后，若模拟卷误差连续 M 次记录都 ≥ 全过线时误差 ×(1+δ)，判"地板"并在第 M 次记录处停；
跑到收敛仍未触发则判"收敛"。按 30 格对照离线判定数正确率，并统计地板格的省时。

用法：cd residual-evolution && .venv/bin/python logs_eps/fission/analyze_dense.py [--plot]
输出：logs_eps/scores/fission_dense.md（表格），--plot 时另存 logs_eps/fission/fission_dense_{1A,1B}.png
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DENSE = ROOT / "results/fission_dense"
AUTOSTOP = ROOT / "results/autostop"
OUT_MD = ROOT / "logs_eps/scores/fission_dense.md"

EPS_ORDER = ["0p1", "0p3", "1", "3", "10"]
DELTAS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.01]
MS = [1, 2, 3, 5, 10]


@dataclass
class Curve:
    cell: str
    exam: str
    eps: str
    seed: int
    rounds: np.ndarray
    secs: np.ndarray
    l1: np.ndarray
    i_g: int
    final_l1: float
    final_sec: float
    ref_decision: str
    ref_l1_g: float
    ref_l1_final: float

    @property
    def l1_g(self) -> float:
        return float(self.l1[self.i_g])

    @property
    def ratio(self) -> np.ndarray:
        return self.l1 / self.l1_g


def load_curve(path: Path) -> Curve | None:
    cell = path.name[: -len("_probe.csv")]
    dec_path = AUTOSTOP / f"{cell}_decision.json"
    if not dec_path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    body = [r for r in rows if r["round"] != "final"]
    fin = [r for r in rows if r["round"] == "final"]
    if not fin:
        return None  # 还在跑
    flags = np.array([r["at_group_floor"] == "1" for r in body])
    if not flags.any():
        return None
    dec = json.loads(dec_path.read_text())
    exam, eps, seed = cell.split("_")
    return Curve(
        cell=cell,
        exam=exam,
        eps=eps[len("eps"):],
        seed=int(seed[1:]),
        rounds=np.array([int(r["round"]) for r in body]),
        secs=np.array([float(r["elapsed_s"]) for r in body]),
        l1=np.array([float(r["l1_b"]) for r in body]),
        i_g=int(np.where(flags)[0][0]),
        final_l1=float(fin[0]["l1_b"]),
        final_sec=float(fin[0]["elapsed_s"]),
        ref_decision=dec["decision"],
        ref_l1_g=float(dec["L1_B_G"]),
        ref_l1_final=float(dec["L1_B_final"]),
    )


def apply_rule(c: Curve, delta: float, m: int) -> tuple[str, float, int]:
    """返回 (判定, 停止秒, 停止轮)。地板：第 M 次连续超阈处；收敛：最后一轮。"""
    thr = c.l1_g * (1.0 + delta)
    run = 0
    for i in range(c.i_g + 1, len(c.l1)):
        run = run + 1 if c.l1[i] >= thr else 0
        if run >= m:
            return "floor", float(c.secs[i]), int(c.rounds[i])
    return "conv", float(c.secs[-1]), int(c.rounds[-1])


def max_run_above(c: Curve, delta: float) -> int:
    thr = c.l1_g * (1.0 + delta)
    best = run = 0
    for i in range(c.i_g + 1, len(c.l1)):
        run = run + 1 if c.l1[i] >= thr else 0
        best = max(best, run)
    return best


def sort_key(c: Curve):
    return (c.exam, EPS_ORDER.index(c.eps), c.seed)


def fmt_eps(e: str) -> str:
    return e.replace("p", ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    curves = [c for c in (load_curve(p) for p in sorted(DENSE.glob("*_probe.csv"))) if c is not None]
    curves.sort(key=sort_key)
    if not curves:
        raise SystemExit("没有完成的探针曲线")

    lines: list[str] = []
    w = lines.append
    w("# 第七十三步 密集探针曲线分析（adult）")
    w("")
    w(f"已完成 {len(curves)} 格。每格每 5 轮记一次模拟卷误差 L1_B（起点表）；"
      "离线判定来自第七十二步（地板停点表 与 收敛终点表 的模拟卷误差直接比较）。")
    w("")

    # ---------- 每格概况 ----------
    w("## 一、每格曲线概况")
    w("")
    w("| 格 | 离线判定 | 全过线轮 / 秒 | L1_B(全过线) | 之后最低 ×全过线 (轮) | 收敛终 ×全过线 | 收敛秒 | 之后最高 ×全过线 | 校验 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in curves:
        r = c.ratio
        after = r[c.i_g + 1:]
        i_min = c.i_g + 1 + int(after.argmin()) if len(after) else c.i_g
        chk = "一致" if (abs(c.l1_g - c.ref_l1_g) <= 1e-9 * max(1.0, c.ref_l1_g)
                      and abs(c.final_l1 - c.ref_l1_final) <= 1e-9 * max(1.0, c.ref_l1_final)) else "不一致"
        w(f"| {c.exam} ε={fmt_eps(c.eps)} s{c.seed} | {c.ref_decision} | {c.rounds[c.i_g]} / {c.secs[c.i_g]:.0f} "
          f"| {c.l1_g:.3f} | {after.min() if len(after) else 1.0:.4f} ({c.rounds[i_min]}) "
          f"| {c.final_l1 / c.l1_g:.4f} | {c.final_sec:.0f} | {after.max() if len(after) else 1.0:.4f} | {chk} |")
    w("")

    # ---------- 分离度 ----------
    w("## 二、两类格在各阈值下的分离度")
    w("")
    w("地板格：全过线后首次连续 M 次 ≥ 阈值的时刻（秒）；收敛格：全过线后 ≥ 阈值的最长连续记录数（越小越安全）。")
    w("")
    floor_cells = [c for c in curves if c.ref_decision == "floor"]
    conv_cells = [c for c in curves if c.ref_decision == "conv"]
    w("| δ | 收敛格最长连续超阈 (最大 / 各格) | 地板格 M=1 首次超阈秒 (最大 / 各格) |")
    w("|---|---|---|")
    for d in DELTAS:
        runs = [max_run_above(c, d) for c in conv_cells]
        firsts = []
        for c in floor_cells:
            dec, sec, _ = apply_rule(c, d, 1)
            firsts.append(sec if dec == "floor" else float("inf"))
        runs_s = ", ".join(str(x) for x in runs) if runs else "—"
        firsts_s = ", ".join("∞" if np.isinf(x) else f"{x:.0f}" for x in firsts) if firsts else "—"
        w(f"| {d:g} | {max(runs) if runs else '—'} / {runs_s} | "
          f"{('∞' if any(np.isinf(x) for x in firsts) else f'{max(firsts):.0f}') if firsts else '—'} / {firsts_s} |")
    w("")

    # ---------- 规则格点 ----------
    w("## 三、规则 R(δ, M) 格点：判对数 与 地板格平均停止秒")
    w("")
    w("单元格格式：判对数/总数（地板格平均停止秒 ｜ 地板格原收敛秒均值）。判错含两种：地板格未触发（白跑到收敛）、收敛格误触发（交了更差的表）。")
    w("")
    w("| δ \\ M | " + " | ".join(str(m) for m in MS) + " |")
    w("|---|" + "---|" * len(MS))
    best = None
    for d in DELTAS:
        cells = []
        for m in MS:
            ok = 0
            floor_secs = []
            for c in curves:
                dec, sec, _ = apply_rule(c, d, m)
                if dec == c.ref_decision:
                    ok += 1
                if c.ref_decision == "floor":
                    floor_secs.append(sec)
            mean_stop = float(np.mean(floor_secs)) if floor_secs else float("nan")
            mean_full = float(np.mean([c.final_sec for c in floor_cells])) if floor_cells else float("nan")
            cells.append(f"{ok}/{len(curves)} ({mean_stop:.0f} ｜ {mean_full:.0f})")
            cand = (ok, -mean_stop, -d, -m)
            if best is None or cand > best[0]:
                best = (cand, d, m)
        w(f"| {d:g} | " + " | ".join(cells) + " |")
    w("")

    # ---------- 推荐规则逐格 ----------
    _, bd, bm = best
    w(f"## 四、推荐规则 R(δ={bd:g}, M={bm}) 逐格结果")
    w("")
    w("| 格 | 离线判定 | 规则判定 | 停止轮 / 秒 | 原收敛秒 | 省时 | 停止处 ×全过线 |")
    w("|---|---|---|---|---|---|---|")
    saved_total = 0.0
    full_total = 0.0
    for c in curves:
        dec, sec, rnd = apply_rule(c, bd, bm)
        i = int(np.where(c.rounds == rnd)[0][0])
        mark = "" if dec == c.ref_decision else " ✗"
        if c.ref_decision == "floor":
            saved_total += c.final_sec - sec
            full_total += c.final_sec
        w(f"| {c.exam} ε={fmt_eps(c.eps)} s{c.seed} | {c.ref_decision} | {dec}{mark} | {rnd} / {sec:.0f} "
          f"| {c.final_sec:.0f} | {c.final_sec - sec:.0f} | {c.ratio[i]:.4f} |")
    w("")
    if full_total > 0:
        w(f"地板格合计：原收敛 {full_total:.0f} 秒 → 规则停 {full_total - saved_total:.0f} 秒，省 {saved_total / full_total:.0%}。")
        w("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n已写 {OUT_MD.relative_to(ROOT)}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for exam in ("1A", "1B"):
            sub = [c for c in curves if c.exam == exam]
            if not sub:
                continue
            fig, axes = plt.subplots(1, len(EPS_ORDER), figsize=(4 * len(EPS_ORDER), 3.6), sharey=False)
            for ax, eps in zip(axes, EPS_ORDER):
                for c in [c for c in sub if c.eps == eps]:
                    t = c.secs - c.secs[c.i_g]
                    ax.plot(t, c.ratio, lw=1, label=f"s{c.seed} ({c.ref_decision})")
                ax.axhline(1.0, color="k", lw=0.6, ls="--")
                ax.axhline(1.0 + bd, color="r", lw=0.6, ls=":")
                ax.set_title(f"{exam} ε={fmt_eps(eps)}")
                ax.set_xlabel("秒（自全过线起）")
                ax.set_xscale("symlog", linthresh=10)
                ax.legend(fontsize=7)
            axes[0].set_ylabel("L1_B / L1_B(全过线)")
            fig.tight_layout()
            out = ROOT / f"logs_eps/fission/fission_dense_{exam}.png"
            fig.savefig(out, dpi=110)
            print(f"已存 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
