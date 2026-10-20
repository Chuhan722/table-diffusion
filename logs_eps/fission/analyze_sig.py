"""第七十三步 探针提前停：分析第二趟密集记账（带配对差统计列），评估"显著回升停"规则。

读取 results/fission_dense2/<cell>_probe.csv（每 5 轮一行：轮次、秒、训练损失、模拟卷 L1_B、是否分档全过线轮、
配对差均值 diff_mean、标准误 diff_se；全过线前两列为空），对照 results/autostop/<cell>_decision.json 的离线判定
（floor / conv），并读 results/fission_dense2/<cell>_check.json 的核对结果（轨迹列与第一趟逐字同、终表行统计与离线重算同）。

规则族 R(z, M)：分档全过线之后，若连续 M 次记账都满足 diff_mean ≥ z·diff_se（起点表比全过线表显著更差），
判"地板"并在第 M 次记账处停；跑到收敛仍未触发则判"收敛"。按 30 格对照离线判定数正确率，并统计地板格的省时。

用法：cd residual-evolution && .venv/bin/python logs_eps/fission/analyze_sig.py [--plot]
输出：logs_eps/scores/fission_sig.md（表格），--plot 时另存 logs_eps/fission/fission_sig_{1A,1B}.png
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DENSE = ROOT / "results/fission_dense2"
AUTOSTOP = ROOT / "results/autostop"
OUT_MD = ROOT / "logs_eps/scores/fission_sig.md"

EPS_ORDER = ["0p1", "0p3", "1", "3", "10"]
ZS = [1.0, 1.5, 2.0, 2.5, 3.0]
MS = [1, 2, 3, 5, 10]
PICK_Z, PICK_M = 2.0, 3


@dataclass
class Curve:
    cell: str
    exam: str
    eps: str
    seed: int
    rounds: np.ndarray
    secs: np.ndarray
    l1: np.ndarray
    z: np.ndarray  # 全过线及之前为 nan
    i_g: int
    final_sec: float
    final_z: float
    ref_decision: str
    check: dict

    @property
    def l1_g(self) -> float:
        return float(self.l1[self.i_g])


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
    z = np.array([float(r["diff_mean"]) / float(r["diff_se"]) if r["diff_mean"] != "" and float(r["diff_se"]) > 0 else np.nan
                  for r in body])
    exam, eps, seed = cell.split("_")
    chk_path = DENSE / f"{cell}_check.json"
    return Curve(
        cell=cell, exam=exam, eps=eps[len("eps"):], seed=int(seed[1:]),
        rounds=np.array([int(r["round"]) for r in body]),
        secs=np.array([float(r["elapsed_s"]) for r in body]),
        l1=np.array([float(r["l1_b"]) for r in body]),
        z=z, i_g=int(np.where(flags)[0][0]),
        final_sec=float(fin[0]["elapsed_s"]),
        final_z=float(fin[0]["diff_mean"]) / float(fin[0]["diff_se"]) if fin[0]["diff_mean"] != "" else float("nan"),
        ref_decision=json.loads(dec_path.read_text())["decision"],
        check=json.loads(chk_path.read_text()) if chk_path.exists() else {},
    )


def apply_rule(c: Curve, z: float, m: int) -> tuple[str, float, int]:
    """返回 (判定, 停止秒, 停止轮)。地板：第 M 次连续显著处；收敛：最后一轮。"""
    run = 0
    for i in range(c.i_g + 1, len(c.z)):
        run = run + 1 if (not np.isnan(c.z[i]) and c.z[i] >= z) else 0
        if run >= m:
            return "floor", float(c.secs[i]), int(c.rounds[i])
    return "conv", float(c.secs[-1]), int(c.rounds[-1])


def max_run(c: Curve, z: float) -> int:
    best = run = 0
    for i in range(c.i_g + 1, len(c.z)):
        run = run + 1 if (not np.isnan(c.z[i]) and c.z[i] >= z) else 0
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
    floor_cells = [c for c in curves if c.ref_decision == "floor"]
    conv_cells = [c for c in curves if c.ref_decision == "conv"]

    lines: list[str] = []
    w = lines.append
    w("# 第七十三步 显著回升停：第二趟密集记账分析（adult）")
    w("")
    w(f"已完成 {len(curves)} 格（地板 {len(floor_cells)}、收敛 {len(conv_cells)}，按第七十二步离线判定）。"
      "每格每 5 轮记一次：起点表对模拟卷的逐题误差 e，与全过线表的 e_G 作配对差 d = e − e_G，"
      "记 z = mean(d) / (std(d, ddof=1)/√n)；z > 0 表示比全过线表更差。")
    w("")

    # ---------- 核对与概况 ----------
    w("## 一、每格核对与概况")
    w("")
    w("核对：轨迹列（轮、损失、L1_B、过线标志）与第一趟密集记账逐字同；终表行的配对差统计与离线按表重算同（误差 ≤ 1e-9）。")
    w("")
    w("| 格 | 离线判定 | 全过线轮 / 秒 | 收敛秒 | 终点 z | 全过线后 z 最高 (轮) | 全过线后 z 最低 (轮) | 轨迹同 | 统计同 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in curves:
        after = c.z[c.i_g + 1:]
        i_max = c.i_g + 1 + int(np.nanargmax(after)) if len(after) else c.i_g
        i_min = c.i_g + 1 + int(np.nanargmin(after)) if len(after) else c.i_g
        same = c.check.get("ref_same")
        stat = (abs(c.check["diff_mean_abs_err"]) <= 1e-9 * max(1.0, abs(c.check["offline_diff_mean"]))
                and abs(c.check["diff_se_reldiff"]) < 1e-9) if "diff_se_reldiff" in c.check else None
        tick = lambda v: "—" if v is None else ("是" if v else "否 ✗")
        w(f"| {c.exam} ε={fmt_eps(c.eps)} s{c.seed} | {c.ref_decision} | {c.rounds[c.i_g]} / {c.secs[c.i_g]:.0f} | {c.final_sec:.0f} "
          f"| {c.final_z:+.1f} | {np.nanmax(after):+.1f} ({c.rounds[i_max]}) | {np.nanmin(after):+.1f} ({c.rounds[i_min]}) "
          f"| {tick(same)} | {tick(stat)} |")
    w("")

    # ---------- 分离度 ----------
    w("## 二、两类格在各 z 门限下的分离度")
    w("")
    w("收敛格：全过线后 z ≥ 门限 的最长连续记账数（越小越安全，规则 M 须大于它）；地板格：M=1 时首次 z ≥ 门限 的秒数。")
    w("")
    w("| z 门限 | 收敛格最长连续显著 (最大 / 各格) | 地板格 M=1 首次显著秒 (最大 / 各格) |")
    w("|---|---|---|")
    for z in ZS:
        runs = [max_run(c, z) for c in conv_cells]
        firsts = []
        for c in floor_cells:
            dec, sec, _ = apply_rule(c, z, 1)
            firsts.append(sec if dec == "floor" else float("inf"))
        runs_s = ", ".join(str(x) for x in runs) if runs else "—"
        firsts_s = ", ".join("∞" if np.isinf(x) else f"{x:.0f}" for x in firsts) if firsts else "—"
        w(f"| {z:g} | {max(runs) if runs else '—'} / {runs_s} | "
          f"{('∞' if any(np.isinf(x) for x in firsts) else f'{max(firsts):.0f}') if firsts else '—'} / {firsts_s} |")
    w("")

    # ---------- 规则格点 ----------
    w("## 三、规则 R(z, M) 格点：判对数 与 地板格平均停止秒")
    w("")
    w("单元格格式：判对数/总数（地板格平均停止秒 ｜ 地板格原收敛秒均值）。判错含两种：地板格未触发（白跑到收敛）、收敛格误触发（交了更差的表）。")
    w("")
    w("| z \\ M | " + " | ".join(str(m) for m in MS) + " |")
    w("|---|" + "---|" * len(MS))
    mean_full = float(np.mean([c.final_sec for c in floor_cells])) if floor_cells else float("nan")
    for z in ZS:
        cells = []
        for m in MS:
            ok = 0
            floor_secs = []
            for c in curves:
                dec, sec, _ = apply_rule(c, z, m)
                ok += dec == c.ref_decision
                if c.ref_decision == "floor":
                    floor_secs.append(sec)
            mean_stop = float(np.mean(floor_secs)) if floor_secs else float("nan")
            cells.append(f"{ok}/{len(curves)} ({mean_stop:.0f} ｜ {mean_full:.0f})")
        w(f"| {z:g} | " + " | ".join(cells) + " |")
    w("")

    # ---------- 选定规则逐格 ----------
    w(f"## 四、选定规则 R(z={PICK_Z:g}, M={PICK_M}) 逐格结果（z=2 ≈ 单侧 95%，M=3 为连续性要求，均为事先约定）")
    w("")
    w("| 格 | 离线判定 | 规则判定 | 停止轮 / 秒 | 全过线秒 | 原收敛秒 | 省时 | 停止处 z | 停止处 L1_B ×全过线 |")
    w("|---|---|---|---|---|---|---|---|---|")
    saved_total = full_total = 0.0
    for c in curves:
        dec, sec, rnd = apply_rule(c, PICK_Z, PICK_M)
        i = int(np.where(c.rounds == rnd)[0][0])
        mark = "" if dec == c.ref_decision else " ✗"
        if c.ref_decision == "floor":
            saved_total += c.final_sec - sec
            full_total += c.final_sec
        zi = c.z[i]
        w(f"| {c.exam} ε={fmt_eps(c.eps)} s{c.seed} | {c.ref_decision} | {dec}{mark} | {rnd} / {sec:.0f} | {c.secs[c.i_g]:.0f} "
          f"| {c.final_sec:.0f} | {c.final_sec - sec:.0f} | {'—' if np.isnan(zi) else f'{zi:+.1f}'} | {c.l1[i] / c.l1_g:.4f} |")
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
        from matplotlib import font_manager as fm

        cjk = [f.name for f in fm.fontManager.ttflist if "Noto Sans CJK" in f.name]
        if cjk:
            plt.rcParams["font.family"] = [cjk[0], "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        for exam in ("1A", "1B"):
            sub = [c for c in curves if c.exam == exam]
            if not sub:
                continue
            fig, axes = plt.subplots(1, len(EPS_ORDER), figsize=(4 * len(EPS_ORDER), 3.6), sharey=False)
            for ax, eps in zip(axes, EPS_ORDER):
                for c in [c for c in sub if c.eps == eps]:
                    t = c.secs - c.secs[c.i_g]
                    ax.plot(t, c.z, lw=1, label=f"s{c.seed} ({c.ref_decision})")
                    dec, sec, _ = apply_rule(c, PICK_Z, PICK_M)
                    if dec == "floor":
                        ax.axvline(sec - c.secs[c.i_g], color="gray", lw=0.5, ls=":")
                ax.axhline(0.0, color="k", lw=0.6, ls="--")
                ax.axhline(PICK_Z, color="r", lw=0.6, ls=":")
                ax.set_title(f"{exam} ε={fmt_eps(eps)}")
                ax.set_xlabel("秒（自全过线起）")
                ax.set_xscale("symlog", linthresh=10)
                ax.legend(fontsize=7)
            axes[0].set_ylabel("z = 配对差均值 / 标准误")
            fig.tight_layout()
            out = ROOT / f"logs_eps/fission/fission_sig_{exam}.png"
            fig.savefig(out, dpi=110)
            print(f"已存 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
