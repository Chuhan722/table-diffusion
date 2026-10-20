"""第七十三步 补充：用"随观察次数增长的门槛"（任意时刻有效的 Bonferroni 序贯检验）离线重评显著回升停。

动机：现行规则 R(z=2, M=3) 的 z ≥ 2 只对"单次观察"是单侧 95% 检验；探针每 5 轮看一次、一格看几百到两千多次，
反复观察下"连续 3 次"没有正式的总体误报率保证（只有经验余量）。严格做法：把总误报额度 α 分摊到每一次观察，
第 k 次观察（自分档全过线后起数）用 α_k = α·6/(π²k²)（Σα_k = α），门槛 c_k = Φ⁻¹(1 − α_k)，任一次 z_k ≥ c_k 即停（M=1）。
由联合界 P(∃k: z_k ≥ c_k) ≤ Σ α_k = α，与观察之间的相关性无关。

数据与现行规则同：results/fission_dense2/<cell>_probe.csv；对照 results/autostop/<cell>_decision.json。
用法：cd residual-evolution && .venv/bin/python logs_eps/fission/analyze_strict.py
输出：logs_eps/scores/fission_strict.md
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sig import DENSE, ROOT, apply_rule, fmt_eps, load_curve, sort_key  # noqa: E402

OUT_MD = ROOT / "logs_eps/scores/fission_strict.md"
ALPHA = 0.05
PICK_Z, PICK_M = 2.0, 3
_ND = NormalDist()


def threshold(k: int, alpha: float = ALPHA) -> float:
    return _ND.inv_cdf(1.0 - alpha * 6.0 / (math.pi ** 2 * k * k))


def apply_strict(c, alpha: float = ALPHA) -> tuple[str, float, int, int]:
    """返回 (判定, 停止秒, 停止轮, 触发时的观察序号 k；未触发为 0)。"""
    k = 0
    for i in range(c.i_g + 1, len(c.z)):
        k += 1
        if not np.isnan(c.z[i]) and c.z[i] >= threshold(k, alpha):
            return "floor", float(c.secs[i]), int(c.rounds[i]), k
    return "conv", float(c.secs[-1]), int(c.rounds[-1]), 0


def main() -> None:
    curves = [c for c in (load_curve(p) for p in sorted(DENSE.glob("*_probe.csv"))) if c is not None]
    curves.sort(key=sort_key)
    if not curves:
        raise SystemExit("没有完成的探针曲线")

    lines: list[str] = []
    w = lines.append
    w("# 第七十三步 补充：随观察次数增长的门槛（任意时刻有效）离线重评")
    w("")
    w(f"规则：第 k 次观察门槛 c_k = Φ⁻¹(1 − α·6/(π²k²))，α = {ALPHA:g}，任一次 z_k ≥ c_k 即判地板停；"
      "跑到收敛未触发判收敛停。门槛示例：" + "，".join(f"k={k} → {threshold(k):.2f}" for k in (1, 5, 10, 20, 50, 100, 500, 2000)) + "。")
    w("")
    w("| 格 | 离线判定 | 全过线后观察次数 | 全过线后最高 z | 现行 R(2,3)：判定 / 停止秒 | 严格：判定 / 停止秒 / 触发 k / 当时门槛 | 收敛秒 |")
    w("|---|---|---|---|---|---|---|")
    ok_cur = ok_str = 0
    for c in curves:
        n_looks = len(c.z) - c.i_g - 1
        zmax = float(np.nanmax(c.z[c.i_g + 1:])) if n_looks > 0 else float("nan")
        dec_c, sec_c, _ = apply_rule(c, PICK_Z, PICK_M)
        dec_s, sec_s, _, k_s = apply_strict(c)
        ok_cur += dec_c == c.ref_decision
        ok_str += dec_s == c.ref_decision
        mark_c = "" if dec_c == c.ref_decision else " ✗"
        mark_s = "" if dec_s == c.ref_decision else " ✗"
        strict_txt = (f"{dec_s}{mark_s} / {sec_s:.0f} / {k_s} / {threshold(k_s):.2f}" if k_s
                      else f"{dec_s}{mark_s} / {sec_s:.0f} / — / —")
        w(f"| {c.exam} ε={fmt_eps(c.eps)} s{c.seed} | {c.ref_decision} | {n_looks} | {zmax:+.1f} "
          f"| {dec_c}{mark_c} / {sec_c:.0f} | {strict_txt} | {c.final_sec:.0f} |")
    w("")
    w(f"判对数：现行 {ok_cur}/{len(curves)}，严格 {ok_str}/{len(curves)}。严格规则未触发的地板格只能跑到收敛；按现行接法（探针跑到收敛即判收敛停）"
      "这些格会交出收敛表、判错。所以若改用严格规则，收敛后的兜底必须换回第七十二步的终点比较（G 表与终表对模拟卷谁更近，离线 30/30），"
      "这样这些格交表仍对，只是一秒不省。")
    w("")
    floor = [c for c in curves if c.ref_decision == "floor"]
    if floor:
        w("地板格按 ε 汇总（三种子，秒；“退回”= 严格规则未触发、跑到收敛）：")
        w("")
        w("| ε | 现行停止秒 | 严格停止秒 | 收敛秒 |")
        w("|---|---|---|---|")
        for eps in sorted({c.eps for c in floor}, key=lambda e: float(e.replace("p", "."))):
            sub = [c for c in floor if c.eps == eps]
            cur = "/".join(f"{apply_rule(c, PICK_Z, PICK_M)[1]:.0f}" for c in sub)
            strict = "/".join((f"{apply_strict(c)[1]:.0f}" if apply_strict(c)[3] else "退回") for c in sub)
            full = "/".join(f"{c.final_sec:.0f}" for c in sub)
            w(f"| {fmt_eps(eps)} | {cur} | {strict} | {full} |")
        w("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n已写 {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
