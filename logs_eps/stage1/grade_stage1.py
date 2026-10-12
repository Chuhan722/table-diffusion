"""第一阶段配对阅卷：1A 我们(ours2way) vs GSD、1B 我们(hybrid) vs 独立 PGM 收敛臂(pgmrep10k)，
1000 迭代臂(pgmrep，AIM 默认预算)与 AIM 终表(aimfull)作参考列。

每张表用 scripts/score_any.py 阅全轴，按噪声种子配对算相对差 (我们-对手)/对手，3 种子均；
主轴二阶 |均差| ≤ 2% 记平，否则记胜(我们误差更小)/负。协议见 docs/第一阶段可比性协议.md。

用法: .venv/bin/python logs_eps/stage1/grade_stage1.py [--eps 0p1,0p3,1,3,10] [--seeds 1,2,3] [--out logs_eps/scores/stage1]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLS = ["measured", "二阶", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]
PRIMARY = COLS.index("二阶")
ARMS = {
    "ours2way": "results/stage1/adult_eps{tag}_ours2way_seed{s}.csv",
    "gsd": "results/baseline2/adult_eps{tag}_gsd_seed{s}.csv",
    "hybrid": "results/stage1/adult_eps{tag}_hybrid_seed{s}.csv",
    "pgmrep": "results/stage1/adult_eps{tag}_pgmrep_seed{s}.csv",
    "pgmrep10k": "results/stage1/adult_eps{tag}_pgmrep10k_seed{s}.csv",
    # ε=0.3/3 的 AIM 终表与录像同一场跑出（远端时间戳一致），落在 results/ 根目录
    "aimfull": ("results/aimreplay/adult_eps{tag}_aimfull_seed{s}.csv", "results/adult_eps{tag}_aimfull_seed{s}.csv"),
}
EXPERIMENTS = [
    ("1A 我们 vs GSD（同一份全二阶带噪文件）", "ours2way", "gsd", []),
    ("1B 我们 vs 独立 PGM（AIM 录像同一份测量）", "hybrid", "pgmrep10k", []),
]


def arm_path(arm: str, tag: str, s: int) -> Path:
    pats = ARMS[arm] if isinstance(ARMS[arm], tuple) else (ARMS[arm],)
    cands = [ROOT / pat.format(tag=tag, s=s) for pat in pats]
    return next((c for c in cands if c.exists()), cands[0])


def score(paths: list[Path]) -> dict[str, list[float]]:
    """一次阅多张表；score_any 按 stem 报分，这里各臂文件名互不重名。"""
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return {}
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/score_any.py"), "adult",
         str(ROOT / "data/adult/adult.csv")] + [str(p) for p in paths],
        capture_output=True, text=True, check=True).stdout
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


def fmt(v: float) -> str:
    return f"{v:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="logs_eps/scores/stage1", help="markdown 输出前缀（不含后缀）")
    args = ap.parse_args()
    tags = args.eps.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    lines_all = []
    for title, ours, rival, refs in EXPERIMENTS:
        lines = [f"## {title}", "",
                 "只比这两臂，其余全同（同一份带噪文件、同噪声种子配对）；**每臂都跑到收敛**："
                 "我们平台自停（未撞 30000 轮上限），PGM 固定 10000 轮（5000 轮已收敛，留一倍余量），GSD 原生早停。",
                 "单位 ×10⁻⁶（L1/N），3 种子均；Δ = (我们−对手)/对手，主轴二阶 |Δ| ≤ 2% 记平。", "",
                 "| ε | 臂 | n | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 3) + "|"]
        verdicts = []
        for tag in tags:
            arms = [ours, rival] + refs
            paths = {a: {s: arm_path(a, tag, s) for s in seeds} for a in arms}
            sc = score([paths[a][s] for a in arms for s in seeds])
            per = {a: {s: sc.get(paths[a][s].stem) for s in seeds} for a in arms}
            for a in arms:
                rows = [per[a][s] for s in seeds if per[a][s] is not None]
                if rows:
                    lines.append(f"| {tag.replace('p', '.')} | {a} | {len(rows)} | "
                                 + " | ".join(fmt(v) for v in np.mean(rows, axis=0)) + " |")
            paired = [s for s in seeds if per[ours][s] is not None and per[rival][s] is not None]
            if paired:
                d = np.array([(np.array(per[ours][s]) - np.array(per[rival][s])) / np.array(per[rival][s])
                              for s in paired])
                m = d.mean(axis=0)
                primary = m[PRIMARY]
                verdict = "平" if abs(primary) <= 0.02 else ("胜" if primary < 0 else "负")
                lines.append(f"| {tag.replace('p', '.')} | Δ 配对均({len(paired)}种子) | | "
                             + " | ".join(f"{v:+.1%}" for v in m) + " |")
                verdicts.append((tag, primary, verdict, len(paired),
                                 " ".join(f"s{s}:{d[i][PRIMARY]:+.1%}" for i, s in enumerate(paired))))
            else:
                verdicts.append((tag, float("nan"), "缺表", 0, ""))
        lines += ["", "| ε | 主轴二阶 Δ | 判定 | 配对种子 | 逐种子 |", "|---|---|---|---|---|"]
        for tag, primary, verdict, n, detail in verdicts:
            lines.append(f"| {tag.replace('p', '.')} | {primary:+.1%} | {verdict} | {n} | {detail} |"
                         if n else f"| {tag.replace('p', '.')} | — | {verdict} | 0 | |")
        lines.append("")
        lines_all += lines
    text = "\n".join(lines_all)
    print(text)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
