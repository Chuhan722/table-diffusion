"""第二阶段首战 A 阅卷：测量后处理（平分清洗）消融。

臂：hybridp1 = 第一阶段 hybrid + --oneway-pool 2；hybridp12 = 再加 --pair-pool 2；hybridp1u = --oneway-pool-scope uncovered；
对照：pgmrep10k（独立 PGM 收敛臂）、hybrid（第一阶段同引擎无清洗）；
归属对照 pgmrep10kp1u = pgmrep10k 命令 + 同样的 --oneway-pool 2 --oneway-pool-scope uncovered（run_pgm_replay.py 同名开关，清洗向量与我们臂逐位同）。
每场只比两臂，其余全同；主轴二阶 |Δ| ≤ 2% 记平（同第一阶段协议）。
附一阶覆盖表：仅一阶测量字段上各臂终表的一阶误差，直接看病灶有没有治好。

用法: .venv/bin/python logs_eps/stage2/grade_stage2.py [--eps 0p1,0p3,1] [--seeds 1,2,3] [--out logs_eps/scores/stage2a]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps/stage1"))
import grade_stage1 as g1  # noqa: E402

g1.ARMS.update({
    "hybridp1": "results/stage2/adult_eps{tag}_hybridp1_seed{s}.csv",
    "hybridp12": "results/stage2/adult_eps{tag}_hybridp12_seed{s}.csv",
    "hybridp1u": "results/stage2/adult_eps{tag}_hybridp1u_seed{s}.csv",
    "pgmrep10kp1u": "results/stage2/adult_eps{tag}_pgmrep10kp1u_seed{s}.csv",
})
EXPERIMENTS = [
    ("2A-1 一阶清洗 vs 独立 PGM（AIM 录像同一份测量；hybrid 列为第一阶段参考）", "hybridp1", "pgmrep10k", ["hybrid"]),
    ("2A-2 一阶清洗 vs 第一阶段 hybrid（同引擎同考卷，只差一个开关）", "hybridp1", "hybrid", []),
    ("2A-3 一阶+二阶清洗 vs 仅一阶清洗（只差二阶开关）", "hybridp12", "hybridp1", []),
    ("2A-4 一阶+二阶清洗 vs 独立 PGM", "hybridp12", "pgmrep10k", []),
    ("2A-5 仅未覆盖字段清洗 vs 独立 PGM（ε≥1 该臂与 hybrid 逐位同，见 1B）", "hybridp1u", "pgmrep10k", []),
    ("2A-6 仅未覆盖字段清洗 vs 全字段清洗（只差范围开关）", "hybridp1u", "hybridp1", []),
    ("2A-7 仅未覆盖字段清洗 vs 第一阶段 hybrid", "hybridp1u", "hybrid", []),
    ("2A-8 归属对照：PGM 也吃清洗后一阶 vs PGM 吃原始（同录像同种子同 10000 轮，只差清洗）", "pgmrep10kp1u", "pgmrep10k", []),
    ("2A-9 归属对照：我们(仅未覆盖字段清洗) vs PGM(同样清洗)（两臂吃完全相同的清洗后测量）", "hybridp1u", "pgmrep10kp1u", []),
]
INTRO = ("只比这两臂，其余全同（同一份 AIM 录像考卷、同噪声种子配对）；我们臂命令 = 第一阶段 hybrid 逐字 + 清洗开关，"
         "PGM 臂为第一阶段 10000 轮收敛终表不重跑。")
COVER_ARMS = ("hybrid", "hybridp1", "hybridp12", "hybridp1u", "pgmrep10k", "pgmrep10kp1u")


def oneway_cover(tags: list[str], seeds: list[int]) -> str:
    real = pd.read_csv(ROOT / "data/adult/adult.csv")
    cols = list(real.columns)
    n = len(real)
    sizes = {c: int(real[c].max()) + 1 for c in cols}
    ro = {c: np.bincount(real[c], minlength=sizes[c]).astype(float) for c in cols}
    lines = ["## 一阶覆盖：各臂终表一阶误差按字段是否被测过的对覆盖分组（×10⁻⁶ L1/N，组内字段均值，种子均）", "",
             "| ε | 字段组 | 字段数 | n | " + " | ".join(COVER_ARMS) + " | p1−PGM | p1−hybrid |",
             "|---|---|---|---|" + "---|" * len(COVER_ARMS) + "---|---|"]
    for tag in tags:
        acc: dict[str, list] = {}
        for s in seeds:
            paths = {a: g1.arm_path(a, tag, s) for a in COVER_ARMS}
            if not all(paths[a].exists() for a in ("hybrid", "hybridp1", "pgmrep10k")):
                continue
            dump = json.loads((ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{s}_measurements.json").read_text())
            covered: set[str] = set()
            for m in dump["measurements"]:
                if len(m["clique"]) >= 2:
                    covered.update(m["clique"])
            syn = {a: pd.read_csv(p) if p.exists() else None for a, p in paths.items()}
            for g, fs in {"仅一阶测量": [c for c in cols if c not in covered], "被对覆盖": [c for c in cols if c in covered]}.items():
                if not fs:
                    continue
                r = [[np.abs(np.bincount(syn[a][c], minlength=sizes[c]) - ro[c]).sum() / n * 1e6 if syn[a] is not None else np.nan
                      for a in COVER_ARMS] + [len(fs)] for c in fs]
                acc.setdefault(g, []).append(np.mean(r, axis=0))
        for g in ("仅一阶测量", "被对覆盖"):
            if g not in acc:
                lines.append(f"| {tag.replace('p', '.')} | {g} | 0 | 0 |" + " — |" * (len(COVER_ARMS) + 2))
                continue
            m = np.mean(acc[g], axis=0)
            i1, ih, ip = COVER_ARMS.index("hybridp1"), COVER_ARMS.index("hybrid"), COVER_ARMS.index("pgmrep10k")
            lines.append(f"| {tag.replace('p', '.')} | {g} | {m[-1]:.1f} | {len(acc[g])} | "
                         + " | ".join("—" if np.isnan(v) else f"{v:,.0f}" for v in m[:-1])
                         + f" | {(m[i1] - m[ip]) / m[ip]:+.1%} | {(m[i1] - m[ih]) / m[ih]:+.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="logs_eps/scores/stage2a")
    args = ap.parse_args()
    tags = args.eps.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    text = g1.grade(EXPERIMENTS, tags, seeds, INTRO) + "\n" + oneway_cover(tags, seeds)
    print(text)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
