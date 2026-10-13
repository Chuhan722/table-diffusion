"""2B 机制核查阅卷：PGM 同一拟合「不出表直接算边际」vs「出表数边际」vs 我们的表。

臂：hybrid = 第一阶段我们的表；pgm表 = pgm_exact_marginals.py 重拟合（与第一阶段 pgmrep10k 逐字同配置）后出的表，
与 pgm模型 是同一个拟合；pgm表(1B) = 第一阶段 pgmrep10k 原表（JAX 多线程浮点顺序不定，重拟合表与它不逐格同，
但各轴差 <1.5%，是同一分布的两次出表；两表互距也顺带报出，就是出表随机性的直接度量）；pgm模型 = 精确边际，不经过出表；
pgm大表 = 同模型出 20 倍行数再除 20（半空间题精确算不了，pgm模型 行的半空间取自此近似，带 *）。
轴与 scripts/score_any.py 同口径（×10⁻⁶ L1/N，二阶全 105 对均，三阶全 455 三元组，等值/半空间为留出题 |err|/N 均），
另加一阶与 AIM 测过/没测的对分组（与 stage1_pairgroups.md 同法）。
三条 Δ：我们 vs pgm模型（引擎 vs 对手的分布本身）、pgm表 vs pgm模型（出表损耗）、我们 vs pgm表（第一阶段 1B 原记分，核对用）。

用法: .venv/bin/python logs_eps/stage2/grade_pgm_exact.py [--eps 3,10] [--seeds 1,2,3] [--out logs_eps/scores/stage2b_pgm_exact.md]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AXES = ["一阶", "二阶", "二阶·测过", "二阶·没测", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]
TAGS = {"0p1": "0.1", "0p3": "0.3", "1": "1", "3": "3", "10": "10"}


def load_int_table(path: Path, columns: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    return np.stack([df[c].to_numpy(dtype=np.int64) for c in columns], axis=1)


def marg(X: np.ndarray, cols: tuple[int, ...], doms: list[int]) -> np.ndarray:
    key = X[:, cols[0]].astype(np.int64)
    size = doms[cols[0]]
    for c in cols[1:]:
        key = key * doms[c] + X[:, c]
        size *= doms[c]
    return np.bincount(key, minlength=size).astype(np.float64)


def answers(queries: list[dict], X: np.ndarray, fpos: dict[str, int], doms: list[int]) -> np.ndarray:
    out = np.zeros(len(queries), dtype=np.float64)
    for i, q in enumerate(queries):
        c0 = q["conditions"][0]
        if c0["operator"] == "halfspace":
            w = np.zeros((len(doms), max(doms)), dtype=np.int64)
            for name, tab in c0["scores"].items():
                for v, sc in tab.items():
                    w[fpos[name], int(v)] = int(sc)
            rs = np.zeros(len(X), dtype=np.int64)
            for j in range(len(doms)):
                rs += w[j, X[:, j]]
            out[i] = float((rs >= int(c0["threshold"])).sum())
        else:
            mask = np.ones(len(X), dtype=bool)
            for c in q["conditions"]:
                mask &= X[:, fpos[c["attribute"]]] == int(c["value"])
            out[i] = float(mask.sum())
    return out


class Real:
    def __init__(self) -> None:
        df = pd.read_csv(ROOT / "data/adult/adult.csv")
        self.columns = list(df.columns)
        self.fpos = {c: j for j, c in enumerate(self.columns)}
        self.X = np.stack([df[c].to_numpy(dtype=np.int64) for c in self.columns], axis=1)
        self.N = len(self.X)
        self.doms = [int(self.X[:, j].max()) + 1 for j in range(len(self.columns))]
        F = len(self.columns)
        self.pairs = list(combinations(range(F), 2))
        self.trips = list(combinations(range(F), 3))
        assert len(self.trips) <= 2000
        self.m1 = [marg(self.X, (j,), self.doms) for j in range(F)]
        self.m2 = [marg(self.X, p, self.doms) for p in self.pairs]
        self.m3 = [marg(self.X, t, self.doms) for t in self.trips]
        hq = json.load(open(ROOT / "data/adult/heldout_1224query.json", encoding="utf-8"))["queries"]
        self.eq = [q for q in hq if q["conditions"][0]["operator"] != "halfspace"]
        self.hs = [q for q in hq if q["conditions"][0]["operator"] == "halfspace"]
        self.y_eq = np.array([float(q["result"]) for q in self.eq])
        self.y_hs = np.array([float(q["result"]) for q in self.hs])

    def axes(self, m1, m2, m3, eq_ans, hs_ans, measured_pairs: set[tuple[int, int]]) -> dict[str, float]:
        N = self.N
        e1 = np.array([np.abs(a - b).sum() / N for a, b in zip(m1, self.m1)])
        e2 = np.array([np.abs(a - b).sum() / N for a, b in zip(m2, self.m2)])
        e3 = np.array([np.abs(a - b).sum() / N for a, b in zip(m3, self.m3)])
        meas = np.array([p in measured_pairs for p in self.pairs])
        top5 = np.sort(e3)[-max(1, len(e3) // 20):].mean()
        return {
            "一阶": e1.mean() * 1e6, "二阶": e2.mean() * 1e6,
            "二阶·测过": e2[meas].mean() * 1e6 if meas.any() else float("nan"),
            "二阶·没测": e2[~meas].mean() * 1e6 if (~meas).any() else float("nan"),
            "三阶均": e3.mean() * 1e6, "三阶top5%": top5 * 1e6, "三阶max": e3.max() * 1e6,
            "等值": (np.abs(eq_ans - self.y_eq) / N).mean() * 1e6,
            "半空间": (np.abs(hs_ans - self.y_hs) / N).mean() * 1e6,
        }

    def axes_table(self, X: np.ndarray, measured_pairs) -> dict[str, float]:
        m1 = [marg(X, (j,), self.doms) for j in range(len(self.columns))]
        m2 = [marg(X, p, self.doms) for p in self.pairs]
        m3 = [marg(X, t, self.doms) for t in self.trips]
        return self.axes(m1, m2, m3, answers(self.eq, X, self.fpos, self.doms),
                         answers(self.hs, X, self.fpos, self.doms), measured_pairs)

    def axes_npz(self, z, prefix: str, hs_key: str, eq_key: str, measured_pairs) -> dict[str, float]:
        m1 = [z[f"{prefix}m1_{j}"] for j in range(len(self.columns))]
        m2 = [z[f"{prefix}m2_" + "_".join(map(str, p))] for p in self.pairs]
        m3 = [z[f"{prefix}m3_" + "_".join(map(str, t))] for t in self.trips]
        return self.axes(m1, m2, m3, z[eq_key], z[hs_key], measured_pairs)


def fmt(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:,.0f}"


def pct(a: float, b: float) -> str:
    return "—" if (np.isnan(a) or np.isnan(b) or b == 0) else f"{(a - b) / b * 100:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0p1,0p3,1,3,10")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="logs_eps/scores/stage2b_pgm_exact.md")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    real = Real()

    lines = ["## 2B 机制核查：PGM 同一拟合，不出表 vs 出表 vs 我们的表（×10⁻⁶ L1/N，种子均）", "",
             "hybrid = 我们的表（第一阶段）；pgm表 = 独立 PGM 10000 轮重拟合后出表；pgm模型 = 同一个拟合不出表的精确边际；pgm表(1B) = 第一阶段原表（同配置另一次拟合+出表，核对用）；",
             "pgm大表 = 同模型出 20 倍行数再除 20。半空间精确算不了，pgm模型 行的半空间取 pgm大表 的近似值（带 *）。",
             "二阶·测过/没测 = AIM 录像测过的字段对 / 没测的对（与 stage1_pairgroups.md 同法）。Δ 为逐种子配对均。", ""]
    verdicts = []
    for tag in args.eps.split(","):
        rows: dict[str, list[dict[str, float]]] = {"hybrid": [], "pgm表(1B)": [], "pgm表": [], "pgm模型": [], "pgm大表": []}
        notes, used = [], []
        for s in seeds:
            npz = ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz"
            ours = ROOT / f"results/stage1/adult_eps{tag}_hybrid_seed{s}.csv"
            pgm = ROOT / f"results/stage1/adult_eps{tag}_pgmrep10k_seed{s}.csv"
            refit = ROOT / f"results/stage2/adult_eps{tag}_pgmexact_seed{s}.csv"
            if not (npz.exists() and ours.exists() and pgm.exists() and refit.exists()):
                continue
            z = np.load(npz)
            assert list(z["columns"]) == real.columns and list(z["doms"]) == real.doms and int(z["rows"]) == real.N
            mp = {tuple(int(v) for v in p) for p in z["measured_pairs"]}
            Xo = load_int_table(ours, real.columns)
            Xp = load_int_table(pgm, real.columns)
            Xr = load_int_table(refit, real.columns)
            dist = np.mean([np.abs(marg(Xp, p, real.doms) - marg(Xr, p, real.doms)).sum() / real.N for p in real.pairs]) * 1e6
            notes.append(f"s{s} 重拟合表与 1B 原表二阶互距 {dist:,.0f}")
            rows["hybrid"].append(real.axes_table(Xo, mp))
            rows["pgm表(1B)"].append(real.axes_table(Xp, mp))
            rows["pgm表"].append(real.axes_table(Xr, mp))
            big = real.axes_npz(z, "big_", "hs_big", "eq_big", mp)
            exact = real.axes_npz(z, "", "hs_big", "eq_exact", mp)
            rows["pgm模型"].append(exact)
            rows["pgm大表"].append(big)
            used.append(s)
        if not used:
            continue
        eps = TAGS[tag]
        nm = f"{len(mp)}/{len(real.pairs) - len(mp)}"
        lines.append(f"### ε = {eps}（种子 {','.join(map(str, used))}；最后一个种子测过/没测对数 {nm}）")
        lines.append("")
        lines.append("| 臂 | " + " | ".join(AXES) + " |")
        lines.append("|---|" + "---|" * len(AXES))
        mean = {k: {a: float(np.mean([r[a] for r in v])) for a in AXES} for k, v in rows.items()}
        for k in ["hybrid", "pgm表(1B)", "pgm表", "pgm模型", "pgm大表"]:
            cells = [fmt(mean[k][a]) + ("*" if (k == "pgm模型" and a == "半空间") else "") for a in AXES]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")

        def drow(label: str, a: str, b: str) -> str:
            cells = []
            for ax in AXES:
                ds = [(ra[ax] - rb[ax]) / rb[ax] * 100 for ra, rb in zip(rows[a], rows[b]) if rb[ax]]
                cells.append(f"{np.mean(ds):+.1f}%" if ds else "—")
            return f"| {label} | " + " | ".join(cells) + " |"

        lines.append(drow("Δ 我们 vs pgm模型", "hybrid", "pgm模型"))
        lines.append(drow("Δ pgm表 vs pgm模型（出表损耗）", "pgm表", "pgm模型"))
        lines.append(drow("Δ 我们 vs pgm表(1B)（第一阶段原记分）", "hybrid", "pgm表(1B)"))
        lines.append("")
        per = " ".join(f"s{s}:{(ra['二阶'] - rb['二阶']) / rb['二阶'] * 100:+.1f}%"
                       for s, ra, rb in zip(used, rows["hybrid"], rows["pgm模型"]))
        per_loss = " ".join(f"s{s}:{(ra['二阶'] - rb['二阶']) / rb['二阶'] * 100:+.1f}%"
                            for s, ra, rb in zip(used, rows["pgm表"], rows["pgm模型"]))
        lines.append(f"逐种子主轴二阶 我们 vs pgm模型：{per}；出表损耗：{per_loss}。{'；'.join(notes)}。")
        lines.append("")
        d_main = float(np.mean([(ra['二阶'] - rb['二阶']) / rb['二阶'] * 100 for ra, rb in zip(rows["hybrid"], rows["pgm模型"])]))
        d_loss = float(np.mean([(ra['二阶'] - rb['二阶']) / rb['二阶'] * 100 for ra, rb in zip(rows["pgm表"], rows["pgm模型"])]))
        d_1b = float(np.mean([(ra['二阶'] - rb['二阶']) / rb['二阶'] * 100 for ra, rb in zip(rows["hybrid"], rows["pgm表(1B)"])]))
        verdicts.append((eps, d_main, d_loss, d_1b))

    lines.append("### 主轴二阶汇总")
    lines.append("")
    lines.append("| ε | 我们 vs pgm模型 | 判定 | pgm 出表损耗 | 我们 vs pgm表(1B) |")
    lines.append("|---|---|---|---|---|")
    for eps, d_main, d_loss, d_1b in verdicts:
        v = "平" if abs(d_main) <= 2 else ("胜" if d_main < 0 else "负")
        lines.append(f"| {eps} | {d_main:+.1f}% | {v} | {d_loss:+.1f}% | {d_1b:+.1f}% |")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
