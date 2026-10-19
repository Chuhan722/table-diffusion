"""第七十一步 噪声分裂探针阅卷：学习卷 y+e 上跑出的各张表，对 模拟卷 y−e 的误差能否替代对真表的误差判"有谷 / 没谷"、定停点。

每格（1A/1B × ε × 种子）要阅的探针表：快照 0..12（地板_A × 2.0/1.5/1.3/1.2/1.1/1.0/0.95/0.9/0.8/0.7/0.6/0.5/0.4）、
分档全过线快照 G（= 地板停会交出的表）、终表（= 收敛停会交出的表）。每张表算：
  平方：L_A = ½Σw(y_A−q)²（学习卷，应≈引擎 old_loss）、L_B = ½Σw(y_B−q)²（模拟卷）、L_T = ½Σw(y_真−q)²（真表）、L_y（原卷）；
  绝对：L1_B = mean|y_B−q|（模拟卷）、L1_T = mean|y_真−q|；
  以及七轴阅卷的 二阶 L1/N（scripts/score_any.py）与 1B 的 测过的对（scripts/score_pair_groups.py）。
原理：q 只依赖 y_A，y_B 的噪声与 y_A 独立 → E[L_B] = L_T + 地板_A（常数），L_B 与 L_T 同形；
  L1_B 是真 L1 误差经 N(0,σ'²) 平滑后的量（E|a+Z| 随 |a| 单调），幅度被压缩但次序保留；汇报用的二阶 L1/N 是 L1 口径，故以绝对值版为准。
判据（探针内）：G 的 L1_B（或 L_B）低于终表 → 有谷 → 地板停；否则 → 收敛停。与同一场的真表判据（L_T、L1_T、二阶、测过的对）对照。
迁移（原卷）：探针选的点（argmin L1_B 的倍数 / G / 终）映射到原卷同法跑出的同一倍数快照
  （1A：results/floor 第六十二、六十三步；1B：results/groupfloor/snap1b 第六十八步），
  看原卷真二阶在该点相对原卷网格最优、地板停、收敛停差多少。

用法：cd residual-evolution && .venv/bin/python logs_eps/fission/score_fission.py
输出：logs_eps/scores/fission_probe.md、logs_eps/fission/fission_curves_1A.png / _1B.png；七轴缓存 results/fission/score_cache.json
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
CACHE = RES / "score_cache.json"
FLOOR_CACHE = ROOT / "logs_eps/floor/scores.json"  # 第六十二、六十三步 1A 原卷快照的七轴缓存（原始单位）
MULT = [2.0, 1.5, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
MULT_1B = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
TAGS = ["0p1", "0p3", "1", "3", "10"]
REAL = pd.read_csv(ROOT / "data/adult/adult.csv")


# ---------- 工具 ----------
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


def mlabel(m: float) -> str:
    return f"×{m:g}"


_cache: dict | None = None


def two_way(paths: list[Path]) -> dict[str, float]:
    """各表的二阶 L1/N（×10⁻⁶），先查两处缓存，缺的喂 score_any 并写入本步缓存。"""
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    floor_cache = json.load(open(FLOOR_CACHE, encoding="utf-8")) if FLOOR_CACHE.exists() else {}
    out, todo = {}, []
    for p in paths:
        rel = str(p.relative_to(ROOT))
        if rel in _cache:
            out[rel] = _cache[rel]
        elif rel in floor_cache:
            out[rel] = float(floor_cache[rel]["二阶"]) * 1e6
        else:
            todo.append(p)
    for i in range(0, len(todo), 12):
        chunk = todo[i:i + 12]
        sc = score_any(chunk)
        for p in chunk:
            v = sc.get(p.stem)
            if v is None:
                raise SystemExit(f"score_any 未返回 {p}")
            _cache[str(p.relative_to(ROOT))] = v[1]
            out[str(p.relative_to(ROOT))] = v[1]
        json.dump(_cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    return out


# ---------- 探针场 ----------
def probe_points(name: str) -> list[dict]:
    """探针场全部要阅的表：dict(label, kind∈{mult,G,final}, mult, path, round, engine, sec)。"""
    out = []
    rec = {}
    snaps = RES / f"{name}_snaps.csv"
    if snaps.exists():
        for r in csv.DictReader(open(snaps, encoding="utf-8")):
            rec[r["index"]] = (int(r["round"]), float(r["old_loss"]), float(r["elapsed_s"]))
    for i, m in enumerate(MULT):
        p = RES / f"{name}_snap{i}.csv"
        if p.exists() and str(i) in rec:
            rd, ol, el = rec[str(i)]
            out.append(dict(label=mlabel(m), kind="mult", mult=m, path=p, round=rd, engine=ol, sec=el))
    pg = RES / f"{name}_snapG.csv"
    if pg.exists() and "G" in rec:
        rd, ol, el = rec["G"]
        out.append(dict(label="G(地板停)", kind="G", mult=None, path=pg, round=rd, engine=ol, sec=el))
    fin = RES / f"{name}.csv"
    if fin.exists():
        n_rounds, loss, reason, el = read_final(RES / f"{name}_curve.csv")
        out.append(dict(label=f"终(收敛停,{reason})", kind="final", mult=None, path=fin, round=n_rounds, engine=loss, sec=el))
    return out


# ---------- 原卷场（迁移对照） ----------
def orig_points(exam: str, tag: str, seed: int, floor_source: float) -> list[dict] | None:
    """原卷同法跑出的网格：dict(label, kind, mult, path)。1A 地板停 = ×1.0（单档，G 即 ×1.0）；1B 地板停 = 分档 G。"""
    if exam == "1A":
        base = ROOT / "results/floor"
        name = f"adult_eps{tag}_ours2way_seed{seed}"
        snaps = base / f"{name}_snaps.csv"
        if not snaps.exists() or not (base / f"{name}.csv").exists():
            return None
        pts = []
        for r in csv.DictReader(open(snaps, encoding="utf-8")):
            m = round(float(r["threshold"]) / floor_source, 3)
            p = base / f"{name}_snap{r['index']}.csv"
            if p.exists():
                pts.append(dict(label=mlabel(m), kind="mult", mult=m, path=p, round=int(r["round"])))
        one = [x for x in pts if abs(x["mult"] - 1.0) < 1e-9]
        if one:
            pts.append(dict(label="G(地板停)", kind="G", mult=None, path=one[0]["path"], round=one[0]["round"]))
        n_rounds, _, reason, _ = read_final(base / f"{name}_curve.csv")
        pts.append(dict(label=f"终(收敛停,{reason})", kind="final", mult=None, path=base / f"{name}.csv", round=n_rounds))
        return pts
    base = ROOT / "results/groupfloor/snap1b"
    name = f"adult_eps{tag}_snap_seed{seed}"
    snaps = base / f"{name}_snaps.csv"
    if not snaps.exists() or not (base / f"{name}.csv").exists():
        return None
    rec = {r["index"]: int(r["round"]) for r in csv.DictReader(open(snaps, encoding="utf-8"))}
    pts = []
    for i, m in enumerate(MULT_1B):
        p = base / f"{name}_snap{i}.csv"
        if p.exists() and str(i) in rec:
            pts.append(dict(label=mlabel(m), kind="mult", mult=m, path=p, round=rec[str(i)]))
    pg = base / f"{name}_snapG.csv"
    if pg.exists() and "G" in rec:
        pts.append(dict(label="G(地板停)", kind="G", mult=None, path=pg, round=rec["G"]))
    n_rounds, _, reason, _ = read_final(base / f"{name}_curve.csv")
    pts.append(dict(label=f"终(收敛停,{reason})", kind="final", mult=None, path=base / f"{name}.csv", round=n_rounds))
    return pts


def map_point(choice: dict, orig: list[dict]) -> tuple[dict | None, str]:
    """探针选的点映射到原卷网格：同倍数优先，缺则取最近倍数（标"近似"）。"""
    if choice["kind"] in ("G", "final"):
        hit = [x for x in orig if x["kind"] == choice["kind"]]
        return (hit[0], "") if hit else (None, "缺")
    mults = [x for x in orig if x["kind"] == "mult"]
    if not mults:
        return None, "缺"
    best = min(mults, key=lambda x: abs(x["mult"] - choice["mult"]))
    return best, ("" if abs(best["mult"] - choice["mult"]) < 1e-6 else f"近似{best['label']}")


# ---------- 单格阅卷 ----------
def grade(cfg: dict) -> dict | None:
    exam, tag, seed, src = cfg["exam"], cfg["tag"], int(cfg["seed"]), cfg["src"]
    name = f"fis_{exam}_eps{tag}_s{seed}"
    pts = probe_points(name)
    if not pts or pts[-1]["kind"] != "final":
        return None
    qa = json.load(open(ROOT / "data" / cfg["data"] / cfg["exam_file"], encoding="utf-8"))["queries"]
    qb = json.load(open(RES / f"{src}_fisB.json", encoding="utf-8"))["queries"]
    qy = json.load(open(ROOT / "data" / src / cfg["exam_file"], encoding="utf-8"))["queries"]
    sig = np.array([q["sigma"] for q in qy], dtype=np.float64)
    w = np.ones_like(sig) if cfg["mode"] == "none" else (sig.min() / sig) ** 2
    ya = np.array([q["result"] for q in qa]); yb = np.array([q["result"] for q in qb]); yy = np.array([q["result"] for q in qy])
    yt = counts(REAL, qy)
    floor_a = cfg["floor_split"]
    two = two_way([x["path"] for x in pts])
    groups = {}
    if exam == "1B" and (ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{seed}_measurements.json").exists():
        keys = [f"k{i}" for i in range(len(pts))]
        gr = pair_groups(tag, seed, {k: x["path"] for k, x in zip(keys, pts)})
        groups = {x["label"]: gr.get(k) for k, x in zip(keys, pts)}
    for x in pts:
        q = counts(pd.read_csv(x["path"]), qy)
        x.update(L_A=half_wsq(w, ya, q), L_B=half_wsq(w, yb, q), L_T=half_wsq(w, yt, q), L_y=half_wsq(w, yy, q),
                 L1_B=float(np.mean(np.abs(yb - q))), L1_T=float(np.mean(np.abs(yt - q))),
                 two=two[str(x["path"].relative_to(ROOT))],
                 measured=(groups[x["label"]][0] if groups.get(x["label"]) else float("nan")))
        x["L_B_adj"] = x["L_B"] - floor_a
        x["opt"] = (x["L_B"] - x["L_A"]) / floor_a
    g = next((x for x in pts if x["kind"] == "G"), None)
    fin = pts[-1]
    dec = None
    if g is not None:
        def pick(key):
            return "地板停" if g[key] < fin[key] else "收敛停"
        dec = dict(probe_L1=pick("L1_B"), probe_L2=pick("L_B"), truth_LT=pick("L_T"), truth_L1T=pick("L1_T"),
                   truth_two=pick("two"), truth_measured=(pick("measured") if groups else "-"),
                   dL1B=(g["L1_B"] - fin["L1_B"]) / fin["L1_B"], dLB=(g["L_B"] - fin["L_B"]) / fin["L_B"],
                   dLT=(g["L_T"] - fin["L_T"]) / fin["L_T"], d_two=(g["two"] - fin["two"]) / fin["two"])
    corr = float(np.corrcoef([x["L_B_adj"] for x in pts], [x["L_T"] for x in pts])[0, 1]) if len(pts) > 2 else float("nan")
    arg = {k: min(pts, key=lambda x: x[k]) for k in ("L1_B", "L_B", "L_T", "L1_T", "two")}
    # 迁移：探针选点 → 原卷网格
    transfer = None
    orig = orig_points(exam, tag, seed, cfg["floor_source"])
    if orig and dec is not None:
        otwo = two_way([x["path"] for x in orig])
        for x in orig:
            x["two"] = otwo[str(x["path"].relative_to(ROOT))]
        o_g = next(x for x in orig if x["kind"] == "G"); o_fin = orig[-1]
        o_best = min(orig, key=lambda x: x["two"])
        chosen, note = map_point(arg["L1_B"], orig)
        rule_pick = o_g if dec["probe_L1"] == "地板停" else o_fin  # 只用探针判的停法（不传倍数）
        truth_rule = "地板停" if o_g["two"] < o_fin["two"] else "收敛停"
        transfer = dict(
            orig_G=o_g["two"], orig_final=o_fin["two"], orig_best=o_best["two"], orig_best_label=o_best["label"],
            truth_rule=truth_rule, rule_hit=(dec["probe_L1"] == truth_rule), rule_two=rule_pick["two"],
            chosen_label=(chosen["label"] if chosen else "-"), chosen_note=note, chosen_two=(chosen["two"] if chosen else float("nan")),
            orig_grid=[(x["label"], x["two"]) for x in orig],
        )
    return dict(exam=exam, tag=tag, seed=seed, name=name, floor_a=floor_a, rows=pts, decision=dec, corr=corr,
                arg={k: v["label"] for k, v in arg.items()}, transfer=transfer)


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{nd}f}"


def pct(a, b):
    return f"{(a - b) / b:+.1%}" if (a == a and b == b and b) else "-"


def mean_pct(pairs):
    v = [(a - b) / b for a, b in pairs if a == a and b == b and b]
    return f"{np.mean(v):+.1%}" if v else "-"


# ---------- 汇总 ----------
def main() -> None:
    cfgs = json.load(open(RES / "floors.json", encoding="utf-8"))
    results = [r for r in (grade(c) for c in cfgs) if r]
    if not results:
        raise SystemExit("无结果可阅")
    results.sort(key=lambda r: (r["exam"], TAGS.index(r["tag"]), r["seed"]))
    lines = ["# 第七十一步 噪声分裂探针（adult，两卷 × 5 ε × 3 种子）", "",
             "学习卷 y+e 上跑第一段到收敛；每张表对 模拟卷 y−e 的每题绝对误差 L1_B = mean|y_B−q|（绝对值版，主判据）与平方损失 L_B（减去地板_A 后≈对真表的 L_T）。",
             "探针内判据：地板停表 G 的 L1_B 低于终表 → 有谷 → 地板停；否则收敛停。对照同一场真表判据：二阶 L1/N×10⁻⁶（汇报口径）、L1_T、L_T、测过的对（1B）。",
             "迁移：探针判出的停法 / 探针选的倍数，套到原卷同法网格（1A 第六十二、六十三步，1B 第六十八步，均为第一段表）上看原卷真二阶。", ""]

    # 表一：探针内判定 + 迁移
    lines += ["## 一、逐格判定与迁移（原卷真二阶 ×10⁻⁶）", "",
              "| 考卷 | ε | 种子 | 探针判 | 探针内真判(二阶) | 原卷真判(二阶) | G 对终 L1_B | G 对终 真二阶(探针场) | 探针选点 | 原卷最优点 | 原卷二阶@探针停法 | @探针选点 | @地板停 | @收敛停 | @最优 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, t = r["decision"], r["transfer"]
        if d is None:
            lines.append(f"| {r['exam']} | {r['tag']} | {r['seed']} | 无 G 快照 | | | | | | | | | | | |")
            continue
        if t is None:
            lines.append(f"| {r['exam']} | {r['tag']} | {r['seed']} | **{d['probe_L1']}** | {d['truth_two']} | 原卷网格缺 | {d['dL1B']:+.1%} | {d['d_two']:+.1%} | {r['arg']['L1_B']} | | | | | | |")
            continue
        chosen = t["chosen_label"] + (f"（{t['chosen_note']}）" if t["chosen_note"] else "")
        lines.append(f"| {r['exam']} | {r['tag']} | {r['seed']} | **{d['probe_L1']}** | {d['truth_two']} | {t['truth_rule']}{'' if t['rule_hit'] else ' ✗'} | "
                     f"{d['dL1B']:+.1%} | {d['d_two']:+.1%} | {r['arg']['L1_B']}→{chosen} | {t['orig_best_label']} | "
                     f"{t['rule_two']:,.0f} | {t['chosen_two']:,.0f} | {t['orig_G']:,.0f} | {t['orig_final']:,.0f} | {t['orig_best']:,.0f} |")
    lines.append("")

    # 表二：按考卷汇总
    lines += ["## 二、汇总", "",
              "| 考卷 | 格数 | 探针判=探针内真判 | 探针判=原卷真判 | 原卷二阶：探针停法 vs 较好的固定停法 | 探针停法 vs 地板停 | 探针停法 vs 收敛停 | 探针选点 vs 原卷最优 | 探针选点 vs 较好的固定停法 | 平方版判=探针内真判 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for exam in ("1A", "1B"):
        rs = [r for r in results if r["exam"] == exam and r["decision"] and r["transfer"]]
        if not rs:
            continue
        n = len(rs)
        a1 = sum(r["decision"]["probe_L1"] == r["decision"]["truth_two"] for r in rs)
        a2 = sum(r["transfer"]["rule_hit"] for r in rs)
        a3 = sum(r["decision"]["probe_L2"] == r["decision"]["truth_two"] for r in rs)
        bestfix = [min(r["transfer"]["orig_G"], r["transfer"]["orig_final"]) for r in rs]
        lines.append(f"| {exam} | {n} | {a1}/{n} | {a2}/{n} | "
                     f"{mean_pct([(r['transfer']['rule_two'], b) for r, b in zip(rs, bestfix)])} | "
                     f"{mean_pct([(r['transfer']['rule_two'], r['transfer']['orig_G']) for r in rs])} | "
                     f"{mean_pct([(r['transfer']['rule_two'], r['transfer']['orig_final']) for r in rs])} | "
                     f"{mean_pct([(r['transfer']['chosen_two'], r['transfer']['orig_best']) for r in rs])} | "
                     f"{mean_pct([(r['transfer']['chosen_two'], b) for r, b in zip(rs, bestfix)])} | {a3}/{n} |")
    lines += ["", "说明：'探针停法' = 只用探针判出的地板停/收敛停二选一，套到原卷；'探针选点' = 探针上 L1_B 最低的倍数（或 G/终）套到原卷同倍数快照；",
              "'较好的固定停法' = 该格原卷上地板停与收敛停中真二阶较低者（事后才知道）；'原卷最优' = 原卷网格全部快照中真二阶最低者（事后才知道）。百分比为各格相对差的平均，正为更差。", ""]

    # 表三：按 ε 看探针选倍数
    lines += ["## 三、探针选的倍数（按 ε，三种子）", "", "| 考卷 | ε | 探针选点（s1 / s2 / s3） | 原卷最优点（s1 / s2 / s3） | 原卷真判（s1 / s2 / s3） |", "|---|---|---|---|---|"]
    for exam in ("1A", "1B"):
        for tag in TAGS:
            rs = [r for r in results if r["exam"] == exam and r["tag"] == tag and r["transfer"]]
            if rs:
                lines.append(f"| {exam} | {tag} | {' / '.join(r['arg']['L1_B'] for r in rs)} | {' / '.join(r['transfer']['orig_best_label'] for r in rs)} | "
                             f"{' / '.join(r['transfer']['truth_rule'] for r in rs)} |")
    lines.append("")

    # 附：逐格明细
    lines += ["## 附、逐格明细（探针场；二阶为真表 L1/N ×10⁻⁶）", ""]
    for r in results:
        fa = r["floor_a"]
        lines += [f"### {r['exam']} ε={r['tag']} 种子 {r['seed']}（地板_A = {fa:,.0f}；corr(L_B−地板_A, L_T) = {r['corr']:.3f}）", "",
                  "| 表 | 轮 | 秒 | 引擎损失÷地板_A | L_B÷地板_A | L_T÷地板_A | (L_B−L_A)÷地板_A | 每题 L1_B | 每题 L1_T | 二阶 | 测过的对 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            lines.append(f"| {x['label']} | {fmt(x['round'])} | {fmt(x['sec'], 1)} | {fmt(x['engine'] / fa if x['engine'] else None, 3)} | "
                         f"{x['L_B'] / fa:.3f} | {x['L_T'] / fa:.3f} | {x['opt']:+.3f} | {x['L1_B']:.2f} | {x['L1_T']:.2f} | {x['two']:.1f} | {fmt(x['measured'], 1)} |")
        if r["transfer"]:
            lines.append("")
            lines.append("原卷网格真二阶：" + "，".join(f"{lab} {v:,.0f}" for lab, v in r["transfer"]["orig_grid"]))
        lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: lines.index("## 三、探针选的倍数（按 ε，三种子）")]))
    print(f"写入 {OUT_MD}")
    plot(results)


def plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for exam in ("1A", "1B"):
        rs = [r for r in results if r["exam"] == exam]
        if not rs:
            continue
        seeds = sorted({r["seed"] for r in rs})
        fig, axes = plt.subplots(len(seeds), len(TAGS), figsize=(4.0 * len(TAGS), 3.2 * len(seeds)), squeeze=False)
        for j, tag in enumerate(TAGS):
            for i, seed in enumerate(seeds):
                ax = axes[i][j]
                r = next((x for x in rs if x["tag"] == tag and x["seed"] == seed), None)
                if r is None:
                    ax.set_axis_off(); continue
                rows = sorted([x for x in r["rows"] if x["round"] is not None], key=lambda x: x["round"])
                rd = [x["round"] + 1 for x in rows]
                ax.plot(rd, [x["L1_B"] for x in rows], "o-", color="C0", ms=3, label="每题 L1_B（模拟卷）")
                ax2 = ax.twinx()
                ax2.plot(rd, [x["two"] for x in rows], "s--", color="C1", ms=3, label="二阶 L1/N（真表）")
                for x in rows:
                    if x["kind"] == "G":
                        ax.axvline(x["round"] + 1, color="gray", ls=":", lw=1)
                ax.set_xscale("log"); ax.set_title(f"{exam} ε={tag} 种子 {seed}", fontsize=9); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
                if i == 0 and j == 0:
                    ax.legend(loc="upper left", fontsize=6); ax2.legend(loc="upper right", fontsize=6)
        fig.suptitle(f"{exam}：探针 L1_B（蓝，左轴）与真表二阶（橙，右轴）随轮数；竖虚线 = 地板停 G", fontsize=10)
        fig.tight_layout()
        png = ROOT / f"logs_eps/fission/fission_curves_{exam}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"图 {png}")


if __name__ == "__main__":
    main()
