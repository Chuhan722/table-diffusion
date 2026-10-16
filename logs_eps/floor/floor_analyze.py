"""第六十二步：沿噪声地板倍数取快照、定停点（adult，种子 1，五档 ε，两臂）。

我们臂：results/floor 的快照（地板 × 5/3/2/1.5/1.3/1.2/1.1/1.05/1/0.95/0.9/0.8/0.7）+ 第六十一步快照
（终损失 × 1.2/1.1/1.05/1.02/1.01，同一条逐位相同的轨迹）+ 起点表 + 终表，合成一条"真值误差 vs 损失/地板"曲线。
GSD 臂：results/floor 的快照（同一倍数网格，fitness 口径阈值 = 倍数 × Σσ²/N²，每 1000 代检查）+ 终表。
核对：我们终表与 results/effort 逐位同、曲线除时间列外逐列同、快照轮 = 曲线上首个 old_loss ≤ 阈值的轮、
快照表 ℒ = 2·记账 old_loss；GSD 快照表 ℒ = N²·记账 fitness，终 ℒ 与第六十步之比。
阅卷 scripts/score_any.py（对真表、干净查询），成绩缓存 logs_eps/floor/scores.json。
输出 logs_eps/scores/floor_step62.md。
"""
from __future__ import annotations

import csv
import filecmp
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps" / "effort"))
sys.path.insert(0, str(ROOT / "logs_eps" / "plateau"))
from align_loss import cell_counts, load_codes, load_exam  # noqa: E402
from plateau_analyze import AXES, EPS, TAGS, noise_floor, read_curve  # noqa: E402

FLOOR = ROOT / "results" / "floor"
EFFORT = ROOT / "results" / "effort"
PLATEAU = ROOT / "results" / "plateau"
INIT = PLATEAU / "init"
OUT = ROOT / "logs_eps" / "scores" / "floor_step62.md"
CACHE = ROOT / "logs_eps" / "floor" / "scores.json"
PY = ROOT / ".venv" / "bin" / "python"
SCORE = ROOT / "scripts" / "score_any.py"
N = 32561
MAIN = ["二阶", "三阶均", "三阶top5%", "半空间"]

CONCLUSION = """\
## 结论（种子 1，五档 ε；数字见下表）

（待填）
"""


# ---------- 读取 ----------
def ours_name(tag: str) -> str:
    return f"adult_eps{tag}_ours2way_seed1"


def gsd_name(tag: str) -> str:
    return f"adult_eps{tag}_gsd_seed1"


def read_snaps(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def table_loss(path: Path, tag: str) -> float:
    """表对噪声目标的 ℒ = Σ(y − c)²（与 align_loss 同法）。"""
    _, pairs, tgt = load_exam(ROOT / f"data/adult_alloc{tag}base_s1/measured_2way.json")
    cols, rows = load_codes(path)
    return float(np.sum((tgt - cell_counts(cols, rows, pairs)) ** 2))


# ---------- 阅卷（带缓存） ----------
def load_cache() -> dict:
    return json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}


def save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)


def grade(tables: list[Path], cache: dict) -> None:
    """score_any 一次最多喂若干张；结果按仓库相对路径缓存。"""
    todo = [t for t in tables if str(t.relative_to(ROOT)) not in cache and t.exists()]
    for i in range(0, len(todo), 12):
        chunk = todo[i:i + 12]
        out = subprocess.run(
            [str(PY), str(SCORE), "adult", *map(str, chunk)],
            check=True, capture_output=True, text=True, cwd=ROOT,
        ).stdout.splitlines()
        rows = []
        for line in out[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            try:
                rows.append(dict(zip(AXES, map(float, parts[1:8]))))
            except ValueError:
                continue
        if len(rows) != len(chunk):
            raise RuntimeError(f"阅卷行数 {len(rows)} ≠ 表数 {len(chunk)}：\n" + "\n".join(out))
        for t, r in zip(chunk, rows):
            cache[str(t.relative_to(ROOT))] = r
        save_cache(cache)


def sc(cache: dict, p: Path) -> dict | None:
    return cache.get(str(p.relative_to(ROOT)))


# ---------- 我们臂 ----------
def ours_points(tag: str, cache: dict) -> tuple[list[dict], list[str], dict]:
    """返回 [点]（按损失从高到低）、核对说明、全程量。点 = dict(label, mult, round, cand_share, round_share, t_share, scores, path)。"""
    notes = []
    name = ours_name(tag)
    floor = noise_floor(tag)
    same = filecmp.cmp(EFFORT / f"{name}.csv", FLOOR / f"{name}.csv", shallow=False)
    notes.append("终表与第六十步逐位一致" if same else "终表不一致！")
    a, b = read_curve(EFFORT / f"{name}_curve.csv"), read_curve(FLOOR / f"{name}_curve.csv")
    cols = [c for c in a[0].keys() if c != "elapsed_s"]
    ok = len(a) == len(b) and all(ra[c] == rb[c] for ra, rb in zip(a, b) for c in cols)
    notes.append("曲线除时间列外逐列一致" if ok else "曲线不一致！")
    losses = np.array([float(r["old_loss"]) for r in a])
    cand = np.cumsum([float(r["candidates"]) for r in a])
    tsec = np.array([float(r["elapsed_s"]) for r in a])
    total_rounds, total_cand, total_t = len(a), cand[-1], tsec[-1]
    final_loss = losses[-1]

    pts = []

    def add(label, path, k, loss):
        pts.append(dict(
            label=label, mult=loss / floor, round=k,
            round_share=k / total_rounds, cand_share=(cand[k - 1] if k > 0 else 0.0) / total_cand,
            t_share=(tsec[k - 1] if k > 0 else 0.0) / total_t, loss=loss, path=path,
        ))

    init_snaps = read_snaps(INIT / f"{name}_snaps.csv")
    add("起点", INIT / f"{name}_snap0.csv", 0, float(init_snaps[0]["old_loss"]))
    bad_round, bad_loss = 0, 0.0
    for src, snaps_path, pref in [
        ("地板×", FLOOR / f"{name}_snaps.csv", FLOOR / name),
        ("终×", PLATEAU / f"{name}_snaps.csv", PLATEAU / name),
    ]:
        for r in read_snaps(snaps_path):
            k, thr, loss = int(r["round"]), float(r["threshold"]), float(r["old_loss"])
            pred = int(np.argmax(losses <= thr))
            if pred != k or losses[k] != loss:
                bad_round += 1
            p = Path(f"{pref}_snap{r['index']}.csv")
            if abs(table_loss(p, tag) - 2 * loss) > 1e-6 * 2 * loss:
                bad_loss += 1
            if src == "地板×":
                label = f"地板×{thr / floor:.3g}"
            else:
                label = f"终×{thr / final_loss:.3g}"
            add(label, p, k, loss)
    notes.append("快照轮与曲线首穿越轮一致" if bad_round == 0 else f"{bad_round} 张快照轮号不一致！")
    notes.append("快照表 ℒ = 2·记账损失" if bad_loss == 0 else f"{bad_loss} 张快照表 ℒ 不符！")
    add("终表", FLOOR / f"{name}.csv", total_rounds, final_loss)
    pts.sort(key=lambda p: -p["loss"])
    grade([p["path"] for p in pts], cache)
    for p in pts:
        p["scores"] = sc(cache, p["path"])
    return pts, notes, dict(rounds=total_rounds, cand=total_cand, t=total_t, floor=floor, final=final_loss)


# ---------- GSD 臂 ----------
def gsd_net_time(t: np.ndarray) -> np.ndarray:
    """第六十步同一规则：>10×中位数的记录间隔换成中位数（扣 GSD 每 32,561 代的记账突发）。"""
    dt = np.diff(t)
    med = float(np.median(dt))
    burst = dt > 10 * med
    return np.concatenate([[t[0] if t[0] <= 10 * med else med], np.where(burst, med, dt)]).cumsum()


def gsd_points(tag: str, cache: dict):
    name = gsd_name(tag)
    if not (FLOOR / f"{name}.csv").exists():
        return None, ["GSD 臂未收"], None
    floor_l = 2.0 * noise_floor(tag)
    gens = read_snaps(FLOOR / f"{name}_gens.csv")
    g = np.array([int(r["gen"]) for r in gens])
    fit = np.array([float(r["best_fitness"]) for r in gens])
    t = np.array([float(r["elapsed_s"]) for r in gens])
    cand = np.array([float(r["cum_candidates"]) for r in gens])
    t_net = gsd_net_time(t)
    total = dict(gens=int(g[-1]), cand=cand[-1], t=t[-1], t_net=t_net[-1], floor_l=floor_l)
    notes = []
    final_l = table_loss(FLOOR / f"{name}.csv", tag)
    old = table_loss(EFFORT / f"{name}.csv", tag)
    notes.append(f"终 ℒ 与第六十步之比 {final_l / old:.4f}（GSD 的 GPU 搜索非逐位确定）")
    rel = abs(final_l - N * N * fit[-1]) / final_l
    notes.append(f"终表 ℒ 与记账末行 N²·fitness 相对差 {rel:.1e}")
    pts = []

    def add(label, path, gen_idx, loss_l):
        pts.append(dict(
            label=label, mult=loss_l / floor_l, gen=int(g[gen_idx]),
            gen_share=g[gen_idx] / g[-1], cand_share=cand[gen_idx] / cand[-1],
            t_share=t[gen_idx] / t[-1], tnet_share=t_net[gen_idx] / t_net[-1], tnet=t_net[gen_idx],
            loss=loss_l, path=path,
        ))

    bad = 0
    snaps_path = FLOOR / f"{name}_snaps.csv"
    if snaps_path.exists():
        for r in read_snaps(snaps_path):
            gen, thr, bf = int(r["gen"]), float(r["threshold"]), float(r["best_fitness"])
            idx = int(np.where(g == gen)[0][0])
            p = FLOOR / f"{name}_snap{r['index']}.csv"
            tl = table_loss(p, tag)
            if abs(tl - N * N * bf) > 1e-5 * tl:
                bad += 1
            add(f"地板×{thr * N * N / floor_l:.3g}", p, idx, tl)
    notes.append("快照表 ℒ = N²·记账 fitness" if bad == 0 else f"{bad} 张快照表 ℒ 不符！")
    add("终表", FLOOR / f"{name}.csv", len(g) - 1, final_l)
    pts.sort(key=lambda p: -p["loss"])
    grade([p["path"] for p in pts], cache)
    for p in pts:
        p["scores"] = sc(cache, p["path"])
    return pts, notes, total


# ---------- 表格 ----------
def pct(x: float) -> str:
    return f"{x:+.1%}"


def cell(v: float, base: float) -> str:
    return f"{v:.0f} ({pct((v - base) / base)})"


def best_label(pts: list[dict], axis: str) -> str:
    return min(pts, key=lambda p: p["scores"][axis])["label"]


def ours_table(tag: str, pts: list[dict], notes: list[str], tot: dict) -> list[str]:
    final = pts[-1]["scores"]
    lines = [
        f"## 我们 ε = {EPS[tag]}（{'；'.join(notes)}；全程 {tot['rounds']} 轮、{tot['cand']:.3g} 候选、"
        f"{tot['t']:.0f} 秒；地板 {tot['floor']:.4g}，终损失 = 地板 × {tot['final'] / tot['floor']:.3f}）",
        "",
        "| 层级 | 损失/地板 | 轮 | 轮占比 | 候选占比 | 秒占比 | " + " | ".join(AXES) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(AXES),
    ]
    bests = {ax: best_label(pts, ax) for ax in AXES}
    for p in pts:
        cells = []
        for ax in AXES:
            c = cell(p["scores"][ax], final[ax]) if p["label"] != "终表" else f"{final[ax]:.0f}"
            if bests[ax] == p["label"]:
                c = f"**{c}**"
            cells.append(c)
        lines.append(
            f"| {p['label']} | {p['mult']:.3f} | {p['round']} | {p['round_share']:.1%} | {p['cand_share']:.1%} | "
            f"{p['t_share']:.1%} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def gsd_table(tag: str, pts: list[dict], notes: list[str], tot: dict) -> list[str]:
    final = pts[-1]["scores"]
    lines = [
        f"## GSD ε = {EPS[tag]}（{'；'.join(notes)}；全程 {tot['gens']:,} 代、{tot['cand']:.3g} 候选、"
        f"原始 {tot['t']:.0f} 秒 / 净 {tot['t_net']:.0f} 秒；终 ℒ = 地板 × {pts[-1]['mult']:.3f}）",
        "",
        "| 层级 | 损失/地板 | 代 | 代占比 | 候选占比 | 净秒 | 净秒占比 | " + " | ".join(AXES) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(AXES),
    ]
    bests = {ax: best_label(pts, ax) for ax in AXES}
    for p in pts:
        cells = []
        for ax in AXES:
            c = cell(p["scores"][ax], final[ax]) if p["label"] != "终表" else f"{final[ax]:.0f}"
            if bests[ax] == p["label"]:
                c = f"**{c}**"
            cells.append(c)
        lines.append(
            f"| {p['label']} | {p['mult']:.3f} | {p['gen']:,} | {p['gen_share']:.1%} | {p['cand_share']:.1%} | "
            f"{p['tnet']:.0f} | {p['tnet_share']:.1%} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def find(pts: list[dict], label: str) -> dict | None:
    for p in pts:
        if p["label"] == label:
            return p
    return None


def summary_ours(all_pts: dict[str, list[dict]]) -> list[str]:
    lines = [
        "## 汇总（我们）：最优停点在哪、地板停（×1.0）与跑到收敛（终表）差多少",
        "",
        "| ε | 二阶最优层级 | 三阶均最优 | 三阶top5%最优 | 半空间最优 | ×1.0 轮占比 | ×1.0 候选占比 | ×1.0 二阶 Δ终 | ×1.0 三阶均 Δ终 | ×1.0 三阶top5% Δ终 | ×1.0 半空间 Δ终 | 二阶最优 Δ终 | ×1.0 二阶 Δ最优 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        pts = all_pts[tag]
        final = pts[-1]["scores"]
        f1 = find(pts, "地板×1")
        bl = best_label(pts, "二阶")
        bp = find(pts, bl)["scores"]
        d = lambda ax: pct((f1["scores"][ax] - final[ax]) / final[ax])  # noqa: E731
        lines.append(
            f"| {EPS[tag]} | {bl} | {best_label(pts, '三阶均')} | {best_label(pts, '三阶top5%')} | {best_label(pts, '半空间')} | "
            f"{f1['round_share']:.1%} | {f1['cand_share']:.1%} | {d('二阶')} | {d('三阶均')} | {d('三阶top5%')} | {d('半空间')} | "
            f"{pct((bp['二阶'] - final['二阶']) / final['二阶'])} | {pct((f1['scores']['二阶'] - bp['二阶']) / bp['二阶'])} |"
        )
    lines.append("")
    return lines


def arms_table(ours: dict[str, list[dict]], gsd: dict[str, list[dict] | None]) -> list[str]:
    lines = [
        "## 两臂对照：地板停（×1.0 快照）与跑到收敛（终表），我们 ÷ GSD（<1 我们好）",
        "",
        "| ε | 层级 | " + " | ".join(f"{ax} 我们" for ax in MAIN) + " | " + " | ".join(f"{ax} GSD" for ax in MAIN)
        + " | " + " | ".join(f"{ax} 比" for ax in MAIN) + " |",
        "|---|---|" + "---|" * (3 * len(MAIN)),
    ]
    for tag in TAGS:
        if not gsd.get(tag):
            continue
        for label in ["地板×1.5", "地板×1.2", "地板×1", "终表"]:
            po, pg = find(ours[tag], label), find(gsd[tag], label)
            if po is None or pg is None:
                continue
            o, g = po["scores"], pg["scores"]
            lines.append(
                f"| {EPS[tag]} | {label if label != '地板×1' else '地板停 ×1.0'} | "
                + " | ".join(f"{o[ax]:.0f}" for ax in MAIN) + " | "
                + " | ".join(f"{g[ax]:.0f}" for ax in MAIN) + " | "
                + " | ".join(f"{o[ax] / g[ax]:.3f}" for ax in MAIN) + " |"
            )
    lines.append("")
    return lines


PNG = ROOT / "logs_eps" / "floor" / "adult_地板停点_2026-10-16.png"


def plot(ours: dict[str, list[dict]], gsd: dict[str, list[dict] | None]) -> None:
    """上行二阶、下行三阶均：横轴 损失/地板（对数、从高到低），两臂各一条；竖线 = 地板。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = ["二阶", "三阶均"]
    fig, axes = plt.subplots(len(rows), len(TAGS), figsize=(3.4 * len(TAGS), 3.1 * len(rows)), squeeze=False)
    for j, tag in enumerate(TAGS):
        for i, ax_name in enumerate(rows):
            ax = axes[i][j]
            for arm, pts, color in [("我们", ours[tag], "C0"), ("GSD", gsd.get(tag), "C3")]:
                if not pts:
                    continue
                sel = [p for p in pts if p["label"] != "起点"]
                x = [p["mult"] for p in sel]
                y = [p["scores"][ax_name] for p in sel]
                ax.plot(x, y, "o-", ms=3, lw=1, color=color, label=arm)
                ax.plot([pts[-1]["mult"]], [pts[-1]["scores"][ax_name]], "s", ms=6, color=color, mfc="none")
            ax.axvline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"ε = {EPS[tag]}")
            if i == len(rows) - 1:
                ax.set_xlabel("损失 / 噪声地板（→ 搜索方向）")
            if j == 0:
                ax.set_ylabel(f"{ax_name} 误差 ×10⁻⁶")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("adult 种子 1：真值误差随损失下降的形状（空心方块 = 跑到收敛的终表；虚线 = 地板）", fontsize=10)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    ours, gsd = {}, {}
    body: list[str] = []
    for tag in TAGS:
        pts, notes, tot = ours_points(tag, cache)
        ours[tag] = pts
        body += ours_table(tag, pts, notes, tot)
    for tag in TAGS:
        pts, notes, tot = gsd_points(tag, cache)
        gsd[tag] = pts
        if pts is not None:
            body += gsd_table(tag, pts, notes, tot)
    head = [
        "# 第六十二步：沿噪声地板倍数取快照、定停点（adult，种子 1，五档 ε，两臂）",
        "",
        CONCLUSION,
        "问题：第六十一步只在终损失 ×1.2～1.01 采样，最优停点（损失 ≈ 噪声地板）没采到。本步沿地板倍数 "
        "5 / 3 / 2 / 1.5 / 1.3 / 1.2 / 1.1 / 1.05 / 1 / 0.95 / 0.9 / 0.8 / 0.7 取快照（够不到的档不取），"
        "两臂同法，看真值误差在地板附近的形状、定停点；×1.0 快照即两臂各自加地板停会交出的表。",
        "",
        "口径：地板 = ½Σσ²（引擎 old_loss 口径）或 Σσ²（ℒ = Σ(y − c)² 口径），两者同一条线；损失/地板列两臂同尺。"
        "我们的快照由引擎在起点损失首次 ≤ 阈值时写出该轮起点表（轨迹逐位不变，第六十一步的终×倍数快照来自同一条轨迹、一并列入）；"
        "GSD 的快照由补丁在每 1000 代的记账点上最优 fitness 首次 ≤ 阈值时写出当时的最优表（复用已读的 fitness，不多同步），"
        "GSD 的 GPU 搜索非逐位确定、本场与第六十步是两条轨迹。误差单位 ×10⁻⁶（L1/N），括号内 Δ = (该层 − 终表)/终表，"
        "每轴最优层级加粗。轮/代/候选/秒占比以本臂全程为分母；GSD 净秒按第六十步同一规则扣记账突发。",
        "",
    ]
    tail = summary_ours(ours) + arms_table(ours, gsd)
    OUT.write_text("\n".join(head + tail + body), encoding="utf-8")
    plot(ours, gsd)
    print(f"written {OUT} and {PNG}")


if __name__ == "__main__":
    main()
