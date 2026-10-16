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

1. **形状**：两臂五档 ε 的真值误差（二阶 L1）沿"损失/地板"都是一条谷：地板以上单调下降，谷底在地板附近，谷底以下回升到终表。
   我们的谷底（二阶）在地板 ×0.985 / 0.95 / 0.9 / 0.8 / 0.56（ε = 0.1 / 0.3 / 1 / 3 / 10），比终表好 34.6 / 33.0 / 21.9 / 11.6 / 4.0%；
   GSD 的谷底在 ×0.9 / 0.9 / 0.9 / 0.7 / 终表，比终表好 24.6 / 22.9 / 16.6 / 6.0 / 0%。ε 越大谷越浅、越靠下：
   ε = 10 两臂都是终表（或 0.56×）最好，地板停反而比终表差 4.0%（我们）/ 5.8%（GSD）。
   平方口径的真值误差 T 同样有谷、深度相同（我们 −35.5 / −34.5 / −19.5 / −10.7 / −4.6%），谷底略深（0.905 / 0.86 / 0.72 / 0.57 / 0.52×）。
2. **地板停（×1.0）对我们**：比终表好 33.8 / 32.0 / 21.3 / 9.1%，ε = 10 差 4.0%；离各档最优 1.3 / 1.6 / 0.8 / 2.9 / 8.4%；
   代价是全程候选的 9.2 / 3.2 / 2.5 / 3.9 / 16.2%（23 / 60 / 202 / 558 / 1754 轮，搜索循环内 10 / 10 / 22 / 93 / 121 秒，机器负载 7–49）。
   一条不带参数的规则（损失到 ½Σσ² 即停）在 ε ≤ 3 拿到几乎全部收益；ε = 10 是例外，需多看种子再定是否按 ε 调倍数。
3. **地板停对 GSD**：比终表好 8.0 / 12.8 / 14.9 / 5.2%，ε = 10 差 5.8%；但离其自身最优 22.0 / 13.2 / 2.0 / 0.9 / 5.8%
   （GSD 到地板时真值误差还在明显下降，谷底在 ×0.9 以下）；花全程代数的 19.9 / 20.0 / 20.5 / 14.8 / 13.5%（净 102–200 秒）。
4. **两臂同法对照（本步主结果）**：同一测量、同一噪声目标、同一停法下只换生成器。地板停处我们 ÷ GSD：
   二阶 0.729 / 0.780 / 0.916 / 0.954 / 0.982，三阶均 0.781 / 0.823 / 0.938 / 0.976 / 1.003，半空间 0.753 / 0.829 / 0.912 / 0.952 / 1.043，
   三阶top5% 0.978 / 0.899 / 0.906 / 0.934 / 1.012。各取自己二阶最优层级（上帝视角）：二阶 0.879 / 0.869 / 0.926 / 0.935 / 0.959，
   三阶均 0.895 / 0.887 / 0.946 / 0.971 / 0.999。跑到收敛的终表：二阶 1.013 / 1.000 / 0.990 / 0.995 / 0.999，各轴都在 ±6% 内（ε = 10 半空间 1.09 除外）。
   即：两个生成器收敛到同一处；差别全在路径上——同一损失水平下我们的表离真值更近，ε 越小差距越大。
5. **为什么**（分解 ℒ = ‖e‖² + T − 2⟨d,e⟩，e 为注入噪声、d 为表与真值之差）：到地板时两臂 T − 2⟨d,e⟩ 必然相等，
   但我们的 T = 0.070 / 0.151 / 0.323 / 0.544 / 0.752，GSD 为 0.093 / 0.200 / 0.395 / 0.663 / 0.878（低 25 / 25 / 18 / 18 / 14%）——
   GSD 从起点降到地板的同样一段损失里，多花了一截在追噪声上；终表处两臂 T 与追噪项都相同（0.072 / 0.071 … 0.449 / 0.451）。
   地板处追噪项占 ℒ 的份额随 ε 上升（我们 6% → 74%），而 T 的最低点随 ε 向地板以下移动：σ 小时到地板还有大块结构误差没消、残差仍含信号，
   所以继续往下仍在改善真值，这是 ε = 10 例外的来源。
6. **到地板的代价**：我们 9.6e6 / 1.1e7 / 2.4e7 / 6.1e7 / 1.9e8 候选、10 / 10 / 22 / 93 / 121 秒；GSD 1.2e7 / 1.4e7 / 1.6e7 / 1.7e7 / 2.1e7 候选、净 102 / 152 / 163 / 139 / 200 秒。
   ε ≥ 1 我们的候选数更多（单个候选便宜、融合核），墙上时间 ε ≤ 1 快 7–15 倍、ε ≥ 3 快 1.5–1.7 倍（我们这边负载很重，时间偏保守）。
7. **其他观察**：ε = 10 地板以上（×5～×2）GSD 比我们好（×5 处 15049 对 18500），×1.5 处交叉，地板以下我们略好；
   三阶top5% 与半空间在 ε = 0.1 / 10 两端两臂互有胜负（这两轴本身噪）。
8. **核对**：我们 5 张终表与第六十步逐位同、曲线除时间列逐列同、56 张快照轮号 = 曲线首穿越轮、快照 ℒ = 2·记账损失；
   GSD 62 张快照与 5 张终表 ℒ = N²·记账 fitness（终表相对差 ≤ 3e-7），终 ℒ 与第六十步之比 0.9978–1.0000（非逐位确定）。
9. **局限**：单种子；GSD 每 1000 代才检查一次阈值（×1.0 实际落在 0.992–0.998）；GSD 在 ε = 0.1 的网格只到 ×0.9（终 0.825，谷底位置未分辨）；
   阅卷轴为 L1，T 为平方口径，两者谷深一致但位置略有差异。
10. **下一步建议**：把"损失到噪声地板即停"作为我们的默认停法（无参数、ε ≤ 3 近最优），两臂三种子 × 五档 ε 同法复跑以确认第 4 条的比值；
    ε = 10 的倍数问题等多种子后再定；汇报的收敛终表数字按地板停改写。
"""


# ---------- 读取 ----------
def ours_name(tag: str, seed: int = 1) -> str:
    return f"adult_eps{tag}_ours2way_seed{seed}"


def gsd_name(tag: str, seed: int = 1) -> str:
    return f"adult_eps{tag}_gsd_seed{seed}"


def read_snaps(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


_EXAM: dict[tuple[str, int], tuple] = {}
_COUNTS: dict[tuple[str, str, int], np.ndarray] = {}


def exam(tag: str, seed: int = 1):
    """该 ε、该种子的噪声考卷（查询对顺序 + 噪声目标）。"""
    key = (tag, seed)
    if key not in _EXAM:
        _, pairs, tgt = load_exam(ROOT / f"data/adult_alloc{tag}base_s{seed}/measured_2way.json")
        _EXAM[key] = (pairs, tgt)
    return _EXAM[key]


def counts(path: Path, tag: str, seed: int = 1) -> np.ndarray:
    """表在该 ε、该种子噪声考卷查询顺序下的格计数 c（按路径缓存，核对与分解共用）。"""
    key = (str(path), tag, seed)
    if key not in _COUNTS:
        cols, rows = load_codes(path)
        _COUNTS[key] = cell_counts(cols, rows, exam(tag, seed)[0])
    return _COUNTS[key]


def table_loss(path: Path, tag: str, seed: int = 1) -> float:
    """表对噪声目标的 ℒ = Σ(y − c)²（与 align_loss 同法）。"""
    return float(np.sum((exam(tag, seed)[1] - counts(path, tag, seed)) ** 2))


def decompose(path: Path, tag: str, seed: int = 1) -> dict:
    """ℒ = ‖e‖² + T − 2⟨d, e⟩：e = y − c*（注入的噪声），d = c − c*，T = ‖d‖²（对真值的平方误差），
    2⟨d, e⟩ = 表与噪声同向的部分（追噪项）。全部除以 Σσ²（ℒ 口径地板）。"""
    pairs, y = exam(tag, seed)
    c_true = counts(ROOT / "data/adult/adult.csv", tag, seed)
    c = counts(path, tag, seed)
    e, d = y - c_true, c - c_true
    s2 = 2.0 * noise_floor(tag)
    return dict(e2=float(e @ e) / s2, T=float(d @ d) / s2, cross=2.0 * float(d @ e) / s2)


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
def ours_points(tag: str, cache: dict, seed: int = 1) -> tuple[list[dict], list[str], dict]:
    """返回 [点]（按损失从高到低）、核对说明、全程量。点 = dict(label, mult, round, cand_share, round_share, t_share, scores, path)。
    种子 1 另与第六十步终表/曲线逐位核对、并入第六十一步的终×快照与起点表；其他种子只有本场的地板×快照与终表。"""
    notes = []
    name = ours_name(tag, seed)
    floor = noise_floor(tag)
    if seed == 1:
        same = filecmp.cmp(EFFORT / f"{name}.csv", FLOOR / f"{name}.csv", shallow=False)
        notes.append("终表与第六十步逐位一致" if same else "终表不一致！")
        a, b = read_curve(EFFORT / f"{name}_curve.csv"), read_curve(FLOOR / f"{name}_curve.csv")
        cols = [c for c in a[0].keys() if c != "elapsed_s"]
        ok = len(a) == len(b) and all(ra[c] == rb[c] for ra, rb in zip(a, b) for c in cols)
        notes.append("曲线除时间列外逐列一致" if ok else "曲线不一致！")
    else:
        a = read_curve(FLOOR / f"{name}_curve.csv")
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
            t_share=(tsec[k - 1] if k > 0 else 0.0) / total_t, loss=loss, path=path, seed=seed,
        ))

    if seed == 1:
        init_snaps = read_snaps(INIT / f"{name}_snaps.csv")
        add("起点", INIT / f"{name}_snap0.csv", 0, float(init_snaps[0]["old_loss"]))
    bad_round, bad_loss = 0, 0.0
    sources = [("地板×", FLOOR / f"{name}_snaps.csv", FLOOR / name)]
    if seed == 1:
        sources.append(("终×", PLATEAU / f"{name}_snaps.csv", PLATEAU / name))
    for src, snaps_path, pref in sources:
        for r in read_snaps(snaps_path):
            k, thr, loss = int(r["round"]), float(r["threshold"]), float(r["old_loss"])
            pred = int(np.argmax(losses <= thr))
            if pred != k or losses[k] != loss:
                bad_round += 1
            p = Path(f"{pref}_snap{r['index']}.csv")
            if abs(table_loss(p, tag, seed) - 2 * loss) > 1e-6 * 2 * loss:
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


def gsd_points(tag: str, cache: dict, seed: int = 1):
    name = gsd_name(tag, seed)
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
    final_l = table_loss(FLOOR / f"{name}.csv", tag, seed)
    if seed == 1:
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
            loss=loss_l, path=path, seed=seed,
        ))

    bad = 0
    snaps_path = FLOOR / f"{name}_snaps.csv"
    if snaps_path.exists():
        for r in read_snaps(snaps_path):
            gen, thr, bf = int(r["gen"]), float(r["threshold"]), float(r["best_fitness"])
            idx = int(np.where(g == gen)[0][0])
            p = FLOOR / f"{name}_snap{r['index']}.csv"
            tl = table_loss(p, tag, seed)
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
    return f"{v * 1e6:.0f} ({pct((v - base) / base)})"


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
            c = cell(p["scores"][ax], final[ax]) if p["label"] != "终表" else f"{final[ax] * 1e6:.0f}"
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
            c = cell(p["scores"][ax], final[ax]) if p["label"] != "终表" else f"{final[ax] * 1e6:.0f}"
            if bests[ax] == p["label"]:
                c = f"**{c}**"
            cells.append(c)
        lines.append(
            f"| {p['label']} | {p['mult']:.3f} | {p['gen']:,} | {p['gen_share']:.1%} | {p['cand_share']:.1%} | "
            f"{p['tnet']:.0f} | {p['tnet_share']:.1%} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def decomp_table(arm: str, tag: str, pts: list[dict]) -> list[str]:
    def dec(p: dict) -> dict:
        return decompose(p["path"], tag, p.get("seed", 1))

    final_T = dec(pts[-1])["T"]
    lines = [
        f"## 分解 {arm} ε = {EPS[tag]}：ℒ = ‖e‖² + T − 2⟨d,e⟩（均除以 Σσ²；"
        f"‖e‖²/Σσ² = {dec(pts[-1])['e2']:.3f}）",
        "",
        "| 层级 | 损失/地板 | T（真值平方误差） | 2⟨d,e⟩（追噪项） | 追噪项/ℒ | T Δ终 |",
        "|---|---|---|---|---|---|",
    ]
    best = min(pts, key=lambda p: dec(p)["T"])["label"]
    for p in pts:
        dc = dec(p)
        t_cell = f"{dc['T']:.3f}"
        if p["label"] == best:
            t_cell = f"**{t_cell}**"
        lines.append(
            f"| {p['label']} | {p['mult']:.3f} | {t_cell} | {dc['cross']:.3f} | {dc['cross'] / p['mult']:.1%} | "
            f"{pct((dc['T'] - final_T) / final_T)} |"
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


def summary_gsd(all_pts: dict[str, list[dict] | None]) -> list[str]:
    lines = [
        "## 汇总（GSD）：最优停点在哪、地板停（×1.0）与跑到收敛（终表）差多少",
        "",
        "| ε | 二阶最优层级 | 三阶均最优 | 三阶top5%最优 | 半空间最优 | ×1.0 代占比 | ×1.0 净秒占比 | ×1.0 二阶 Δ终 | ×1.0 三阶均 Δ终 | ×1.0 三阶top5% Δ终 | ×1.0 半空间 Δ终 | 二阶最优 Δ终 | ×1.0 二阶 Δ最优 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        pts = all_pts.get(tag)
        if not pts:
            continue
        final = pts[-1]["scores"]
        f1 = find(pts, "地板×1")
        bl = best_label(pts, "二阶")
        bp = find(pts, bl)["scores"]
        d = lambda ax: pct((f1["scores"][ax] - final[ax]) / final[ax])  # noqa: E731
        lines.append(
            f"| {EPS[tag]} | {bl} | {best_label(pts, '三阶均')} | {best_label(pts, '三阶top5%')} | {best_label(pts, '半空间')} | "
            f"{f1['gen_share']:.1%} | {f1['tnet_share']:.1%} | {d('二阶')} | {d('三阶均')} | {d('三阶top5%')} | {d('半空间')} | "
            f"{pct((bp['二阶'] - final['二阶']) / final['二阶'])} | {pct((f1['scores']['二阶'] - bp['二阶']) / bp['二阶'])} |"
        )
    lines.append("")
    return lines


def arms_cost_decomp(ours: dict[str, list[dict]], gsd: dict[str, list[dict] | None],
                     tot_o: dict[str, dict], tot_g: dict[str, dict | None]) -> list[str]:
    lines = [
        "## 两臂到地板的代价与分解：×1.0 快照处各自花了多少、真值平方误差 T 与追噪项 2⟨d,e⟩ 各多少（均除以 Σσ²）",
        "",
        "| ε | 我们 轮 | 我们 候选 | 我们 秒（搜索循环内） | GSD 代 | GSD 候选 | GSD 净秒（搜索循环内） | T 我们 ×1.0 | T GSD ×1.0 | 追噪 我们 ×1.0 | 追噪 GSD ×1.0 | T 我们 终表 | T GSD 终表 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        if not gsd.get(tag):
            continue
        po, pg = find(ours[tag], "地板×1"), find(gsd[tag], "地板×1")
        do, dg = decompose(po["path"], tag), decompose(pg["path"], tag)
        fo, fg = decompose(ours[tag][-1]["path"], tag), decompose(gsd[tag][-1]["path"], tag)
        to, tg = tot_o[tag], tot_g[tag]
        lines.append(
            f"| {EPS[tag]} | {po['round']} | {po['cand_share'] * to['cand']:.2g} | {po['t_share'] * to['t']:.0f} | "
            f"{pg['gen']:,} | {pg['cand_share'] * tg['cand']:.2g} | {pg['tnet']:.0f} | "
            f"{do['T']:.3f} | {dg['T']:.3f} | {do['cross']:.3f} | {dg['cross']:.3f} | {fo['T']:.3f} | {fg['T']:.3f} |"
        )
    lines.append("")
    return lines


def arms_table(ours: dict[str, list[dict]], gsd: dict[str, list[dict] | None]) -> list[str]:
    lines = [
        "## 两臂对照：地板停（×1.0 快照）、各自二阶最优层级、跑到收敛（终表），我们 ÷ GSD（<1 我们好）",
        "",
        "| ε | 层级 | " + " | ".join(f"{ax} 我们" for ax in MAIN) + " | " + " | ".join(f"{ax} GSD" for ax in MAIN)
        + " | " + " | ".join(f"{ax} 比" for ax in MAIN) + " |",
        "|---|---|" + "---|" * (3 * len(MAIN)),
    ]
    for tag in TAGS:
        if not gsd.get(tag):
            continue
        bo, bg = best_label(ours[tag], "二阶"), best_label(gsd[tag], "二阶")
        for label in ["地板×1.5", "地板×1.2", "地板×1", "各自最优", "终表"]:
            if label == "各自最优":
                po, pg, shown = find(ours[tag], bo), find(gsd[tag], bg), f"各自二阶最优（我们 {bo} / GSD {bg}）"
            else:
                po, pg, shown = find(ours[tag], label), find(gsd[tag], label), (label if label != "地板×1" else "地板停 ×1.0")
            if po is None or pg is None:
                continue
            o, g = po["scores"], pg["scores"]
            lines.append(
                f"| {EPS[tag]} | {shown} | "
                + " | ".join(f"{o[ax] * 1e6:.0f}" for ax in MAIN) + " | "
                + " | ".join(f"{g[ax] * 1e6:.0f}" for ax in MAIN) + " | "
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
    import matplotlib.ticker
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
                y = [p["scores"][ax_name] * 1e6 for p in sel]
                ax.plot(x, y, "o-", ms=3, lw=1, color=color, label=arm)
                ax.plot([pts[-1]["mult"]], [pts[-1]["scores"][ax_name] * 1e6], "s", ms=6, color=color, mfc="none")
            ax.axvline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_xscale("log")
            xs = [p["mult"] for p in ours[tag] if p["label"] != "起点"]
            ticks = [t for t in (5, 3, 2, 1.5, 1.2, 1, 0.9, 0.8, 0.7, 0.6, 0.5) if min(xs) * 0.98 <= t <= max(xs) * 1.02]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            lo, hi = min(xs) * 0.93, max(xs) * 1.08
            ax.set_xlim(hi, lo)
            ys = [p["scores"][ax_name] * 1e6 for arm_pts in (ours[tag], gsd.get(tag) or [])
                  for p in arm_pts if p["label"] != "起点" and lo <= p["mult"] <= hi]
            pad = 0.05 * (max(ys) - min(ys))
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
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
    ours, gsd, tot_o, tot_g = {}, {}, {}, {}
    body: list[str] = []
    for tag in TAGS:
        pts, notes, tot = ours_points(tag, cache)
        ours[tag], tot_o[tag] = pts, tot
        body += ours_table(tag, pts, notes, tot) + decomp_table("我们", tag, pts)
    for tag in TAGS:
        pts, notes, tot = gsd_points(tag, cache)
        gsd[tag], tot_g[tag] = pts, tot
        if pts is not None:
            body += gsd_table(tag, pts, notes, tot) + decomp_table("GSD", tag, pts)
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
    tail = summary_ours(ours) + summary_gsd(gsd) + arms_table(ours, gsd) + arms_cost_decomp(ours, gsd, tot_o, tot_g)
    OUT.write_text("\n".join(head + tail + body), encoding="utf-8")
    plot(ours, gsd)
    print(f"written {OUT} and {PNG}")


if __name__ == "__main__":
    main()
