"""第六十一步：平台段价值量化（adult 1A，种子 1，五档 ε）。

读 results/plateau 的快照触发记录与快照表，先核对终表与第六十步（results/effort）逐位一致、
曲线除 elapsed_s 外逐列一致，再用 scripts/score_any.py（阅卷同一把尺）给
终表与五张快照（终损失 ×1.2/1.1/1.05/1.02/1.01 时刻的起点表）阅卷，
按 ε 列出：到达该层级时已用轮数/候选数/循环秒数占全程比例，以及各轴误差与终表的相对差。
轮数、候选数、秒数一律取第六十步曲线（时间口径与搜索代价记录同一份）。
用法：.venv/bin/python logs_eps/plateau/plateau_analyze.py
输出：logs_eps/scores/plateau_step61.md
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "effort"))
from effort_analyze import first_reach, gsd_series  # noqa: E402  第六十步的 GSD 时间轴（含净秒规则）

CONCLUSION = """## 结论（种子 1，五档 ε；数字见下表）

1. **尾段不是白算，是在追噪。** 五档 ε 的终损失都压到了噪声地板以下（地板 × 0.82 / 0.72 / 0.60 / 0.52 / 0.47，ε = 0.1 → 10）。损失一过地板，各轴对真表的误差掉头变差：终表二阶误差比 ×1.2 快照差 35% / 30% / 17% / 10% / 4%，measured、三阶均、半空间同向；三阶 top5% / max 在 ε = 10 持平（±0.7%）。
2. **砍掉尾段既省算力又提质量。** ×1.2 快照只用了全程 1% / 2% / 6% / 12% / 34% 的轮数、10% / 5% / 6% / 12% / 34% 的候选。
3. **尾段之前的搜索是有价值的。** 起点表（洗牌平均 + 清洗）到 ×1.2 快照，二阶误差相对终表从 +2% / +43% / +186% / +516% / +1408% 降到 −35% / −30% / −17% / −10% / −4%；ε = 0.1 的起点表二阶已与终表相当（+2%），但三阶 top5% / 半空间差 34% / 46%，搜索前 28 轮把它们补上。
4. **最优停点未被采到。** 只有 ε = 0.1 的 ×1.2 快照恰在地板上（0.99×），其余 ε 的 ×1.2 快照已在地板下（0.86 / 0.72 / 0.62 / 0.56×），真值误差的最优点应更早、在地板附近。按偏差原理，"损失 = 噪声地板"是只用公开参数（σ）的现成停止规则；引擎已有硬地板停，1A 配置未开。
5. **对第六十步的影响。** 搜索代价对比按"到同一终损失"算，两家终损失都在地板下，比的是追噪速度。按"到地板"算（下表）：我们 23 / 60 / 202 / 558 / 1,754 轮、循环内 10 / 10 / 22 / 94 / 121 秒即到（另有约 20 秒装载与初始化）；GSD 25～42 万代、原始 326～479 秒、净 134～181 秒，净秒我们快 12.8× / 17.5× / 7.6× / 1.7× / 1.5×。候选数 ε ≤ 0.3 我们更少（0.76× / 0.78×），ε = 1 / 3 / 10 多 1.5× / 3.6× / 9×（第六十步到终损失口径是 5～20×）。GSD 同样追噪、终表同在地板下；地板停对两家都可用，属停止规则贡献，不记入生成器。
6. **下一步（未开始）：** 沿地板倍数取快照（×2 / ×1.5 / ×1.2 / ×1 / ×0.8）看真值误差在地板附近的形状、定停点；再给 1A 两臂都加地板停重算。
"""
EFFORT = ROOT / "results" / "effort"
PLATEAU = ROOT / "results" / "plateau"
INIT = PLATEAU / "init"  # 第 0 轮起点表（洗牌平均 + 清洗），CPU 单轮跑用 1e18 阈值快照取得
DATA = ROOT / "data"
OUT = ROOT / "logs_eps" / "scores" / "plateau_step61.md"
PY = ROOT / ".venv" / "bin" / "python"
SCORE = ROOT / "scripts" / "score_any.py"
TAGS = ["0p1", "0p3", "1", "3", "10"]
EPS = {"0p1": "0.1", "0p3": "0.3", "1": "1", "3": "3", "10": "10"}
LEVELS = ["1.2", "1.1", "1.05", "1.02", "1.01"]
AXES = ["measured", "二阶", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]


def name(tag: str) -> str:
    return f"adult_eps{tag}_ours2way_seed1"


def noise_floor(tag: str) -> float:
    """噪声地板（引擎 old_loss 口径 = ½Σσ²）：真表对带噪目标的期望损失，低于它的下降全是追噪。"""
    q = json.load(open(DATA / f"adult_alloc{tag}base_s1" / "measured_2way.json", encoding="utf-8"))
    qs = q["queries"] if isinstance(q, dict) and "queries" in q else q
    return 0.5 * sum(float(x["sigma"]) ** 2 for x in qs)


def read_curve(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [r for r in rows if r["round"] != "final"]


def verify(tag: str) -> list[str]:
    """终表逐位一致 + 曲线除 elapsed_s 外逐列一致。"""
    notes = []
    same = filecmp.cmp(EFFORT / f"{name(tag)}.csv", PLATEAU / f"{name(tag)}.csv", shallow=False)
    notes.append("终表逐位一致" if same else "终表不一致！")
    a = read_curve(EFFORT / f"{name(tag)}_curve.csv")
    b = read_curve(PLATEAU / f"{name(tag)}_curve.csv")
    cols = [c for c in a[0].keys() if c != "elapsed_s"]
    ok = len(a) == len(b) and all(ra[c] == rb[c] for ra, rb in zip(a, b) for c in cols)
    notes.append("曲线除时间列外逐列一致" if ok else "曲线不一致！")
    return notes


def score(tag: str) -> dict[str, dict[str, float]]:
    """调用 score_any.py，终表在前作基表，后接五张快照；返回 {表名: {轴: 值}}。"""
    tables = [PLATEAU / f"{name(tag)}.csv"] + [
        PLATEAU / f"{name(tag)}_snap{i}.csv" for i in range(len(LEVELS))
    ]
    init_table = INIT / f"{name(tag)}_snap0.csv"
    if init_table.exists():
        tables.append(init_table)
    out = subprocess.run(
        [str(PY), str(SCORE), "adult", *map(str, tables)],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout.splitlines()
    res = {}
    for i, line in enumerate(out[1:]):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        vals = list(map(float, parts[1:8]))
        key = parts[0]
        if i == len(tables) - 1 and init_table.exists():
            key = "init"  # 起点表与快照 0 同名，按位置改名
        res[key] = dict(zip(AXES, vals))
    return res


def floor_reach_table() -> list[str]:
    """两臂到噪声地板的代价：我们按曲线（轮 / 候选 / 循环秒），GSD 按代数日志（代 / 候选 / 原始秒与净秒，线性插值）。"""
    out = [
        "## 到噪声地板的代价（两臂，种子 1；ℒ = Σ(y − c)² 口径，地板 = Σσ²）",
        "",
        "| ε | 我们：轮 / 全程 | 候选 / 全程 | 循环秒 / 全程 | GSD：代 / 全程 | 候选 / 全程 | 原始秒 / 全程 | 净秒 / 全程 | 候选比 GSD÷我们 | 净秒比 GSD÷我们 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        floor_l = 2.0 * noise_floor(tag)
        curve = read_curve(EFFORT / f"{name(tag)}_curve.csv")
        cum = 0
        hit = None
        for r in curve:
            cum += max(0, int(r["candidates"]))
            if hit is None and 2.0 * float(r["old_loss"]) <= floor_l:
                hit = (int(r["round"]), cum, float(r["elapsed_s"]))
        tot = (len(curve), cum, float(curve[-1]["elapsed_s"]))
        g = gsd_series(tag)
        raw = first_reach(g, floor_l)
        net = first_reach(dict(g, t=g["t_net"]), floor_l)
        gen_hit = int(np.interp(raw[1], g["cand"], np.arange(len(g["cand"])) * 1000)) if raw else None
        out.append(
            f"| {EPS[tag]} | {hit[0]} / {tot[0]} | {hit[1]:.2g} / {tot[1]:.2g} | {hit[2]:.0f} / {tot[2]:.0f} | "
            f"{gen_hit:,} / {g['gens']:,} | {raw[1]:.2g} / {g['cand'][-1]:.2g} | {raw[0]:.0f} / {g['t'][-1]:.0f} | "
            f"{net[0]:.0f} / {g['t_net'][-1]:.0f} | {raw[1] / hit[1]:.2f}× | {net[0] / hit[2]:.1f}× |"
        )
    out.append("")
    return out


def main() -> None:
    lines = [
        "# 第六十一步：平台段价值量化（adult 1A，种子 1）",
        "",
        CONCLUSION,
        "问题：第六十步发现进入终损失 1.1 倍以内后的尾段占总候选数 53%～91%，每轮只挪 1～70 行。"
        "本步量这段尾巴对质量有没有贡献：在终损失 ×1.2 / 1.1 / 1.05 / 1.02 / 1.01 时刻各存一张起点表，与终表同尺阅卷。",
        "",
        "口径：快照由引擎在起点损失首次不高于阈值时写出（纯记账，轨迹逐位不变，终表与第六十步逐位一致）；"
        "阈值取第六十步该 ε 终损失的倍数；轮数 / 候选数 / 秒数的占比取第六十步曲线（与搜索代价记录同一份时间口径），"
        "秒数为演化循环内累计秒；误差单位 ×10⁻⁶（L1/N），Δ = (快照 − 终表)/终表。"
        "起点表 = 第 0 轮的表（洗牌平均 + 清洗），单独由 CPU 单轮跑快照取得，初始损失与 GPU 曲线第 0 轮一致。"
        "噪声地板 = ½Σσ²（引擎口径），即真表对这份带噪目标的期望损失；损失低于地板后的下降在期望意义上全是追噪。",
        "",
    ]
    summary_rows = []
    for tag in TAGS:
        eps = EPS[tag]
        notes = verify(tag)
        curve = read_curve(EFFORT / f"{name(tag)}_curve.csv")
        total_rounds = len(curve)
        cand = [max(0, int(r["candidates"])) for r in curve]
        cum_cand = []
        s = 0
        for c in cand:
            s += c
            cum_cand.append(s)
        total_cand = cum_cand[-1]
        total_time = float(curve[-1]["elapsed_s"])
        final_loss = float(curve[-1]["old_loss"])
        snaps = {
            int(r["index"]): r
            for r in csv.DictReader(open(PLATEAU / f"{name(tag)}_snaps.csv", encoding="utf-8"))
        }
        sc = score(tag)
        fin = sc[name(tag)]
        floor = noise_floor(tag)
        lines.append(
            f"## ε = {eps}（{'；'.join(notes)}；全程 {total_rounds} 轮、{total_cand:.3g} 候选、{total_time:.0f} 秒；"
            f"噪声地板 {floor:.4g}，终损失 = 地板 × {final_loss / floor:.2f}）"
        )
        lines.append("")
        lines.append("| 层级 | 轮 | 轮数占比 | 候选占比 | 秒数占比 | 损失/终 | 损失/地板 | " + " | ".join(AXES) + " |")
        lines.append("|---|---|---|---|---|---|---|" + "---|" * len(AXES))
        if "init" in sc:
            row = sc["init"]
            init_loss = float(curve[0]["old_loss"])
            cells = [f"{row[ax] * 1e6:.0f} ({(row[ax] - fin[ax]) / fin[ax] * 100.0:+.1f}%)" for ax in AXES]
            lines.append(
                f"| 起点表 | 0 | 0% | 0% | 0% | {init_loss / final_loss:.4f} | {init_loss / floor:.2f} | "
                + " | ".join(cells) + " |"
            )
            summary_rows.append((eps, "起点", 0.0, 0.0,
                                 (row["二阶"] - fin["二阶"]) / fin["二阶"] * 100.0,
                                 (row["measured"] - fin["measured"]) / fin["measured"] * 100.0,
                                 (row["三阶均"] - fin["三阶均"]) / fin["三阶均"] * 100.0,
                                 (row["半空间"] - fin["半空间"]) / fin["半空间"] * 100.0))
        for i, lv in enumerate(LEVELS):
            if i not in snaps:
                lines.append(f"| ×{lv} | 未触发 | | | | | | " + " | ".join("" for _ in AXES) + " |")
                continue
            k = int(snaps[i]["round"])
            loss = float(snaps[i]["old_loss"])
            el = float(curve[k]["elapsed_s"])
            row = sc[f"{name(tag)}_snap{i}"]
            cells = []
            for ax in AXES:
                d = (row[ax] - fin[ax]) / fin[ax] * 100.0
                cells.append(f"{row[ax] * 1e6:.0f} ({d:+.1f}%)")
            lines.append(
                f"| ×{lv} | {k} | {k / total_rounds:.1%} | {cum_cand[k - 1] / total_cand if k > 0 else 0:.1%} | "
                f"{el / total_time:.1%} | {loss / final_loss:.4f} | {loss / floor:.2f} | " + " | ".join(cells) + " |"
            )
            summary_rows.append((eps, lv, k / total_rounds, cum_cand[k - 1] / total_cand if k > 0 else 0.0,
                                 (row["二阶"] - fin["二阶"]) / fin["二阶"] * 100.0,
                                 (row["measured"] - fin["measured"]) / fin["measured"] * 100.0,
                                 (row["三阶均"] - fin["三阶均"]) / fin["三阶均"] * 100.0,
                                 (row["半空间"] - fin["半空间"]) / fin["半空间"] * 100.0))
        lines.append(
            f"| 终表 | {total_rounds} | 100% | 100% | 100% | 1.0000 | {final_loss / floor:.2f} | "
            + " | ".join(f"{fin[ax] * 1e6:.0f}" for ax in AXES) + " |"
        )
        lines.append("")
    lines.extend(floor_reach_table())
    lines.append("## 汇总：砍掉尾段省多少、二阶误差变多少")
    lines.append("")
    lines.append("| ε | 层级 | 已用轮数 | 已用候选 | 二阶 Δ | measured Δ | 三阶均 Δ | 半空间 Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for eps, lv, fr, fc, d2, dm, d3, dh in summary_rows:
        tag_lv = lv if lv == "起点" else f"×{lv}"
        lines.append(f"| {eps} | {tag_lv} | {fr:.0%} | {fc:.0%} | {d2:+.1f}% | {dm:+.1f}% | {d3:+.1f}% | {dh:+.1f}% |")
    lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
