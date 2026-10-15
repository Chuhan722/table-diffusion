"""第六十步 第 4 步：搜索代价曲线与到达同一损失的代价对比（adult 1A，种子 1，五档 ε）。

共同损失 ℒ = Σ_{14123 格}(噪声目标 − 计数)²；我们 ℒ = 2·old_loss（--exam-weights none），GSD ℒ = N²·best_fitness。
横轴两种：演化/搜索循环墙钟秒；累计候选评估数（我们：每轮菜单条数或行级候选数；GSD：每代种群 50 条）。
产出：logs_eps/scores/effort_step60.md（表）与 汇报/adult_搜索代价_2026-10-15.png（图）。
用法：.venv/bin/python logs_eps/effort/effort_analyze.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TAGS = ["0p1", "0p3", "1", "3", "10"]
EPS = {"0p1": "0.1", "0p3": "0.3", "1": "1", "3": "3", "10": "10"}
SEED = 1
OUT_MD = REPO / "logs_eps/scores/effort_step60.md"
OUT_PNG = REPO.parent / "汇报" / "adult_搜索代价_2026-10-15.png"


# 结论随数字手写，重跑后若数字变动需同步改（判读见笔记第六十步）。
CONCLUSION = """## 结论（10-15 阅毕）

1. 两臂终损失相差 ≤0.5%（Δ −0.25% / +0.02% / +0.48% / +0.12% / −0.05%），"到共同终损失的代价"是公平门槛。
2. 到共同终损失的秒数之比（GSD÷我们）：原始墙钟 27.6× / 6.4× / 1.8× / 2.2× / 3.8×；扣掉 GSD 记账突发后 **10.7× / 2.5× / 0.7× / 0.9× / 1.5×**；
   再扣 GSD 从随机表降到我们起点的那段 8.8× / 2.1× / 0.5× / 0.8× / 1.4×。即 ε ≤ 0.3 我们真快，ε ≥ 1 净搜索时间持平或略慢；
   此前整场墙钟表里 ε ≥ 1 的 2.2～2.7× 主要是 GSD 的实现开销（每 32,561 代一次约 25 s 的设备标量回收，占其循环 62%）与随机起点。
3. 按候选评估数：ε=0.1 两家相当（0.88×），ε ≥ 0.3 GSD 少 5～20 倍（0.20× / 0.05× / 0.06× / 0.14×）；我们每条候选便宜 10.6～14.6 倍。
   时间优势全部来自每条候选的评估成本（整批 GPU 评估），不来自搜索步数。候选粒度不同（我们一条候选是一批行同移，GSD 一条候选改一行），候选数只说明搜索步数。
4. 代价集中在平台段：进入终值 1.1 倍以内后，我们每轮仍评估 4×10⁴～10⁵ 条候选、只移动 74 / 30 / 8.6 / 2.1 / 0.9 行（下降段 1,336 / 601 / 175 / 66 / 38 行），
   这段占总候选 86% / 91% / 88% / 80% / 53%、占总轮 97% / 95% / 89% / 80% / 54%；GSD 同段（门槛 1.1 倍→1 倍）用的候选 ε=0.1 与我们相当（0.85×），ε=0.3 / 1 / 3 / 10 少 6 / 32 / 17 / 5 倍。这是本步找到的问题，未动。
5. 全程行改动数：我们 0.20～0.30M（每行均 6～9 次）；GSD 每代至多收 1 个候选，代数 1.2M～3.3M 是其行改动次数上限（未记录接受数）。
"""


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ours_series(tag: str):
    rows = read(REPO / f"results/effort/adult_eps{tag}_ours2way_seed{SEED}_curve.csv")
    final = rows[-1]
    assert final["round"] == "final"
    body = rows[:-1]
    t = [0.0]
    loss = [2.0 * float(body[0]["old_loss"])]
    cand = [0]
    cum = 0
    modes: dict[str, int] = {}
    for k, r in enumerate(body):
        c = int(r["candidates"]) if r["candidates"] not in ("", "-1") else 0
        cum += c
        modes[r["mode"] or "?"] = modes.get(r["mode"] or "?", 0) + 1
        nxt = 2.0 * float(body[k + 1]["old_loss"]) if k + 1 < len(body) else 2.0 * float(final["old_loss"])
        t.append(float(r["elapsed_s"]))
        loss.append(nxt)
        cand.append(cum)
    return dict(t=np.array(t), loss=np.array(loss), cand=np.array(cand, dtype=float),
                rounds=len(body), modes=modes, stop=final["status"])


def gsd_series(tag: str):
    rows = read(REPO / f"results/effort/adult_eps{tag}_gsd_seed{SEED}_gens.csv")
    exam = json.load(open(REPO / f"data/adult_alloc{tag}base_s{SEED}/measured_2way.json", encoding="utf-8"))
    n = float(exam["record_count"])
    t = np.array([float(r["elapsed_s"]) for r in rows])
    loss = n * n * np.array([float(r["best_fitness"]) for r in rows])
    cand = np.array([float(r["cum_candidates"]) for r in rows])
    gens = int(rows[-1]["gen"])
    # 净搜索时间：GSD 每 32,561 代做一次 jnp.array(32,561 个设备标量) 的记账，突发 ~25 s（首窗另含 JIT 编译）；
    # 把 >10×中位数的记录间隔换成中位数，得到扣除突发后的时间轴。
    dt = np.diff(t)
    med = float(np.median(dt))
    burst = dt > 10 * med
    t_net = np.concatenate([[t[0] if t[0] <= 10 * med else med], np.where(burst, med, dt)]).cumsum()
    return dict(t=t, loss=loss, cand=cand, gens=gens, t_net=t_net, bursts=int(burst.sum()),
                burst_s=float(dt[burst].sum()), med_ms=1e3 * med)


def first_reach(s, thr: float):
    """第一次 ℒ ≤ thr 的 (秒, 累计候选)；相邻记录点间按线性插值；未达到返回 None。"""
    loss, t, c = s["loss"], s["t"], s["cand"]
    idx = np.nonzero(loss <= thr)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return t[0], c[0]
    l0, l1 = loss[i - 1], loss[i]
    f = (l0 - thr) / (l0 - l1) if l0 > l1 else 1.0
    return t[i - 1] + f * (t[i] - t[i - 1]), c[i - 1] + f * (c[i] - c[i - 1])


def launcher_wall() -> dict[str, float]:
    """发射日志 START→END 的整场分钟。"""
    out: dict[str, float] = {}
    log = REPO / "logs_eps/effort/launch_effort.log"
    if not log.exists():
        return out
    starts: dict[str, float] = {}
    from datetime import datetime
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+) (START|END) (\S+)", line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").timestamp()
        if m.group(2) == "START":
            starts[m.group(3)] = ts
        elif m.group(3) in starts:
            out[m.group(3)] = (ts - starts[m.group(3)]) / 60.0
    return out


def fmt_n(x: float) -> str:
    return f"{x:,.0f}"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    wall = launcher_wall()
    tags = [t for t in TAGS
            if (REPO / f"results/effort/adult_eps{t}_ours2way_seed{SEED}_curve.csv").exists()
            and (REPO / f"results/effort/adult_eps{t}_gsd_seed{SEED}_gens.csv").exists()]
    fig, axes = plt.subplots(2, len(tags), figsize=(3.4 * len(tags), 6.2), squeeze=False)
    md = ["# 第六十步 搜索代价对比：我们 vs GSD（adult 1A，种子 1，五档 ε）", "", CONCLUSION, "",
          "同一份 measured_2way.json，共同损失 ℒ = Σ(噪声目标 − 计数)²；我们 ℒ = 2·old_loss，GSD ℒ = N²·best_fitness（N = 32,561）。",
          "候选评估 = 一次“提出一个改动并给它打分”：我们每轮打分的菜单条数（行级模式为行数×(取值数−1)），GSD 每代 50 个种群成员。",
          "两家候选粒度不同（我们一条候选是一批行同移，GSD 一条候选是改一行），所以候选数只能说明搜索步数，不是算力。",
          "循环秒 = 演化/搜索循环内墙钟（我们从第 0 轮起算；GSD 从 fit 内循环起算，含首代 JIT 编译）。整场分钟 = 发射脚本 START→END。",
          "GSD 净秒 = 扣除其每 32,561 代一次的记账突发（原版代码把 32,561 个设备标量逐个拉回主机建数组，每次约 25 s，首窗另含 JIT 编译）后的循环秒，"
          "即把 >10×中位数的 1000 代间隔换成中位数；这是 GSD 代码的实现开销，不是搜索本身，另列以免把它算成我们的优势。",
          "", "## 一 总量", "",
          "起点不同：我们从洗牌平均初始化（--init-avg --init-clean）起算，GSD 从随机表起算（首代 ℒ 约为我们起点的 9 倍），"
          "所以第二节另给“GSD 自我们 ℒ₀ 起”一列——从 GSD 第一次降到我们起点损失之后再计秒与候选，把初始化的先手扣掉。",
          "",
          "| ε | 臂 | 起点 ℒ | 终 ℒ | 终/我们 ℒ₀ | 轮/代数 | 累计候选 | 循环秒 | 净秒 | 整场分钟 | 每候选 µs（净） |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    burst_rows = ["", "GSD 记账突发：", ""]
    reach_rows = ["", "## 二 到达共同终损失所需代价", "",
                  "门槛 = 两臂终 ℒ 中较高者（两臂都到得了，不偏谁）；终 Δ = 我们终 ℒ ÷ GSD 终 ℒ − 1。",
                  "“自 ℒ₀ 起”= 扣除 GSD 从随机表降到我们起点损失所花的秒与候选（把初始化先手扣掉）。", "",
                  "| ε | 门槛 ℒ | 终 Δ | 我们到达轮/总轮 | 我们秒 | GSD 秒 | GSD 净秒 | GSD 净秒（自 ℒ₀） | 我们候选 | GSD 候选 | GSD 候选（自 ℒ₀） |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
    ratio_rows = ["", "## 二′ 比值分解（GSD ÷ 我们）：秒之比 = 候选之比 × 每候选耗时之比", "",
                  "| ε | 秒之比（原） | 秒之比（净） | 秒之比（净，自 ℒ₀） | 候选之比 | 候选之比（自 ℒ₀） | 每候选耗时之比（净） |",
                  "|---|---|---|---|---|---|---|"]
    tail_rows = ["", "## 二″ 收尾段：到达门槛 ×m 所需（秒 / 候选；GSD 用净秒、不扣 ℒ₀）", "",
                 "两臂早停都是“相对改善 < 1e-4”，大头时间都花在最后几成的改善上；此表看谁在平台段更省。", "",
                 "| ε | m | 我们秒 | GSD 净秒 | 秒之比 | 我们候选 | GSD 候选 | 候选之比 |", "|---|---|---|---|---|---|---|---|"]
    level_rows = ["", "## 三 到达同一相对损失 ℒ/ℒ₀ 的代价（秒 / 候选；GSD 用净秒）", "",
                  "| ε | ℒ/ℒ₀ | 我们秒 | GSD 净秒 | 我们候选 | GSD 候选 |", "|---|---|---|---|---|---|"]
    for j, tag in enumerate(tags):
        o, g = ours_series(tag), gsd_series(tag)
        l0 = o["loss"][0]
        o["t_net"] = o["t"]
        for name, s, n_steps, key in (("我们", o, o["rounds"], f"adult_eps{tag}_ours2way_seed{SEED}"),
                                       ("GSD", g, g["gens"], f"adult_eps{tag}_gsd_seed{SEED}")):
            tt, tn, cc = s["t"][-1], s["t_net"][-1], s["cand"][-1]
            md.append(f"| {EPS[tag]} | {name} | {s['loss'][0]:.4g} | {s['loss'][-1]:.4g} | {s['loss'][-1] / l0:.4f} | {fmt_n(n_steps)} | "
                      f"{fmt_n(cc)} | {tt:,.0f} | {tn:,.0f} | {wall.get(key, float('nan')):.1f} | {1e6 * tn / cc:.1f} |")
        burst_rows.append(f"- ε={EPS[tag]}：{g['bursts']} 次突发共 {g['burst_s']:,.0f} s（占循环 {g['burst_s'] / g['t'][-1]:.0%}），"
                          f"正常每 1000 代 {g['med_ms']:.0f} ms。")
        gn = dict(g, t=g["t_net"])
        warm = first_reach(gn, l0) or (float("nan"), float("nan"))
        thr = max(o["loss"][-1], g["loss"][-1])
        delta = o["loss"][-1] / g["loss"][-1] - 1
        (to, co), (tg, _), (tgn, cg) = first_reach(o, thr), first_reach(g, thr), first_reach(gn, thr)
        k = int(np.nonzero(o["loss"] <= thr)[0][0])
        tgn0, cg0 = tgn - warm[0], cg - warm[1]
        cr = cg / co
        reach_rows.append(f"| {EPS[tag]} | {thr:.4g} | {delta:+.2%} | {k} / {o['rounds']} | {to:,.1f} | {tg:,.0f} | {tgn:,.0f} | {tgn0:,.0f} | "
                          f"{fmt_n(co)} | {fmt_n(cg)} | {fmt_n(cg0)} |")
        ratio_rows.append(f"| {EPS[tag]} | {tg / to:.1f}× | {tgn / to:.1f}× | {tgn0 / to:.1f}× | {cr:.2f}× | {cg0 / co:.2f}× | {(tgn / to) / cr:.1f}× |")
        for m in (2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.0):
            a, b = first_reach(o, m * thr), first_reach(gn, m * thr)
            if a is None or b is None:
                continue
            if a[0] <= 0:  # 门槛×m 高于我们起点：我们零代价即满足，比值无意义
                tail_rows.append(f"| {EPS[tag]} | {m:g} | 起点已满足 | {b[0]:,.0f} | — | 0 | {fmt_n(b[1])} | — |")
                continue
            tail_rows.append(f"| {EPS[tag]} | {m:g} | {a[0]:,.1f} | {b[0]:,.0f} | {b[0] / a[0]:.1f}× | {fmt_n(a[1])} | {fmt_n(b[1])} | {b[1] / a[1]:.2f}× |")
        floor = thr / l0
        for lv in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
            if lv < floor:
                break
            a, b = first_reach(o, lv * l0), first_reach(gn, lv * l0)
            if a is None or b is None:
                continue
            level_rows.append(f"| {EPS[tag]} | {lv:g} | {a[0]:,.1f} | {b[0]:,.0f} | {fmt_n(a[1])} | {fmt_n(b[1])} |")
        for i, (x_o, x_g, xl) in enumerate(((o["t"], g["t_net"], "循环墙钟（秒）"), (o["cand"], g["cand"], "累计候选评估数"))):
            ax = axes[i][j]
            if i == 0:
                ax.plot(np.maximum(g["t"], g["t"][g["t"] > 0].min()), g["loss"] / l0, color="#bbbbbb", lw=1.0, ls="--", label="GSD 原秒")
            ax.plot(np.maximum(x_g, x_g[x_g > 0].min() if (x_g > 0).any() else 1), g["loss"] / l0, color="#888888", lw=1.4, label="GSD" + ("（净秒）" if i == 0 else ""))
            ax.plot(np.maximum(x_o, x_o[x_o > 0].min()), o["loss"] / l0, color="#c0392b", lw=1.4, label="我们")
            ax.axhline(thr / l0, color="#888888", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"ε = {EPS[tag]}")
            if j == 0:
                ax.set_ylabel("ℒ / ℒ₀")
            ax.set_xlabel(xl)
            ax.grid(alpha=0.3, which="both", lw=0.4)
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle("adult 二阶工作负载：到达同一损失的搜索代价（种子 1，虚线 = 两臂共同终损失）", y=1.0)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    md += burst_rows + reach_rows + ratio_rows + tail_rows + level_rows + ["", f"图：{OUT_PNG.name}（上行横轴墙钟秒，下行横轴累计候选数，纵轴 ℒ/ℒ₀，双对数）。"]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n写入 {OUT_MD}\n图 {OUT_PNG}")


if __name__ == "__main__":
    main()
