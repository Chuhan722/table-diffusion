#!/usr/bin/env python
"""第七十二步 探针并行自动停：阅卷（只核对逐位同与计时，不重新算误差）。

用法：grade_autostop.py [变体名]   （变体名空 → results/autostop、logs_eps/autostop；给 pilot1 → results/autostop_pilot1 …）
每格核对：
  判停法 = 第七十一步按原卷真值定下的停法（1A ε≤3 地板停、1A ε=10 收敛停、1B 全收敛停）；
  P 的 G 快照与终表 = results/fission 探针臂逐位同；F 的 G 快照与终表 = 现有地板停 / 收敛停臂逐位同；交出的表 = 所判停法该臂的表。
计时：并行墙钟 = 两进程启动到交表（含载入、编译、判定）；探针到判定 = tD − t0；
      顺序估计 = 探针单跑循环内秒（第七十一步，重负载下测）+ 所判停法臂循环内秒（1A 地板停 = ×1.0 快照时刻；1B/收敛停 = 三段之和），不含载入。
输出：logs_eps/scores/autostop_times[_变体].md
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
TAGS = ["0p1", "0p3", "1", "3", "10"]
EPS = {"0p1": "0.1", "0p3": "0.3", "1": "1", "3": "3", "10": "10"}
GSD_SEC = {"0p1": 554, "0p3": 601, "1": 713, "3": 1031, "10": 1492}  # 汇报 §3.1：GSD 跑到底，扣记账突发
FLOORS = {(d["exam"], d["tag"], d["seed"]): d["floor_source"]
          for d in json.load(open(ROOT / "results/fission/floors.json", encoding="utf-8"))}


def expect(exam: str, tag: str) -> str:
    return "floor" if exam == "1A" and tag != "10" else "conv"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "missing"


def same(a: Path, b: Path) -> str:
    return "同" if a.exists() and b.exists() and md5(a) == md5(b) else ("缺" if not (a.exists() and b.exists()) else "异")


def last_elapsed(curve: Path) -> float | None:
    if not curve.exists():
        return None
    val = None
    for r in csv.DictReader(open(curve, encoding="utf-8")):
        if r.get("elapsed_s"):
            val = float(r["elapsed_s"])
    return val


def stage_secs(log: Path) -> dict:
    """从引擎日志抽三段循环内秒：第一段（最终损失…耗时）、纤维洗牌、第二段。"""
    out = {}
    if not log.exists():
        return out
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"耗时 ([\d.]+) 秒", line)
        if not m:
            continue
        if line.startswith("第二段"):
            out["s2"] = float(m.group(1))
        elif line.startswith("纤维洗牌"):
            out["pool"] = float(m.group(1))
        elif line.startswith("最终损失") and "s1" not in out:
            out["s1"] = float(m.group(1))
    return out


def snap_x1(tag: str, seed: int) -> tuple[Path | None, float | None]:
    """1A 地板停臂的 ×1.0 快照路径与其循环内时刻。"""
    base = ROOT / "results/floor"
    name = f"adult_eps{tag}_ours2way_seed{seed}"
    snaps = base / f"{name}_snaps.csv"
    if not snaps.exists():
        return None, None
    floor = FLOORS[("1A", tag, seed)]
    for r in csv.DictReader(open(snaps, encoding="utf-8")):
        if round(float(r["threshold"]) / floor, 3) == 1.0:
            return base / f"{name}_snap{r['index']}.csv", float(r["elapsed_s"])
    return None, None


def arm_secs(exam: str, tag: str, seed: int, dec: str) -> float | None:
    if exam == "1A":
        if dec == "floor":
            return snap_x1(tag, seed)[1]
        return last_elapsed(ROOT / f"results/floor/adult_eps{tag}_ours2way_seed{seed}_curve.csv")
    sub = "whole" if dec == "floor" else "whole_conv"
    kind = "oursgstop" if dec == "floor" else "oursconv"
    s = stage_secs(ROOT / f"logs_eps/groupfloor/{sub}/adult_eps{tag}_{kind}_seed{seed}.log")
    return sum(s.values()) if s else None


def grade_cell(out: Path, log: Path, t: dict) -> dict:
    exam, tag, seed, cell, dec = t["exam"], t["tag"], int(t["seed"]), t["cell"], t["decision"]
    src = f"adult_alloc{tag}base_s{seed}" if exam == "1A" else f"adult_aimrep{tag}_s{seed}"
    fis = ROOT / "results/fission"
    early = t.get("reasonP") == "fission_rise"  # 第七十三步：P 显著回升停提前收工，终表自然不同于跑到收敛的探针臂
    checks = {
        "P_G": same(out / f"P_{cell}_snapG.csv", fis / f"fis_{cell}_snapG.csv"),
        "P_终": "早停" if early else same(out / f"P_{cell}.csv", fis / f"fis_{cell}.csv"),
    }
    if exam == "1A":
        x1, _ = snap_x1(tag, seed)
        checks["F_G"] = same(out / f"F_{cell}_snapG.csv", x1) if x1 else "缺"
        checks["F_终"] = same(out / f"F_{cell}.csv", ROOT / f"results/floor/adult_eps{tag}_ours2way_seed{seed}.csv") if dec == "conv" else "—"
        deliver = x1 if dec == "floor" else ROOT / f"results/floor/adult_eps{tag}_ours2way_seed{seed}.csv"
    else:
        checks["F_G"] = same(out / f"F_{cell}_snapG.csv", ROOT / f"results/groupfloor/snap1b/adult_eps{tag}_snap_seed{seed}_snapG.csv")
        checks["F_终"] = same(out / f"F_{cell}.csv", ROOT / f"results/groupfloor/whole1b_conv/adult_eps{tag}_oursconv_seed{seed}.csv") if dec == "conv" else "—"
        deliver = (ROOT / f"results/groupfloor/whole1b/adult_eps{tag}_oursgstop_seed{seed}.csv" if dec == "floor"
                   else ROOT / f"results/groupfloor/whole1b_conv/adult_eps{tag}_oursconv_seed{seed}.csv")
    checks["交表"] = same(out / f"{cell}_out.csv", deliver) if deliver else "缺"
    d = json.load(open(out / f"{cell}_decision.json", encoding="utf-8")) if (out / f"{cell}_decision.json").exists() else {}
    pin = stage_secs(log / f"P_{cell}.log")
    fin = stage_secs(log / f"F_{cell}.log")
    p_solo = last_elapsed(fis / f"fis_{cell}_curve.csv")
    arm = arm_secs(exam, tag, seed, dec)
    gv = d.get("G_vs_final")
    if gv is None and d.get("L1_B_G") and d.get("L1_B_final"):
        gv = (d["L1_B_G"] - d["L1_B_final"]) / d["L1_B_final"]
    return dict(cell=cell, exam=exam, tag=tag, seed=seed, dec=dec, dec_ok=dec == expect(exam, tag),
                g_vs_final=gv, z_final=d.get("z_final"), P_stop=d.get("P_last_elapsed_s"), wall=t["wall"], tP=t["tP"] - t["t0"], tD=t["tD"] - t["t0"],
                killedF=t.get("killedF"), rcP=t.get("rcP"), rcF=t.get("rcF"), rcS=t.get("rcS"),
                P_in=pin.get("s1"), F_in=sum(fin.values()) if fin else None, F_stages=fin,
                seq=(p_solo + arm) if (p_solo is not None and arm is not None) else None, p_solo=p_solo, arm=arm,
                checks=checks, load_end=t.get("load_end"), cores=f"{t.get('cores_P')}/{t.get('cores_F')} nice{t.get('nice_F', 0)}")


def fmt(x, nd=0) -> str:
    return "—" if x is None else (f"{x:,.{nd}f}")


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    suf = f"_{variant}" if variant else ""
    out, log = ROOT / f"results/autostop{suf}", ROOT / f"logs_eps/autostop{suf}"
    rows = [grade_cell(out, log, json.load(open(p, encoding="utf-8"))) for p in sorted(out.glob("*_times.json"))]
    if not rows:
        sys.exit(f"{out} 下没有 *_times.json")
    lines = [f"# 第七十二步 探针并行自动停：计时与逐位核对{'（变体 ' + variant + '）' if variant else ''}", "",
             f"格数 {len(rows)}；判停法与第七十一步真值一致 {sum(r['dec_ok'] for r in rows)}/{len(rows)}；"
             f"逐位核对全同 {sum(all(v in ('同', '—', '早停') for v in r['checks'].values()) for r in rows)}/{len(rows)}；"
             f"rc 全 0 {sum((r['rcP'] == 0) and (r['rcF'] in (0, None) or r['killedF']) and (r['rcS'] in (0, None)) for r in rows)}/{len(rows)}。", "",
             "并行墙钟 = 两进程启动到交表（含载入、编译、判定，秒）；探针到判定 = 探针进程启动到判停法；"
             "顺序估计 = 探针单跑循环内秒 + 所判停法臂循环内秒（不含载入；探针单跑是重负载下测的）。三种子均值。", ""]
    for exam in ("1A", "1B"):
        sub = [r for r in rows if r["exam"] == exam]
        if not sub:
            continue
        lines += [f"## {exam}", "",
                  "| ε | 判停法 | 并行墙钟 | 探针到判定 | 顺序估计 | " + ("GSD 跑到底 | " if exam == "1A" else "") + "G 对终 L1_B | 逐位 |",
                  "|---|---|---|---|---|" + ("---|" if exam == "1A" else "") + "---|---|"]
        for tag in TAGS:
            g = [r for r in sub if r["tag"] == tag]
            if not g:
                continue
            decs = sorted({r["dec"] for r in g})
            seqs = [r["seq"] for r in g if r["seq"] is not None]
            gv = [r["g_vs_final"] for r in g if r["g_vs_final"] is not None]
            ok = all(all(v in ("同", "—", "早停") for v in r["checks"].values()) for r in g)
            lines.append(f"| {EPS[tag]} | {'/'.join('地板停' if d == 'floor' else ('收敛停' if d == 'conv' else d) for d in decs)}"
                         f"{'' if all(r['dec_ok'] for r in g) else ' ✗'} | {fmt(mean(r['wall'] for r in g))} | {fmt(mean(r['tD'] for r in g))} | "
                         f"{fmt(mean(seqs)) if seqs else '—'} | " + (f"{GSD_SEC[tag]:,} | " if exam == "1A" else "")
                         + f"{'/'.join(f'{v:+.1%}' for v in gv)} | {'全同' if ok else '有异'} |")
        lines.append("")
    lines += ["## 附：逐格", "",
              "| 格 | 判停法 | G 对终 | 终点 z | 墙钟 | 到判定 | P 循环内 | F 循环内（段） | 顺序估计 = 探针单跑 + 臂 | 杀 F | rc P/F/S | 核对 P_G/P_终/F_G/F_终/交表 | 核 | 收尾负载 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = r["F_stages"]
        stg = "+".join(f"{st[k]:.0f}" for k in ("s1", "pool", "s2") if k in st) if st else "—"
        lines.append(f"| {r['cell']} | {r['dec']}{'' if r['dec_ok'] else ' ✗'} | {fmt(r['g_vs_final'] * 100, 2) + '%' if r['g_vs_final'] is not None else '—'} | "
                     f"{fmt(r['z_final'], 1) if r['z_final'] is not None else '—'} | "
                     f"{fmt(r['wall'])} | {fmt(r['tD'])} | {fmt(r['P_in'])} | {fmt(r['F_in'])}（{stg}） | "
                     f"{fmt(r['seq'])} = {fmt(r['p_solo'])} + {fmt(r['arm'])} | {r['killedF']} | {r['rcP']}/{r['rcF']}/{r['rcS']} | "
                     f"{'/'.join(r['checks'][k] for k in ('P_G', 'P_终', 'F_G', 'F_终', '交表'))} | {r['cores']} | {r['load_end']} |")
    dst = ROOT / f"logs_eps/scores/autostop_times{suf}.md"
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
