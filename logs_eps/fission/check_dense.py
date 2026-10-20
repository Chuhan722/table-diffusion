#!/usr/bin/env python
"""第七十三步 密集记账核对（只用 numpy，远端 venv 没 pandas）。

用法：check_dense.py 格 密集目录 第七十一步目录 第七十二步目录 [参照密集目录] [模拟卷.json]
核：终表 md5 = fis_<格>.csv，G 表 md5 = fis_<格>_snapG.csv；
    记账 CSV 的 G 行 L1_B = decision.json 的 L1_B_G、final 行 = L1_B_final（相对差）；
    G 行轮号 = 第七十一步 fis_<格>_snaps.csv 里 G 的轮号。
    给参照密集目录时：两趟记账的 round/old_loss/l1_b/at_group_floor 列逐字相同（轨迹零改变）；
    给模拟卷时：final 行的 diff_mean/diff_se = 终表对 G 表逐题误差配对差的离线重算（decide.py 的计数法）。
输出一行 JSON 到 stdout。
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autostop"))
from decide import counts, load  # noqa: E402


def offline_diff(exam_b: Path, g_csv: Path, fin_csv: Path):
    """终表对 G 表在模拟卷上的配对差均值与标准误（与引擎同式：d = e_final − e_G，se = std(d, ddof=1)/√n）。"""
    qb = json.load(open(exam_b, encoding="utf-8"))["queries"]
    yb = np.array([q["result"] for q in qb], dtype=np.float64)
    d = np.abs(yb - counts(load(str(fin_csv)), qb)) - np.abs(yb - counts(load(str(g_csv)), qb))
    return float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(d.size))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else ""


def _z_rule(rows, z: float, runs: int):
    """按记账离线复算判停：参照后连续 runs 条 diff_mean ≥ z·diff_se 的首个终点（轮号、秒），没有则 None。"""
    streak = 0
    for r in rows:
        if r["round"] == "final" or r.get("diff_mean", "") == "":
            streak = 0
            continue
        dm, ds = float(r["diff_mean"]), float(r["diff_se"])
        streak = streak + 1 if (ds > 0.0 and dm >= z * ds) else 0
        if streak >= runs:
            return {"round": int(r["round"]), "elapsed_s": float(r["elapsed_s"])}
    return None


def main() -> None:
    cell, dense, fis, auto = sys.argv[1:5]
    ref = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
    exam_b = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
    dense, fis, auto = Path(dense), Path(fis), Path(auto)
    res = {"cell": cell}
    res["final_md5_same"] = md5(dense / f"{cell}.csv") == md5(fis / f"fis_{cell}.csv") != ""
    res["snapG_md5_same"] = md5(dense / f"{cell}_snapG.csv") == md5(fis / f"fis_{cell}_snapG.csv") != ""
    with open(dense / f"{cell}_probe.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    res["rows"] = len(rows)
    g_rows = [r for r in rows if r["at_group_floor"] == "1"]
    fin = [r for r in rows if r["round"] == "final"]
    res["g_rows"] = len(g_rows)
    res["has_final"] = len(fin) == 1
    dec_path = auto / f"{cell}_decision.json"
    if dec_path.exists():
        dec = json.load(open(dec_path, encoding="utf-8"))
        if g_rows:
            g = float(g_rows[0]["l1_b"])
            res["G_round"] = int(g_rows[0]["round"])
            res["G_elapsed_s"] = float(g_rows[0]["elapsed_s"])
            res["L1_B_G"] = g
            res["G_reldiff"] = (g - dec["L1_B_G"]) / dec["L1_B_G"]
        if fin:
            f = float(fin[0]["l1_b"])
            res["L1_B_final"] = f
            res["final_reldiff"] = (f - dec["L1_B_final"]) / dec["L1_B_final"]
        res["decision_ref"] = dec["decision"]
    snaps = fis / f"fis_{cell}_snaps.csv"
    if snaps.exists():
        with open(snaps, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["index"] == "G":
                    res["G_round_ref"] = int(r["round"])
    num = [r for r in rows if r["round"] != "final"]
    if num:
        res["last_round"] = int(num[-1]["round"])
        res["last_elapsed_s"] = float(num[-1]["elapsed_s"])
    if ref is not None and (ref / f"{cell}_probe.csv").exists():
        with open(ref / f"{cell}_probe.csv", encoding="utf-8") as fh:
            ref_rows = list(csv.DictReader(fh))
        keys = ("round", "old_loss", "l1_b", "at_group_floor")
        res["ref_rows"] = len(ref_rows)
        res["ref_same"] = len(ref_rows) == len(rows) and all(
            all(a[k] == b[k] for k in keys) for a, b in zip(rows, ref_rows)
        )
    if exam_b is not None and fin and g_rows and fin[0].get("diff_mean", "") != "":
        m, se = offline_diff(exam_b, dense / f"{cell}_snapG.csv", dense / f"{cell}.csv")
        res["final_diff_mean"] = float(fin[0]["diff_mean"])
        res["final_diff_se"] = float(fin[0]["diff_se"])
        res["final_z"] = res["final_diff_mean"] / res["final_diff_se"]
        res["offline_diff_mean"], res["offline_diff_se"] = m, se
        res["diff_mean_abs_err"] = abs(res["final_diff_mean"] - m)
        res["diff_se_reldiff"] = (res["final_diff_se"] - se) / se
        res["z_rule_2x3"] = _z_rule(rows, 2.0, 3)
    ok = res["final_md5_same"] and res["snapG_md5_same"] and res["has_final"] and res["g_rows"] == 1
    if "ref_same" in res:
        ok = ok and res["ref_same"]
    if "offline_diff_se" in res:
        ok = ok and res["diff_mean_abs_err"] <= 1e-9 * max(1.0, abs(m)) and abs(res["diff_se_reldiff"]) < 1e-9
    if "G_reldiff" in res:
        ok = ok and abs(res["G_reldiff"]) < 1e-9 and abs(res["final_reldiff"]) < 1e-9
    if "G_round_ref" in res and "G_round" in res:
        ok = ok and res["G_round"] == res["G_round_ref"]
    res["ok"] = bool(ok)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
