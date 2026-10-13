"""给 AIM 录像考卷补"没测的对"的推断目标（第二阶段 B：没测对的填法）。

输入：录像考卷目录 data/<src>/（measured_*query.json）+ PGM 同一录像拟合后的精确边际 npz
（scripts/pgm_exact_marginals.py 产物，键 measured_pairs、m2_i_j、columns、doms）。
对每个 AIM 没测的字段对，把 PGM 模型推出来的二阶边际（精确，不出表）逐格写成 eq2way 题，
result = 推断计数，sigma = 原考卷二阶题 sigma 的中位数（当作同等精度的伪测量；--sigma-mult 可放大）。
纯后处理零预算：推断只用已花预算的测量。输出新目录 data/<name>/（CSV 复制、考卷、README）。

用法: .venv/bin/python scripts/add_infer_targets.py --src adult_aimrep10_s1 --npz results/stage2/adult_eps10_pgmexact_seed1_exact.npz --name adult_aimrep10_s1_infer
"""
from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="录像考卷补没测对的 PGM 推断目标")
    ap.add_argument("--src", required=True, help="源考卷数据目录名（data/ 下）")
    ap.add_argument("--npz", required=True, help="PGM 精确边际 npz")
    ap.add_argument("--name", required=True, help="输出数据目录名")
    ap.add_argument("--sigma-mult", type=float, default=1.0, help="伪测量 sigma = 中位数 × 此倍数")
    args = ap.parse_args()

    src = ROOT / "data" / args.src
    exams = sorted(src.glob("measured_*query.json"))
    if len(exams) != 1:
        raise SystemExit(f"源目录须恰有一份 measured_*query.json，见 {exams}")
    payload = json.load(open(exams[0], encoding="utf-8"))
    npz_path = Path(args.npz).resolve()
    z = np.load(npz_path, allow_pickle=True)
    columns = [str(c) for c in z["columns"]]
    doms = [int(d) for d in z["doms"]]
    measured = {tuple(int(v) for v in p) for p in z["measured_pairs"]}
    pairs = list(combinations(range(len(columns)), 2))
    unmeasured = [p for p in pairs if p not in measured]

    s2 = np.array([q["sigma"] for q in payload["queries"] if q["type"] == "eq2way"], dtype=np.float64)
    sigma = float(np.median(s2)) * args.sigma_mult
    queries = list(payload["queries"])
    n0 = len(queries)
    k = 0
    for i, j in unmeasured:
        m = np.asarray(z[f"m2_{i}_{j}"], dtype=np.float64).reshape(doms[i], doms[j])
        for a in range(doms[i]):
            for b in range(doms[j]):
                k += 1
                queries.append({
                    "id": f"P2_{k:05d}", "type": "eq2way",
                    "expression": f"{columns[i]} == {a} AND {columns[j]} == {b}",
                    "conditions": [
                        {"attribute": columns[i], "operator": "==", "value": str(a)},
                        {"attribute": columns[j], "operator": "==", "value": str(b)},
                    ],
                    "result": float(m[a, b]), "sigma": sigma, "repeats": 0, "source": "pgm_infer",
                })
    out = ROOT / "data" / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / f"{args.src}.csv", out / f"{args.name}.csv")
    payload["dataset"] = f"{args.name}.csv"
    payload["query_count"] = len(queries)
    payload["queries"] = queries
    payload["description"] = payload.get("description", "") + (
        f"；另补 AIM 没测的 {len(unmeasured)} 对的 PGM 推断二阶边际（精确，不出表）共 {k} 题，"
        f"sigma={sigma:.4g}（原二阶 sigma 中位数 ×{args.sigma_mult:g}），零预算后处理"
    )
    payload["infer"] = {"npz": str(npz_path.relative_to(ROOT)), "pairs": [[columns[i], columns[j]] for i, j in unmeasured],
                        "sigma": sigma, "sigma_mult": args.sigma_mult}
    exam_name = f"measured_{len(queries)}query.json"
    json.dump(payload, open(out / exam_name, "w", encoding="utf-8"), ensure_ascii=False)
    (out / "README.md").write_text(
        f"# {args.name}\n\n1. 来源，{args.src} 考卷原样（{n0} 题）+ PGM 推断目标。\n"
        f"2. 口径，AIM 没测的对 {len(unmeasured)}/{len(pairs)}，每对精确二阶边际逐格成题共 {k} 题，"
        f"sigma {sigma:.4g}（原二阶中位数 ×{args.sigma_mult:g}）；推断来自 {npz_path.name}（同录像、同配置 PGM 拟合，不出表）。\n"
        f"3. 用法，run_plants.py --data {args.name} --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。\n",
        encoding="utf-8")
    print(f"写出 {out / exam_name}：原 {n0} 题 + 推断 {k} 题（{len(unmeasured)} 对），sigma={sigma:.4g}")


if __name__ == "__main__":
    main()
