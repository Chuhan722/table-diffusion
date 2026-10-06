"""把 AIM 测量录像转成本仓混 sigma 考卷数据目录（选测录像回放，第 0 步）。

输入 run_aim_full.py --dump-measurements 录下的 JSON：每条测量 = 团、sigma、
C 序展平的加噪边际计数向量。同一团的重复测量按逆方差合并
（y = sum(y_i/s_i^2)/sum(1/s_i^2)，sigma = 1/sqrt(sum(1/s_i^2))，
高斯似然下与分开入卷等价），一阶团出 eq1way 题，二阶团出 eq2way 题，
每题带 sigma 字段供 run_plants.py --exam-weights sigma 逆方差加权，
一阶题供 --init-oneway 直接初始化。

数据目录结构与 make_noisy_exam 产物一致：<name>/<name>.csv（真表副本，
引擎只用其字段域与行数）、measured_<n>query.json、README.md。
引擎不再自己加噪不花预算，隐私账完全等于 AIM 这一跑。

用法：
  python scripts/aim_replay_exam.py --dump results/aimreplay/adult_eps0p1_aimfull_seed1_measurements.json \
      --real data/adult/adult.csv --name adult_aimrep0p1_s1
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="AIM 测量录像转混 sigma 考卷")
    ap.add_argument("--dump", required=True, help="run_aim_full.py --dump-measurements 产物")
    ap.add_argument("--real", required=True, help="真表 CSV，复制进数据目录作字段域与行数来源")
    ap.add_argument("--name", required=True, help="数据目录名，亦为表名")
    ap.add_argument("--data-root", default=str(REPO / "data"))
    ap.add_argument("--force", action="store_true", help="目录已存在则覆盖")
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    columns = list(dump["columns"])
    sizes = {c: int(n) for c, n in zip(columns, dump["domain_sizes"])}
    col_pos = {c: i for i, c in enumerate(columns)}

    # 同团合并：键为按表列序排好的团，向量转置到同一顺序
    acc: dict[tuple[str, ...], dict] = defaultdict(lambda: {"num": None, "prec": 0.0, "n": 0})
    for m in dump["measurements"]:
        clique = tuple(m["clique"])
        shape = tuple(int(x) for x in m["shape"])
        if shape != tuple(sizes[a] for a in clique):
            raise SystemExit(f"录像团 {clique} 形状 {shape} 与域不符")
        y = np.asarray(m["y"], dtype=np.float64).reshape(shape)
        order = sorted(range(len(clique)), key=lambda i: col_pos[clique[i]])
        key = tuple(clique[i] for i in order)
        y = np.transpose(y, order)
        sigma = float(m["sigma"])
        if sigma <= 0.0:
            raise SystemExit(f"录像团 {clique} sigma 非正: {sigma}")
        slot = acc[key]
        prec = 1.0 / sigma**2
        slot["num"] = y * prec if slot["num"] is None else slot["num"] + y * prec
        slot["prec"] += prec
        slot["n"] += 1

    queries = []
    n1 = n2 = 0
    sig_min, sig_max = float("inf"), 0.0
    for key in sorted(acc, key=lambda k: (len(k), tuple(col_pos[a] for a in k))):
        slot = acc[key]
        y = slot["num"] / slot["prec"]
        sigma = 1.0 / np.sqrt(slot["prec"])
        sig_min, sig_max = min(sig_min, sigma), max(sig_max, sigma)
        if len(key) == 1:
            (a,) = key
            for va in range(sizes[a]):
                n1 += 1
                queries.append({
                    "id": f"R1_{n1:05d}", "type": "eq1way",
                    "expression": f"{a} == {va}",
                    "conditions": [{"attribute": a, "operator": "==", "value": str(va)}],
                    "result": float(y[va]), "sigma": float(sigma),
                    "repeats": int(slot["n"]),
                })
        elif len(key) == 2:
            a, b = key
            for va in range(sizes[a]):
                for vb in range(sizes[b]):
                    n2 += 1
                    queries.append({
                        "id": f"R2_{n2:05d}", "type": "eq2way",
                        "expression": f"{a} == {va} AND {b} == {vb}",
                        "conditions": [
                            {"attribute": a, "operator": "==", "value": str(va)},
                            {"attribute": b, "operator": "==", "value": str(vb)},
                        ],
                        "result": float(y[va, vb]), "sigma": float(sigma),
                        "repeats": int(slot["n"]),
                    })
        else:
            raise SystemExit(f"录像出现 {len(key)} 阶团 {key}，本转换器只处理一阶二阶")

    cliques1 = [k for k in acc if len(k) == 1]
    cliques2 = [k for k in acc if len(k) == 2]
    missing1 = [c for c in columns if (c,) not in acc]
    if missing1:
        raise SystemExit(f"一阶覆盖不全，缺 {missing1}，--init-oneway 无法初始化")

    out_dir = Path(args.data_root) / args.name
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} 已存在，加 --force 覆盖")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copyfile(args.real, out_dir / f"{args.name}.csv")
    payload = {
        "dataset": f"{args.name}.csv",
        "record_count": int(dump["num_rows"]),
        "query_count": len(queries),
        "result_unit": "records",
        "description": (
            "AIM 选测录像回放考卷：题目与加噪答案全部来自官方 AIM 一跑的测量，"
            "同团重复测量逆方差合并，每题带 sigma；引擎不再加噪不花预算"
        ),
        "source_dump": str(Path(args.dump).resolve()),
        "aim": {k: dump[k] for k in ("data", "epsilon", "delta", "seed", "rho", "max_model_size")},
        "queries": queries,
    }
    exam_path = out_dir / f"measured_{len(queries)}query.json"
    exam_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    all_pairs = len(columns) * (len(columns) - 1) // 2
    repeats = sum(1 for s in acc.values() if s["n"] > 1)
    readme = (
        f"# {args.name}\n\n"
        f"1. 来源，官方 AIM 录像 {Path(args.dump).name}，ε={dump['epsilon']} δ={dump['delta']} "
        f"seed={dump['seed']} ρ={dump['rho']:.6g}，测量 {len(dump['measurements'])} 条。\n"
        f"2. 口径，一阶团 {len(cliques1)}/{len(columns)}，二阶团 {len(cliques2)}/{all_pairs}，"
        f"重复测量合并 {repeats} 团，合并后 sigma 范围 {sig_min:.4g}~{sig_max:.4g}，"
        f"题数 {len(queries)}（一阶 {n1} 二阶 {n2}）。\n"
        f"3. 用法，run_plants.py --data {args.name} --init-oneway --exam-weights sigma，"
        f"不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(readme, end="")
    print(f"-> {exam_path}")


if __name__ == "__main__":
    main()
