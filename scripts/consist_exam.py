"""噪声考卷一致化：非负截断加 IPF 边际咬合加总量归一，输出理顺版考卷。

动机（第三十九步定案）：加噪场卡死病根在噪声答案自相矛盾——
负格、各联表对同一字段的边际互相打架，促聚方向被矛盾目标顶死。
本脚本对已购噪声答案做标准后处理（PGM/MST 流派同款），
评价与买单记账全不动，只把拟合目标理顺，红线内合规。

工序：
1. 非负截断（make_noisy_exam 已裁 0~1，此处兜底）加极小地板防死格；
2. IPF 调和：公共边际取各联表投影的平均，各表迭代拉平到公共边际，
   循环至各表对同字段的边际最大分歧收敛；
3. 总量归一到表行数 N。

输出 data/<name>c/ 目录：理顺卷（格式同原卷）加表原样复制加 README。
--truth 给真卷时打印一致化前后到真答案的距离对比（自查用，不参与工序）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from resevo.dataset import load_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="噪声考卷一致化")
    parser.add_argument("--data", required=True, help="噪声数据目录名，如 nltcs_eps1_s1")
    parser.add_argument("--iters", type=int, default=200, help="IPF 外层迭代上限")
    parser.add_argument("--tol", type=float, default=1e-9, help="边际分歧收敛阈（频率口径）")
    parser.add_argument("--truth", type=str, default="", help="真卷 json，给出则打印距离自查")
    args = parser.parse_args()

    src_dir = REPO / "data" / args.data
    found = sorted(src_dir.glob("measured_*query.json"))
    if len(found) != 1:
        raise SystemExit(f"{src_dir} 下 measured_*query.json 须恰有一个")
    payload = json.loads(found[0].read_text(encoding="utf-8"))
    queries = payload["queries"]
    schema, rows = load_table(str(src_dir / f"{args.data}.csv"))
    n_rows = len(rows)
    code = [{v: i for i, v in enumerate(d)} for d in schema.domains]
    sizes = [len(d) for d in schema.domains]

    # 重组联表，格子按 conditions 的取值定位，不赖条目顺序
    tables: dict[tuple[int, int], np.ndarray] = {}
    slots: dict[tuple[int, int], np.ndarray] = {}  # 格子到考卷条目号
    for qi, q in enumerate(queries):
        conds = q["conditions"]
        if len(conds) != 2 or any(c["operator"] != "==" for c in conds):
            raise SystemExit("一致化只支持全二阶等值训练卷")
        a = schema.field_index(conds[0]["attribute"])
        b = schema.field_index(conds[1]["attribute"])
        va, vb = code[a][conds[0]["value"]], code[b][conds[1]["value"]]
        if a > b:
            a, b, va, vb = b, a, vb, va
        if (a, b) not in tables:
            tables[(a, b)] = np.zeros((sizes[a], sizes[b]))
            slots[(a, b)] = np.full((sizes[a], sizes[b]), -1, dtype=np.int64)
        tables[(a, b)][va, vb] = max(0.0, float(q["result"])) / n_rows
        slots[(a, b)][va, vb] = qi
    if any((s < 0).any() for s in slots.values()):
        raise SystemExit("考卷二阶格子不完整，一致化要求全格")
    F = schema.num_fields

    # 极小地板防死格死行，质量占比不足亿分之一
    floor = 1e-12
    for t in tables.values():
        np.maximum(t, floor, out=t)

    def _marginal_gap() -> float:
        gap = 0.0
        for a in range(F):
            projs = []
            for (x, y), t in tables.items():
                if x == a:
                    projs.append(t.sum(axis=1))
                elif y == a:
                    projs.append(t.sum(axis=0))
            m = np.mean(projs, axis=0)
            gap = max(gap, max(float(np.abs(p - m).max()) for p in projs))
        return gap

    gap0 = _marginal_gap()
    for it in range(args.iters):
        # 公共边际，各表投影平均后归一
        marg = []
        for a in range(F):
            projs = [
                t.sum(axis=1) if x == a else t.sum(axis=0)
                for (x, y), t in tables.items() if a in (x, y)
            ]
            m = np.mean(projs, axis=0)
            marg.append(m / m.sum())
        # 每表行列各拉平一次到公共边际
        for (a, b), t in tables.items():
            rs = t.sum(axis=1)
            t *= (marg[a] / np.maximum(rs, 1e-300))[:, None]
            cs = t.sum(axis=0)
            t *= (marg[b] / np.maximum(cs, 1e-300))[None, :]
        gap = _marginal_gap()
        if gap < args.tol:
            break
    # 总量归一到 N（IPF 后每表和为 1）
    for t in tables.values():
        t *= n_rows / t.sum()

    # 答案写回原格式
    new_results = np.empty(len(queries))
    for key, t in tables.items():
        s = slots[key]
        for va in range(t.shape[0]):
            for vb in range(t.shape[1]):
                new_results[s[va, vb]] = t[va, vb]
    for q, r in zip(queries, new_results):
        q["result"] = float(r)
    payload["description"] = (
        payload.get("description", "")
        + f" 一致化版，非负截断加 IPF 边际咬合（{it + 1} 轮，边际分歧 "
        f"{gap0:.3e} 到 {gap:.3e}）加总量归一 N，评价与隐私记账不变。"
    )

    out_name = f"{args.data}c"
    out_dir = REPO / "data" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_dir / f"{args.data}.csv", out_dir / f"{out_name}.csv")
    with open(out_dir / found[0].name, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    md5 = hashlib.md5((out_dir / f"{out_name}.csv").read_bytes()).hexdigest()
    with open(out_dir / "README.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# {out_name} 一致化噪声考卷\n\n"
            f"1. 来源，data/{args.data} 的噪声卷做后处理，表原样复制，md5 {md5}。\n"
            f"2. 工序，非负截断（地板 {floor:g}），IPF 边际咬合 {it + 1} 轮"
            f"（各表对同字段边际最大分歧 {gap0:.3e} 收敛到 {gap:.3e}），总量归一 N={n_rows}。\n"
            "3. 隐私口径，只对已购噪声答案做后处理，不再碰真表，评价一律用真实目录考卷。\n"
        )
    print(f"已写出 {out_dir}，IPF {it + 1} 轮，边际分歧 {gap0:.3e} -> {gap:.3e}")

    if args.truth:
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))["queries"]
        y_t = np.array([float(q["result"]) for q in truth])
        noisy = json.loads(found[0].read_text(encoding="utf-8"))["queries"]
        y_n = np.array([float(q["result"]) for q in noisy])
        d_n = float(np.linalg.norm(y_n - y_t))
        d_c = float(np.linalg.norm(new_results - y_t))
        print(f"距真答案 L2：原噪声卷 {d_n:.2f}，一致化后 {d_c:.2f}"
              f"（{'变近' if d_c < d_n else '变远'} {abs(d_c - d_n) / d_n * 100:.1f}%）")


if __name__ == "__main__":
    main()
