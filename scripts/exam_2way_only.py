"""把分配考卷裁成"只含二阶题"的基线卷，给 GSD / PGM 同餐用。

GSD 官方 oneshot 口径只挂全二阶统计量，PGM 基线脚本也只吃二阶边际，两家的加载器都要求
每题恰好两个条件、且全部字段对格子齐全；分配考卷里还带一阶题，直接喂会炸。本脚本只做裁剪：
顶层字段原样保留（record_count 等），queries 只留 2 条件题，query_count 改成二阶题数，
并把 allocation.sigma2 抄到顶层 sigma2 方便启动脚本读。噪声答案一个字不改。

基线卷本身由 make_alloc_exam.py --f1 0.001 --pairs <全部对数> --tag base 生成：
Δ₂=1（增删一行）口径下预算 99.9% 全给二阶、一阶名义上花 0.1%（被裁掉），
二阶 σ = √(P / (2·0.999ρ))，与"全预算全二阶"只差 0.05%。

用法：
  python scripts/exam_2way_only.py data/adult_alloc1base_s1/measured_14316query.json
  → 同目录写出 measured_2way.json，并打印 σ2 与题数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    exam = json.load(open(src, encoding="utf-8"))
    two = [q for q in exam["queries"] if len(q["conditions"]) == 2]
    if not two:
        raise SystemExit("考卷里没有二阶题")
    sigmas = sorted({round(float(q["sigma"]), 9) for q in two})
    if len(sigmas) != 1:
        raise SystemExit(f"二阶 σ 不唯一：{sigmas}")
    out = {k: v for k, v in exam.items() if k != "queries"}
    out["query_count"] = len(two)
    out["sigma2"] = sigmas[0]
    out["description"] = (exam.get("description", "") + "；本卷为 exam_2way_only 裁剪版，只含二阶题，供 GSD/PGM 同餐")
    out["queries"] = two
    dst = src.with_name("measured_2way.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{dst} 二阶题 {len(two)} σ2={sigmas[0]:.6g} record_count={out['record_count']}")


if __name__ == "__main__":
    main()
