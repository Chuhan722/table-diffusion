"""噪声考卷清洗器，负值截断之上补归一化，纯后处理零隐私预算。

GSD 官方口径加噪后只裁剪到零一区间，每个边际的频率总和不再等于一，
考卷自相矛盾，本脚本把每个边际拉回合法分布，两种摊法，
scale 等比例缩放总和回一，proj 是 L2 单纯形投影等额摊差截负重摊，
高斯噪声每格同方差与格子大小无关，等额摊在最小二乘意义下更对口。
真答案必在单纯形内，投影非扩张性保证清洗后离真答案只近不远，
与 GSD 库内置的 clip 同族，AIM 系 PGM 全局一致化同理，文献标准后处理。

验证模式对照三口径离真实考卷的距离，
./.venv/bin/python scripts/clean_noisy_exam.py --data nltcs --seeds 1 2 3
写出模式产清洗版考卷目录 <原目录>_clean，方法由 --method 选，
./.venv/bin/python scripts/clean_noisy_exam.py --data nltcs --seeds 1 --write --method proj
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_exam(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def exam_path(dirname: str) -> Path:
    hits = sorted((DATA_ROOT / dirname).glob("measured_*query.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{dirname} 训练卷不唯一 {hits}")
    return hits[0]


def group_by_margin(exam: dict) -> dict[tuple, list[int]]:
    """按字段组合分组，值为该边际全部格子的条目下标，保持条目序。"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, q in enumerate(exam["queries"]):
        key = tuple(c["attribute"] for c in q["conditions"])
        groups[key].append(i)
    return groups


def project_simplex(v: np.ndarray, total: float = 1.0) -> np.ndarray:
    """欧氏投影到总和 total 的单纯形，Duchi 排序法，输入可含负值。"""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    idx = np.arange(1, len(u) + 1)
    cond = u - cssv / idx > 0
    rho = idx[cond][-1]
    tau = cssv[rho - 1] / rho
    return np.maximum(v - tau, 0.0)


def clean_freqs(freqs: np.ndarray, method: str) -> np.ndarray:
    """单边际频率向量清洗，输入已在零一区间，输出非负总和一。"""
    if method == "scale":
        s = freqs.sum()
        if s <= 0.0:
            return np.full_like(freqs, 1.0 / len(freqs))
        return freqs / s
    if method == "proj":
        return project_simplex(freqs.astype(np.float64))
    raise ValueError(f"未知清洗方法 {method}")


def margin_distances(
    noisy: dict, real: dict, methods: list[str]
) -> dict[str, tuple[float, float, float]]:
    """逐边际算频率向量到真实考卷的距离，返回各口径 L1 均值 L2 均值 最大格误差。"""
    n = float(real["record_count"])
    groups = group_by_margin(real)
    real_res = np.array([q["result"] for q in real["queries"]], dtype=np.float64)
    noisy_res = np.array([q["result"] for q in noisy["queries"]], dtype=np.float64)
    for i, (qr, qn) in enumerate(zip(real["queries"], noisy["queries"])):
        if qr["expression"] != qn["expression"]:
            raise ValueError(f"条目 {i} 顺序不一致")
    out = {}
    for method in ["raw"] + methods:
        l1s, l2s, mx = [], [], 0.0
        for idx in groups.values():
            rf = real_res[idx] / n
            nf = noisy_res[idx] / n
            cf = nf if method == "raw" else clean_freqs(nf, method)
            diff = np.abs(cf - rf)
            l1s.append(diff.sum())
            l2s.append(float(np.sqrt((diff**2).sum())))
            mx = max(mx, float(diff.max()))
        out[method] = (float(np.mean(l1s)), float(np.mean(l2s)), mx)
    return out


def write_clean(noisy_dir: str, method: str) -> Path:
    """写清洗版考卷目录，除训练卷逐格替换外其余文件原样复制。"""
    src = DATA_ROOT / noisy_dir
    dst = DATA_ROOT / f"{noisy_dir}_clean"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    old_table = dst / f"{noisy_dir}.csv"
    if old_table.exists():
        old_table.rename(dst / f"{noisy_dir}_clean.csv")
    path = exam_path(f"{noisy_dir}_clean")
    exam = load_exam(path)
    n = float(exam["record_count"])
    res = np.array([q["result"] for q in exam["queries"]], dtype=np.float64)
    for idx in group_by_margin(exam).values():
        cf = clean_freqs(res[idx] / n, method)
        res[idx] = cf * n
    for q, r in zip(exam["queries"], res):
        q["result"] = float(r)
    exam["description"] += f" 清洗版，负截断后按 {method} 归一化每边际总和回一，纯后处理零预算。"
    with open(path, "w") as f:
        json.dump(exam, f, ensure_ascii=False)
    readme = dst / "README.md"
    if readme.exists():
        with open(readme, "a") as f:
            f.write(
                f"\n5. 清洗版，每边际频率负截断后 {method} 归一化总和回一，"
                "scale 为等比例缩放，proj 为 L2 单纯形投影，纯后处理零预算。\n"
            )
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="噪声考卷清洗与离线验证")
    parser.add_argument("--data", required=True, help="数据名，如 nltcs")
    parser.add_argument("--epsilon-tag", default="eps1", help="噪声目录 eps 段，默认 eps1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--write", action="store_true", help="写清洗版考卷目录")
    parser.add_argument("--method", default="proj", choices=["scale", "proj"])
    args = parser.parse_args()

    real = load_exam(exam_path(args.data))
    for seed in args.seeds:
        dirname = f"{args.data}_{args.epsilon_tag}_s{seed}"
        noisy = load_exam(exam_path(dirname))
        dists = margin_distances(noisy, real, ["scale", "proj"])
        print(f"{dirname}  每边际频率离真卷距离，三口径对照")
        for m, (l1, l2, mx) in dists.items():
            print(f"  {m:5s}  L1均值 {l1:.6f}  L2均值 {l2:.6f}  最大格 {mx:.6f}")
        if args.write:
            dst = write_clean(dirname, args.method)
            print(f"  已写 {dst.name}  方法 {args.method}")


if __name__ == "__main__":
    main()
