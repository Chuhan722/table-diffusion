"""第七十一步 噪声分裂探针：把带噪考卷 y 拆成 学习卷 y+e 与 模拟卷 y−e（e~N(0,σ²) 逐题独立）。

两卷噪声独立（Cov(n+e, n−e) = σ²−σ² = 0，高斯即独立），各自 σ' = √2·σ；纯后处理，不花隐私预算。
学习卷落到 data/<源目录>_fisA/（真表 CSV 原样复制，考卷文件名与源同名，权重 w 不变，地板翻倍）；
模拟卷落到 results/fission/<源目录>_fisB.json，只供阅卷（score_fission.py），引擎不读。

用法：cd residual-evolution && .venv/bin/python logs_eps/fission/make_fission.py
      打印每格 地板_原 / 地板_A，并与已知地板（launch_snap1b.sh FLOOR、launch_floor.sh SNAP ×1.0）对账。
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FISSION_SEED = 20261019
# (考卷, ε标签, 源数据目录, 考卷文件（None=自动发现唯一 measured_*query.json）, --exam-weights, 已知地板)
CONFIGS = [
    ("1B", "0p1", "adult_aimrep0p1_s1", None, "sigma", 6287575.166310625),
    ("1B", "1", "adult_aimrep1_s1", None, "sigma", 299038.1812299605),
    ("1A", "0p1", "adult_alloc0p1base_s1", "measured_2way.json", "none", 857055944.8114933),
    ("1A", "1", "adult_alloc1base_s1", "measured_2way.json", "none", 12144672.776841227),
]


def exam_file(src_dir: Path, name: str | None) -> Path:
    if name:
        return src_dir / name
    found = sorted(src_dir.glob("measured_*query.json"))
    if len(found) != 1:
        raise SystemExit(f"{src_dir} 下 measured_*query.json 须恰有一个，实有 {len(found)}")
    return found[0]


def floor_of(sig: np.ndarray, mode: str) -> float:
    w = np.ones_like(sig) if mode == "none" else (sig.min() / sig) ** 2
    return float((0.5 * w * sig * sig).sum())


def split(payload: dict, rng: np.random.Generator) -> tuple[dict, dict, np.ndarray]:
    sig = np.array([float(q["sigma"]) for q in payload["queries"]], dtype=np.float64)
    if (sig <= 0.0).any():
        raise SystemExit("每题须有正 sigma")
    e = rng.normal(0.0, sig)
    scale = math.sqrt(2.0)
    a = json.loads(json.dumps(payload))
    b = json.loads(json.dumps(payload))
    for i, (qa, qb) in enumerate(zip(a["queries"], b["queries"])):
        qa["result"] = float(qa["result"]) + float(e[i])
        qb["result"] = float(qb["result"]) - float(e[i])
        qa["sigma"] = float(sig[i]) * scale
        qb["sigma"] = float(sig[i]) * scale
    for d in (a, b):
        if "sigma2" in d and isinstance(d["sigma2"], (int, float)):
            d["sigma2"] = float(d["sigma2"]) * 2.0
    return a, b, sig


def main() -> None:
    out_b = ROOT / "results/fission"
    out_b.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (exam, tag, src, fname, mode, known) in enumerate(CONFIGS):
        src_dir = ROOT / "data" / src
        xp = exam_file(src_dir, fname)
        payload = json.load(open(xp, encoding="utf-8"))
        rng = np.random.default_rng([FISSION_SEED, idx])
        a, b, sig = split(payload, rng)
        f0 = floor_of(sig, mode)
        fa = floor_of(np.array([q["sigma"] for q in a["queries"]]), mode)
        meta = dict(source=str(xp.relative_to(ROOT)), fission_seed=FISSION_SEED, index=idx,
                    sigma_scale="sqrt(2)", floor_source=f0, floor_split=fa)
        name_a = f"{src}_fisA"
        dir_a = ROOT / "data" / name_a
        dir_a.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / f"{src}.csv", dir_a / f"{name_a}.csv")
        a["dataset"] = name_a
        a["fission"] = dict(role="学习卷 A = y+e", **meta)
        b["dataset"] = src
        b["fission"] = dict(role="模拟卷 B = y−e（只供阅卷）", **meta)
        json.dump(a, open(dir_a / xp.name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(b, open(out_b / f"{src}_fisB.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rel = abs(f0 - known) / known
        rows.append(dict(exam=exam, tag=tag, src=src, data=name_a, exam_file=xp.name, mode=mode,
                         questions=int(sig.size), floor_source=f0, floor_known=known, floor_rel_diff=rel,
                         floor_split=fa, floor_split_str=repr(fa)))
        print(f"{exam} ε={tag} {name_a}/{xp.name} 题 {sig.size} 地板_原 {f0!r}（已知 {known!r}，相对差 {rel:.1e}）"
              f" 地板_A {fa!r}（÷原 {fa / f0:.6f}）")
    json.dump(rows, open(out_b / "floors.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {out_b / 'floors.json'}")


if __name__ == "__main__":
    sys.exit(main())
