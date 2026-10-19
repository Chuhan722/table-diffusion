"""第七十一步 噪声分裂探针：把带噪考卷 y 拆成 学习卷 y+e 与 模拟卷 y−e（e~N(0,σ²) 逐题独立）。

两卷噪声独立（Cov(n+e, n−e) = σ²−σ² = 0，高斯即独立），各自 σ' = √2·σ；纯后处理，不花隐私预算。
学习卷落到 data/<源目录>_fisA/（真表 CSV 原样复制，考卷文件名与源同名，权重 w 不变，地板翻倍）；
模拟卷落到 results/fission/<源目录>_fisB.json，只供阅卷（score_fission.py），引擎不读。
幂等：已存在的学习卷/模拟卷须与重算结果逐字相同，否则报错退出，绝不覆盖（试点四格的分裂序号固定在 0～3）。

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
TAGS = ["0p1", "0p3", "1", "3", "10"]
# 配置 = (考卷, ε标签, 种子)。前四格是试点（10-18）的原序，分裂随机数按序号取，序号不得变；其后为全网格补齐（10-19）。
PILOT = [("1B", "0p1", 1), ("1B", "1", 1), ("1A", "0p1", 1), ("1A", "1", 1)]
CONFIGS = PILOT + [(e, t, s) for e in ("1B", "1A") for t in TAGS for s in (1, 2, 3) if (e, t, s) not in PILOT]
# 1B 已知地板（logs_eps/groupfloor/launch_snap1b.sh FLOOR 表，逐字）；1A 的已知地板取各场 *_snaps.csv 里最接近的阈值（×1.0 快照）
KNOWN_1B = {
    ("0p1", 1): 6287575.166310625, ("0p1", 2): 1823340.3762813245, ("0p1", 3): 5123813.774480508,
    ("0p3", 1): 1399681.5247673127, ("0p3", 2): 1800172.6178743835, ("0p3", 3): 1630659.5345760211,
    ("1", 1): 299038.1812299605, ("1", 2): 225753.08438055194, ("1", 3): 190205.15277737722,
    ("3", 1): 68465.50849395874, ("3", 2): 48037.311033374186, ("3", 3): 51669.8821903612,
    ("10", 1): 9924.017934371434, ("10", 2): 14727.64151114602, ("10", 3): 14291.893133479229,
}


def source_of(exam: str, tag: str, seed: int) -> tuple[str, str | None, str]:
    """返回 (源数据目录, 考卷文件名或 None=自动发现, --exam-weights)。"""
    if exam == "1B":
        return f"adult_aimrep{tag}_s{seed}", None, "sigma"
    return f"adult_alloc{tag}base_s{seed}", "measured_2way.json", "none"


def known_floor(exam: str, tag: str, seed: int, computed: float) -> float | None:
    if exam == "1B":
        return KNOWN_1B.get((tag, seed))
    snaps = ROOT / f"results/floor/adult_eps{tag}_ours2way_seed{seed}_snaps.csv"
    if not snaps.exists():
        return None
    import csv
    ths = [float(r["threshold"]) for r in csv.DictReader(open(snaps, encoding="utf-8"))]
    return min(ths, key=lambda x: abs(x - computed)) if ths else None


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


def dumps(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=1)


def main() -> None:
    out_b = ROOT / "results/fission"
    out_b.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (exam, tag, seed) in enumerate(CONFIGS):
        src, fname, mode = source_of(exam, tag, seed)
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
        if not (dir_a / f"{name_a}.csv").exists():
            shutil.copyfile(src_dir / f"{src}.csv", dir_a / f"{name_a}.csv")
        a["dataset"] = name_a
        a["fission"] = dict(role="学习卷 A = y+e", **meta)
        b["dataset"] = src
        b["fission"] = dict(role="模拟卷 B = y−e（只供阅卷）", **meta)
        status = "新写"
        for path, text in ((dir_a / xp.name, dumps(a)), (out_b / f"{src}_fisB.json", dumps(b))):
            if path.exists():
                if path.read_text(encoding="utf-8") != text:
                    raise SystemExit(f"{path} 已存在且内容不同，拒绝覆盖（分裂序号或源考卷变了？）")
                status = "已存在逐字同"
            else:
                path.write_text(text, encoding="utf-8")
        known = known_floor(exam, tag, seed, f0)
        rel = abs(f0 - known) / known if known else float("nan")
        rows.append(dict(exam=exam, tag=tag, seed=seed, src=src, data=name_a, exam_file=xp.name, mode=mode,
                         questions=int(sig.size), floor_source=f0, floor_known=known, floor_rel_diff=rel,
                         floor_split=fa, floor_split_str=repr(fa), index=idx))
        print(f"[{idx:2d}] {exam} ε={tag} s{seed} {name_a}/{xp.name} 题 {sig.size} 地板_原 {f0!r}"
              f"（已知 {known!r}，相对差 {rel:.1e}） 地板_A {fa!r}（÷原 {fa / f0:.6f}） {status}")
    json.dump(rows, open(out_b / "floors.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {out_b / 'floors.json'}（{len(rows)} 格）")


if __name__ == "__main__":
    sys.exit(main())
