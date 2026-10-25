"""acsmob 1A 噪声分裂探针出卷：沿用第七十一步 make_fission.py 的 split/floor_of/exam_file/dumps，
只换数据集与随机数序列（种子 [FISSION_SEED, 2, idx]，与 adult 的 [FISSION_SEED, idx] 互不干扰，adult 既有卷一字不动）。

学习卷 → data/acsmob_alloc<tag>base_s<seed>_fisA/measured_2way.json（真表 CSV 原样复制，σ' = √2σ，地板_A = 2×原地板）
模拟卷 → results/fission/acsmob_alloc<tag>base_s<seed>_fisB.json（只供阅卷）
地板表 → results/fission/floors_acsmob.json（floor_split_str 供启动脚本逐字取用）
幂等：已存在文件须逐字相同，否则报错不覆盖。

用法：cd residual-evolution && .venv/bin/python logs_eps/acsmob/make_fission_acsmob.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps/fission"))
from make_fission import FISSION_SEED, TAGS, dumps, exam_file, floor_of, split  # noqa: E402

DATASET_ID = 2  # 1 = adult（原脚本未带此位），2 = acsmob
CONFIGS = [("1A", t, s) for t in TAGS for s in (1, 2, 3)]


def main() -> None:
    out_b = ROOT / "results/fission"
    out_b.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (exam, tag, seed) in enumerate(CONFIGS):
        src, fname, mode = f"acsmob_alloc{tag}base_s{seed}", "measured_2way.json", "none"
        src_dir = ROOT / "data" / src
        xp = exam_file(src_dir, fname)
        payload = json.load(open(xp, encoding="utf-8"))
        rng = np.random.default_rng([FISSION_SEED, DATASET_ID, idx])
        a, b, sig = split(payload, rng)
        f0 = floor_of(sig, mode)
        fa = floor_of(np.array([q["sigma"] for q in a["queries"]]), mode)
        meta = dict(source=str(xp.relative_to(ROOT)), fission_seed=[FISSION_SEED, DATASET_ID], index=idx,
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
                    raise SystemExit(f"{path} 已存在且内容不同，拒绝覆盖")
                status = "已存在逐字同"
            else:
                path.write_text(text, encoding="utf-8")
        rows.append(dict(exam=exam, tag=tag, seed=seed, src=src, data=name_a, exam_file=xp.name, mode=mode,
                         questions=int(sig.size), floor_source=f0, floor_split=fa, floor_split_str=repr(fa), index=idx))
        print(f"[{idx:2d}] {exam} ε={tag} s{seed} {name_a}/{xp.name} 题 {sig.size} 地板_原 {f0!r} 地板_A {fa!r}（÷原 {fa / f0:.6f}） {status}")
    json.dump(rows, open(out_b / "floors_acsmob.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {out_b / 'floors_acsmob.json'}（{len(rows)} 格）")


if __name__ == "__main__":
    sys.exit(main())
