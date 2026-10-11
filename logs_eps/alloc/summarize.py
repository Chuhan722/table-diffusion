"""第一步/第二步阅卷汇总：对每组表跑 scripts/score_any.py，按组取 3 种子均，输出 markdown 表。

用法: .venv/bin/python logs_eps/alloc/summarize.py adult 0p1 标签=glob [标签=glob ...]
"""
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ds, eps = sys.argv[1], sys.argv[2]
groups = [a.split("=", 1) for a in sys.argv[3:]]
cols = ["measured", "二阶", "三阶均", "三阶top5%", "三阶max", "等值", "半空间"]

def score_group(pat):
    """每组单独阅卷：score_any 按文件 stem 报分，不同目录同名表（如 results/ 与 results/baseline2/）放一起会互相覆盖。"""
    fs = sorted(f for f in glob.glob(str(ROOT / pat)) if "curve" not in f)
    if not fs:
        return []
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/score_any.py"), ds, str(ROOT / f"data/{ds}/{ds}.csv")] + fs,
        capture_output=True, text=True, check=True,
    ).stdout
    scores = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 9 or parts[0] in ("表", ds):
            continue
        try:
            scores[parts[0]] = [float(x) * 1e6 for x in parts[1:8]]
        except ValueError:
            continue
    return [scores[Path(f).stem] for f in fs if Path(f).stem in scores]

group_rows = {label: score_group(pat) for label, pat in groups}
print(f"| ε | 表 | n | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 3) + "|")
means = {}
for label, _ in groups:
    rows = group_rows[label]
    if not rows:
        continue
    m = np.mean(rows, axis=0)
    means[label] = m
    print(f"| {eps.replace('p', '.')} | {label} | {len(rows)} | " + " | ".join(f"{v:.0f}" for v in m) + " |")
