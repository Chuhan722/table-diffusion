"""通用全轴阅卷器，四数据集，含零空间校验（一阶二阶与基表逐格相同）。

用法: .venv/bin/python scripts/score_any.py 数据集 基表.csv [待阅表.csv ...]
原为 /tmp/score_any.py（09-28 起各步阅卷同一把尺），10-11 收进仓库以免丢失，逻辑逐字未动。
输出每表一行全分项，二阶为全字段对 TVD 均值，三阶抽样 2000 个三元组（固定种子），
峰度用每字段值随机嵌入的线性分数，全表同一把尺。
第一张表为基表，其余表逐一与基表比对全部一阶二阶计数，完全相同才算零空间合法。
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resevo.dataset import load_queries, load_table  # noqa: E402
from resevo.metrics import load_heldout_queries  # noqa: E402
from eval_plants import spec_answers, to_matrix  # noqa: E402

CONF = {
    "nltcs": ("data/nltcs/nltcs.csv", "data/nltcs/measured_480query.json",
              "data/nltcs/heldout_1224query.json"),
    "plants": ("data/plants/plants.csv", "data/plants/measured_9248query.json",
               "data/plants/heldout_1224query.json"),
    "adult": ("data/adult/adult.csv", "data/adult/measured_14123query.json",
              "data/adult/heldout_1224query.json"),
    "acsmob": ("data/acsmob/acsmob.csv", "data/acsmob/measured_10895query.json",
               "data/acsmob/heldout_1224query.json"),
    "test300": ("data/test_300x10/test_300x10.csv",
                "data/test_300x10/measured_698query.json",
                "data/test_300x10/heldout_1224query.json"),
}

ds = sys.argv[1]
tables = sys.argv[2:]
tp, mp, hp = CONF[ds]
schema, real_rows = load_table(str(ROOT / tp))
N = len(real_rows)
Xr = to_matrix(schema, real_rows).astype(np.int64)
F = schema.num_fields
doms = [len(d) for d in schema.domains]
measured = load_queries(str(ROOT / mp))
heldout = load_heldout_queries(str(ROOT / hp))
hs = [q for q in heldout if any(c["operator"] == "halfspace" for c in q.conditions)]
eq = [q for q in heldout if not any(c["operator"] == "halfspace" for c in q.conditions)]
y_m = np.array([q.result for q in measured])
y_hs = np.array([q.result for q in hs])
y_eq = np.array([q.result for q in eq])

pairs = list(combinations(range(F), 2))
all_trips = list(combinations(range(F), 3))
tr_rng = np.random.default_rng(11)
if len(all_trips) > 2000:
    idx = tr_rng.choice(len(all_trips), size=2000, replace=False)
    trips = [all_trips[i] for i in sorted(idx)]
else:
    trips = all_trips


def marg(X, cols):
    key = X[:, cols[0]].astype(np.int64)
    size = doms[cols[0]]
    for c in cols[1:]:
        key = key * doms[c] + X[:, c]
        size *= doms[c]
    return np.bincount(key, minlength=size)


real_m2 = [marg(Xr, p) for p in pairs]
real_m3 = [marg(Xr, t) for t in trips]

emb_rng = np.random.default_rng(7)
W = [emb_rng.normal(size=(64, d)) for d in doms]


def kurt(X):
    S = np.zeros((len(X), 64))
    for f in range(F):
        S += W[f].T[X[:, f]]
    mu, sd = S.mean(0), S.std(0)
    z = (S - mu) / np.where(sd > 0, sd, 1)
    return float((z ** 4).mean() - 3)


base_m1 = base_m2 = None
print("表 | measured | 二阶 | 三阶均 | 三阶top5% | 三阶最大 | 等值 | 半空间 | 半空间偏 | 峰度 | 零空间")
for path in tables:
    _s, rows = load_table(path)
    X = to_matrix(schema, rows).astype(np.int64)
    m1 = [np.bincount(X[:, f], minlength=doms[f]) for f in range(F)]
    m2 = [marg(X, p) for p in pairs]
    if base_m1 is None:
        base_m1, base_m2 = m1, m2
        legal = "基表"
    else:
        ok1 = all(np.array_equal(a, b) for a, b in zip(m1, base_m1))
        ok2 = all(np.array_equal(a, b) for a, b in zip(m2, base_m2))
        legal = "通过" if (ok1 and ok2) else "失败"
    m = np.abs(spec_answers(measured, schema, X) - y_m).mean() / N
    e2 = np.array([np.abs(c - r).sum() / N for c, r in zip(m2, real_m2)])
    e3 = np.array([np.abs(marg(X, t) - r).sum() / N for t, r in zip(trips, real_m3)])
    top5 = np.sort(e3)[-max(1, len(e3) // 20):].mean()
    d_hs = (spec_answers(hs, schema, X) - y_hs) / N if hs else np.zeros(1)
    d_eq = np.abs(spec_answers(eq, schema, X) - y_eq) / N if eq else np.zeros(1)
    name = Path(path).stem
    print(
        f"{name} | {m:.6f} | {e2.mean():.6f} | {e3.mean():.6f} | {top5:.6f} | "
        f"{e3.max():.6f} | {d_eq.mean():.6f} | {np.abs(d_hs).mean():.6f} | "
        f"{d_hs.mean():+.6f} | {kurt(X):+.3f} | {legal}"
    )
