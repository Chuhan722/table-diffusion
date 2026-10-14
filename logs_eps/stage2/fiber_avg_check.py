"""纤维洗牌离线核查：单张洗牌表 vs K 张洗牌平均，没测的对离真值/离 PGM 分布（只诊断，不参与方法）。
用法: .venv/bin/python logs_eps/stage2/fiber_avg_check.py <eps_tag> <seed> <K> <passes>
"""
import sys, json, numpy as np, pandas as pd
from itertools import combinations
sys.path.insert(0, "scripts")
from fiber_shuffle import load_exam_sets, neighbors_from_sets, group_ids, shuffle_column, marg
tag, s, K, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
import glob
df = pd.read_csv(f"results/stage1/adult_eps{tag}_hybrid_seed{s}.csv"); cols = list(df.columns)
X0 = np.stack([df[c].to_numpy(np.int64) for c in cols], 1); N, F = X0.shape
sets = load_exam_sets(glob.glob(f"data/adult_aimrep{tag}_s{s}/measured_*query.json")[0]); nb = neighbors_from_sets(cols, sets)
R = pd.read_csv("data/adult/adult.csv"); R = np.stack([R[c].to_numpy(np.int64) for c in cols], 1)
doms = [int(max(R[:, j].max(), X0[:, j].max())) + 1 for j in range(F)]
pairs = list(combinations(range(F), 2)); meas = {tuple(sorted((cols.index(a), cols.index(b)))) for a, b in sets if len((a, b)) == 2 and True} if False else None
meas = {tuple(sorted((cols.index(q[0]), cols.index(q[1])))) for q in sets if len(q) == 2}
un = [p for p in pairs if p not in meas]
z = np.load(f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz")
truth = {p: marg(R, p, doms) for p in pairs}; pgm = {p: z["m2_" + "_".join(map(str, p))] for p in pairs}
def score(mm, ref): return np.mean([np.abs(mm[p] - ref[p]).sum() / N for p in un]) * 1e6
cur = {p: marg(X0, p, doms) for p in pairs}
print(f"eps={tag} s{s}: 起点 没测 vs真值 {score(cur,truth):,.0f}  vs PGM分布 {score(cur,pgm):,.0f} | PGM分布 vs 真值 {score(pgm,truth):,.0f}")
acc = {p: np.zeros_like(truth[p]) for p in un}; singles = []
for k in range(K):
    rng = np.random.default_rng(1000 + k); X = X0.copy()
    for t in range(P):
        for a in rng.permutation(F): shuffle_column(X, int(a), group_ids(X, nb[int(a)]), rng)
    m = {p: marg(X, p, doms) for p in un}; singles.append((score(m, truth), score(m, pgm)))
    for p in un: acc[p] += m[p]
    if k + 1 in (1, 2, 5, 10, 20, 50, 100, K):
        avg = {p: acc[p] / (k + 1) for p in un}
        print(f"  K={k+1:2d} 单张均: vs真值 {np.mean([a for a,_ in singles]):,.0f} vsPGM {np.mean([b for _,b in singles]):,.0f} | K 张平均: vs真值 {score(avg,truth):,.0f} vsPGM {score(avg,pgm):,.0f}")
