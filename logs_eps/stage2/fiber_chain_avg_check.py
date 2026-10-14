import sys, glob, numpy as np, pandas as pd, time
from itertools import combinations
sys.path.insert(0, "scripts")
from fiber_shuffle import load_exam_sets, neighbors_from_sets, group_ids, shuffle_column, marg
tag, s, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
df = pd.read_csv(f"results/stage1/adult_eps{tag}_hybrid_seed{s}.csv"); cols = list(df.columns)
X = np.stack([df[c].to_numpy(np.int64) for c in cols], 1); N, F = X.shape
sets = load_exam_sets(glob.glob(f"data/adult_aimrep{tag}_s{s}/measured_*query.json")[0]); nb = neighbors_from_sets(cols, sets)
R = pd.read_csv("data/adult/adult.csv"); R = np.stack([R[c].to_numpy(np.int64) for c in cols], 1)
doms = [int(max(R[:, j].max(), X[:, j].max())) + 1 for j in range(F)]
pairs = list(combinations(range(F), 2)); meas = {tuple(sorted((cols.index(q[0]), cols.index(q[1])))) for q in sets if len(q) == 2}
un = [p for p in pairs if p not in meas]
z = np.load(f"results/stage2/adult_eps{tag}_pgmexact_seed{s}_exact.npz")
truth = {p: marg(R, p, doms) for p in pairs}; pgm = {p: z["m2_" + "_".join(map(str, p))] for p in pairs}
def score(mm, ref): return np.mean([np.abs(mm[p] - ref[p]).sum() / N for p in un]) * 1e6
rng = np.random.default_rng(7); acc = {p: np.zeros_like(truth[p]) for p in un}; t0 = time.time(); n = 0
for t in range(1, P + 1):
    for a in rng.permutation(F): shuffle_column(X, int(a), group_ids(X, nb[int(a)]), rng)
    if t >= 2:
        for p in un: acc[p] += marg(X, p, doms)
        n += 1
    if t in (21, 51, 101, P):
        avg = {p: acc[p] / n for p in un}
        print(f"单链 趟 2..{t} 共 {n} 张平均: vs真值 {score(avg,truth):,.0f} vsPGM {score(avg,pgm):,.0f}  用时 {time.time()-t0:.0f}s")
