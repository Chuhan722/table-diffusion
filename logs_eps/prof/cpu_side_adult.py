"""本机 CPU 量引擎每轮 CPU 侧（分组 / 出菜单 / 码本同步）在 1A 终表规模上的耗时，并数终表不同状态数 G（10-14 提速空间评估用）。用法: .venv/bin/python logs_eps/prof/cpu_side_adult.py 3 1"""
import sys, time, cProfile, pstats
sys.path.insert(0, "src")
import numpy as np
from resevo.dataset import load_table, load_queries, StateRegistry, query_field_sets
from resevo.batchmenu import CodeBook, generate_batch_menu
from resevo.grouping import group_state_ids
from resevo.editspace import EditBudget

tag, seed = sys.argv[1], sys.argv[2]
d = f"data/adult_alloc{tag}base_s{seed}"
schema, real_rows = load_table(f"{d}/adult_alloc{tag}base_s{seed}.csv")
specs = load_queries(f"{d}/measured_2way.json")
_, rows = load_table(f"results/stage1/adult_eps{tag}_ours2way_seed{seed}.csv")
seen, field_sets = set(), []
for fs in query_field_sets(specs, schema):
    if 2 <= len(fs) <= 4 and fs not in seen:
        seen.add(fs); field_sets.append(fs)
reg = StateRegistry(schema, specs); reg.set_lazy_features(True)
ids = reg.register_table(rows)
cb = CodeBook(reg)
g = group_state_ids(ids)
G, Q = g.num_groups, len(specs)
print(f"eps={tag} s{seed}: 行 {len(rows)} 组(不同状态) G={G} 题 Q={Q}  G×Q float64 = {G*Q*8/1e9:.2f} GB, int8 {G*Q/1e9:.2f} GB")
budget = EditBudget()
rng = np.random.default_rng(1)
def t(fn, n=20):
    fn(); t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1000
print(f"分组 group_state_ids: {t(lambda: group_state_ids(ids)):.1f} ms")
print(f"码本 sync: {t(lambda: cb.sync()):.2f} ms")
codes = cb.sync(); codes_g = codes[g.unique_ids]
full = lambda: generate_batch_menu(codes_g, g.counts, cb.domain_sizes, rng, budget, field_sets)
m = full(); print(f"全量出菜单 (G={G}): {t(full, 5):.1f} ms, 路径 {m.num_paths}")
# 工作批：随机挑组凑 4096 行
perm = rng.permutation(G); cum = np.cumsum(g.counts[perm]); j = int(np.searchsorted(cum, 4096)); sel = np.sort(perm[:j+1])
sub = lambda: generate_batch_menu(codes_g[sel], g.counts[sel], cb.domain_sizes, rng, budget, field_sets)
m2 = sub(); print(f"工作批出菜单 (选 {len(sel)} 组 {int(g.counts[sel].sum())} 行): {t(sub):.1f} ms, 路径 {m2.num_paths}")
pr = cProfile.Profile(); pr.enable()
for _ in range(20): sub()
pr.disable()
st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(12)
