"""考卷冗余度：1 − rank(A)/m。

A 为"测量格子 ← 全表单元格"的线性映射。rank 的闭式：对测过的团取下闭包（所有子集），
每个属性子集 S 贡献 Π_{i∈S}(d_i − 1)（空集贡献 1）；加权不改变 rank。
m 为测量格数：1A 全二阶 = Σ_{i<j} d_i d_j；1B 分"原始录像"（重复团分开计）与
"重放考卷"（同团重复测量逆方差合并，生成器实际看到的）两种。
只读文件，不写结果；输出到标准输出。
"""
import itertools
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TAGS = ["0p1", "0p3", "1", "3", "10"]


def rank_of(cliques, domain):
    subsets = set()
    for cl in cliques:
        for r in range(len(cl) + 1):
            subsets.update(itertools.combinations(sorted(cl), r))
    total = 0
    for s in subsets:
        p = 1
        for i in s:
            p *= domain[i] - 1
        total += p
    return total


def main():
    path = os.path.join(ROOT, "results", "aimreplay", "adult_eps1_aimfull_seed1_measurements.json")
    d = json.load(open(path))["domain_sizes"]
    n = len(d)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m1a = sum(d[i] * d[j] for i, j in pairs)
    r1a = rank_of(pairs, d)
    print(f"1A 全二阶: m={m1a} rank={r1a} 冗余度={1 - r1a / m1a:.3f}")
    for tag in TAGS:
        for s in (1, 2, 3):
            f = os.path.join(ROOT, "results", "aimreplay", f"adult_eps{tag}_aimfull_seed{s}_measurements.json")
            if not os.path.exists(f):
                continue
            ms = json.load(open(f))["measurements"]
            cols = json.load(open(f))["columns"]
            idx = {c: k for k, c in enumerate(cols)}
            cl_all = [tuple(idx[c] for c in mm["clique"]) for mm in ms]
            cells = {cl: 1 for cl in cl_all}
            for cl in cells:
                p = 1
                for i in cl:
                    p *= d[i]
                cells[cl] = p
            m_raw = sum(cells[cl] for cl in cl_all)
            m_merged = sum(cells.values())
            r = rank_of(cells.keys(), d)
            print(f"1B ε={tag} s{s}: 团 {len(cl_all)}（去重 {len(cells)}）"
                  f" 原始 m={m_raw} 冗余度={1 - r / m_raw:.3f};"
                  f" 合并 m={m_merged} rank={r} 冗余度={1 - r / m_merged:.3f}")


if __name__ == "__main__":
    sys.exit(main())
