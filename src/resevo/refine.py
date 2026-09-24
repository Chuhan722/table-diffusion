"""零空间聚簇精修，保全部一阶二阶边缘只动三阶以上，带内在门槛与停止规则。

移动原语，选字段对 (j,k)，找两对行，每对内部除 (j,k) 外全列相同，
甲对在 (j,k) 上占 (a,b1)(a2,b2)，乙对占交叉图案 (a,b2)(a2,b1)，a≠a2 且 b1≠b2，
动作为两对同时在 j 或 k 上做对内互换。对内互换不动本对一阶与 (j,k) 对外联表，
两对在 (j,k) 联表上的增量互补抵消，故全部一阶二阶边缘精确不变，只动三阶以上，
对卷面 measured 答案的损失（含带噪答案）构造性严格不变。

三条内在规则全部只看合成表自身，不看真表不看 heldout 答案，红线干净：
门槛，嵌入峰度显著负（表比高斯参照更平，欠聚簇病征）才施药，否则原样输出；
目标，辛普森集中度（完整行模式计数平方和）贪心爬山；
停止，单扫接受率跌破阈值或无接受即收手，防止过量把质量堆进错误尾巴。

门槛阈值 -0.05 与停止阈值 0.006 系六个数据集实例（nltcs plants adult 乘零噪声
与 eps1）事后标定，方向判断 18/18，详见工作笔记第二十六、二十七步。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

KURT_GATE_DEFAULT = -0.05
STOP_RATE_DEFAULT = 0.006
HASH_SEED = 20260924
EMBED_SEED = 7
EMBED_DIM = 64


@dataclass
class SweepStat:
    """单扫统计，接受数提议数与表聚簇状态。"""

    sweep: int
    accepted: int
    proposals: int
    rate: float
    simpson: int
    unique_rows: int


@dataclass
class RefineReport:
    """一次精修的完整回执。"""

    kurtosis: float
    gate: float
    treated: bool
    forced: bool
    sweeps: list[SweepStat] = field(default_factory=list)

    @property
    def accepted_total(self) -> int:
        return sum(s.accepted for s in self.sweeps)


def embed_kurtosis(
    X: np.ndarray,
    domain_sizes: Sequence[int],
    dim: int = EMBED_DIM,
    seed: int = EMBED_SEED,
) -> float:
    """体温计，每字段值随机嵌入的线性分数超额峰度，负值表示比高斯参照更平。"""
    rng = np.random.default_rng(seed)
    n, num_fields = X.shape
    scores = np.zeros((n, dim))
    for f in range(num_fields):
        w = rng.normal(size=(dim, int(domain_sizes[f])))
        scores += w.T[X[:, f]]
    sd = scores.std(axis=0)
    z = (scores - scores.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return float((z**4).mean() - 3.0)


def _marginal_counts(
    X: np.ndarray, domain_sizes: Sequence[int]
) -> list[np.ndarray]:
    """全部一阶与二阶联表计数，验证用。"""
    num_fields = X.shape[1]
    out = [
        np.bincount(X[:, f], minlength=int(domain_sizes[f]))
        for f in range(num_fields)
    ]
    for j, k in combinations(range(num_fields), 2):
        dj, dk = int(domain_sizes[j]), int(domain_sizes[k])
        flat = X[:, j].astype(np.int64) * dk + X[:, k]
        out.append(np.bincount(flat, minlength=dj * dk))
    return out


def verify_null_space(
    X0: np.ndarray, X1: np.ndarray, domain_sizes: Sequence[int]
) -> bool:
    """两表全部一阶二阶计数逐格相同才为真，精修输出的强制自检。"""
    if X0.shape != X1.shape:
        return False
    a = _marginal_counts(X0, domain_sizes)
    b = _marginal_counts(X1, domain_sizes)
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def refine_rows(
    X: np.ndarray,
    domain_sizes: Sequence[int],
    *,
    seed: int,
    kurt_gate: float = KURT_GATE_DEFAULT,
    stop_rate: float = STOP_RATE_DEFAULT,
    max_sweeps: int = 32,
    pairs_per_sweep: int = 0,
    force: bool = False,
    on_sweep: Callable[[SweepStat], None] | None = None,
) -> tuple[np.ndarray, RefineReport]:
    """门槛量测加零空间爬山加停止规则的整套精修，返回新表与回执。

    输入 X 为值下标矩阵不被修改，pairs_per_sweep 为 0 表示每扫遍历全部字段对，
    force 为真时跳过门槛强制施药（验证实验用，常规调用勿开）。
    """
    X = np.ascontiguousarray(X, dtype=np.int16)
    n, num_fields = X.shape
    kurt = embed_kurtosis(X, domain_sizes)
    treated = force or kurt < kurt_gate
    report = RefineReport(kurtosis=kurt, gate=kurt_gate, treated=treated, forced=force)
    if not treated:
        return X.copy(), report

    X = X.copy()
    rng = np.random.default_rng(seed)
    hk = np.random.default_rng(HASH_SEED)
    maxdom = max(int(d) for d in domain_sizes)
    keys = hk.integers(1, 2**63, size=(num_fields, maxdom), dtype=np.uint64)
    # 六十四位上下文哈希，行哈希为全字段钥匙之和，uint64 回绕加法
    with np.errstate(over="ignore"):
        hall = keys[np.arange(num_fields)[None, :], X].sum(axis=1, dtype=np.uint64)

    pat_cnt: dict[bytes, int] = {}
    for r in range(n):
        key = X[r].tobytes()
        pat_cnt[key] = pat_cnt.get(key, 0) + 1

    def ctx_equal(r1: int, r2: int, j: int, k: int) -> bool:
        """哈希撞车防线，真实校验两行除 (j,k) 外逐列相同。"""
        mask = np.ones(num_fields, dtype=bool)
        mask[[j, k]] = False
        return bool(np.array_equal(X[r1, mask], X[r2, mask]))

    def swap_delta(two_pairs, g: int) -> int:
        """两对行各自对内互换字段 g 后的辛普森增量，不落地试算。"""
        tmp: dict[bytes, int] = {}
        delta = 0
        for r1, r2 in two_pairs:
            v1, v2 = int(X[r1, g]), int(X[r2, g])
            for r, newv in ((r1, v2), (r2, v1)):
                old = X[r].tobytes()
                cnt_old = pat_cnt.get(old, 0) + tmp.get(old, 0)
                sav = X[r, g]
                X[r, g] = newv
                new = X[r].tobytes()
                X[r, g] = sav
                cnt_new = pat_cnt.get(new, 0) + tmp.get(new, 0)
                delta += 2 * (cnt_new - cnt_old + 1)
                tmp[old] = tmp.get(old, 0) - 1
                tmp[new] = tmp.get(new, 0) + 1
        return delta

    def apply_swap(two_pairs, g: int) -> None:
        for r1, r2 in two_pairs:
            v1, v2 = int(X[r1, g]), int(X[r2, g])
            for r, newv in ((r1, v2), (r2, v1)):
                old = X[r].tobytes()
                pat_cnt[old] -= 1
                if not pat_cnt[old]:
                    del pat_cnt[old]
                with np.errstate(over="ignore"):
                    hall[r] = hall[r] - keys[g, X[r, g]] + keys[g, newv]
                X[r, g] = np.int16(newv)
                new = X[r].tobytes()
                pat_cnt[new] = pat_cnt.get(new, 0) + 1

    all_pairs = list(combinations(range(num_fields), 2))
    for sweep in range(max_sweeps):
        order = rng.permutation(len(all_pairs))
        if pairs_per_sweep > 0:
            order = order[:pairs_per_sweep]
        got = proposals = 0
        for pi in order:
            j, k = all_pairs[pi]
            with np.errstate(over="ignore"):
                ctx = hall - keys[j, X[:, j]] - keys[k, X[:, k]]
            srt = np.argsort(ctx, kind="stable")
            vals = ctx[srt]
            bounds = np.flatnonzero(np.diff(vals)) + 1
            starts = np.concatenate(([0], bounds))
            ends = np.concatenate((bounds, [len(vals)]))
            # 桶内收集对角候选对，值对在 j 与 k 上都不同
            cands: dict[frozenset, list] = {}
            for s0, s1 in zip(starts, ends):
                if s1 - s0 < 2:
                    continue
                rows_b = srt[s0:s1]
                groups: dict[tuple, list] = {}
                for r in rows_b:
                    groups.setdefault(
                        (int(X[r, j]), int(X[r, k])), []
                    ).append(int(r))
                kinds = list(groups)
                if len(kinds) < 2:
                    continue
                rng.shuffle(kinds)
                picked = 0
                for i1 in range(len(kinds)):
                    for i2 in range(i1 + 1, len(kinds)):
                        va, vb = kinds[i1], kinds[i2]
                        if va[0] == vb[0] or va[1] == vb[1]:
                            continue
                        r1 = groups[va][int(rng.integers(len(groups[va])))]
                        r2 = groups[vb][int(rng.integers(len(groups[vb])))]
                        pk = frozenset((va, vb))
                        cands.setdefault(pk, []).append((r1, r2, va, vb))
                        picked += 1
                        if picked >= 4:
                            break
                    if picked >= 4:
                        break
            # 甲对配交叉图案的乙对，贪心接受正增量
            used: set[int] = set()
            for pk, lst in cands.items():
                va, vb = tuple(pk)
                cross = frozenset(((va[0], vb[1]), (vb[0], va[1])))
                partner = cands.get(cross)
                if not partner:
                    continue
                for r1, r2, _a, _b in lst:
                    if r1 in used or r2 in used:
                        continue
                    best = None
                    for r3, r4, _c, _d in partner[:20]:
                        if r3 in used or r4 in used or len({r1, r2, r3, r4}) < 4:
                            continue
                        proposals += 1
                        for g in (k, j):
                            d = swap_delta([(r1, r2), (r3, r4)], g)
                            if d > 0 and (best is None or d > best[0]):
                                best = (d, r3, r4, g)
                    if best:
                        _d, r3, r4, g = best
                        if not (
                            ctx_equal(r1, r2, j, k) and ctx_equal(r3, r4, j, k)
                        ):
                            continue
                        apply_swap([(r1, r2), (r3, r4)], g)
                        used.update((r1, r2, r3, r4))
                        got += 1
        rate = got / proposals if proposals else 0.0
        stat = SweepStat(
            sweep=sweep + 1,
            accepted=got,
            proposals=proposals,
            rate=rate,
            simpson=sum(c * c for c in pat_cnt.values()),
            unique_rows=len(pat_cnt),
        )
        report.sweeps.append(stat)
        if on_sweep is not None:
            on_sweep(stat)
        if not got or rate < stop_rate:
            break
    return X, report
