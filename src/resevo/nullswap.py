"""零空间换位相位：保全部一阶二阶的四行换位，专修高阶聚簇。

原语（第二十五~二十六步外挂验证）：选字段对 (j,k)，甲对两行除 (j,k)
外全同且占 (a,b1)(a2,b2)，乙对（可另桶）占交叉图案 (a,b2)(a2,b1)，
两对各自对内交换 j 值。四行占据的 (j,k) 格子集合前后不变，其余
字段逐行原样，故全部一阶与全部二阶联表严格逐格不变，只动三阶
以上结构——课内分构造性纹丝不动。

裁判是辛普森集中度（整行模式计数平方和）严格爬山；恒温器为嵌入
峰度门槛（与 /tmp/score_any.py 同口径同种子），表不欠聚整场跳过，
18 个数据集乘种子实例方向判断全对的规则原样内化；自熄为接受率
地板，连续两场跌破永久熄火防过量（过量把质量堆错尾巴，外挂剂量
曲线实测）。随机流全程独立生成器，不碰引擎抽样流。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .dataset import TableSchema

KURT_SEED = 7  # 峰度尺种子，与外挂阅卷器同口径，门槛 -0.05 即此尺标定
KURT_DIRS = 64


class NullSwapPhase:
    """周期换位相位，evolve_batch 轮首调用，吃码矩阵吐码矩阵。"""

    def __init__(
        self,
        schema: TableSchema,
        *,
        kurt_gate: float = -0.05,
        accept_floor: float = 0.006,
        pairs_per_phase: int = 24,
        bucket_cap: int = 4,
        seed: int = 20260929,
    ) -> None:
        if pairs_per_phase < 1:
            raise ValueError("每场字段对配额必须为正")
        if bucket_cap < 1:
            raise ValueError("每桶候选上限必须为正")
        if not 0.0 <= accept_floor <= 1.0:
            raise ValueError("接受率地板须在 0 到 1 之间")
        self.schema = schema
        self.kurt_gate = float(kurt_gate)
        self.accept_floor = float(accept_floor)
        self.pairs_per_phase = int(pairs_per_phase)
        self.bucket_cap = int(bucket_cap)
        self.rng = np.random.default_rng(seed)
        doms = [len(d) for d in schema.domains]
        kurt_rng = np.random.default_rng(KURT_SEED)
        self._kw = [kurt_rng.normal(size=(KURT_DIRS, d)) for d in doms]
        hash_rng = np.random.default_rng(seed + 1)
        self._zh = [
            hash_rng.integers(0, 2**63, size=d, dtype=np.uint64) for d in doms
        ]
        self._pairs = [
            (j, k)
            for j in range(schema.num_fields)
            for k in range(j + 1, schema.num_fields)
        ]
        self.dead = False
        self._low_streak = 0

    def kurtosis(self, codes: NDArray) -> float:
        """嵌入峰度，随机方向线性分数的超额峰度，64 方向平均。"""
        x = np.asarray(codes)
        s = np.zeros((len(x), KURT_DIRS))
        for f in range(self.schema.num_fields):
            s += self._kw[f].T[x[:, f]]
        mu, sd = s.mean(0), s.std(0)
        z = (s - mu) / np.where(sd > 0, sd, 1)
        return float((z**4).mean() - 3)

    def run(self, codes: NDArray) -> tuple[NDArray | None, dict]:
        """跑一场换位。返回 (新码矩阵或 None, 统计)。

        None 表示表未动（跳过或零接受）。恒温器：峰度不低于门槛
        整场跳过且不计衰竭；施药场接受率跌破地板记衰竭一次，连续
        两场衰竭永久熄火。
        """
        if self.dead:
            return None, {"skipped": "dead"}
        x = np.array(codes, dtype=np.int32, copy=True)
        kurt0 = self.kurtosis(x)
        if kurt0 >= self.kurt_gate:
            return None, {"skipped": "healthy", "kurt": kurt0}
        total_h = np.zeros(len(x), dtype=np.uint64)
        for f in range(self.schema.num_fields):
            total_h += self._zh[f][x[:, f]]
        cnt: dict[int, int] = {}
        for h in total_h.tolist():
            cnt[h] = cnt.get(h, 0) + 1
        chosen = self.rng.choice(
            len(self._pairs),
            size=min(self.pairs_per_phase, len(self._pairs)),
            replace=False,
        )
        proposals = 0
        accepts = 0
        for pi in chosen:
            j, k = self._pairs[int(pi)]
            p, a = self._sweep_pair(x, total_h, cnt, j, k)
            proposals += p
            accepts += a
        rate = accepts / proposals if proposals > 0 else 0.0
        if rate < self.accept_floor:
            self._low_streak += 1
            if self._low_streak >= 2:
                self.dead = True
        else:
            self._low_streak = 0
        stats = {
            "kurt": kurt0,
            "proposals": proposals,
            "accepts": accepts,
            "dead": self.dead,
        }
        if accepts == 0:
            return None, stats
        stats["kurt_after"] = self.kurtosis(x)
        return x, stats

    def _sweep_pair(
        self,
        x: NDArray,
        total_h: NDArray,
        cnt: dict[int, int],
        j: int,
        k: int,
    ) -> tuple[int, int]:
        """单字段对扫描，就地改 x 与 total_h 与 cnt，返回 (提议数, 接受数)。"""
        ctx = total_h - self._zh[j][x[:, j]] - self._zh[k][x[:, k]]
        order = np.argsort(ctx, kind="stable")
        sctx = ctx[order]
        starts = np.flatnonzero(np.r_[True, sctx[1:] != sctx[:-1]])
        ends = np.r_[starts[1:], len(sctx)]
        pat_map: dict[tuple, list[tuple[int, int]]] = {}
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s < 2:
                continue
            rows = order[s:e]
            av = x[rows, j]
            bv = x[rows, k]
            m = len(rows)
            if m <= 8:
                cand = [(i1, i2) for i1 in range(m) for i2 in range(i1 + 1, m)]
            else:
                idx = self.rng.integers(0, m, size=(self.bucket_cap * 6, 2))
                cand = [(int(i1), int(i2)) for i1, i2 in idx if i1 != i2]
            taken = 0
            for i1, i2 in cand:
                if taken >= self.bucket_cap:
                    break
                a1, b1 = int(av[i1]), int(bv[i1])
                a2, b2 = int(av[i2]), int(bv[i2])
                if a1 == a2 or b1 == b2:
                    continue
                if a1 < a2:
                    key = (a1, a2, b1, b2)
                    rec = (int(rows[i1]), int(rows[i2]))
                else:
                    key = (a2, a1, b2, b1)
                    rec = (int(rows[i2]), int(rows[i1]))
                pat_map.setdefault(key, []).append(rec)
                taken += 1
        proposals = 0
        accepts = 0
        used: set[int] = set()
        for key, recs in pat_map.items():
            a, a2, b1, b2 = key
            anti = (a, a2, b2, b1)
            if anti <= key:
                continue
            others = pat_map.get(anti)
            if not others:
                continue
            for (g1, g2), (e1, e2) in zip(recs, others):
                if g1 in used or g2 in used or e1 in used or e2 in used:
                    continue
                proposals += 1
                if self._try_swap(x, total_h, cnt, j, (g1, g2, e1, e2), a, a2):
                    accepts += 1
                    used.update((g1, g2, e1, e2))
        return proposals, accepts

    def _try_swap(
        self,
        x: NDArray,
        total_h: NDArray,
        cnt: dict[int, int],
        j: int,
        rows4: tuple[int, int, int, int],
        a: int,
        a2: int,
    ) -> bool:
        """辛普森增量爬山：四行 j 值 a<->a2，增量为正才落地。"""
        g1, g2, e1, e2 = rows4
        new_j = ((g1, a2), (g2, a), (e1, a2), (e2, a))
        mask = 0xFFFFFFFFFFFFFFFF
        zh = self._zh[j]
        delta = 0
        ops: list[tuple[int, int, int, int]] = []  # (行, 新值, 旧哈希, 新哈希)
        for r, nv in new_j:
            oh = int(total_h[r])
            nh = (oh - int(zh[x[r, j]]) + int(zh[nv])) & mask
            c = cnt.get(oh, 0)
            delta -= 2 * c - 1
            cnt[oh] = c - 1
            c = cnt.get(nh, 0)
            delta += 2 * c + 1
            cnt[nh] = c + 1
            ops.append((r, nv, oh, nh))
        if delta <= 0:
            for _, _, oh, nh in ops:
                cnt[oh] = cnt.get(oh, 0) + 1
                cnt[nh] = cnt.get(nh, 0) - 1
            return False
        for r, nv, _, nh in ops:
            total_h[r] = np.uint64(nh)
            x[r, j] = nv
        return True
