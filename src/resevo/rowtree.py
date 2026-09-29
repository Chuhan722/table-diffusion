"""整行树采样器，考卷二阶答案搭 Chow-Liu 骨架，祖先采样打印整行。

设计（整行采样棋步战役）：引擎的菜单棋步全是逐字段修补（改零件，
抄供体，探索），行内多字段联动只能靠抄真行，长尾模式抄不出来，
加噪场促聚棋还会被噪声残差顶死（组合尺战役验尸定案）。本器
只用已购统计量（训练卷全二阶答案，加噪场即噪声答案，同餐合规，
不看真表一眼）搭字段依赖树，按树条件分布一次生成整行——
行内搭配出生自带，聚性来自生成而非抄行。

树口粮为恰好两条件等值题（eq2way），负答案截零（噪声卷），
互信息定边权，最大生成树定骨架，根取度数最大字段，
条件概率表由联表行归一（零行退化为父边际的均匀混合）。
采样为标准祖先采样：根按边际抽，子按父值行抽，BFS 序到底。
接受与否不归本器管，照旧走残差核算与签筒，红线不动。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TreeRowSampler:
    """Chow-Liu 树整行采样器，fit 吃考卷答案，sample 打印整行元组。"""

    def __init__(self) -> None:
        self.order: list[int] | None = None  # BFS 采样序
        self.parent: list[int] | None = None  # 每字段的父，根为 -1
        self.root_marginal: NDArray[np.float64] | None = None
        self.cpt: dict[int, NDArray[np.float64]] | None = None  # child -> [dp, dc]
        self._domains: list[list[str]] | None = None

    def fit(self, specs, schema) -> "TreeRowSampler":
        F = schema.num_fields
        if F < 2:
            raise ValueError("树采样至少两个字段")
        doms = [list(d) for d in schema.domains]
        code = [{v: i for i, v in enumerate(d)} for d in doms]
        sizes = [len(d) for d in doms]
        pair: dict[tuple[int, int], NDArray[np.float64]] = {}
        for sp in specs:
            conds = sp.conditions
            if len(conds) != 2:
                continue
            if any(c["operator"] != "==" for c in conds):
                continue
            a = schema.field_index(conds[0]["attribute"])
            b = schema.field_index(conds[1]["attribute"])
            va = code[a][conds[0]["value"]]
            vb = code[b][conds[1]["value"]]
            if a > b:
                a, b, va, vb = b, a, vb, va
            t = pair.get((a, b))
            if t is None:
                t = np.zeros((sizes[a], sizes[b]))
                pair[(a, b)] = t
            # 噪声卷负答案截零，树是提议分布，接受另有残差把关
            t[va, vb] = max(0.0, float(sp.result))
        if not pair:
            raise ValueError("考卷里没有两条件等值题，树无口粮")

        # 互信息边权，联表各自归一防考卷缺对时总量口径不一，
        # 截零后全零的对记零信息边，可入树但最后才选，条件表退化均匀
        mi = np.full((F, F), -1.0)
        for (a, b), t in pair.items():
            s = t.sum()
            if s <= 0:
                mi[a, b] = mi[b, a] = 0.0
                continue
            p = t / s
            pa = p.sum(axis=1, keepdims=True)
            pb = p.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = np.where(p > 0, np.log(p / (pa @ pb)), 0.0)
            mi[a, b] = mi[b, a] = float((p * logs).sum())

        # Prim 最大生成树，缺边（考卷没这对）不可入树
        in_tree = [0]
        parent = [-1] * F
        while len(in_tree) < F:
            best, ba, bb = -np.inf, -1, -1
            for a in in_tree:
                for b in range(F):
                    if b in in_tree or mi[a, b] < 0:
                        continue
                    if mi[a, b] > best:
                        best, ba, bb = mi[a, b], a, b
            if bb < 0:
                raise ValueError("考卷二阶覆盖不连通，树搭不起来")
            parent[bb] = ba
            in_tree.append(bb)
        order = in_tree  # Prim 加入序天然父先子后

        # 根边际与条件概率表
        def _pair_prob(a: int, b: int) -> NDArray[np.float64]:
            """归一联表按 (a,b) 方向取出。"""
            t = pair[(a, b)] if a < b else pair[(b, a)].T
            s = t.sum()
            return t / s if s > 0 else np.full_like(t, 1.0 / t.size)

        root = order[0]
        anyb = next(b for (a, b) in [
            (min(root, j), max(root, j)) for j in range(F) if j != root
            and (min(root, j), max(root, j)) in pair
        ])
        pr = _pair_prob(root, anyb).sum(axis=1)
        pr = pr / pr.sum() if pr.sum() > 0 else np.full(sizes[root], 1.0 / sizes[root])
        cpt: dict[int, NDArray[np.float64]] = {}
        for c in order[1:]:
            p = parent[c]
            joint = _pair_prob(p, c)  # [dp, dc]
            rows = joint.sum(axis=1, keepdims=True)
            uniform = np.full((1, sizes[c]), 1.0 / sizes[c])
            cpt[c] = np.where(rows > 0, joint / np.maximum(rows, 1e-300), uniform)

        self.order, self.parent = order, parent
        self.root_marginal, self.cpt, self._domains = pr, cpt, doms
        return self

    def sample(self, rng: np.random.Generator, n: int) -> list[tuple[str, ...]]:
        """祖先采样 n 行，输出与真表同构的字段值元组。"""
        if self.order is None:
            raise ValueError("先 fit 再采样")
        F = len(self.parent)
        codes = np.empty((n, F), dtype=np.int64)
        root = self.order[0]
        codes[:, root] = rng.choice(len(self.root_marginal), size=n, p=self.root_marginal)
        for c in self.order[1:]:
            cp = self.cpt[c]  # [dp, dc]
            pv = codes[:, self.parent[c]]
            u = rng.random(n)
            cum = np.cumsum(cp, axis=1)
            codes[:, c] = (u[:, None] > cum[pv]).sum(axis=1)
        doms = self._domains
        return [
            tuple(doms[j][int(codes[i, j])] for j in range(F)) for i in range(n)
        ]
