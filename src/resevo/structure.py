"""结构签筒，引擎两级接受的第二级，残差打平的方向按结构好坏重分签。

设计动机见工作笔记第二十九步，引擎在残差看不见的零空间方向上按签筒
（参考分布）盲走，把表越走越平，半空间类考题因此失分。两级设计：
第一级永远是残差，方向收益由 β 校准保证达标，签筒改形不碰增益不碰步长；
第二级只改签筒在各方向之间的相对质量，结构变好的方向签多，变坏的签少。

恒温开关，体温计低于门槛（表已过平）才启用第二级，回线即关，
健康表全程零干预。体温计与 refine.embed_kurtosis 同刻度同种子，
门槛 -0.05 的六数据集实例标定（18/18）直接沿用。

两把尺，甲整行抱团（完整行模式计数平方和的单行移动增量符号），
丙表形状（自带 seed 1234 一百二十八维嵌入的超额峰度增量符号，
与体温计 seed 7 六十四维刻度分离，防止目标与仪表合一自我欺骗）。

签筒改形每组离开质量精确保持，与距离签筒同一原则，同松紧只换形状，
因此每组保持概率不变，随机矩阵约束与步长语义零改动。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .refine import EMBED_DIM, EMBED_SEED, HASH_SEED, KURT_GATE_DEFAULT

STRUCT_BOOST_DEFAULT = 4.0
BING_SEED = 1234
BING_DIM = 128
_CHUNK = 16384


def _field_weights(
    domain_sizes: Sequence[int], dim: int, seed: int
) -> list[NDArray[np.float64]]:
    """每字段一张 dim 乘域宽的高斯权重表，生成顺序与 refine 逐位一致。"""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(dim, int(k))) for k in domain_sizes]


def _embed_scores(
    codes_g: NDArray[np.int32], weights: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """组码矩阵的嵌入分数 G 乘 dim。"""
    scores = np.zeros((codes_g.shape[0], weights[0].shape[0]))
    for f, w in enumerate(weights):
        scores += w.T[codes_g[:, f]]
    return scores


def _weighted_kurtosis(scores: NDArray[np.float64], counts) -> float:
    """按组重数加权的超额峰度，数学上等于展开表逐行计算。"""
    w = np.asarray(counts, dtype=np.float64)
    n = w.sum()
    mean = (scores * w[:, None]).sum(axis=0) / n
    d = scores - mean
    var = (d * d * w[:, None]).sum(axis=0) / n
    sd = np.sqrt(var)
    z = d / np.where(sd > 0, sd, 1.0)
    return float(((z**4 * w[:, None]).sum(axis=0) / n).mean() - 3.0)


def _moments_kurt(
    mom: NDArray[np.float64], n: float
) -> NDArray[np.float64]:
    """由原点矩批量算超额峰度，mom 形如 (..., 4, dim)，返回 (...,)。"""
    mu = mom[..., 0, :] / n
    ex2 = mom[..., 1, :] / n
    ex3 = mom[..., 2, :] / n
    ex4 = mom[..., 3, :] / n
    var = np.maximum(ex2 - mu * mu, 1e-12)
    mu4 = ex4 - 4 * mu * ex3 + 6 * mu * mu * ex2 - 3 * mu**4
    return (mu4 / (var * var)).mean(axis=-1) - 3.0


def shape_reference(
    ref_paths: NDArray[np.float64],
    offsets: NDArray[np.int64],
    group: NDArray[np.int64],
    factors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """签筒改形，路径质量乘结构因子后按组归还原离开质量。

    每组路径质量总和精确回到改形前，保持概率与步长语义零改动，
    只在组内重新分签，空组无路径天然不受影响。
    """
    shaped = ref_paths * factors
    starts = np.minimum(offsets[:-1], len(ref_paths))
    orig = np.add.reduceat(np.r_[ref_paths, 0.0], starts)
    new = np.add.reduceat(np.r_[shaped, 0.0], starts)
    empty = np.diff(offsets) == 0
    orig[empty] = 1.0
    new[empty] = 1.0
    scale = np.where(new > 0, orig / np.where(new > 0, new, 1.0), 1.0)
    return shaped * scale[group]


class StructShaper:
    """结构签筒状态机，量体温，门内算因子，门外零干预。

    ruler 取 watch 只记体温不改签筒（侦察模式，轨迹与不装完全一致），
    取 jia 或 bing 时体温低于 gate 才产出路径因子，否则返回 None。
    所有随机权重表构造一次复用，体温计自带独立生成器不碰引擎随机流。
    """

    def __init__(
        self,
        ruler: str,
        domain_sizes: Sequence[int],
        boost: float = STRUCT_BOOST_DEFAULT,
        gate: float = KURT_GATE_DEFAULT,
    ):
        if ruler not in ("watch", "jia", "bing"):
            raise ValueError(f"未知结构刻度 {ruler}")
        if boost <= 1.0:
            raise ValueError("签数增幅必须大于 1")
        self.ruler = ruler
        self.boost = float(boost)
        self.gate = float(gate)
        sizes = [int(k) for k in domain_sizes]
        self._thermo_w = _field_weights(sizes, EMBED_DIM, EMBED_SEED)
        if ruler == "jia":
            hk = np.random.default_rng(HASH_SEED)
            self._keys = hk.integers(
                1, 2**63, size=(len(sizes), max(sizes)), dtype=np.uint64
            )
        elif ruler == "bing":
            bw = _field_weights(sizes, BING_DIM, BING_SEED)
            # padded 张量供逐路径花式索引，字段真实域宽之外恒零不被引用
            self._bing_w = np.zeros((len(sizes), BING_DIM, max(sizes)))
            for f, w in enumerate(bw):
                self._bing_w[f, :, : w.shape[1]] = w

    def temperature(self, codes_g: NDArray[np.int32], counts) -> float:
        """体温计读数，refine.embed_kurtosis 同刻度的加权实现。"""
        return _weighted_kurtosis(_embed_scores(codes_g, self._thermo_w), counts)

    def active(self, temperature: float) -> bool:
        """恒温开关，只有病表（体温低于门槛）才启用第二级。"""
        return self.ruler in ("jia", "bing") and temperature < self.gate

    def path_factors(
        self, menu, codes_g: NDArray[np.int32], counts
    ) -> NDArray[np.float64]:
        """每条菜单路径的签数因子，结构变好乘 boost，变坏除 boost。"""
        if self.ruler == "jia":
            delta = self._jia_delta(menu, codes_g, counts)
        else:
            delta = self._bing_delta(menu, codes_g, counts)
        factors = np.ones(menu.num_paths, dtype=np.float64)
        factors[delta > 0] = self.boost
        factors[delta < 0] = 1.0 / self.boost
        return factors

    def _jia_delta(self, menu, codes_g, counts) -> NDArray[np.int64]:
        """整行抱团增量，单行从源模式移到目标模式的 Σc² 变化符号。

        分组表按状态去重，模式计数恰等于组重数，目标模式计数用
        源哈希加字段差分哈希在组哈希表里查，查无此组即计数为零。
        Δ(Σc²) = 2(c_目标 - c_源 + 1)，返回其符号载体 c_目标 - c_源 + 1。
        """
        keys = self._keys
        num_fields = codes_g.shape[1]
        cnts = np.asarray(counts, dtype=np.int64)
        with np.errstate(over="ignore"):
            h_g = keys[np.arange(num_fields)[None, :], codes_g].sum(
                axis=1, dtype=np.uint64
            )
            dh = np.zeros(menu.num_paths, dtype=np.uint64)
            for s in range(menu.fields.shape[1]):
                fld = menu.fields[:, s]
                mask = fld >= 0
                f = fld[mask]
                old = codes_g[menu.group[mask], f]
                new = menu.values[mask, s]
                dh[mask] += keys[f, new] - keys[f, old]
            target = h_g[menu.group] + dh
        order = np.argsort(h_g, kind="stable")
        h_sorted = h_g[order]
        c_sorted = cnts[order]
        idx = np.searchsorted(h_sorted, target)
        idx_c = np.minimum(idx, len(h_sorted) - 1)
        found = h_sorted[idx_c] == target
        c_target = np.where(found, c_sorted[idx_c], 0)
        return c_target - cnts[menu.group] + 1

    def _bing_delta(self, menu, codes_g, counts) -> NDArray[np.float64]:
        """表形状增量，单行移动后自带嵌入超额峰度的变化，分块矢量化。"""
        w = np.asarray(counts, dtype=np.float64)
        n = w.sum()
        scores = np.zeros((codes_g.shape[0], BING_DIM))
        for f in range(codes_g.shape[1]):
            scores += self._bing_w[f].T[codes_g[:, f]]
        mom = np.stack(
            [(scores ** (p + 1) * w[:, None]).sum(axis=0) for p in range(4)]
        )
        cur = float(_moments_kurt(mom, n))
        out = np.empty(menu.num_paths, dtype=np.float64)
        for lo in range(0, menu.num_paths, _CHUNK):
            hi = min(lo + _CHUNK, menu.num_paths)
            grp = menu.group[lo:hi]
            s_old = scores[grp]
            s_new = s_old.copy()
            for s in range(menu.fields.shape[1]):
                fld = menu.fields[lo:hi, s]
                mask = fld >= 0
                f = fld[mask]
                new = menu.values[lo:hi][mask, s]
                old = codes_g[grp[mask], f]
                s_new[mask] += self._bing_w[f, :, new] - self._bing_w[f, :, old]
            m_new = np.broadcast_to(
                mom, (hi - lo, 4, BING_DIM)
            ).copy()
            po = s_old.copy()
            pn = s_new.copy()
            for p in range(4):
                m_new[:, p, :] += pn - po
                if p < 3:
                    po = po * s_old
                    pn = pn * s_new
            out[lo:hi] = _moments_kurt(m_new, n) - cur
        return out
