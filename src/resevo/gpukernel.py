"""GPU 批量核，cupy 后端，数学定义与 CPU 批量核逐项相同。

设计纪律，
- CPU 路径一行不动，本模块独立实现，跨后端对拍测试把守一致性。
- 确定性三原则，浮点聚合不用原子加，分段求和一律前缀和差分，
  并行前缀和是确定性归约且误差深度对数级，
  漂移合并用键排序加分段归约，段最大用 scatter_max，
  最大值与顺序无关逐位精确，同种子同卡两遍逐位可复现。
- 抽样、状态注册、菜单生成留在 CPU，随机数语义与 CPU 路径完全一致。

残差在卡上由条件计数重构特征算出，零噪声整数数据下与 CPU 逐位相同，
不需要把注册表的大特征矩阵搬上卡。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .batchkernel import BatchKernelResult
from .batchmenu import BatchMenu, QueryStructure
from .grouping import GroupedTable
from .state import Workload
from .stepsize import analytic_step


def _cupy():
    import cupy as cp

    return cp


_SEG_KERNEL_SRC = r"""
extern "C" {

// 一段一线程块，线程按跨步串行吃元素再做定序树规约，
// 归约顺序只由块宽决定与数据无关，同输入同卡逐位可复现。
__global__ void seg_sum(const double* v, const long long* offsets,
                        double* out, const long long nseg) {
    const long long s = blockIdx.x;
    if (s >= nseg) return;
    const long long lo = offsets[s], hi = offsets[s + 1];
    __shared__ double buf[128];
    double acc = 0.0;
    for (long long k = lo + threadIdx.x; k < hi; k += blockDim.x) acc += v[k];
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] += buf[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[s] = buf[0];
}

__global__ void seg_max(const double* v, const long long* offsets,
                        double* out, const long long nseg) {
    const long long s = blockIdx.x;
    if (s >= nseg) return;
    const long long lo = offsets[s], hi = offsets[s + 1];
    __shared__ double buf[128];
    double acc = -1.0 / 0.0;
    for (long long k = lo + threadIdx.x; k < hi; k += blockDim.x)
        acc = max(acc, v[k]);
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step)
            buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x + step]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[s] = buf[0];
}

// 一游程一线程串行求和，游程极短时省去整块线程的浪费，串行定序。
__global__ void run_sum(const double* v, const long long* starts,
                        const long long* bounds, double* out,
                        const long long nruns) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nruns) return;
    double acc = 0.0;
    for (long long k = starts[r]; k < bounds[r]; ++k) acc += v[k];
    out[r] = acc;
}

}
"""

_seg_module_cache = {}


def _seg_kernels(cp):
    """按当前设备缓存编译好的分段归约核。"""
    dev = cp.cuda.device.get_device_id()
    mod = _seg_module_cache.get(dev)
    if mod is None:
        raw = cp.RawModule(code=_SEG_KERNEL_SRC)
        mod = {
            "sum": raw.get_function("seg_sum"),
            "max": raw.get_function("seg_max"),
            "run": raw.get_function("run_sum"),
        }
        _seg_module_cache[dev] = mod
    return mod


def _guarded_scan(cp, values, starts_of, kind: str):
    """段内 Hillis-Steele 迭代倍增扫描，返回每元素到段首的前缀聚合。

    starts_of 是每元素所属段的起始下标，越界读用护栏挡掉，
    不做跨段前缀相减，无灾难性相消，误差深度对数级，
    不用浮点原子操作，同输入同卡逐位可复现。
    """
    total = int(values.size)
    m = values.astype(cp.float64, copy=True)
    if total == 0:
        return m
    pos = cp.arange(total, dtype=cp.int64)
    max_len = int((pos - starts_of).max()) + 1
    fill = 0.0 if kind == "sum" else -cp.inf
    shift = 1
    while shift < max_len:
        shifted = cp.concatenate(
            [cp.full(shift, fill, dtype=cp.float64), m[:-shift]]
        )
        valid = pos - shift >= starts_of
        if kind == "sum":
            m = cp.where(valid, m + shifted, m)
        else:
            m = cp.where(valid, cp.maximum(m, shifted), m)
        shift *= 2
    return m


def _seg_sum(cp, values, offsets):
    """按段求和，一段一线程块的定序树规约核，空段为零，确定性。"""
    num_seg = int(offsets.size) - 1
    out = cp.zeros(num_seg, dtype=cp.float64)
    if num_seg == 0 or values.size == 0:
        return out
    kern = _seg_kernels(cp)["sum"]
    values = cp.ascontiguousarray(values, dtype=cp.float64)
    offsets = offsets.astype(cp.int64, copy=False)
    kern((num_seg,), (128,), (values, offsets, out, np.int64(num_seg)))
    return out


def _seg_sum2d(cp, values, offsets):
    """二维按段沿行求和，段内倍增扫描，空段为零行。"""
    num_seg = offsets.size - 1
    width = values.shape[1]
    total = int(values.shape[0])
    if total == 0:
        return cp.zeros((num_seg, width), dtype=cp.float64)
    seg_ids = _seg_ids(cp, offsets, total)
    starts_of = offsets[seg_ids]
    pos = cp.arange(total, dtype=cp.int64)
    max_len = int((pos - starts_of).max()) + 1
    m = values.astype(cp.float64, copy=True)
    shift = 1
    while shift < max_len:
        shifted = cp.concatenate(
            [cp.zeros((shift, width), dtype=cp.float64), m[:-shift]]
        )
        valid = (pos - shift >= starts_of)[:, None]
        m = cp.where(valid, m + shifted, m)
        shift *= 2
    out = m[cp.maximum(offsets[1:] - 1, 0)]
    return cp.where((offsets[1:] == offsets[:-1])[:, None], 0.0, out)


def _seg_ids(cp, offsets, total: int):
    """由段边界展开每元素所属段号，等价数组重数的 repeat，cupy 缺此接口。"""
    return (
        cp.searchsorted(offsets, cp.arange(total, dtype=cp.int64), side="right") - 1
    )


def _seg_max(cp, values, offsets):
    """按段最大，一段一线程块的定序树规约核，与顺序无关逐位精确。

    cupy 的 maximum.at 对 float64 非原子实现有竞态不可用。
    """
    num_seg = int(offsets.size) - 1
    out = cp.zeros(num_seg, dtype=cp.float64)
    if num_seg == 0 or values.size == 0:
        return out
    kern = _seg_kernels(cp)["max"]
    values = cp.ascontiguousarray(values, dtype=cp.float64)
    offsets = offsets.astype(cp.int64, copy=False)
    kern((num_seg,), (128,), (values, offsets, out, np.int64(num_seg)))
    return out


class GpuBatchContext:
    """一次跑批的 GPU 上下文，静态量上传一次，每轮只传菜单与组编码。

    静态量含查询结构的依赖表通过表半空间表、目标与权重，
    每轮上传组编码与菜单数组，回传保持概率拼接向量与标量指标，
    条件计数在卡上整轮重算，省掉跨轮缓存的显存增长与大块传输。
    """

    def __init__(self, qs: QueryStructure, target, weights):
        cp = _cupy()
        self.cp = cp
        self.qs = qs
        self.dep = [cp.asarray(d) for d in qs.dep]
        self.allow = [cp.asarray(a) for a in qs.allow]
        self.ncond = cp.asarray(qs.ncond)
        self.hs_cols = cp.asarray(qs.hs_cols)
        self.hs_threshold = cp.asarray(qs.hs_threshold)
        self.hs_score = [cp.asarray(t) for t in qs.hs_score]
        self.num_hs = qs.num_halfspaces
        self.target = cp.asarray(np.asarray(target, dtype=np.float64))
        self.weights = cp.asarray(np.asarray(weights, dtype=np.float64))
        self.num_queries = int(self.target.size)
        # 依赖表拼接缓存，供条目流展开一次取数，免去按字段的小核循环
        sizes_np = np.array([d.size for d in qs.dep], dtype=np.int64)
        off_np = np.zeros(len(sizes_np) + 1, dtype=np.int64)
        np.cumsum(sizes_np, out=off_np[1:])
        total_dep = int(off_np[-1])
        dep_cat = (
            np.concatenate([np.asarray(d, dtype=np.int32) for d in qs.dep])
            if total_dep else np.zeros(0, dtype=np.int32)
        )
        maxdom = max((a.shape[1] for a in qs.allow if a.size), default=1)
        allow_cat = np.zeros((total_dep, maxdom), dtype=np.int8)
        for j, a in enumerate(qs.allow):
            if a.size:
                allow_cat[off_np[j]:off_np[j + 1], : a.shape[1]] = a
        self.dep_sizes = cp.asarray(sizes_np)
        self.dep_off = cp.asarray(off_np)
        self.dep_all = cp.asarray(dep_cat)
        self.allow_flat = cp.asarray(allow_cat)

    # ------------------------------------------------------------------
    def _condition_counts(self, codes_g):
        """条件计数与半空间分数，语义同 CPU 版，列表唯一无写冲突。"""
        cp = self.cp
        num_groups = codes_g.shape[0]
        cnt = cp.zeros((num_groups, self.num_queries), dtype=cp.int16)
        for j in range(codes_g.shape[1]):
            if self.dep[j].size == 0:
                continue
            pj = self.allow[j][:, codes_g[:, j]].T
            cnt[:, self.dep[j]] += pj.astype(cp.int16)
        score = cp.zeros((num_groups, self.num_hs), dtype=cp.int64)
        if self.num_hs:
            for j in range(codes_g.shape[1]):
                tab = self.hs_score[j]
                if tab.size and bool(tab.any()):
                    score += tab[:, codes_g[:, j]].T
        return cnt, score

    # ------------------------------------------------------------------
    def _residual_loss(self, cnt, score_g, counts):
        """由条件计数重构特征算残差损失与组错位分，整数数据下与 CPU 逐位相同。"""
        cp = self.cp
        feats = (cnt == self.ncond[None, :]).astype(cp.float64)
        if self.num_hs:
            feats[:, self.hs_cols] = (
                score_g >= self.hs_threshold[None, :]
            ).astype(cp.float64)
        answers = counts @ feats
        residual = self.target - answers
        loss = float((residual * self.weights) @ residual / 2)
        # 组错位分，组覆盖的题按绝对加权残差求和，供工作批挑组
        scores = feats @ (self.weights * cp.abs(residual))
        return residual, loss, scores

    def loss_of(self, codes: np.ndarray, state_ids) -> float:
        """当前表的损失，全程在卡上算，等价 CPU 的 table_loss。"""
        from .grouping import group_state_ids

        cp = self.cp
        grouped = group_state_ids(np.asarray(state_ids, dtype=np.int64))
        codes_g = cp.asarray(codes[grouped.unique_ids])
        counts = cp.asarray(grouped.counts.astype(np.float64))
        cnt, score_g = self._condition_counts(codes_g)
        _, loss, _ = self._residual_loss(cnt, score_g, counts)
        return loss

    # ------------------------------------------------------------------
    def _entry_expand(self, j_of, cur_codes, new_codes):
        """条目流展开，按路径序铺开各自字段的依赖列并取通过差。

        返回 CSR 段边界，列号，通过差与条目行归属，
        段内列序即依赖表原序，与旧的按字段小核循环逐位同序，
        全程整批索引取数，无按字段的 Python 循环与主机同步。
        """
        cp = self.cp
        j64 = j_of.astype(cp.int64)
        row_nnz = self.dep_sizes[j64]
        indptr = cp.zeros(int(j_of.size) + 1, dtype=cp.int64)
        cp.cumsum(row_nnz, out=indptr[1:])
        total = int(indptr[-1])
        row_of = _seg_ids(cp, indptr, total)
        within = cp.arange(total, dtype=cp.int64) - indptr[row_of]
        dep_pos = self.dep_off[j64[row_of]] + within
        indices = self.dep_all[dep_pos]
        adj = (
            self.allow_flat[dep_pos, new_codes[row_of]]
            - self.allow_flat[dep_pos, cur_codes[row_of]]
        )
        return indptr, indices, adj, row_of

    def _delta_parts(self, menu_gpu, cnt, codes_g):
        """答案差稀疏块，结构复刻 CPU 版，坐标合并用键排序整型分段和。"""
        cp = self.cp
        group, fields, values = menu_gpu["group"], menu_gpu["fields"], menu_gpu["values"]
        parts = []
        valid = fields >= 0
        nslots = valid.sum(axis=1)

        def _finish(indptr, indices, adj, path_idx):
            grp_of = group[path_idx]
            row_of = _seg_ids(cp, indptr, int(indptr[-1]))
            cnt_vals = cnt[grp_of[row_of], indices]
            need = self.ncond[indices]
            delta_i8 = (cnt_vals + adj == need).astype(cp.int8)
            delta_i8 -= (cnt_vals == need).astype(cp.int8)
            nz = cp.flatnonzero(delta_i8)
            row_nnz = cp.bincount(row_of[nz], minlength=path_idx.size)
            new_indptr = cp.zeros(path_idx.size + 1, dtype=cp.int64)
            cp.cumsum(row_nnz, out=new_indptr[1:])
            parts.append({
                "data": delta_i8[nz].astype(cp.float64),
                "indices": indices[nz],
                "indptr": new_indptr,
                "row_of": row_of[nz],
                "path_idx": path_idx,
                "grp_of": grp_of,
            })

        # 单字段路径，条目流一次展开，行内列取依赖表本身有序
        singles = cp.flatnonzero(nslots == 1)
        if singles.size:
            j_of = fields[singles, 0]
            cur = codes_g[group[singles], j_of]
            indptr, indices, adj, _ = self._entry_expand(
                j_of, cur, values[singles, 0]
            )
            _finish(indptr, indices, adj, singles.astype(cp.int64))

        # 多字段路径，各槽位条目流展开后坐标键排序合并，整型前缀和精确无误差
        multis = cp.flatnonzero(nslots >= 2)
        if multis.size:
            rows_parts, cols_parts, adj_parts = [], [], []
            for slot in range(3):
                live_mask = fields[multis, slot] >= 0
                live = multis[live_mask]
                if live.size == 0:
                    continue
                j_of = fields[live, slot]
                cur = codes_g[group[live], j_of]
                loc = cp.flatnonzero(live_mask)
                _, cols_s, adj_s, row_of_s = self._entry_expand(
                    j_of, cur, values[live, slot]
                )
                rows_parts.append(loc[row_of_s])
                cols_parts.append(cols_s)
                adj_parts.append(adj_s.astype(cp.int64))
            keys_rows = cp.concatenate(rows_parts)
            keys_cols = cp.concatenate(cols_parts)
            vals = cp.concatenate(adj_parts)
            # 键位宽够时压成 int32 排序，radix 位数减半，值域超界回退 int64
            if int(multis.size) * self.num_queries < 2 ** 31:
                keys = (
                    keys_rows.astype(cp.int32) * np.int32(self.num_queries)
                    + keys_cols.astype(cp.int32)
                )
            else:
                keys = keys_rows.astype(cp.int64) * self.num_queries + keys_cols
            order = cp.argsort(keys)
            sk = keys[order]
            sv = vals[order]
            newseg = cp.concatenate(
                [cp.ones(1, dtype=cp.bool_), sk[1:] != sk[:-1]]
            )
            starts = cp.flatnonzero(newseg)
            bounds = cp.concatenate([starts, cp.asarray([sk.size])])
            csum = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(sv)])
            merged = (csum[bounds[1:]] - csum[bounds[:-1]]).astype(cp.int8)
            uk = sk[starts]
            urows = uk // self.num_queries
            ucols = uk - urows * self.num_queries
            row_nnz = cp.bincount(urows, minlength=int(multis.size))
            indptr = cp.zeros(int(multis.size) + 1, dtype=cp.int64)
            cp.cumsum(row_nnz, out=indptr[1:])
            _finish(indptr, ucols, merged, multis.astype(cp.int64))
        return parts

    # ------------------------------------------------------------------
    def _halfspace_delta(self, menu_gpu, score_g, codes_g):
        """半空间通道答案差，路径乘半空间稠密矩阵，无半空间返回 None。"""
        cp = self.cp
        if self.num_hs == 0 or menu_gpu["num_paths"] == 0:
            return None
        group, fields, values = menu_gpu["group"], menu_gpu["fields"], menu_gpu["values"]
        adj = cp.zeros((menu_gpu["num_paths"], self.num_hs), dtype=cp.int64)
        for slot in range(3):
            live = cp.flatnonzero(fields[:, slot] >= 0)
            if live.size == 0:
                continue
            j_of = fields[live, slot]
            j_host = cp.asnumpy(j_of)
            for j in np.unique(j_host):
                tab = self.hs_score[int(j)]
                if not (tab.size and bool(tab.any())):
                    continue
                sel = live[j_of == int(j)]
                cur = codes_g[group[sel], int(j)]
                new = values[sel, slot]
                adj[sel] += (tab[:, new] - tab[:, cur]).T
        base = score_g[group]
        thresh = self.hs_threshold[None, :]
        delta = ((base + adj) >= thresh).astype(cp.float64)
        delta -= (base >= thresh).astype(cp.float64)
        return delta

    # ------------------------------------------------------------------
    def _calibrate_flat(self, gains_flat, ref_flat, offsets, seg_ids, counts,
                        old_loss, alpha, numerical_tol, beta_hint):
        """calibrate_beta_flat 的 GPU 版，判据容差与括根求根逻辑逐句对应。

        段最大用 scatter_max，最大与顺序无关逐位精确，
        分段归一化与分段增益和用前缀和差分，布伦特法标量循环在 CPU，
        每次评估只在卡上做一轮指数扫描。
        """
        cp = self.cp

        num_groups = offsets.size - 1
        gmax = _seg_max(cp, gains_flat, offsets)
        max_gain_sum = float(counts @ cp.maximum(gmax, 0.0))
        requirement = alpha * max_gain_sum
        scale = max(1.0, old_loss)
        if max_gain_sum <= numerical_tol * scale:
            return None, max_gain_sum, requirement

        centered = gains_flat - gmax[seg_ids]

        def evaluate(beta: float):
            zw = ref_flat * cp.exp(beta * centered)
            norm = _seg_sum(cp, zw, offsets)
            P = zw / norm[seg_ids]
            per_group = _seg_sum(cp, P * gains_flat, offsets)
            return P, float(counts @ per_group)

        ps, D = evaluate(0.0)
        if D >= requirement:
            beta = 0.0
        else:
            lo = 0.0
            hi = 0.0
            probe = 1.0 / max(max_gain_sum, 1e-300)
            if beta_hint is not None and np.isfinite(beta_hint) and beta_hint > 0.0:
                probe = float(beta_hint)
            ps, D = evaluate(probe)
            if D >= requirement:
                hi = probe
                for _ in range(100):
                    probe /= 2.0
                    ps, D = evaluate(probe)
                    if D < requirement:
                        lo = probe
                        break
                    hi = probe
                else:
                    raise FloatingPointError("无法括住 beta 根，检查数值条件")
            else:
                lo = probe
                for _ in range(100):
                    probe *= 2.0
                    ps, D = evaluate(probe)
                    if D >= requirement:
                        hi = probe
                        break
                    lo = probe
                else:
                    raise FloatingPointError("无法括住 beta 根，检查数值条件")
            beta = float(
                brentq(
                    lambda b: evaluate(b)[1] - requirement,
                    lo, hi, xtol=1e-300, rtol=1e-13, maxiter=200,
                )
            )
            ps, D = evaluate(beta)
            for _ in range(8):
                if D >= requirement:
                    break
                beta = beta * (1.0 + 4e-13) if beta > 0 else hi * 1e-16
                ps, D = evaluate(beta)
        if D <= 0:
            raise FloatingPointError("方向增益必须严格为正")
        return (beta, ps, D), max_gain_sum, requirement

    # ------------------------------------------------------------------
    def build(
        self,
        grouped: GroupedTable,
        menu: BatchMenu,
        codes: np.ndarray,
        stay_probability: float = 0.9,
        alpha: float = 0.5,
        damping: float = 1.0,
        max_expected_rows: float | None = None,
        numerical_tol: float = 1e-12,
        beta_hint: float | None = None,
    ) -> BatchKernelResult:
        """构造一轮批量核，输入输出与 CPU build_batch_kernel 对齐。"""
        cp = self.cp
        if not (0 < stay_probability < 1):
            raise ValueError("stay_probability 必须落在 (0,1)")
        num_groups = grouped.num_groups
        num_queries = self.num_queries

        codes_g = cp.asarray(codes[grouped.unique_ids])
        counts = cp.asarray(grouped.counts.astype(np.float64))
        menu_gpu = {
            "group": cp.asarray(menu.group),
            "fields": cp.asarray(menu.fields),
            "values": cp.asarray(menu.values),
            "mass": cp.asarray(menu.mass),
            "offsets": cp.asarray(menu.offsets),
            "num_paths": menu.num_paths,
        }

        # 残差由条件计数重构特征得出，整数数据下与 CPU 逐位相同
        cnt, score_g = self._condition_counts(codes_g)
        residual, old_loss, scores_gpu = self._residual_loss(cnt, score_g, counts)
        group_scores = cp.asnumpy(scores_gpu)

        parts = self._delta_parts(menu_gpu, cnt, codes_g)
        delta_hs = self._halfspace_delta(menu_gpu, score_g, codes_g)
        we = self.weights * residual
        gains_paths = cp.zeros(menu.num_paths, dtype=cp.float64)
        for part in parts:
            lin = _seg_sum(cp, part["data"] * we[part["indices"]], part["indptr"])
            quad = _seg_sum(
                cp,
                part["data"] * part["data"] * self.weights[part["indices"]],
                part["indptr"],
            )
            gains_paths[part["path_idx"]] = lin - quad / 2
        if delta_hs is not None:
            w_hs = self.weights[self.hs_cols]
            gains_paths += delta_hs @ we[self.hs_cols]
            gains_paths -= (delta_hs * delta_hs) @ w_hs / 2

        # flat 结构，每段索引 0 是保持项
        lengths_np = np.diff(menu.offsets) + 1
        offsets_np = np.zeros(num_groups + 1, dtype=np.int64)
        np.cumsum(lengths_np, out=offsets_np[1:])
        total = int(offsets_np[-1])
        offsets = cp.asarray(offsets_np)
        seg_starts = offsets[:-1]
        seg_ids = _seg_ids(cp, offsets, total)
        path_pos = cp.arange(menu.num_paths, dtype=cp.int64) + menu_gpu["group"] + 1
        gains_flat = cp.zeros(total, dtype=cp.float64)
        gains_flat[path_pos] = gains_paths
        ref_flat = cp.zeros(total, dtype=cp.float64)
        mass_sum = _seg_sum(cp, menu_gpu["mass"], menu_gpu["offsets"])
        empty = cp.diff(menu_gpu["offsets"]) == 0
        mass_sum = cp.where(empty, 1.0, mass_sum)
        ref_flat[path_pos] = (
            (1 - stay_probability) * menu_gpu["mass"] / mass_sum[menu_gpu["group"]]
        )
        ref_flat[seg_starts] = cp.where(empty, 1.0, stay_probability)

        tilt, max_gain_sum, requirement = self._calibrate_flat(
            gains_flat, ref_flat, offsets, seg_ids, counts,
            old_loss, alpha, numerical_tol, beta_hint,
        )
        if tilt is None:
            frozen = np.zeros(total, dtype=np.float64)
            frozen[offsets_np[:-1]] = 1.0
            return BatchKernelResult(
                menu, grouped, offsets_np, frozen, old_loss,
                0.0, 0.0, 0.0, 0.0, old_loss, old_loss, 0.0,
                max_gain_sum, requirement, "no_positive_direction",
                group_scores,
            )
        beta, ps, direction_gain = tilt

        rates_flat = ps.copy()
        rates_flat[seg_starts] = 0.0
        leave = _seg_sum(cp, rates_flat, offsets)
        rates_paths = rates_flat[path_pos]

        # 漂移，键排序加前缀和分段归约，确定性合并
        key_parts, weight_parts = [], []
        for part in parts:
            keys = part["grp_of"][part["row_of"]] * num_queries + part["indices"]
            key_parts.append(keys)
            weight_parts.append(
                part["data"] * rates_paths[part["path_idx"]][part["row_of"]]
            )
        drifts = cp.zeros(num_groups * num_queries, dtype=cp.float64)
        if key_parts:
            keys = cp.concatenate(key_parts)
            wvals = cp.concatenate(weight_parts)
            order = cp.argsort(keys)
            sk = keys[order]
            sw = wvals[order]
            newseg = cp.concatenate([cp.ones(1, dtype=cp.bool_), sk[1:] != sk[:-1]])
            starts = cp.flatnonzero(newseg)
            bounds = cp.concatenate(
                [starts[1:], cp.asarray([int(sk.size)], dtype=cp.int64)]
            )
            sums = cp.zeros(starts.size, dtype=cp.float64)
            kern = _seg_kernels(cp)["run"]
            nruns = int(starts.size)
            threads = 256
            kern(
                ((nruns + threads - 1) // threads,), (threads,),
                (sw, starts, bounds, sums, np.int64(nruns)),
            )
            drifts[sk[starts]] = sums
        drifts = drifts.reshape(num_groups, num_queries)
        if delta_hs is not None:
            weighted = delta_hs * rates_paths[:, None]
            seg = _seg_sum2d(cp, weighted, menu_gpu["offsets"])
            drifts[:, self.hs_cols] += seg
        drift_total = counts @ drifts
        self_cross = float(counts @ ((drifts * drifts) @ self.weights))
        interaction = float(
            ((drift_total * self.weights) @ drift_total - self_cross) / 2
        )

        # 标量与拼接向量回传，步长与越界修正走 CPU 原语义
        leave_np = cp.asnumpy(leave)
        rates_np = cp.asnumpy(rates_flat)
        h_star = analytic_step(float(leave_np.max()), direction_gain, interaction)
        unit_rows = float(np.dot(cp.asnumpy(counts), leave_np))
        if max_expected_rows is not None and unit_rows > 0:
            step = damping * min(h_star, max_expected_rows / unit_rows)
        else:
            step = damping * h_star

        probs_flat = step * rates_np
        stay = 1.0 - np.add.reduceat(probs_flat, offsets_np[:-1])
        if np.any(stay < -1e-12):
            raise FloatingPointError("步长违反随机矩阵约束")
        probs_flat[offsets_np[:-1]] = np.maximum(stay, 0.0)
        negative = np.flatnonzero(stay < 0)
        for b in negative:
            lo, hi = int(offsets_np[b]), int(offsets_np[b + 1])
            probs_flat[lo:hi] /= probs_flat[lo:hi].sum()

        expected_loss = old_loss - step * direction_gain + step * step * interaction
        upper = old_loss - step * direction_gain / 2
        return BatchKernelResult(
            menu, grouped, offsets_np, probs_flat, old_loss,
            beta, direction_gain, interaction, step,
            expected_loss, upper, step * unit_rows,
            max_gain_sum, requirement, "ok",
            group_scores,
        )
