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

单元格直算通路（第五十六步提速 A/B），
- 考卷全部等值题都是单元格（每字段恰允许一个值，典型如二阶边际全格）时，
  一条路径对某边际的答案差只在旧格 −1 新格 +1 两处非零，
  直接按组码查格号得到这两项，不再把字段牵连的全部题展开再筛非零；
- 特征矩阵同样由格号散射得到，省掉逐字段累加的条件计数矩阵；
- 产出的稀疏块与通用通路逐条同序同值，分块边界也照通用通路的条目预算切，
  之后的增益、括根、漂移合并一行不动，同种子同卡与通用通路逐位同结果；
- 含非单元格题（区间、多值）时自动退回通用通路，半空间通道不受影响。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .batchkernel import BatchKernelResult
from .batchmenu import BatchMenu, QueryStructure
from .grouping import GroupedTable
from .gain import ExactGainOracle, exact_sparse_path_gains, integral_values, validate_numerical_tol
from .state import Workload
from .stepsize import (
    analytic_step, expected_improvement, final_probabilities, kernel_statistics, validate_step_parameters,
)
from .tilt import calibration_requirement, validate_tilt_parameters, validate_tilt_result


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

_PAIR_KERNEL_SRC = r"""
extern "C" {

// 直算条目的融合核（第五十九步 甲 GPU 侧）：一对（路径，边际）一线程，
// 全整数运算，与原先几十步 cupy 逐步计算逐位相同。
// 多字段路径的键：按槽序铺开各槽字段所含边际，键 = 段内行号 × 边际数 + 边际号。
__global__ void multi_keys(const long long total, const long long lo,
                           const long long* __restrict__ row_loc,
                           const long long* __restrict__ indptr,
                           const int* __restrict__ fsub,
                           const long long* __restrict__ fm_cnt,
                           const long long* __restrict__ fm_off,
                           const long long* __restrict__ fm_list,
                           const long long m, long long* __restrict__ key) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long r = row_loc[t];
    const long long ra = r + lo;
    const long long within = t - indptr[r];
    const int f0 = fsub[ra * 3], f1 = fsub[ra * 3 + 1], f2 = fsub[ra * 3 + 2];
    const long long c0 = f0 >= 0 ? fm_cnt[f0] : 0;
    const long long c01 = c0 + (f1 >= 0 ? fm_cnt[f1] : 0);
    const int slot = (within >= c0 ? 1 : 0) + (within >= c01 ? 1 : 0);
    const long long start = slot == 0 ? 0 : (slot == 1 ? c0 : c01);
    const int f = slot == 0 ? f0 : (slot == 1 ? f1 : f2);
    key[t] = r * m + fm_list[fm_off[f] + within - start];
}

// 每对算旧格新格的题号，写成交错的 (qa, qb) / (da, -da) / 有效位 / 行号，
// 新码不落地：字段命中改动槽（后槽优先，与逐槽赋值顺序一致）即取新值码。
// use_midx 为 0 时（单字段路径）边际号由字段所含边际表按序取，否则读 m_idx。
__global__ void direct_pairs(const long long total, const long long lo,
                             const long long* __restrict__ row_loc,
                             const long long* __restrict__ indptr,
                             const long long* __restrict__ m_idx, const int use_midx,
                             const int* __restrict__ cur, const int F,
                             const int* __restrict__ fsub,
                             const int* __restrict__ vsub,
                             const long long* __restrict__ m_off,
                             const long long* __restrict__ m_fields,
                             const long long* __restrict__ m_strides, const int kmax,
                             const long long* __restrict__ fm_off,
                             const long long* __restrict__ fm_list,
                             const long long* __restrict__ cellq,
                             long long* __restrict__ q2, signed char* __restrict__ d2,
                             bool* __restrict__ v2, long long* __restrict__ r2) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long r = row_loc[t];
    const long long ra = r + lo;
    long long mi;
    if (use_midx) {
        mi = m_idx[t];
    } else {
        mi = fm_list[fm_off[fsub[ra * 3]] + (t - indptr[r])];
    }
    long long fo = m_off[mi], fn = fo;
    for (int k = 0; k < kmax; ++k) {
        const long long f = m_fields[mi * kmax + k];
        const long long s = m_strides[mi * kmax + k];
        const long long c = cur[ra * F + f];
        long long cn = c;
        for (int sl = 0; sl < 3; ++sl)
            if (fsub[ra * 3 + sl] == f) cn = vsub[ra * 3 + sl];
        fo += c * s;
        fn += cn * s;
    }
    const long long qo = cellq[fo], qn = cellq[fn];
    const bool change = fo != fn;
    const bool swap = qo > qn;
    const long long qa = swap ? qn : qo, qb = swap ? qo : qn;
    const signed char da = swap ? 1 : -1;
    q2[2 * t] = qa;
    q2[2 * t + 1] = qb;
    d2[2 * t] = da;
    d2[2 * t + 1] = -da;
    v2[2 * t] = change && qa >= 0;
    v2[2 * t + 1] = change && qb >= 0;
    r2[2 * t] = r;
    r2[2 * t + 1] = r;
}

}
"""

_seg_module_cache = {}
_pair_module_cache = {}

# 括根与增益的融合核（第五十九步 甲 GPU 侧）：浮点运算一律显式 __dadd_rn/__dmul_rn/__ddiv_rn，
# 不受编译器 FMA 合并影响；段归约照 seg_sum 的线程跨步串行 + 定序树（128 线程），
# 每个浮点结果与原先"逐步 cupy 运算 + seg_sum"逐位相同。
_TILT_KERNEL_SRC = r"""
extern "C" {

// 括根评估 (1)：zw = ref · exp(beta · centered)，段和即归一化常数
__global__ void tilt_norm(const double* __restrict__ ref, const double* __restrict__ centered,
                          const double beta, const long long* __restrict__ offsets,
                          double* __restrict__ zw, double* __restrict__ norm,
                          const long long nseg) {
    const long long s = blockIdx.x;
    if (s >= nseg) return;
    const long long lo = offsets[s], hi = offsets[s + 1];
    __shared__ double buf[128];
    double acc = 0.0;
    for (long long k = lo + threadIdx.x; k < hi; k += blockDim.x) {
        const double z = __dmul_rn(ref[k], exp(__dmul_rn(beta, centered[k])));
        zw[k] = z;
        acc = __dadd_rn(acc, z);
    }
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] = __dadd_rn(buf[threadIdx.x], buf[threadIdx.x + step]);
        __syncthreads();
    }
    if (threadIdx.x == 0) norm[s] = buf[0];
}

// 括根评估 (2)：P = zw / norm[段]，段和 Σ P·gains
__global__ void tilt_gain(const double* __restrict__ zw, const double* __restrict__ norm,
                          const double* __restrict__ gains, const long long* __restrict__ offsets,
                          double* __restrict__ P, double* __restrict__ per_group,
                          const long long nseg) {
    const long long s = blockIdx.x;
    if (s >= nseg) return;
    const long long lo = offsets[s], hi = offsets[s + 1];
    const double nrm = norm[s];
    __shared__ double buf[128];
    double acc = 0.0;
    for (long long k = lo + threadIdx.x; k < hi; k += blockDim.x) {
        const double p = __ddiv_rn(zw[k], nrm);
        P[k] = p;
        acc = __dadd_rn(acc, __dmul_rn(p, gains[k]));
    }
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] = __dadd_rn(buf[threadIdx.x], buf[threadIdx.x + step]);
        __syncthreads();
    }
    if (threadIdx.x == 0) per_group[s] = buf[0];
}

// 漂移合并的键与权：一条目一线程，键 = 题号 × 组数 + 组号，权 = d · rates[path[row]]，
// 写入拼接缓冲区的 [base, base+n)
__global__ void drift_kv(const long long n, const long long base,
                         const long long* __restrict__ indices, const double* __restrict__ data,
                         const long long* __restrict__ row_of, const long long* __restrict__ grp_of,
                         const long long* __restrict__ path_idx, const double* __restrict__ rates,
                         const long long num_groups,
                         long long* __restrict__ keys, double* __restrict__ vals) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long r = row_of[i];
    keys[base + i] = indices[i] * num_groups + grp_of[r];
    vals[base + i] = __dmul_rn(data[i], rates[path_idx[r]]);
}

// 路径增益：一路径一块，lin = Σ d·we[q]，quad = Σ (d·d)·w[q]，gains[path] = lin − quad/2
__global__ void part_gains(const double* __restrict__ data, const long long* __restrict__ indices,
                           const double* __restrict__ we, const double* __restrict__ w,
                           const long long* __restrict__ indptr,
                           const long long* __restrict__ path_idx,
                           double* __restrict__ gains, double* __restrict__ magnitudes,
                           const long long nseg) {
    const long long s = blockIdx.x;
    if (s >= nseg) return;
    const long long lo = indptr[s], hi = indptr[s + 1];
    __shared__ double buf1[128];
    __shared__ double buf2[128];
    __shared__ double absbuf[128];
    double a1 = 0.0, a2 = 0.0, absolute = 0.0;
    for (long long k = lo + threadIdx.x; k < hi; k += blockDim.x) {
        const double d = data[k];
        const long long q = indices[k];
        const double linear = __dmul_rn(d, we[q]);
        a1 = __dadd_rn(a1, linear);
        a2 = __dadd_rn(a2, __dmul_rn(__dmul_rn(d, d), w[q]));
        absolute = __dadd_rn(absolute, fabs(linear));
    }
    buf1[threadIdx.x] = a1;
    buf2[threadIdx.x] = a2;
    absbuf[threadIdx.x] = absolute;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            buf1[threadIdx.x] = __dadd_rn(buf1[threadIdx.x], buf1[threadIdx.x + step]);
            buf2[threadIdx.x] = __dadd_rn(buf2[threadIdx.x], buf2[threadIdx.x + step]);
            absbuf[threadIdx.x] = __dadd_rn(absbuf[threadIdx.x], absbuf[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gains[path_idx[s]] = __dsub_rn(buf1[0], __ddiv_rn(buf2[0], 2.0));
        magnitudes[path_idx[s]] = __dadd_rn(absbuf[0], __ddiv_rn(buf2[0], 2.0));
    }
}

// 局部复核掩码、异常位和选中数量合并；正常空掩码也必须核验异常位。
__global__ void gain_recheck(const double* __restrict__ gains,
                             const double* __restrict__ magnitudes,
                             const long long* __restrict__ terms,
                             const long long n, const int integral,
                             const double tolerance, const double unit_roundoff,
                             const double smallest, const double exact_limit,
                             bool* __restrict__ mask,
                             unsigned long long* __restrict__ status) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double g = gains[i], m = magnitudes[i];
    unsigned long long flags = 0;
    if (!isfinite(g) || !isfinite(m)) flags |= 1;
    if (m < 0.0) flags |= 2;
    const double operations = __dmul_rn(8.0, __dadd_rn((double)terms[i], 1.0));
    const double ku = __dmul_rn(operations, unit_roundoff);
    if (ku >= 0.5) flags |= 4;
    if (m > exact_limit) flags |= 8;
    bool selected = false;
    if (!(flags & 7) && (!integral || m > exact_limit)) {
        const double gamma = __ddiv_rn(ku, __dsub_rn(1.0, ku));
        const double relative = fmax(tolerance, __ddiv_rn(gamma, __dsub_rn(1.0, gamma)));
        const double threshold = fmax(__dmul_rn(relative, m), __dmul_rn(operations, smallest));
        selected = terms[i] > 0 && fabs(g) <= threshold;
    }
    mask[i] = selected;
    if (selected) atomicAdd(status, 1ULL);
    if (flags) atomicOr(status + 1, flags);
}

}
"""
_tilt_module_cache = {}


def _tilt_kernels(cp):
    """按当前设备缓存编译好的括根/增益融合核。"""
    dev = cp.cuda.device.get_device_id()
    mod = _tilt_module_cache.get(dev)
    if mod is None:
        raw = cp.RawModule(code=_TILT_KERNEL_SRC)
        mod = {
            "norm": raw.get_function("tilt_norm"),
            "gain": raw.get_function("tilt_gain"),
            "part_gains": raw.get_function("part_gains"),
            "drift_kv": raw.get_function("drift_kv"),
            "gain_recheck": raw.get_function("gain_recheck"),
        }
        _tilt_module_cache[dev] = mod
    return mod


def _pair_kernels(cp):
    """按当前设备缓存编译好的直算条目融合核。"""
    dev = cp.cuda.device.get_device_id()
    mod = _pair_module_cache.get(dev)
    if mod is None:
        raw = cp.RawModule(code=_PAIR_KERNEL_SRC)
        mod = {
            "keys": raw.get_function("multi_keys"),
            "pairs": raw.get_function("direct_pairs"),
        }
        _pair_module_cache[dev] = mod
    return mod


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


def _chunk_bounds(cp, sizes, budget: int, with_cums: bool = False):
    """按条目预算把路径序列切成连续块，返回主机端 (lo, hi) 列表。

    预算不正或总量不超预算时整段一块，切点由条目累计和二分得出，
    每块至少一条路径，单路径超预算时独占一块不再细分。
    sizes 可为主机数组或设备数组（设备数组回传一次）；
    with_cums 为真时连同主机端累计和一起返回，供调用方免同步取块内总量。
    """
    n = int(sizes.size)
    if n == 0:
        return ([], None) if with_cums else []
    if budget <= 0 and not with_cums:
        return [(0, n)]
    if isinstance(sizes, np.ndarray):
        cums = np.cumsum(sizes.astype(np.int64, copy=False))
    else:
        cums = cp.asnumpy(cp.cumsum(sizes.astype(cp.int64)))
    if budget <= 0 or int(cums[-1]) <= budget:
        out = [(0, n)]
    else:
        out = []
        lo = 0
        while lo < n:
            base = int(cums[lo - 1]) if lo else 0
            hi = int(np.searchsorted(cums, base + budget, side="right"))
            hi = min(max(hi, lo + 1), n)
            out.append((lo, hi))
            lo = hi
    return (out, cums) if with_cums else out


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


@dataclass(frozen=True)
class MarginalLayout:
    """全单元格查询结构的边际表布局，直算通路的主机端静态量。

    每个边际是一组题的集合，这些题涉及同一字段集且每字段恰允许一个值，
    cellq_all 按边际拼接，把格的扁平号映射到全局题号，缺格记 -1，
    边际按最小题号升序编号，ranges_sorted 为真表示各边际的题号区间
    互不交错，此时按边际序铺开的条目天然按题号升序，免排序。
    """

    num_marginals: int
    kmax: int
    marg_fields: NDArray[np.int64]  # m 乘 kmax，填充位记 0 配零步长不生效
    marg_strides: NDArray[np.int64]  # m 乘 kmax，填充位记 0
    marg_off: NDArray[np.int64]  # m，各边际在 cellq_all 的起点
    cellq_all: NDArray[np.int64]  # 拼接的格号到题号表
    fm_list: NDArray[np.int64]  # 按字段拼接的所含边际号，字段内升序
    fm_off: NDArray[np.int64]  # 字段数加一
    ranges_sorted: bool


def build_marginal_layout(qs: QueryStructure) -> MarginalLayout | None:
    """从查询结构识别单元格边际布局，任一等值题不是单元格则返回 None。"""
    ncond = np.asarray(qs.ncond, dtype=np.int64)
    num_queries = int(ncond.size)
    num_fields = len(qs.dep)
    eq_queries = np.flatnonzero(ncond >= 0)
    if eq_queries.size == 0 or np.any(ncond[eq_queries] == 0):
        return None
    fields_of: list[list[int]] = [[] for _ in range(num_queries)]
    vals_of: list[list[int]] = [[] for _ in range(num_queries)]
    domains = []
    for j in range(num_fields):
        dep_j = np.asarray(qs.dep[j], dtype=np.int64)
        allow_j = np.asarray(qs.allow[j], dtype=np.bool_)
        domains.append(int(allow_j.shape[1]))
        if dep_j.size == 0:
            continue
        if np.any(allow_j.sum(axis=1) != 1):
            return None
        val_j = allow_j.argmax(axis=1)
        for t, q in enumerate(dep_j.tolist()):
            fields_of[q].append(j)
            vals_of[q].append(int(val_j[t]))
    groups: dict[tuple[int, ...], list[int]] = {}
    for q in eq_queries.tolist():
        key = tuple(fields_of[q])
        if len(key) != int(ncond[q]):
            return None
        groups.setdefault(key, []).append(q)
    margs = []
    for key, qlist in groups.items():
        dims = [domains[j] for j in key]
        strides = [int(np.prod(dims[k + 1:], dtype=np.int64)) for k in range(len(dims))]
        cellq = np.full(int(np.prod(dims, dtype=np.int64)), -1, dtype=np.int64)
        for q in qlist:
            flat = sum(v * s for v, s in zip(vals_of[q], strides))
            if cellq[flat] != -1:
                return None
            cellq[flat] = q
        margs.append((min(qlist), max(qlist), key, strides, cellq))
    margs.sort(key=lambda t: t[0])
    ranges_sorted = all(
        margs[i][1] < margs[i + 1][0] for i in range(len(margs) - 1)
    )
    m = len(margs)
    kmax = max(len(t[2]) for t in margs)
    marg_fields = np.zeros((m, kmax), dtype=np.int64)
    marg_strides = np.zeros((m, kmax), dtype=np.int64)
    marg_off = np.zeros(m, dtype=np.int64)
    pieces = []
    pos = 0
    for i, (_, _, key, strides, cellq) in enumerate(margs):
        marg_fields[i, : len(key)] = key
        marg_strides[i, : len(key)] = strides
        marg_off[i] = pos
        pos += cellq.size
        pieces.append(cellq)
    fm_lists = [[] for _ in range(num_fields)]
    for i, (_, _, key, _, _) in enumerate(margs):
        for j in key:
            fm_lists[j].append(i)
    fm_off = np.zeros(num_fields + 1, dtype=np.int64)
    np.cumsum([len(l) for l in fm_lists], out=fm_off[1:])
    fm_list = np.array(
        [i for l in fm_lists for i in l], dtype=np.int64
    ).reshape(-1)
    return MarginalLayout(
        m, kmax, marg_fields, marg_strides, marg_off,
        np.concatenate(pieces), fm_list, fm_off, ranges_sorted,
    )


# 直算通路内部按（路径，边际）对数分块的预算，压住临时数组显存峰值
_DIRECT_PAIR_BUDGET = 1 << 23


class GpuBatchContext:
    """一次跑批的 GPU 上下文，静态量上传一次，每轮只传菜单与组编码。

    静态量含查询结构的依赖表通过表半空间表、目标与权重，
    每轮上传组编码与菜单数组，回传保持概率拼接向量与标量指标，
    条件计数在卡上整轮重算，省掉跨轮缓存的显存增长与大块传输。
    """

    def __init__(
        self, qs: QueryStructure, target, weights,
        direct: bool | None = None, sparse: bool | None = None,
    ):
        """direct 取 None 自动识别单元格结构，True 强制直算（结构不符报错），False 走通用通路。

        sparse 控制稀疏整表项/漂移项（第五十九步 乙）：答案按格号整数计数（与稠密 gemv
        逐位相同），组错位分与漂移项只在非零项上聚合，不铺 (组数×题数) 稠密矩阵，
        浮点求和顺序异于 gemv、末位可异（轨迹如换种子，分布同）；只在直算布局且无半空间题
        时可用，None 可用即开，True 强制（不可用报错），False 走稠密老路逐位复现旧表。
        """
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
        self._target_host = np.array(target, dtype=np.float64, copy=True)
        self._weights_host = np.array(weights, dtype=np.float64, copy=True)
        self._target_host.setflags(write=False)
        self._weights_host.setflags(write=False)
        self._integral_gain_inputs = integral_values(self._target_host, self._weights_host)
        self.target = cp.asarray(self._target_host)
        self.weights = cp.asarray(self._weights_host)
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
        self.dep_sizes_np = sizes_np  # 主机端留一份，直算分块在主机算免同步
        self.dep_off = cp.asarray(off_np)
        self.dep_all = cp.asarray(dep_cat)
        self.allow_flat = cp.asarray(allow_cat)
        # 单元格直算通路的边际布局，识别失败则留在通用通路
        self.layout = None if direct is False else build_marginal_layout(qs)
        if direct is True and self.layout is None:
            raise ValueError("直算通路要求全部等值题都是单元格")
        self.direct = self.layout is not None
        if self.direct:
            lay = self.layout
            self.m_fields = cp.asarray(lay.marg_fields)
            self.m_strides = cp.asarray(lay.marg_strides)
            self.m_off = cp.asarray(lay.marg_off)
            self.cellq = cp.asarray(lay.cellq_all)
            self.fm_list = cp.asarray(lay.fm_list)
            self.fm_off = cp.asarray(lay.fm_off)
            self.fm_cnt_np = np.diff(lay.fm_off)
            self.fm_cnt = cp.asarray(self.fm_cnt_np)
        can_sparse = self.direct and self.num_hs == 0
        if sparse is True and not can_sparse:
            raise ValueError("稀疏通路要求直算布局且无半空间题")
        self.sparse = bool(can_sparse) if sparse is None else bool(sparse)

    # ------------------------------------------------------------------
    def _hs_scores(self, codes_g):
        """半空间整数分，按字段分表累加。"""
        cp = self.cp
        num_groups = codes_g.shape[0]
        score = cp.zeros((num_groups, self.num_hs), dtype=cp.int64)
        if self.num_hs:
            for j in range(codes_g.shape[1]):
                tab = self.hs_score[j]
                if tab.size and bool(tab.any()):
                    score += tab[:, codes_g[:, j]].T
        return score

    def _cells_direct(self, codes_g):
        """由组码查每组每边际的格号（组数×边际数），缺格为 -1；散射特征与稀疏整表项共用。"""
        cp = self.cp
        num_groups = int(codes_g.shape[0])
        m = self.layout.num_marginals
        flat = cp.broadcast_to(self.m_off[None, :], (num_groups, m))
        for k in range(self.layout.kmax):
            flat = flat + (
                codes_g[:, self.m_fields[:, k]].astype(cp.int64)
                * self.m_strides[None, :, k]
            )
        return self.cellq[flat]

    def _feats_direct(self, codes_g):
        """由组码查格号散射出特征矩阵，与条件计数重构的特征逐位相同。"""
        cp = self.cp
        num_groups = int(codes_g.shape[0])
        m = self.layout.num_marginals
        q = self._cells_direct(codes_g)
        feats = cp.zeros((num_groups, self.num_queries), dtype=cp.float64)
        valid = q >= 0
        rows = cp.broadcast_to(cp.arange(num_groups, dtype=cp.int64)[:, None], (num_groups, m))
        feats[rows[valid], q[valid]] = 1.0
        score_g = self._hs_scores(codes_g)
        if self.num_hs:
            feats[:, self.hs_cols] = (
                score_g >= self.hs_threshold[None, :]
            ).astype(cp.float64)
        return feats, score_g

    def _state_terms_sparse(self, codes_g, counts):
        """整表项的稀疏版：不铺 (组数×题数) 特征矩阵，只在每组命中的格上算。

        答案 = 各组计数按格号累加，计数是整数、和不超 2^53，任何求和顺序都精确，
        与稠密 gemv 逐位相同，残差与损失随之逐位相同；
        组错位分 = 每组命中格的加权绝对残差之和，按（组×边际）小矩阵沿行规约，
        求和顺序异于 gemv、末位可异，同形状同输入同卡逐位可复现。
        """
        cp = self.cp
        q = self._cells_direct(codes_g)
        num_groups, m = q.shape
        valid = q >= 0
        nq = self.num_queries
        # 缺格记到多出的一格再丢掉，免去布尔压缩的主机同步
        counts_b = cp.broadcast_to(counts[:, None], (num_groups, m)).ravel()
        answers = cp.bincount(
            cp.where(valid, q, nq).ravel(), weights=counts_b, minlength=nq + 1,
        )[:nq]
        residual = self.target - answers
        loss = float((residual * self.weights) @ residual / 2)
        wr = self.weights * cp.abs(residual)
        scores = cp.where(valid, wr[cp.maximum(q, 0)], 0.0).sum(axis=1)
        return residual, loss, scores, self._hs_scores(codes_g), None, answers

    def _direct_entries(self, sub, menu_gpu, codes_g, multi: bool, pairs_total=None):
        """给定路径子集，直算每条路径的答案差稀疏块，条目按（路径，题号）升序。

        单字段路径只枚举该字段所含的边际，多字段路径枚举各槽字段所含边际的并集
        （按（路径，边际）排序去重），不含动过字段的边际旧格新格必相同、本就不出条目，
        旧格与新格相同的边际不出条目，缺格一侧也不出条目，
        内部再按（路径，边际）对数预算分段，段结果拼接不改条目序。

        第五十九步 甲 GPU 侧：每对的格号/题号/有效位由一个融合核算出（全整数，
        与原先逐步 cupy 计算逐位相同），分块总量取主机端累计和（调用方给出
        pairs_total 且不超预算时整段一块、不回传），每段只剩布尔压缩一次同步；
        每行条目数用整数散射累加（精确）。
        """
        cp = self.cp
        group, fields, values = menu_gpu["group"], menu_gpu["fields"], menu_gpu["values"]
        n = int(sub.size)
        m = self.layout.num_marginals
        kmax = self.layout.kmax
        grp = group[sub]
        cur = cp.ascontiguousarray(codes_g[grp], dtype=cp.int32)
        num_fields = int(cur.shape[1])
        fsub = cp.ascontiguousarray(fields[sub], dtype=cp.int32)
        vsub = cp.ascontiguousarray(values[sub], dtype=cp.int32)
        if multi:
            c_all = cp.where(fsub >= 0, self.fm_cnt[cp.maximum(fsub, 0)], 0)
            pairs_per = c_all.sum(axis=1)
        else:
            pairs_per = self.fm_cnt[fsub[:, 0]]
        if pairs_total is not None and int(pairs_total) <= _DIRECT_PAIR_BUDGET:
            bounds, totals = [(0, n)], [int(pairs_total)]
        else:
            bounds, cums = _chunk_bounds(cp, pairs_per, _DIRECT_PAIR_BUDGET, with_cums=True)
            totals = [int(cums[hi - 1]) - (int(cums[lo - 1]) if lo else 0) for lo, hi in bounds]
        kern = _pair_kernels(cp)
        threads = 256
        datas, inds, nnzs = [], [], []
        total_entries = 0
        for (lo, hi), total in zip(bounds, totals):
            indptr = cp.zeros(hi - lo + 1, dtype=cp.int64)
            cp.cumsum(pairs_per[lo:hi], out=indptr[1:])
            row_loc = _seg_ids(cp, indptr, total)
            if multi:
                key = cp.empty(total, dtype=cp.int64)
                if total:
                    kern["keys"](
                        ((total + threads - 1) // threads,), (threads,),
                        (np.int64(total), np.int64(lo), row_loc, indptr, fsub,
                         self.fm_cnt, self.fm_off, self.fm_list, np.int64(m), key),
                    )
                key = cp.unique(key)
                total = int(key.size)
                row_loc = key // m
                m_idx = key - row_loc * m
            else:
                m_idx = indptr  # 占位，核内不读
            q2 = cp.empty(2 * total, dtype=cp.int64)
            d2 = cp.empty(2 * total, dtype=cp.int8)
            v2 = cp.empty(2 * total, dtype=cp.bool_)
            r2 = cp.empty(2 * total, dtype=cp.int64)
            if total:
                kern["pairs"](
                    ((total + threads - 1) // threads,), (threads,),
                    (np.int64(total), np.int64(lo), row_loc, indptr, m_idx,
                     np.int32(1 if multi else 0), cur, np.int32(num_fields), fsub, vsub,
                     self.m_off, self.m_fields, self.m_strides, np.int32(kmax),
                     self.fm_off, self.fm_list, self.cellq, q2, d2, v2, r2),
                )
            keep = cp.flatnonzero(v2)
            q_k, d_k, r_k = q2[keep], d2[keep], r2[keep]
            if not self.layout.ranges_sorted:
                order = cp.argsort(r_k * self.num_queries + q_k)
                q_k, d_k, r_k = q_k[order], d_k[order], r_k[order]
            nnz = cp.zeros(hi - lo, dtype=cp.uint64)  # add.at 不收 int64，无符号累加后按位重解释
            if r_k.size:
                cp.add.at(nnz, r_k, 1)
            nnzs.append(nnz.view(cp.int64))
            datas.append(d_k.astype(cp.float64))
            inds.append(q_k)
            total_entries += int(keep.size)
        if datas:
            data = cp.concatenate(datas)
            indices = cp.concatenate(inds)
            row_nnz = cp.concatenate(nnzs)
        else:
            data = cp.zeros(0, dtype=cp.float64)
            indices = cp.zeros(0, dtype=cp.int64)
            row_nnz = cp.zeros(n, dtype=cp.int64)
        indptr = cp.zeros(n + 1, dtype=cp.int64)
        cp.cumsum(row_nnz, out=indptr[1:])
        return {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "row_of": _seg_ids(cp, indptr, total_entries),
            "path_idx": sub.astype(cp.int64),
            "grp_of": grp,
        }

    @staticmethod
    def _slice_parts(cp, ent, bounds):
        """把一整段路径的稀疏块按路径区间切成多块，每块与单独直算该区间所得逐位相同。"""
        if len(bounds) == 1 and bounds[0] == (0, int(ent["path_idx"].size)):
            return [ent]
        ip = cp.asnumpy(ent["indptr"])
        out = []
        for lo, hi in bounds:
            a, b = int(ip[lo]), int(ip[hi])
            out.append({
                "data": ent["data"][a:b],
                "indices": ent["indices"][a:b],
                "indptr": ent["indptr"][lo:hi + 1] - ent["indptr"][lo],
                "row_of": ent["row_of"][a:b] - lo,
                "path_idx": ent["path_idx"][lo:hi],
                "grp_of": ent["grp_of"][lo:hi],
            })
        return out

    def _delta_direct(self, menu_gpu, codes_g, entry_budget: int = 0, fields_np=None):
        """答案差稀疏块的直算版，块划分照通用通路的条目预算切，产出逐位同序同值。

        单字段路径与多字段路径各整段直算一次再按块边界切片，
        省掉逐块的核发射与同步，每块内容与逐块直算相同。
        路径分类与分块在主机端按菜单的 fields 做（fields_np 不给则回传一次），
        不再在卡上做布尔压缩同步。
        """
        cp = self.cp
        if fields_np is None:
            fields_np = cp.asnumpy(menu_gpu["fields"])
        fields_np = np.asarray(fields_np)
        # 菜单经规范化后槽位按字段升序左齐、无效槽在后：槽 1 空即单字段路径。
        # 分块只在条目总量超预算时才需逐路径尺寸（adult 一段装下），总量用直方图点积算。
        f0, f1, f2 = fields_np[:, 0], fields_np[:, 1], fields_np[:, 2]
        dep_sizes = self.dep_sizes_np
        nf = dep_sizes.size
        parts = []
        singles = np.flatnonzero(f1 < 0)
        if singles.size:
            j0 = f0[singles]
            hist = np.bincount(j0, minlength=nf)
            total_s = int(hist @ dep_sizes)
            if entry_budget > 0 and total_s > entry_budget:
                bounds = _chunk_bounds(cp, dep_sizes[j0], entry_budget)
            else:
                bounds = [(0, int(singles.size))]
            ent = self._direct_entries(
                cp.asarray(singles), menu_gpu, codes_g, multi=False,
                pairs_total=int(hist @ self.fm_cnt_np),
            )
            parts.extend(self._slice_parts(cp, ent, bounds))
        multis = np.flatnonzero(f1 >= 0)
        if multis.size:
            j0, j1, j2 = f0[multis], f1[multis], f2[multis]
            j2v = j2[j2 >= 0]
            hist = np.bincount(j0, minlength=nf) + np.bincount(j1, minlength=nf)
            hist += np.bincount(j2v, minlength=nf)
            total_m = int(hist @ dep_sizes)
            if entry_budget > 0 and total_m > entry_budget:
                sizes_m = dep_sizes[j0] + dep_sizes[j1]
                sizes_m += np.where(j2 >= 0, dep_sizes[np.maximum(j2, 0)], 0)
                bounds = _chunk_bounds(cp, sizes_m, entry_budget)
            else:
                bounds = [(0, int(multis.size))]
            ent = self._direct_entries(
                cp.asarray(multis), menu_gpu, codes_g, multi=True,
                pairs_total=int(hist @ self.fm_cnt_np),
            )
            parts.extend(self._slice_parts(cp, ent, bounds))
        return parts

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
        return cnt, self._hs_scores(codes_g)

    # ------------------------------------------------------------------
    def _feats_from_counts(self, cnt, score_g):
        """由条件计数重构特征矩阵，整数数据下与 CPU 逐位相同。"""
        cp = self.cp
        feats = (cnt == self.ncond[None, :]).astype(cp.float64)
        if self.num_hs:
            feats[:, self.hs_cols] = (
                score_g >= self.hs_threshold[None, :]
            ).astype(cp.float64)
        return feats

    def _residual_from_feats(self, feats, counts):
        """由特征矩阵算残差损失与组错位分。"""
        cp = self.cp
        answers = counts @ feats
        residual = self.target - answers
        loss = float((residual * self.weights) @ residual / 2)
        # 组错位分，组覆盖的题按绝对加权残差求和，供工作批挑组
        scores = feats @ (self.weights * cp.abs(residual))
        return residual, loss, scores, answers

    def _residual_loss(self, cnt, score_g, counts):
        """由条件计数重构特征算残差损失与组错位分，整数数据下与 CPU 逐位相同。"""
        return self._residual_from_feats(self._feats_from_counts(cnt, score_g), counts)

    def _state_terms(self, codes_g, counts):
        """本轮表状态的残差、损失、组分与答案差所需的中间量，按通路分派。"""
        if self.direct:
            if self.sparse:
                return self._state_terms_sparse(codes_g, counts)
            feats, score_g = self._feats_direct(codes_g)
            residual, loss, scores, answers = self._residual_from_feats(feats, counts)
            return residual, loss, scores, score_g, None, answers
        cnt, score_g = self._condition_counts(codes_g)
        residual, loss, scores, answers = self._residual_loss(cnt, score_g, counts)
        return residual, loss, scores, score_g, cnt, answers

    def loss_of(self, codes: np.ndarray, state_ids) -> float:
        """当前表的损失，全程在卡上算，等价 CPU 的 table_loss。"""
        from .grouping import group_state_ids

        cp = self.cp
        grouped = group_state_ids(np.asarray(state_ids, dtype=np.int64))
        codes_g = cp.asarray(codes[grouped.unique_ids])
        counts = cp.asarray(grouped.counts.astype(np.float64))
        _, loss, _, _, _, _ = self._state_terms(codes_g, counts)
        return loss

    def group_losses_of(self, codes: np.ndarray, state_ids, group_index_dev, num_groups: int):
        """当前表按题分档的损失 ½ Σ_{i∈g} w_i e_i²（第六十八步分档地板），全程在卡上算。

        group_index_dev 为卡上的每题档号向量（0..num_groups-1），返回 (各档损失 numpy 向量, 总损失)，
        各档之和与 loss_of 同值（浮点求和顺序不同、末位可异）；只读不碰任何演化状态。
        """
        from .grouping import group_state_ids

        cp = self.cp
        grouped = group_state_ids(np.asarray(state_ids, dtype=np.int64))
        codes_g = cp.asarray(codes[grouped.unique_ids])
        counts = cp.asarray(grouped.counts.astype(np.float64))
        residual, loss, _, _, _, _ = self._state_terms(codes_g, counts)
        if int(group_index_dev.size) != int(residual.size):
            raise ValueError("分档索引长度须等于题数")
        per = cp.bincount(group_index_dev, weights=self.weights * residual * residual, minlength=num_groups)
        return cp.asnumpy(per) / 2.0, loss

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

    def _delta_parts(self, menu_gpu, cnt, codes_g, entry_budget: int = 0):
        """答案差稀疏块，结构复刻 CPU 版，坐标合并用键排序整型分段和。

        entry_budget 为正时按条目预算把路径切块逐块展开，
        路径不跨块且块内条目序同不分块，每路径的收益逐位不变，
        大域宽数据条目流上亿时靠这个压住显存峰值，默认 0 不切块零改变。
        """
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
            # 空块（整块路径都不改答案）时 bincount 无元素会报错，直接给零计数
            row_nnz = (
                cp.bincount(row_of[nz], minlength=path_idx.size) if nz.size
                else cp.zeros(path_idx.size, dtype=cp.int64)
            )
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
            j_all = fields[singles, 0].astype(cp.int64)
            for lo, hi in _chunk_bounds(cp, self.dep_sizes[j_all], entry_budget):
                sub = singles[lo:hi]
                j_of = fields[sub, 0]
                cur = codes_g[group[sub], j_of]
                indptr, indices, adj, _ = self._entry_expand(
                    j_of, cur, values[sub, 0]
                )
                _finish(indptr, indices, adj, sub.astype(cp.int64))

        # 多字段路径，各槽位条目流展开后坐标键排序合并，整型前缀和精确无误差
        multis = cp.flatnonzero(nslots >= 2)
        if multis.size:
            sizes_m = cp.zeros(multis.size, dtype=cp.int64)
            for slot in range(3):
                jm = fields[multis, slot]
                sizes_m += cp.where(
                    jm >= 0, self.dep_sizes[cp.maximum(jm, 0).astype(cp.int64)], 0
                )
            for lo, hi in _chunk_bounds(cp, sizes_m, entry_budget):
                sub = multis[lo:hi]
                rows_parts, cols_parts, adj_parts = [], [], []
                for slot in range(3):
                    live_mask = fields[sub, slot] >= 0
                    live = sub[live_mask]
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
                if int(sub.size) * self.num_queries < 2 ** 31:
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
                row_nnz = cp.bincount(urows, minlength=int(sub.size))
                indptr = cp.zeros(int(sub.size) + 1, dtype=cp.int64)
                cp.cumsum(row_nnz, out=indptr[1:])
                _finish(indptr, ucols, merged, sub.astype(cp.int64))
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
        validate_tilt_parameters(alpha, numerical_tol)
        if not bool(cp.all(cp.isfinite(gains_flat))):
            raise FloatingPointError("numerical_indeterminate: 输入增益非有限")

        num_groups = offsets.size - 1
        gmax = _seg_max(cp, gains_flat, offsets)
        max_gain_sum = float(counts @ cp.maximum(gmax, 0.0))
        requirement = calibration_requirement(max_gain_sum, alpha, old_loss)
        if max_gain_sum == 0.0:
            if bool(cp.any(gmax > 0)):
                raise FloatingPointError("numerical_indeterminate: 正增益在聚合中丢失")
            return None, max_gain_sum, requirement

        centered = gains_flat - gmax[seg_ids]
        kern = _tilt_kernels(cp)
        total = int(gains_flat.size)
        nseg = int(num_groups)
        off64 = offsets.astype(cp.int64, copy=False)
        gains_c = cp.ascontiguousarray(gains_flat, dtype=cp.float64)
        ref_c = cp.ascontiguousarray(ref_flat, dtype=cp.float64)

        # 评估缓冲区只分配一次：每次评估整体改写，返回的 P 总是最近一次评估所写
        zw = cp.empty(total, dtype=cp.float64)
        norm = cp.zeros(nseg, dtype=cp.float64)
        P = cp.empty(total, dtype=cp.float64)
        per_group = cp.zeros(nseg, dtype=cp.float64)

        def evaluate(beta: float):
            if not np.isfinite(beta):
                raise FloatingPointError("numerical_indeterminate: beta 超出有限表示范围")
            # 两个融合核（见 _TILT_KERNEL_SRC），与 ref·exp(beta·centered) / seg_sum / 归一 / seg_sum 逐位相同
            if nseg and total:
                kern["norm"]((nseg,), (128,), (ref_c, centered, np.float64(beta), off64, zw, norm, np.int64(nseg)))
                kern["gain"]((nseg,), (128,), (zw, norm, gains_c, off64, P, per_group, np.int64(nseg)))
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
        validate_tilt_result(D, requirement, max_gain_sum, numerical_tol)
        return (beta, ps, D), max_gain_sum, requirement

    # ------------------------------------------------------------------
    def _probe_grid(self, gains_flat, ref_flat, offsets, seg_ids, counts,
                    menu_gpu, parts, delta_hs, path_pos, damping,
                    max_expected_rows, max_gain_sum, entry_budget,
                    grid_size=16):
        """beta 网格探针 GPU 版，与 CPU 版 _probe_beta_grid 同口径。

        纯只读确定性，不碰随机流不写主路径任何数组，
        漂移合并用与主路径相同的键排序分段归约，探针自持漂移数组。
        """
        cp = self.cp
        num_queries = self.num_queries
        seg_starts = offsets[:-1]
        num_groups = int(offsets.size - 1)
        gmax = _seg_max(cp, gains_flat, offsets)
        centered = gains_flat - gmax[seg_ids]

        def tilted(beta: float):
            zw = ref_flat * cp.exp(beta * centered)
            norm = _seg_sum(cp, zw, offsets)
            P = zw / norm[seg_ids]
            D = float(counts @ _seg_sum(cp, P * gains_flat, offsets))
            return P, D

        target = 0.99 * max_gain_sum
        beta_sat = 1.0 / max(max_gain_sum, 1e-300)
        _, dv = tilted(beta_sat)
        for _ in range(200):
            if dv >= target:
                break
            beta_sat *= 2.0
            _, dv = tilted(beta_sat)

        kern = _seg_kernels(cp)["run"]
        grid = [0.0] + list(
            np.geomspace(beta_sat / 2 ** 14, beta_sat, grid_size - 1)
        )
        best_beta, best_j = 0.0, 0.0
        for beta in grid:
            P, D = tilted(float(beta))
            if D <= 0.0:
                continue
            rates_flat = P.copy()
            rates_flat[seg_starts] = 0.0
            leave = _seg_sum(cp, rates_flat, offsets)
            max_leave = float(leave.max())
            if max_leave <= 0.0:
                continue
            rates_paths = rates_flat[path_pos]
            drifts = cp.zeros(num_groups * num_queries, dtype=cp.float64)

            def merge_into(keys, wvals, accumulate):
                order = cp.argsort(keys)
                sk = keys[order]
                sw = wvals[order]
                newseg = cp.concatenate(
                    [cp.ones(1, dtype=cp.bool_), sk[1:] != sk[:-1]]
                )
                starts = cp.flatnonzero(newseg)
                bounds = cp.concatenate(
                    [starts[1:], cp.asarray([int(sk.size)], dtype=cp.int64)]
                )
                sums = cp.zeros(starts.size, dtype=cp.float64)
                nruns = int(starts.size)
                if nruns == 0:
                    return
                threads = 256
                kern(
                    ((nruns + threads - 1) // threads,), (threads,),
                    (sw, starts, bounds, sums, np.int64(nruns)),
                )
                uk = sk[starts].astype(cp.int64)
                if accumulate:
                    drifts[uk] += sums
                else:
                    drifts[uk] = sums

            if entry_budget > 0:
                for part in parts:
                    if part["indices"].size == 0:
                        continue
                    keys = (
                        part["grp_of"][part["row_of"]] * num_queries
                        + part["indices"]
                    )
                    merge_into(
                        keys,
                        part["data"]
                        * rates_paths[part["path_idx"]][part["row_of"]],
                        True,
                    )
            else:
                key_parts, weight_parts = [], []
                for part in parts:
                    keys = (
                        part["grp_of"][part["row_of"]] * num_queries
                        + part["indices"]
                    )
                    key_parts.append(keys)
                    weight_parts.append(
                        part["data"]
                        * rates_paths[part["path_idx"]][part["row_of"]]
                    )
                if key_parts:
                    merge_into(
                        cp.concatenate(key_parts),
                        cp.concatenate(weight_parts),
                        False,
                    )
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
            unit_rows = float(counts @ leave)
            step = analytic_step(
                max_leave, D, interaction, damping,
                unit_expected_rows=unit_rows, max_expected_rows=max_expected_rows,
            )
            j = expected_improvement(step, D, interaction)
            if j > best_j:
                best_j, best_beta = j, float(beta)
        return best_beta, best_j

    # ------------------------------------------------------------------
    def _merge_keys(self, keys, wvals):
        """按键稳定排序 + 游程串行求和，返回（唯一键升序，对应和）。

        同键条目按原序相加（cupy argsort 稳定），同输入同卡逐位可复现；
        稠密与稀疏漂移通路共用，二者每个 (组, 题) 项的值逐位相同。
        """
        cp = self.cp
        if int(keys.size) == 0:
            return cp.zeros(0, dtype=cp.int64), cp.zeros(0, dtype=cp.float64)
        order = cp.argsort(keys)
        sk = keys[order]
        sw = wvals[order]
        newseg = cp.concatenate([cp.ones(1, dtype=cp.bool_), sk[1:] != sk[:-1]])
        starts = cp.flatnonzero(newseg)
        bounds = cp.concatenate(
            [starts[1:], cp.asarray([int(sk.size)], dtype=cp.int64)]
        )
        sums = cp.zeros(starts.size, dtype=cp.float64)
        nruns = int(starts.size)
        threads = 256
        _seg_kernels(cp)["run"](
            ((nruns + threads - 1) // threads,), (threads,),
            (sw, starts, bounds, sums, np.int64(nruns)),
        )
        return sk[starts].astype(cp.int64), sums

    def _drift_sparse(self, parts, rates_paths, counts, num_groups, num_queries,
                      entry_budget: int = 0):
        """漂移项的稀疏版：只合并非零 (组, 题) 项，不铺 (组数×题数) 稠密矩阵。

        条目按键 (题号, 组号) 稳定排序合并；答案差的块按条目预算拼成尽量少的段
        （直算条目远少于通用通路的依赖条目，adult 一段装下），每段一次合并，
        多段时各段的唯一键再按段序合并一次；
        键已按题号升序，题号变化处即段界，drift_total 一段一块定序树规约，
        自交叉项一次全量规约，同输入同卡逐位可复现；
        与稠密 gemv 的差只在浮点求和顺序。
        """
        cp = self.cp
        batches, cur, cur_n = [], [], 0
        for part in parts:
            n = int(part["indices"].size)
            if n == 0:
                continue
            if cur and entry_budget > 0 and cur_n + n > entry_budget:
                batches.append(cur)
                cur, cur_n = [], 0
            cur.append(part)
            cur_n += n
        if cur:
            batches.append(cur)
        uks, sums_all = [], []
        kv_kern = _tilt_kernels(cp)["drift_kv"]
        rates_c = cp.ascontiguousarray(rates_paths, dtype=cp.float64)
        for batch in batches:
            # 融合核直接把各块的键与权写进拼接缓冲区，与逐块算完再 concatenate 逐位相同
            n_all = sum(int(p["indices"].size) for p in batch)
            keys = cp.empty(n_all, dtype=cp.int64)
            vals = cp.empty(n_all, dtype=cp.float64)
            base = 0
            for p in batch:
                n = int(p["indices"].size)
                kv_kern(
                    ((n + 255) // 256,), (256,),
                    (np.int64(n), np.int64(base),
                     cp.ascontiguousarray(p["indices"], dtype=cp.int64),
                     cp.ascontiguousarray(p["data"], dtype=cp.float64),
                     cp.ascontiguousarray(p["row_of"], dtype=cp.int64),
                     cp.ascontiguousarray(p["grp_of"], dtype=cp.int64),
                     cp.ascontiguousarray(p["path_idx"], dtype=cp.int64),
                     rates_c, np.int64(num_groups), keys, vals),
                )
                base += n
            uk, sums = self._merge_keys(keys, vals)
            uks.append(uk)
            sums_all.append(sums)
        if not uks:
            return cp.zeros(num_queries, dtype=cp.float64), 0.0
        if len(uks) == 1:
            uk, sums = uks[0], sums_all[0]
        else:
            uk, sums = self._merge_keys(cp.concatenate(uks), cp.concatenate(sums_all))
        q_idx = uk // num_groups
        g_idx = uk - q_idx * num_groups
        cg = counts[g_idx]
        newq = cp.concatenate([cp.ones(1, dtype=cp.bool_), q_idx[1:] != q_idx[:-1]])
        starts = cp.flatnonzero(newq)
        offsets = cp.concatenate([starts, cp.asarray([int(uk.size)], dtype=cp.int64)])
        drift_total = cp.zeros(num_queries, dtype=cp.float64)
        drift_total[q_idx[starts]] = _seg_sum(cp, cg * sums, offsets)
        self_cross = float(cp.sum(cg * self.weights[q_idx] * sums * sums))
        return drift_total, self_cross

    def _gain_recheck_mask(self, gains, magnitudes, terms, numerical_tol):
        """固定量整数性已缓存；检查状态与选中数量只回传一次。"""
        validate_numerical_tol(numerical_tol)
        cp = self.cp
        size = int(gains.size)
        mask = cp.empty(size, dtype=cp.bool_)
        if not size:
            return mask, 0
        status = cp.zeros(2, dtype=cp.uint64)
        _tilt_kernels(cp)["gain_recheck"](
            ((size + 255) // 256,), (256,),
            (gains, magnitudes, terms, np.int64(size), np.int32(self._integral_gain_inputs),
             np.float64(numerical_tol), np.float64(np.finfo(np.float64).eps / 2.0),
             np.float64(np.nextafter(0.0, 1.0)), np.float64(2.0**50), mask, status),
        )
        selected, flags = (int(value) for value in cp.asnumpy(status))
        if flags & 1:
            raise FloatingPointError("numerical_indeterminate: 局部增益或运算规模非有限")
        if flags & 2:
            raise FloatingPointError("numerical_indeterminate: 局部运算规模不能为负")
        if flags & 4 and (not self._integral_gain_inputs or flags & 8):
            raise FloatingPointError("numerical_indeterminate: 无法建立有效的局部舍入界")
        return mask, selected

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
        entry_budget: int = 0,
        probe_grid: int = 0,
        ref_shape: str = "uniform",
        path_factors: np.ndarray | None = None,
        struct_shaper=None,
        struct_temp: float = float("nan"),
    ) -> BatchKernelResult:
        """构造一轮批量核，输入输出与 CPU build_batch_kernel 对齐。"""
        validate_step_parameters(damping, max_expected_rows)
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

        # 残差由条件计数重构特征得出，整数数据下与 CPU 逐位相同；
        # 单元格直算通路改由格号散射特征，答案差直接取旧格新格两项
        residual, old_loss, scores_gpu, score_g, cnt, answers = self._state_terms(codes_g, counts)
        # 组分数回传推迟到出口处，免得在此处同步截断后续核的提交

        if self.direct:
            parts = self._delta_direct(menu_gpu, codes_g, entry_budget, fields_np=menu.fields)
        else:
            parts = self._delta_parts(menu_gpu, cnt, codes_g, entry_budget)
        delta_hs = self._halfspace_delta(menu_gpu, score_g, codes_g)
        we = cp.ascontiguousarray(self.weights * residual, dtype=cp.float64)
        gains_paths = cp.zeros(menu.num_paths, dtype=cp.float64)
        magnitudes = cp.zeros(menu.num_paths, dtype=cp.float64)
        gain_terms = cp.zeros(menu.num_paths, dtype=cp.int64)
        part_gain_kern = _tilt_kernels(cp)["part_gains"]
        for part in parts:
            # 一路径一块的融合核：lin = Σ d·we[q]，quad = Σ (d·d)·w[q]，写入 lin − quad/2，
            # 与原先两次 seg_sum 加散射赋值逐位相同
            nseg_part = int(part["path_idx"].size)
            if nseg_part == 0:
                continue
            part_gain_kern(
                (nseg_part,), (128,),
                (cp.ascontiguousarray(part["data"], dtype=cp.float64),
                 cp.ascontiguousarray(part["indices"], dtype=cp.int64),
                 we, self.weights,
                 cp.ascontiguousarray(part["indptr"], dtype=cp.int64),
                 cp.ascontiguousarray(part["path_idx"], dtype=cp.int64),
                 gains_paths, magnitudes, np.int64(nseg_part)),
            )
            gain_terms[part["path_idx"]] = cp.diff(part["indptr"])
        if delta_hs is not None:
            w_hs = self.weights[self.hs_cols]
            gains_paths += delta_hs @ we[self.hs_cols]
            quadratic_hs = (delta_hs * delta_hs) @ w_hs / 2
            gains_paths -= quadratic_hs
            magnitudes += cp.abs(delta_hs) @ cp.abs(we[self.hs_cols]) + quadratic_hs
            gain_terms += cp.count_nonzero(delta_hs, axis=1)

        recheck, selected_count = self._gain_recheck_mask(
            gains_paths, magnitudes, gain_terms, numerical_tol,
        )
        if selected_count:
            selected_dev = cp.flatnonzero(recheck)
            selected_parts = []
            for part in parts:
                rows = cp.flatnonzero(recheck[part["path_idx"]])
                if not rows.size:
                    continue
                starts = part["indptr"][rows]
                lengths = part["indptr"][rows + 1] - starts
                indptr = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(lengths)])
                total = int(indptr[-1])
                row_of = _seg_ids(cp, indptr, total)
                entries = cp.arange(total) + (starts - indptr[:-1])[row_of]
                selected_parts.append((
                    cp.asnumpy(part["path_idx"][rows]), cp.asnumpy(indptr),
                    cp.asnumpy(part["indices"][entries]), cp.asnumpy(part["data"][entries]),
                ))
            corrected = exact_sparse_path_gains(
                cp.asnumpy(selected_dev), selected_parts,
                ExactGainOracle.from_answers(
                    self._target_host, self._weights_host, cp.asnumpy(answers),
                ),
                None if delta_hs is None else cp.asnumpy(delta_hs[selected_dev]),
                cp.asnumpy(self.hs_cols),
            )
            gains_paths[selected_dev] = cp.asarray(corrected)

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
        if path_factors is not None or ref_shape == "distance":
            # host 端同一实现装配参考分布再上传（CPU 算好的因子或
            # distance 形状走这里），uniform 无因子时留在卡上
            from .batchkernel import _reference_arrays

            ref_src_np, ref_paths_np = _reference_arrays(
                menu, codes[grouped.unique_ids],
                grouped.counts.astype(np.float64),
                num_groups, stay_probability, ref_shape, path_factors,
            )
            ref_flat[path_pos] = cp.asarray(ref_paths_np)
            ref_flat[seg_starts] = cp.asarray(ref_src_np)
        elif ref_shape == "uniform":
            ref_paths_dev = (
                (1 - stay_probability) * menu_gpu["mass"] / mass_sum[menu_gpu["group"]]
            )
            if struct_shaper is not None:
                # 卡内药房：因子与签筒改形全在 GPU，免 host 往返
                factors_dev = struct_shaper.path_factors_dev(
                    menu_gpu, codes_g, counts, temperature=struct_temp,
                )
                shaped = ref_paths_dev * factors_dev
                orig = _seg_sum(cp, ref_paths_dev, menu_gpu["offsets"])
                new = _seg_sum(cp, shaped, menu_gpu["offsets"])
                orig = cp.where(empty, 1.0, orig)
                new = cp.where(empty, 1.0, new)
                scale = cp.where(new > 0, orig / cp.where(new > 0, new, 1.0), 1.0)
                ref_paths_dev = shaped * scale[menu_gpu["group"]]
            ref_flat[path_pos] = ref_paths_dev
            ref_flat[seg_starts] = cp.where(empty, 1.0, stay_probability)
        else:
            raise ValueError(f"未知参考分布形状 {ref_shape}")

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
                cp.asnumpy(scores_gpu),
            )
        beta, ps, direction_gain = tilt

        rates_flat = ps.copy()
        rates_flat[seg_starts] = 0.0
        leave = _seg_sum(cp, rates_flat, offsets)
        rates_paths = rates_flat[path_pos]

        # 漂移，键排序加前缀和分段归约，确定性合并
        if self.sparse:
            drift_total, self_cross = self._drift_sparse(
                parts, rates_paths, counts, num_groups, num_queries, entry_budget,
            )
        else:
            drifts = cp.zeros(num_groups * num_queries, dtype=cp.float64)
            if entry_budget > 0:
                # 分块口径，逐块排序合并后散射累加，免去全量条目流一次性拼接
                for part in parts:
                    if part["indices"].size == 0:
                        continue
                    keys = part["grp_of"][part["row_of"]] * num_queries + part["indices"]
                    uk, sums = self._merge_keys(
                        keys,
                        part["data"] * rates_paths[part["path_idx"]][part["row_of"]],
                    )
                    # 块内键唯一花式加法安全，跨块按块序累加，分块固定结果确定
                    drifts[uk] += sums
            else:
                key_parts, weight_parts = [], []
                for part in parts:
                    keys = part["grp_of"][part["row_of"]] * num_queries + part["indices"]
                    key_parts.append(keys)
                    weight_parts.append(
                        part["data"] * rates_paths[part["path_idx"]][part["row_of"]]
                    )
                if key_parts:
                    uk, sums = self._merge_keys(
                        cp.concatenate(key_parts), cp.concatenate(weight_parts),
                    )
                    drifts[uk] = sums
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
        unit_rows = float(np.dot(grouped.counts.astype(np.float64), leave_np))
        step = analytic_step(
            float(leave_np.max()), direction_gain, interaction, damping,
            unit_expected_rows=unit_rows, max_expected_rows=max_expected_rows,
        )
        probs_flat, has_movement = final_probabilities(rates_np, step, offsets_np)
        expected_loss, upper, expected_rows = kernel_statistics(
            old_loss, direction_gain, interaction, step, unit_rows, has_movement,
        )
        probe = None
        if probe_grid > 0:
            probe = self._probe_grid(
                gains_flat, ref_flat, offsets, seg_ids, counts, menu_gpu,
                parts, delta_hs, path_pos, damping, max_expected_rows,
                max_gain_sum, entry_budget, probe_grid,
            )
        return BatchKernelResult(
            menu, grouped, offsets_np, probs_flat, old_loss,
            beta, direction_gain, interaction, step,
            expected_loss, upper, expected_rows,
            max_gain_sum, requirement, "ok",
            cp.asnumpy(scores_gpu),
            probe,
        )
