"""批量核构造与抽样落地，在差分菜单上直接算增益概率与漂移。

候选不注册，增益的两项都从稀疏答案差算出，
线性项 d@(We) 与二次项 ||d||_W^2 共用一个路径乘查询的稀疏差矩阵，
漂移 v_g=sum_p r_p d_p 按组聚合成稀疏矩阵后与权重直接组装交互项，
熵校准与解析步长复用平铺版实现，数学主线与组核完全一致，
只有抽中落地的新状态才进注册表，每轮注册量不超过实际改动行数。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .batchmenu import (
    BatchMenu,
    CntCache,
    CodeBook,
    QueryStructure,
    build_query_structure,
    condition_counts,
    generate_batch_menu,
    halfspace_scores,
)
from .dataset import StateRegistry
from .editspace import EditBudget, generate_edit_supports, tuples_from_ids
from .engine import EvolveResult, RoundRecord, build_kernel, sample_next
from .grouping import GroupedTable, group_state_ids, grouped_residual
from .pairing import PairingBudget, build_paired_supports, build_rescue_menu
from .state import Workload, loss_from_residual
from .stepsize import analytic_step
from .tilt import calibrate_beta_flat


@dataclass
class BatchKernelResult:
    """批量核输出，flat 数组按组分段，每段索引 0 恒为保持项。"""

    menu: BatchMenu
    grouped: GroupedTable
    offsets: NDArray[np.int64]  # G+1，含保持项的段边界
    probabilities: NDArray[np.float64]  # 段内 0 位是保持概率
    old_loss: float
    beta: float
    direction_gain: float
    interaction: float
    step: float
    expected_loss: float
    analytic_upper_bound: float
    expected_changed_rows: float
    max_gain_sum: float
    required_gain: float
    status: str
    group_scores: NDArray[np.float64] | None = None  # 每组错位分，供工作批挑组


def _delta_parts(
    menu: BatchMenu,
    qs: QueryStructure,
    cnt: NDArray[np.int16],
    codes_g: NDArray[np.int32],
    num_queries: int,
) -> list[tuple[csr_matrix, NDArray[np.int64]]]:
    """构造答案差稀疏块，返回若干 (差矩阵, 对应路径索引) 对。

    单字段路径每行列唯一且有序，按依赖表直接拼出规范稀疏阵免排序免合并，
    多字段路径同一查询可能收到多个字段的计数调整，走坐标格式合并，
    新旧通过判断都由计数与字段组个数的比较得出，与逐查询求值精确同值。
    """
    parts: list[tuple[csr_matrix, NDArray[np.int64]]] = []
    valid = menu.fields >= 0
    nslots = valid.sum(axis=1)
    dep_sizes = np.array([len(d) for d in qs.dep], dtype=np.int64)

    def _finish(indptr, indices, adj, path_idx):
        """计数调整还原成答案差，通过当且仅当计数打满字段组个数。

        计数与调整都是小整数，比较全程走整型，最后才落成浮点差，
        组号先按路径取一次再展开到非零位，省去嵌套花式索引，
        恰为零的答案差直接滤掉，下游矩阵向量积与漂移聚合按序累加，
        去掉精确零加项每行部分和逐位不变。
        """
        grp_of = menu.group[path_idx]
        row_of = np.repeat(np.arange(len(path_idx), dtype=np.int64), np.diff(indptr))
        cnt_vals = cnt[grp_of[row_of], indices]
        need = qs.ncond[indices]
        delta_i8 = (cnt_vals + adj == need).astype(np.int8)
        delta_i8 -= cnt_vals == need
        nz = np.flatnonzero(delta_i8)
        row_nnz = np.bincount(row_of[nz], minlength=len(path_idx))
        new_indptr = np.zeros(len(path_idx) + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=new_indptr[1:])
        sparse = csr_matrix(
            (delta_i8[nz].astype(np.float64), indices[nz], new_indptr),
            shape=(len(path_idx), num_queries),
        )
        parts.append((sparse, path_idx))

    # 单字段路径，行内列直接取依赖表本身有序，免通用排序与合并
    singles = np.flatnonzero(nslots == 1)
    if len(singles):
        j_of = menu.fields[singles, 0]
        v_of = menu.values[singles, 0]
        row_nnz = dep_sizes[j_of]
        indptr = np.zeros(len(singles) + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=np.int64)
        adj = np.empty(int(indptr[-1]), dtype=np.int8)
        order = np.argsort(j_of, kind="stable")
        bounds = np.flatnonzero(
            np.r_[True, j_of[order][1:] != j_of[order][:-1], True]
        )
        for b in range(len(bounds) - 1):
            sel = order[bounds[b] : bounds[b + 1]]
            j = int(j_of[sel[0]])
            dep_j = qs.dep[j]
            if len(dep_j) == 0:
                continue
            g_sel = menu.group[singles[sel]]
            cur_pass = qs.allow[j][:, codes_g[g_sel, j]]  # mj 乘 n
            new_pass = qs.allow[j][:, v_of[sel]]
            block_adj = (new_pass.astype(np.int8) - cur_pass.astype(np.int8)).T
            slots = indptr[sel][:, None] + np.arange(len(dep_j))[None, :]
            indices[slots.ravel()] = np.tile(dep_j, len(sel))
            adj[slots.ravel()] = block_adj.ravel()
        _finish(indptr, indices, adj, singles.astype(np.int64))

    # 多字段路径，坐标格式合并同查询的多字段调整
    multis = np.flatnonzero(nslots >= 2)
    if len(multis):
        rows_parts = []
        cols_parts = []
        adj_parts = []
        for slot in range(3):
            live = multis[menu.fields[multis, slot] >= 0]
            if len(live) == 0:
                continue
            j_of = menu.fields[live, slot]
            order = np.argsort(j_of, kind="stable")
            live = live[order]
            j_sorted = j_of[order]
            bounds = np.flatnonzero(
                np.r_[True, j_sorted[1:] != j_sorted[:-1], True]
            )
            for b in range(len(bounds) - 1):
                sel = live[bounds[b] : bounds[b + 1]]
                j = int(j_sorted[bounds[b]])
                dep_j = qs.dep[j]
                if len(dep_j) == 0:
                    continue
                g_sel = menu.group[sel]
                cur_pass = qs.allow[j][:, codes_g[g_sel, j]]
                new_pass = qs.allow[j][:, menu.values[sel, slot]]
                block_adj = (new_pass.astype(np.int8) - cur_pass.astype(np.int8)).T
                loc = np.searchsorted(multis, sel)  # multis 升序，局部行号直接二分
                rows_parts.append(np.repeat(loc, len(dep_j)))
                cols_parts.append(np.tile(dep_j, len(sel)))
                adj_parts.append(block_adj.ravel())
        coo = coo_matrix(
            (
                np.concatenate(adj_parts),
                (np.concatenate(rows_parts), np.concatenate(cols_parts)),
            ),
            shape=(len(multis), num_queries),
        )
        sparse = coo.tocsr()  # tocsr 已合并重复并排序列下标，无需再 sum_duplicates
        _finish(sparse.indptr, sparse.indices, sparse.data, multis.astype(np.int64))
    return parts


def _halfspace_delta(
    menu: BatchMenu,
    qs: QueryStructure,
    score_g: NDArray[np.int64],
    codes_g: NDArray[np.int32],
) -> NDArray[np.float64] | None:
    """半空间通道的答案差，路径乘半空间的稠密矩阵，无半空间时返回 None。

    半空间一般依赖全部字段，答案差在半空间列上天然稠密，
    每槽的分数调整按字段分桶整列查表，通过判断全程整型，
    新旧过线之差才落成浮点，与逐查询求值精确同值。
    """
    num_hs = qs.num_halfspaces
    if num_hs == 0 or menu.num_paths == 0:
        return None
    adj = np.zeros((menu.num_paths, num_hs), dtype=np.int64)
    for slot in range(3):
        live = np.flatnonzero(menu.fields[:, slot] >= 0)
        if len(live) == 0:
            continue
        j_of = menu.fields[live, slot]
        order = np.argsort(j_of, kind="stable")
        live_s = live[order]
        j_s = j_of[order]
        bounds = np.flatnonzero(np.r_[True, j_s[1:] != j_s[:-1], True])
        for b in range(len(bounds) - 1):
            sel = live_s[bounds[b] : bounds[b + 1]]
            j = int(j_s[bounds[b]])
            tab = qs.hs_score[j]
            if not (tab.size and tab.any()):
                continue
            cur = codes_g[menu.group[sel], j]
            new = menu.values[sel, slot]
            adj[sel] += (tab[:, new] - tab[:, cur]).T
    base = score_g[menu.group]
    thresh = qs.hs_threshold[None, :]
    delta = ((base + adj) >= thresh).astype(np.float64)
    delta -= base >= thresh
    return delta


def build_batch_kernel(
    workload: Workload,
    grouped: GroupedTable,
    menu: BatchMenu,
    qs: QueryStructure,
    codes: NDArray[np.int32],
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    numerical_tol: float = 1e-12,
    cnt_cache: CntCache | None = None,
    beta_hint: float | None = None,
    want_scores: bool = False,
) -> BatchKernelResult:
    """构造一轮批量核，与组核的数学定义逐项相同，浮点顺序不同。

    增益 G=d@(We)-||d||_W^2/2 在稀疏差矩阵上一次算完全部路径，
    参考分布保持概率之外按合并质量比例分配，与迁移率语义一致，
    整代矩按重数加权，交互项的总漂移与自交叉扣除都在稀疏结构上组装。
    """
    if not (0 < stay_probability < 1):
        raise ValueError("stay_probability 必须落在 (0,1)")
    num_groups = grouped.num_groups
    residual = grouped_residual(workload, grouped)
    old_loss = loss_from_residual(workload, residual)
    group_scores = None
    if want_scores:
        # 组错位分，组覆盖的题按绝对加权残差求和，供工作批挑组
        group_scores = workload.features[grouped.unique_ids] @ (
            workload.weights * np.abs(residual)
        )

    codes_g = codes[grouped.unique_ids]
    if cnt_cache is not None:
        cnt_all, score_all = cnt_cache.sync(codes)
        cnt = cnt_all[grouped.unique_ids]
        score_g = score_all[grouped.unique_ids]
    else:
        cnt, _ = condition_counts(qs, codes_g)
        score_g = halfspace_scores(qs, codes_g)
    parts = _delta_parts(menu, qs, cnt, codes_g, workload.num_queries)
    delta_hs = _halfspace_delta(menu, qs, score_g, codes_g)
    we = workload.weights * residual
    gains_paths = np.zeros(menu.num_paths, dtype=np.float64)
    for sparse, path_idx in parts:
        # 平方阵共享下标结构只换数据，免整块拷贝
        squared = csr_matrix(
            (sparse.data * sparse.data, sparse.indices, sparse.indptr),
            shape=sparse.shape,
        )
        gains_paths[path_idx] = sparse @ we - (squared @ workload.weights) / 2
    if delta_hs is not None:
        w_hs = workload.weights[qs.hs_cols]
        gains_paths += delta_hs @ we[qs.hs_cols]
        gains_paths -= (delta_hs * delta_hs) @ w_hs / 2

    # flat 结构，每段索引 0 是保持项，其后依次是该组菜单路径
    lengths = np.diff(menu.offsets) + 1
    offsets = np.zeros(num_groups + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    seg_starts = offsets[:-1]
    path_pos = np.arange(menu.num_paths, dtype=np.int64) + menu.group + 1
    gains_flat = np.zeros(total, dtype=np.float64)
    gains_flat[path_pos] = gains_paths
    ref_flat = np.zeros(total, dtype=np.float64)
    mass_sum = np.add.reduceat(
        np.r_[menu.mass, 0.0], np.minimum(menu.offsets[:-1], len(menu.mass))
    )
    empty = np.diff(menu.offsets) == 0
    mass_sum[empty] = 1.0  # 退化组无路径，保持概率直接为 1
    ref_flat[path_pos] = (1 - stay_probability) * menu.mass / mass_sum[menu.group]
    ref_flat[seg_starts] = np.where(empty, 1.0, stay_probability)

    counts = grouped.counts.astype(np.float64)
    tilt = calibrate_beta_flat(
        gains_flat, ref_flat, offsets, old_loss,
        alpha=alpha, numerical_tol=numerical_tol, multiplicities=counts,
        beta_hint=beta_hint,
    )
    if tilt.frozen:
        return BatchKernelResult(
            menu, grouped, offsets, tilt.probabilities, old_loss,
            0.0, 0.0, 0.0, 0.0, old_loss, old_loss, 0.0,
            tilt.max_gain_sum, tilt.required_gain, "no_positive_direction",
            group_scores,
        )

    rates_flat = tilt.probabilities.copy()
    rates_flat[seg_starts] = 0.0
    leave = np.add.reduceat(rates_flat, seg_starts)
    rates_paths = rates_flat[path_pos]
    # 漂移 v_g=sum_p r_p d_p，组乘查询键上按整数桶累加免排序合并
    key_parts = []
    weight_parts = []
    for sparse, path_idx in parts:
        grp_of = menu.group[path_idx]
        row_of = np.repeat(
            np.arange(len(path_idx), dtype=np.int64), np.diff(sparse.indptr)
        )
        key_parts.append(
            grp_of[row_of] * workload.num_queries + sparse.indices
        )
        weight_parts.append(sparse.data * rates_paths[path_idx][row_of])
    drifts = np.bincount(
        np.concatenate(key_parts) if key_parts else np.zeros(0, dtype=np.int64),
        weights=np.concatenate(weight_parts) if weight_parts else None,
        minlength=num_groups * workload.num_queries,
    ).reshape(num_groups, workload.num_queries)
    if delta_hs is not None:
        # 菜单路径组内连续，半空间漂移按段边界一次 reduceat 聚合
        weighted = delta_hs * rates_paths[:, None]
        starts = np.minimum(menu.offsets[:-1], len(weighted))
        padded = np.vstack([weighted, np.zeros((1, delta_hs.shape[1]))])
        seg = np.add.reduceat(padded, starts, axis=0)
        seg[np.diff(menu.offsets) == 0] = 0.0
        drifts[:, qs.hs_cols] += seg
    drift_total = counts @ drifts
    self_cross = float(counts @ ((drifts * drifts) @ workload.weights))
    interaction = float(
        (np.dot(drift_total * workload.weights, drift_total) - self_cross) / 2
    )
    h_star = analytic_step(float(leave.max()), tilt.direction_gain, interaction)
    unit_rows = float(counts @ leave)
    if max_expected_rows is not None and unit_rows > 0:
        step = damping * min(h_star, max_expected_rows / unit_rows)
    else:
        step = damping * h_star

    probs_flat = step * rates_flat
    stay = 1.0 - np.add.reduceat(probs_flat, seg_starts)
    if np.any(stay < -1e-12):
        raise FloatingPointError("步长违反随机矩阵约束")
    probs_flat[seg_starts] = np.maximum(stay, 0.0)
    negative = np.flatnonzero(stay < 0)
    for b in negative:
        lo, hi = int(offsets[b]), int(offsets[b + 1])
        probs_flat[lo:hi] /= probs_flat[lo:hi].sum()

    expected_loss = old_loss - step * tilt.direction_gain + step * step * interaction
    upper = old_loss - step * tilt.direction_gain / 2
    return BatchKernelResult(
        menu, grouped, offsets, probs_flat, old_loss,
        tilt.beta, tilt.direction_gain, interaction, step,
        expected_loss, upper, step * unit_rows,
        tilt.max_gain_sum, tilt.required_gain, "ok",
        group_scores,
    )


def sample_batch_next(
    result: BatchKernelResult,
    state_ids,
    rng: np.random.Generator,
    registry: StateRegistry,
) -> NDArray[np.int64]:
    """按批量核抽下一代并落地，抽中的新状态此刻才注册。

    每组先按二项分布抽离开行数再对非保持路径抽条件多项分布，
    这是组内独立单行核抽样的精确分解，与直接多项抽样同分布，
    行按组内升序切段分配，先落保持行再落各路径行。
    """
    s = np.asarray(state_ids)
    out = s.astype(np.int64, copy=True)
    grouped = result.grouped
    menu = result.menu
    schema = registry.schema
    offsets = result.offsets
    probs = result.probabilities
    stay = probs[offsets[:-1]]
    movers = rng.binomial(grouped.counts, np.minimum(1.0, np.maximum(0.0, 1.0 - stay)))
    for g in np.flatnonzero(movers):
        lo, hi = int(offsets[g]), int(offsets[g + 1])
        seg = probs[lo + 1 : hi]
        total = seg.sum()
        if total <= 0:
            continue
        counts_g = rng.multinomial(int(movers[g]), seg / total)
        rows = grouped.row_lists[int(g)]
        base_id = int(grouped.unique_ids[int(g)])
        base = registry.state_tuple(base_id)
        pos = len(rows) - int(movers[g])  # 前段保持原状态
        mlo = int(menu.offsets[g])
        for t, c in enumerate(counts_g):
            if c == 0:
                continue
            p = mlo + t
            edited = list(base)
            for slot in range(3):
                j = int(menu.fields[p, slot])
                if j < 0:
                    break
                edited[j] = schema.domains[j][int(menu.values[p, slot])]
            uid = registry.register_edit(base_id, tuple(edited))
            for i in rows[pos : pos + int(c)]:
                out[i] = uid
            pos += int(c)
    return out


@dataclass
class BatchRoundPlan:
    """批量提供器每轮的计划，批量路径行级回退路径或子集救援路径三选一。"""

    mode: str  # "batch" 或 "rows" 或 "rescue"
    workload: Workload
    grouped: GroupedTable | None = None
    menu: BatchMenu | None = None
    supports: list | None = None
    rescue_ids: NDArray[np.int64] | None = None  # 抽中行在小注册表的编号
    rescue_sub_idx: NDArray[np.int64] | None = None  # 抽中行的全表行下标
    rescue_registry: StateRegistry | None = None  # 临时小注册表，用后即弃


def make_batch_provider(
    registry: StateRegistry,
    target,
    weights,
    menu_rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
    pairing_budget: PairingBudget | None = None,
    pairing: bool = False,
    pairing_backoff: int = 4,
    defer_workload: bool = False,
    pair_rescue_rows: int = 0,
    rescue_after: int = 6,
    rescue_gpu: bool = False,
    work_rows: int = 0,
    work_random_frac: float = 0.25,
    select_rng: np.random.Generator | None = None,
    work_below_step: float = 0.0,
):
    """批量候选提供器，平时全矢量出菜单，冻结重试轮可回退行级配对。

    配对退避，冻结重试的前三次都上配对，之后每 pairing_backoff 次上一次，
    其余重试轮只刷新便宜的批量菜单，重试语义与总次数上限不变，
    backoff 取 1 即每次重试都配对，等价旧行为。
    defer_workload 为真时批量轮的计划只携带目标与权重不物化特征矩阵，
    供 GPU 后端配合惰性注册表使用，行级回退轮仍取完整负载。

    子集救援，pair_rescue_rows 为正时惰性模式的冻结重试轮不再只刷菜单，
    抽该数量的行走行级配对，临时小注册表隔离特征物化，
    伪目标修正保持全表残差口径，其余行保持不动，默认 0 关闭零改变，
    非惰性模式不受此参数影响仍走全表行级配对老路。
    救援门槛，rescue_after 控制连败多少次才第一次动用救援，
    之前的重试轮只刷便宜的批量菜单，零星冻结靠换菜单自愈不烧重炮，
    达门槛后每 pairing_backoff 次连败再救一次，救援成功连败清零重计。

    工作批，work_rows 为正时每轮按上一轮核回传的组错位分挑组上场，
    定向额度按分数降序装组，其余名额从剩下的组随机装到 work_rows 行，
    未选组本轮菜单为空段等价保持概率 1，行原地不动不进核，
    没上过场的新状态分数视为无穷大必入选，冻结重试轮回退全量搜索，
    首轮尚无分数也走全量，work_rows 取 0 与旧行为逐位一致。
    work_below_step 为正时早期全量演化，观测到某轮步长低于该值才开闸裁组，
    单向切换不来回抖，早期残差处处大全员上场步长本就顶格，裁组只会自缚手脚。
    """
    if pairing_backoff < 1:
        raise ValueError("配对退避周期必须为正")
    if pair_rescue_rows < 0:
        raise ValueError("救援行数不能为负")
    if rescue_after < 1:
        raise ValueError("救援门槛必须为正")
    if not (0.0 <= work_random_frac <= 1.0):
        raise ValueError("随机名额比例必须落在 [0,1]")
    sel_rng = select_rng if select_rng is not None else np.random.default_rng(0)
    codebook = CodeBook(registry)
    structure_box: list[QueryStructure] = []
    cache_box: list[CntCache] = []
    score_box: list[NDArray[np.float64]] = [np.empty(0, dtype=np.float64)]
    gate_box = [work_below_step <= 0.0]  # 阈值非正则立即启用

    def note_step(step: float) -> None:
        """观测批量轮步长，低于阈值后永久开闸启用工作批。"""
        if not gate_box[0] and step < work_below_step:
            gate_box[0] = True

    def feed_scores(unique_ids, scores) -> None:
        """按状态号记组错位分，数组按需扩容，新号默认无穷大。"""
        arr = score_box[0]
        uids = np.asarray(unique_ids, dtype=np.int64)
        need = int(uids.max()) + 1 if uids.size else 0
        if need > len(arr):
            grown = np.full(max(need, 2 * len(arr)), np.inf, dtype=np.float64)
            grown[: len(arr)] = arr
            score_box[0] = arr = grown
        arr[uids] = scores

    def _select_groups(grouped: GroupedTable) -> NDArray[np.int64] | None:
        """挑本轮上场的组，返回升序组索引，None 表示全量。"""
        arr = score_box[0]
        if work_rows <= 0 or arr.size == 0 or not gate_box[0]:
            return None
        counts = grouped.counts
        if int(counts.sum()) <= work_rows:
            return None
        uids = grouped.unique_ids
        scores = np.full(len(uids), np.inf, dtype=np.float64)
        known = uids < len(arr)
        scores[known] = arr[uids[known]]
        # 分数降序，平分按状态号升序，保证同种子可复现
        order = np.lexsort((uids, -scores))
        directed = int(round(work_rows * (1.0 - work_random_frac)))
        chosen = order[:0]
        taken = 0
        if directed > 0:
            cum = np.cumsum(counts[order])
            j = int(np.searchsorted(cum, directed, side="left"))
            chosen = order[: j + 1]
            taken = int(cum[j])
        rest = order[len(chosen):]
        remaining = work_rows - taken
        if remaining > 0 and len(rest) > 0:
            perm = rest[sel_rng.permutation(len(rest))]
            cum2 = np.cumsum(counts[perm])
            j2 = min(int(np.searchsorted(cum2, remaining, side="left")), len(perm) - 1)
            chosen = np.concatenate([chosen, perm[: j2 + 1]])
        return np.sort(chosen)

    def provider(state_ids, round_index: int, frozen_streak: int = 0) -> BatchRoundPlan:
        # 惰性负载模式的全表行级配对被禁，物化全部历史状态特征会打爆内存，
        # pair_rescue_rows 为正时改走子集救援，小注册表隔离物化其余行保持
        pairing_turn = frozen_streak <= 3 or frozen_streak % pairing_backoff == 0
        if pairing and not defer_workload and frozen_streak > 0 and pairing_turn:
            table = tuples_from_ids(registry, state_ids)
            singles = generate_edit_supports(
                table, registry.schema, registry, menu_rng, budget, joint_field_sets
            )
            menu = build_paired_supports(
                state_ids, singles, registry, target, weights, menu_rng, pairing_budget
            )
            return BatchRoundPlan("rows", menu.workload, supports=menu.supports)
        rescue_turn = (
            frozen_streak >= rescue_after
            and (frozen_streak - rescue_after) % pairing_backoff == 0
        )
        if (pairing and defer_workload and pair_rescue_rows > 0 and rescue_turn):
            small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
                registry, state_ids, target, weights, menu_rng,
                pair_rescue_rows, budget, joint_field_sets, pairing_budget,
                use_gpu=rescue_gpu,
            )
            return BatchRoundPlan(
                "rescue", wl, supports=sups, rescue_ids=sub_ids,
                rescue_sub_idx=sub_idx, rescue_registry=small,
            )
        if not structure_box:
            structure_box.append(build_query_structure(registry, codebook))
            cache_box.append(CntCache(structure_box[0]))
        codes = codebook.sync()
        grouped = group_state_ids(state_ids)
        codes_g = codes[grouped.unique_ids]
        sel = None if frozen_streak > 0 else _select_groups(grouped)
        if sel is None:
            menu = generate_batch_menu(
                codes_g, grouped.counts, codebook.domain_sizes,
                menu_rng, budget, joint_field_sets,
            )
        else:
            # 只给选中组生成路径，再散回全组编号，未选组为空段
            sub = generate_batch_menu(
                codes_g[sel], grouped.counts[sel], codebook.domain_sizes,
                menu_rng, budget, joint_field_sets,
            )
            per_group = np.zeros(grouped.num_groups, dtype=np.int64)
            per_group[sel] = np.diff(sub.offsets)
            offsets_full = np.zeros(grouped.num_groups + 1, dtype=np.int64)
            np.cumsum(per_group, out=offsets_full[1:])
            menu = BatchMenu(
                offsets_full, sel[sub.group], sub.fields, sub.values, sub.mass
            )
        if defer_workload:
            workload = Workload(
                np.empty((0, len(np.asarray(target)))),
                np.asarray(target, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
            )
        else:
            workload = registry.build_workload(target, weights)
        return BatchRoundPlan("batch", workload, grouped, menu)

    provider.codebook = codebook
    provider.structure = structure_box
    provider.cnt_cache = cache_box
    provider.feed_scores = feed_scores
    provider.note_step = note_step
    provider.wants_scores = work_rows > 0
    return provider


def evolve_batch(
    state_ids,
    num_rounds: int,
    rng: np.random.Generator,
    plan_provider,
    registry: StateRegistry,
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    max_frozen_retries: int = 0,
    backend: str = "cpu",
    stop_threshold: float = 0.0,
    stop_lag: int = 100,
    progress_every: int = 0,
) -> EvolveResult:
    """批量路径多轮循环，冻结与重试语义与组路径完全一致。

    backend 取 gpu 时批量轮核构造走 cupy 后端，数学定义一致，
    抽样与行级回退轮仍在 CPU，随机数语义不变。
    stop_threshold 大于零启用平台早停，参考 GSD 的窗口相对改进判据，
    每 stop_lag 轮到窗口边界比一次，上窗最优损失到本窗最优损失的
    相对改进低于阈值即停，停止原因记 loss_plateau。窗口取最优值防抽样抖动。
    判停只读损失，不碰核菜单抽样与随机流，停前轨迹与不停的跑逐位一致，
    返回的表是触发轮的起点表。冻结重试停照旧并存，默认 0 关闭零改变。
    progress_every 为正时每该数轮打印一行进度并立即刷出，
    冻结与救援等非常规轮无条件打印，只写标准输出不碰任何计算，默认 0 静默。
    """
    if num_rounds < 1:
        raise ValueError("轮数必须为正")
    if max_frozen_retries < 0:
        raise ValueError("冻结重试次数不能为负")
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"未知后端 {backend}")
    if stop_threshold < 0.0:
        raise ValueError("早停阈值不能为负")
    if stop_lag < 1:
        raise ValueError("早停窗口轮数必须为正")
    current = np.asarray(state_ids).astype(np.int64, copy=True)
    records: list[RoundRecord] = []
    frozen_streak = 0
    window_best: float | None = None  # 本窗口内最优损失
    prev_window_best: float | None = None  # 上一窗口最优损失
    last_beta: float | None = None  # 上一批量轮的根作下一轮括根热启动
    gpu_ctx = None  # GPU 上下文惰性建，静态量只上传一次
    for k in range(num_rounds):
        plan = plan_provider(current, k, frozen_streak)
        if plan.mode in ("rows", "rescue"):
            kernel_ids = current if plan.mode == "rows" else plan.rescue_ids
            result = build_kernel(
                plan.workload, kernel_ids, plan.supports,
                stay_probability, alpha, damping, max_expected_rows,
            )
        elif plan.mode == "batch":
            if backend == "gpu":
                if gpu_ctx is None:
                    from .gpukernel import GpuBatchContext

                    gpu_ctx = GpuBatchContext(
                        plan_provider.structure[0],
                        plan.workload.target, plan.workload.weights,
                    )
                result = gpu_ctx.build(
                    plan.grouped, plan.menu, plan_provider.codebook.sync(),
                    stay_probability, alpha, damping, max_expected_rows,
                    beta_hint=last_beta,
                )
            else:
                cache = getattr(plan_provider, "cnt_cache", None)
                result = build_batch_kernel(
                    plan.workload, plan.grouped, plan.menu,
                    plan_provider.structure[0], plan_provider.codebook.sync(),
                    stay_probability, alpha, damping, max_expected_rows,
                    cnt_cache=cache[0] if cache else None,
                    beta_hint=last_beta,
                    want_scores=bool(getattr(plan_provider, "wants_scores", False)),
                )
            if result.beta > 0.0:
                last_beta = result.beta
            feed = getattr(plan_provider, "feed_scores", None)
            if feed is not None and result.group_scores is not None:
                feed(result.grouped.unique_ids, result.group_scores)
            note = getattr(plan_provider, "note_step", None)
            if note is not None and result.status == "ok":
                note(result.step)
        else:
            raise ValueError(f"未知计划模式 {plan.mode}")
        records.append(
            RoundRecord(
                k, result.old_loss, result.beta, result.direction_gain,
                result.interaction, result.step, result.expected_loss, result.status,
            )
        )
        if progress_every > 0 and (
            plan.mode != "batch" or result.status != "ok"
            or k % progress_every == 0
        ):
            print(
                f"轮 {k} 模式 {plan.mode} 损失 {result.old_loss:.1f} "
                f"步长 {result.step:.4f} 预测降 "
                f"{result.old_loss - result.expected_loss:.2f} "
                f"{result.status} 连败 {frozen_streak}",
                flush=True,
            )
        if stop_threshold > 0.0:
            loss = result.old_loss
            window_best = loss if window_best is None else min(window_best, loss)
            if (k + 1) % stop_lag == 0:
                if prev_window_best is not None and (
                    prev_window_best <= 0.0
                    or (prev_window_best - window_best) / prev_window_best
                    < stop_threshold
                ):
                    return EvolveResult(current, records, "loss_plateau")
                prev_window_best = window_best
                window_best = None
        if result.status == "no_positive_direction":
            frozen_streak += 1
            if frozen_streak > max_frozen_retries:
                return EvolveResult(current, records, "no_positive_direction")
            continue
        frozen_streak = 0
        if plan.mode == "rows":
            current = sample_next(result, current, rng)
        elif plan.mode == "rescue":
            # 小注册表编号抽样，动过的行翻译回全局编号落回全表对应位置，
            # 全局注册仍走惰性路径不物化特征，小注册表出作用域即弃
            new_small = sample_next(result, plan.rescue_ids, rng)
            moved = new_small != plan.rescue_ids
            if moved.any():
                tuples = [
                    plan.rescue_registry.state_tuple(int(i))
                    for i in new_small[moved]
                ]
                gids = registry.register_table(tuples)
                current = current.copy()
                current[plan.rescue_sub_idx[moved]] = gids
        else:
            current = sample_batch_next(result, current, rng, registry)
    return EvolveResult(current, records, "round_limit")
