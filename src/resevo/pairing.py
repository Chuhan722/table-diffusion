"""配对板块，自动发现需要联合更新的行对并组成双行块。

依据补充设计文档第二部分，配对依据是修改影响互相补偿，不是记录相似。
配对评分 Gamma = M_ik - M_i - M_k，M 为各自支持内含保持的最大增益，
它是分块阶段的代理目标，不是整代真实收益，只用来决定谁和谁绑成块。
配对属于文档明确允许的结构搜索，发生在改表之前的菜单准备阶段，
方法真正取消的是生成下一代之后按损失接受拒绝回退，配对不触碰这条红线，
每行编辑菜单的生成依旧不接收残差，配对阶段按文档用增益给行对打分。

预算全部有限，代表动作每行 4，搭档每行 8 加随机 1，
联合动作每对 16 其中交换 8 组合 8，全表至多 32 对，块至多 2 行。
联合支持完整保留两边全部单边动作，文档要求不能删掉有益单行方向。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import BlockSupport, validate_partition
from .dataset import StateRegistry
from .editspace import EditBudget, generate_edit_supports, tuples_from_ids
from .state import Workload, table_residual


@dataclass(frozen=True)
class PairingBudget:
    """配对预算，8 搭档与 16 联合动作是文档给的数，其余为最简补充决策。"""

    representatives: int = 4  # 每行代表动作数，方向各异
    partners_per_row: int = 8  # 每行经登记本检索的搭档上限
    random_partners: int = 1  # 每行额外随机搭档数
    swap_actions: int = 8  # 每对交换动作上限
    combine_actions: int = 8  # 每对组合动作上限
    max_pairs: int = 32  # 全表配对总预算
    gamma_tol: float = 1e-9  # Gamma 正分门槛，防浮点误差凑数


def _ragged_take(ptr: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按位置取 CSR 段的条目下标与归属，变长段拼接全矢量化。"""
    starts = ptr[pos]
    lens = ptr[pos + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    owner = np.repeat(np.arange(len(pos), dtype=np.int64), lens)
    head = np.concatenate(([0], np.cumsum(lens)[:-1]))
    idx = np.repeat(starts - head, lens) + np.arange(total, dtype=np.int64)
    return idx, owner


def _sparse_flat_deltas(
    registry: StateRegistry,
    flat_rows: list[tuple[str, ...]],
    flat_base_idx: np.ndarray,
    base_rows: list[tuple[str, ...]],
    cnt_base: np.ndarray,
    score_base: np.ndarray,
    we: NDArray[np.float64],
    w: NDArray[np.float64],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单行候选的稀疏差分与增益，免物化免全宽增量矩阵。

    候选都是基行的少字段编辑，按变动字段组合分组矢量化，
    组内基行命中计数加取值向量差得新特征，与基行特征之差即非零条目，
    增益按条目汇总，与全宽稠密差分数学同式，
    返回增益向量与条目 CSR 三件套，条目即动作的影响清单。
    """
    m = len(flat_rows)
    ncond, hs_cols, hs_thr = registry.edit_feature_pack()
    col_to_hs = {int(c): t for t, c in enumerate(hs_cols)}
    fq = registry.field_queries
    rows_arr = np.array(flat_rows, dtype=str)
    base_arr = np.array(base_rows, dtype=str)[flat_base_idx]
    diff = rows_arr != base_arr
    # 按变动字段组合分组，布尔行分组不受字段数超 63 的位宽限制
    uniq_masks, gkey = np.unique(diff, axis=0, return_inverse=True)
    ent_act: list[np.ndarray] = []
    ent_col: list[np.ndarray] = []
    ent_val: list[np.ndarray] = []
    for gi, gmask in enumerate(uniq_masks):
        if not gmask.any():
            continue  # 保持项，零变动零条目
        rows_g = np.flatnonzero(gkey == gi)
        fields = np.flatnonzero(gmask).tolist()
        cols = (
            fq[fields[0]]
            if len(fields) == 1
            else np.unique(np.concatenate([fq[j] for j in fields]))
        )
        if len(cols) == 0:
            continue
        norm_cols = cols[ncond[cols] >= 0]
        hs_sub = cols[ncond[cols] < 0]
        ts = np.array([col_to_hs[int(c)] for c in hs_sub], dtype=np.int64)
        bidx = flat_base_idx[rows_g]
        # 各变动字段的取值向量差在依赖列并集上累加
        dcnt = np.zeros((len(rows_g), len(norm_cols)), dtype=np.int16)
        dsc = (
            np.zeros((len(rows_g), len(ts)), dtype=np.int64)
            if len(ts)
            else None
        )
        for j in fields:
            old_v = base_arr[rows_g, j]
            new_v = rows_arr[rows_g, j]
            vals = sorted(set(old_v.tolist()) | set(new_v.tolist()))
            vpos = {v: p for p, v in enumerate(vals)}
            cnt_tab = np.stack(
                [registry.value_vectors(j, v)[0] for v in vals]
            )
            oc = np.fromiter((vpos[v] for v in old_v), dtype=np.int64)
            nc_ = np.fromiter((vpos[v] for v in new_v), dtype=np.int64)
            if len(norm_cols):
                dcnt += (
                    cnt_tab[nc_][:, norm_cols] - cnt_tab[oc][:, norm_cols]
                )
            if dsc is not None:
                score_tab = np.stack(
                    [registry.value_vectors(j, v)[1] for v in vals]
                )
                dsc += score_tab[nc_][:, ts] - score_tab[oc][:, ts]
        if len(norm_cols):
            nc = ncond[norm_cols][None, :]
            cb = cnt_base[bidx][:, norm_cols]
            d = (cb + dcnt == nc).astype(np.int8) - (cb == nc).astype(np.int8)
            r_nz, c_nz = np.nonzero(d)
            ent_act.append(rows_g[r_nz])
            ent_col.append(norm_cols[c_nz])
            ent_val.append(d[r_nz, c_nz].astype(np.float64))
        if dsc is not None:
            thr = hs_thr[ts][None, :]
            sb = score_base[bidx][:, ts]
            d = (sb + dsc >= thr).astype(np.int8) - (sb >= thr).astype(np.int8)
            r_nz, c_nz = np.nonzero(d)
            ent_act.append(rows_g[r_nz])
            ent_col.append(hs_sub[c_nz])
            ent_val.append(d[r_nz, c_nz].astype(np.float64))
    if ent_act:
        acts = np.concatenate(ent_act)
        cols_all = np.concatenate(ent_col)
        vals_all = np.concatenate(ent_val)
    else:
        acts = np.empty(0, dtype=np.int64)
        cols_all = np.empty(0, dtype=np.int64)
        vals_all = np.empty(0, dtype=np.float64)
    g_flat = np.bincount(
        acts,
        weights=we[cols_all] * vals_all - (w[cols_all] * vals_all * vals_all) / 2,
        minlength=m,
    )
    order = np.argsort(acts, kind="stable")
    cols_s = cols_all[order]
    vals_s = vals_all[order]
    ptr = np.searchsorted(acts[order], np.arange(m + 1, dtype=np.int64))
    return g_flat, ptr, cols_s, vals_s


def _score_combos_sparse(
    ptr: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    num_q: int,
    we: NDArray[np.float64],
    w: NDArray[np.float64],
    a_pos: np.ndarray,
    b_pos: np.ndarray,
) -> NDArray[np.float64]:
    """组合动作总分，两侧稀疏差分合并同类项后按条目汇总。

    组合动作两侧都是已打分的单行候选，稀疏条目直接复用，
    同一查询列的两侧增量先求和再进二次项，丢交叉项的错误由合并规避，
    与稠密路径 gu 加 gv 减交叉项数学同式。
    """
    idx_a, own_a = _ragged_take(ptr, a_pos)
    idx_b, own_b = _ragged_take(ptr, b_pos)
    owner = np.concatenate((own_a, own_b))
    col_cat = np.concatenate((cols[idx_a], cols[idx_b]))
    val_cat = np.concatenate((vals[idx_a], vals[idx_b]))
    if len(owner) == 0:
        return np.zeros(len(a_pos), dtype=np.float64)
    key = owner * np.int64(num_q) + col_cat
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    val_s = val_cat[order]
    seg = np.concatenate(([True], key_s[1:] != key_s[:-1]))
    seg_starts = np.flatnonzero(seg)
    d_sum = np.add.reduceat(val_s, seg_starts)
    key_u = key_s[seg_starts]
    own_u = key_u // np.int64(num_q)
    col_u = key_u % np.int64(num_q)
    g_ent = we[col_u] * d_sum - (w[col_u] * d_sum * d_sum) / 2
    return np.bincount(own_u, weights=g_ent, minlength=len(a_pos))


def _score_swaps_sparse(
    registry: StateRegistry,
    cnt_base: np.ndarray,
    score_base: np.ndarray,
    we: NDArray[np.float64],
    w: NDArray[np.float64],
    num_pairs: int,
    acts: list[tuple[int, int, int, int, str, str]],
) -> NDArray[np.float64]:
    """交换动作免物化打分，命中计数增量只碰受影响查询列。

    交换只改一个字段，两侧增量都落在该字段的依赖查询列上，
    新行命中数为基行命中数加取值向量之差，特征由命中数与条件总数比较得出，
    与先注册物化再全列差分的稠密打分数学同式，
    动作按字段分组矢量化，返回每对交换段最大总分，没有交换动作的对为零。
    acts 每项为 (对序号, 字段, 基行 i 下标, 基行 k 下标, i 值, k 值)。
    """
    ncond, hs_cols, hs_thr = registry.edit_feature_pack()
    col_to_hs = {int(c): t for t, c in enumerate(hs_cols)}
    fq = registry.field_queries
    best = np.zeros(num_pairs, dtype=np.float64)
    by_field: dict[int, list[tuple[int, int, int, str, str]]] = {}
    for pidx, j, bi, bk, vi, vk in acts:
        by_field.setdefault(j, []).append((pidx, bi, bk, vi, vk))
    for j, group in by_field.items():
        cols = fq[j]
        if len(cols) == 0:
            continue
        norm_cols = cols[ncond[cols] >= 0]
        hs_sub = cols[ncond[cols] < 0]
        ts = np.array([col_to_hs[int(c)] for c in hs_sub], dtype=np.int64)
        pair_idx = np.array([g[0] for g in group], dtype=np.int64)
        bi_idx = np.array([g[1] for g in group], dtype=np.int64)
        bk_idx = np.array([g[2] for g in group], dtype=np.int64)
        # 取值向量差按去重取值对建表，组内动作 gather
        vals = sorted({g[3] for g in group} | {g[4] for g in group})
        vpos = {v: p for p, v in enumerate(vals)}
        cnt_tab = np.stack([registry.value_vectors(j, v)[0] for v in vals])
        vi_code = np.fromiter((vpos[g[3]] for g in group), dtype=np.int64)
        vk_code = np.fromiter((vpos[g[4]] for g in group), dtype=np.int64)
        total = np.zeros(len(group), dtype=np.float64)
        if len(norm_cols):
            dvec = (
                cnt_tab[vk_code][:, norm_cols] - cnt_tab[vi_code][:, norm_cols]
            )
            nc = ncond[norm_cols][None, :]
            cb_i = cnt_base[bi_idx][:, norm_cols]
            cb_k = cnt_base[bk_idx][:, norm_cols]
            du = (cb_i + dvec == nc).astype(np.float64) - (
                cb_i == nc
            ).astype(np.float64)
            dv = (cb_k - dvec == nc).astype(np.float64) - (
                cb_k == nc
            ).astype(np.float64)
            d_sum = du + dv
            total += d_sum @ we[norm_cols]
            total -= ((d_sum * d_sum) @ w[norm_cols]) / 2
        if len(hs_sub):
            score_tab = np.stack(
                [registry.value_vectors(j, v)[1] for v in vals]
            )
            dsc = score_tab[vk_code][:, ts] - score_tab[vi_code][:, ts]
            thr = hs_thr[ts][None, :]
            sb_i = score_base[bi_idx][:, ts]
            sb_k = score_base[bk_idx][:, ts]
            du = (sb_i + dsc >= thr).astype(np.float64) - (
                sb_i >= thr
            ).astype(np.float64)
            dv = (sb_k - dsc >= thr).astype(np.float64) - (
                sb_k >= thr
            ).astype(np.float64)
            d_sum = du + dv
            total += d_sum @ we[hs_sub]
            total -= ((d_sum * d_sum) @ w[hs_sub]) / 2
        np.maximum.at(best, pair_idx, total)
    return best


@dataclass
class PairedMenu:
    """配对产出，负载快照，完整分块菜单，以及配对诊断信息。"""

    workload: Workload
    supports: list[BlockSupport]
    pairs: list[tuple[int, int, float]]  # (行 i, 行 k, Gamma)


def _action_keys(cols: np.ndarray, vals: np.ndarray) -> list[tuple[int, int]]:
    """一个动作的影响清单键，碰到的每个查询配上加减方向，来自稀疏条目。"""
    return [(int(q), 1 if v > 0 else -1) for q, v in zip(cols, vals)]


def _select_representatives(
    keys_of,
    gains: NDArray[np.float64],
    budget: PairingBudget,
) -> list[int]:
    """按增益降序贪心占方向组，保证代表动作方向多样。

    文档警告不能只留增益最高的几个，互补对里的动作单独看常是负分，
    这里每个方向组只由一个动作代表，负分动作只要方向组空着照样入选，
    keys_of 按行内动作序号给出影响清单键，稀疏条目惰性切片。
    """
    order = np.argsort(-gains[1:], kind="stable") + 1  # 跳过索引 0 的保持
    taken: set[tuple[int, int]] = set()
    reps: list[int] = []
    for j in order:
        j = int(j)
        keys = keys_of(j)
        free = [key for key in keys if key not in taken]
        if not free:
            continue
        taken.add(free[0])
        reps.append(j)
        if len(reps) >= budget.representatives:
            break
    return reps


def _partners_for_row(
    i: int,
    reps: list[int],
    keys_of,
    ledger: dict[tuple[int, int], list[int]],
    num_rows: int,
    rng: np.random.Generator,
    budget: PairingBudget,
) -> list[int]:
    """翻登记本按你多我少凑搭档，再补少量随机搭档防登记规则堵死。"""
    partners: list[int] = []
    for j in reps:
        for q, sign in keys_of(j):
            for k in ledger.get((q, -sign), ()):
                if k != i and k not in partners:
                    partners.append(k)
                    if len(partners) >= budget.partners_per_row:
                        break
            if len(partners) >= budget.partners_per_row:
                break
        if len(partners) >= budget.partners_per_row:
            break
    for _ in range(budget.random_partners):
        k = int(rng.integers(num_rows))
        if k != i and k not in partners:
            partners.append(k)
    return partners


def _joint_actions_for_pair(
    row_i: tuple[str, ...],
    row_k: tuple[str, ...],
    reps_i: list[int],
    reps_k: list[int],
    gains_i: NDArray[np.float64],
    gains_k: NDArray[np.float64],
    rng: np.random.Generator,
    budget: PairingBudget,
) -> tuple[
    list[tuple[int, tuple[str, ...], tuple[str, ...]]], list[tuple[int, int]]
]:
    """每对的联合动作素材，一半交换一半组合。

    交换互换两行某字段的值，该字段总计数不变，只改与其他字段的搭配，
    交换产生的新行元组连同字段号先收集不注册，打分免物化，
    只有挑中配对的交换赢家才由调用方批量注册，
    组合让两行各出一个代表动作同时动，按单行增益之和取最好的几个，
    组合以行内动作序号对返回，状态号由调用方查表。
    """
    swaps: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    for j in rng.permutation(len(row_i)):
        if len(swaps) >= budget.swap_actions:
            break
        j = int(j)
        if row_i[j] == row_k[j]:
            continue
        u = list(row_i)
        v = list(row_k)
        u[j], v[j] = row_k[j], row_i[j]
        swaps.append((j, tuple(u), tuple(v)))
    combos = [
        (float(gains_i[a] + gains_k[b]), a, b)
        for a in reps_i
        for b in reps_k
    ]
    combos.sort(key=lambda t: (-t[0], t[1], t[2]))
    combo_ab = [(a, b) for _, a, b in combos[: budget.combine_actions]]
    return swaps, combo_ab


def build_paired_supports(
    state_ids,
    single_supports: list[BlockSupport],
    registry: StateRegistry,
    target,
    weights,
    rng: np.random.Generator,
    budget: PairingBudget | None = None,
) -> PairedMenu:
    """从每行单行菜单出发完成牵线打分成对，输出完整分块菜单。

    步骤，算每个动作的影响清单与增益，每行挑方向各异的代表动作，
    建查询加方向的登记本，按相反方向检索搭档，逐对拼联合动作，
    单行交换组合三段打分全走稀疏差分，免物化免全宽增量矩阵，
    Gamma 为正的对按分数贪心绑定，每行至多属于一个块，挑中赢家才注册，
    绑定对输出双行块并完整保留两边全部单边动作，其余行保留原单行菜单。
    """
    if budget is None:
        budget = PairingBudget()
    s = np.asarray(state_ids).astype(np.int64, copy=True)
    n = len(s)
    if len(single_supports) != n or any(
        sp.rows != (i,) for i, sp in enumerate(single_supports)
    ):
        raise ValueError("配对输入必须是按行序排列的单行菜单")

    workload = registry.build_workload(target, weights)
    residual = table_residual(workload, s)
    w = workload.weights
    we = w * residual

    # 每行含保持的增量与增益，索引 0 恒为保持，增益恒 0
    # 全部行平铺一次稀疏差分，免物化免全宽矩阵，与稠密打分数学同式
    ids_rows: list[list[int]] = []
    counts: list[int] = []
    flat_ids: list[int] = []
    flat_base_idx: list[int] = []
    for i, sp in enumerate(single_supports):
        row_ids = [int(s[i])] + [int(o[0]) for o in sp.outcomes]
        ids_rows.append(row_ids)
        counts.append(len(row_ids))
        flat_ids.extend(row_ids)
        flat_base_idx.extend([i] * len(row_ids))
    flat_base_idx_arr = np.array(flat_base_idx, dtype=np.int64)
    flat_rows = [registry.state_tuple(x) for x in flat_ids]
    base_rows = [registry.state_tuple(int(x)) for x in s]
    cnt_base, score_base = registry.counts_scores_cached(base_rows)
    g_flat, ent_ptr, ent_cols, ent_vals = _sparse_flat_deltas(
        registry, flat_rows, flat_base_idx_arr, base_rows,
        cnt_base, score_base, we, w,
    )
    row_offset = np.concatenate(
        ([0], np.cumsum(np.array(counts, dtype=np.int64)))
    )
    gains_rows: list[NDArray[np.float64]] = []
    pos = 0
    for c in counts:
        gains_rows.append(g_flat[pos : pos + c])
        pos += c

    # 影响清单键按平铺位置惰性生成带记忆，挑代表只碰每行前几个动作
    key_cache: dict[int, list[tuple[int, int]]] = {}

    def keys_at(p: int) -> list[tuple[int, int]]:
        got = key_cache.get(p)
        if got is None:
            got = _action_keys(
                ent_cols[ent_ptr[p] : ent_ptr[p + 1]],
                ent_vals[ent_ptr[p] : ent_ptr[p + 1]],
            )
            key_cache[p] = got
        return got

    reps_rows = [
        _select_representatives(
            lambda j, i=i: keys_at(int(row_offset[i]) + j),
            gains_rows[i],
            budget,
        )
        for i in range(n)
    ]

    # 登记本，键为查询编号加方向，值为登记过该方向代表动作的行
    ledger: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        for j in reps_rows[i]:
            for key in keys_at(int(row_offset[i]) + j):
                ledger.setdefault(key, []).append(i)

    pair_set: set[tuple[int, int]] = set()
    for i in range(n):
        for k in _partners_for_row(
            i,
            reps_rows[i],
            lambda j, i=i: keys_at(int(row_offset[i]) + j),
            ledger,
            n,
            rng,
            budget,
        ):
            pair_set.add((min(i, k), max(i, k)))

    # 先收集全部联合动作素材，交换动作只记字段与新行元组，先不注册
    pairs_sorted = sorted(pair_set)
    swaps_of: dict[tuple[int, int], list] = {}
    combos_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, k in pairs_sorted:
        swaps, combo_ab = _joint_actions_for_pair(
            registry.state_tuple(int(s[i])),
            registry.state_tuple(int(s[k])),
            reps_rows[i],
            reps_rows[k],
            gains_rows[i],
            gains_rows[k],
            rng,
            budget,
        )
        swaps_of[(i, k)] = swaps
        combos_of[(i, k)] = combo_ab

    # 交换段免物化稀疏打分，落选交换状态自始至终不注册不物化
    best = np.zeros(len(pairs_sorted), dtype=np.float64)
    acts: list[tuple[int, int, int, int, str, str]] = []
    for idx, (i, k) in enumerate(pairs_sorted):
        row_i = base_rows[i]
        row_k = base_rows[k]
        for j, _, _ in swaps_of[(i, k)]:
            acts.append((idx, j, i, k, row_i[j], row_k[j]))
    if acts:
        best = _score_swaps_sparse(
            registry, cnt_base, score_base, we, w, len(pairs_sorted), acts
        )

    # 组合段两侧都是已打分的单行候选，稀疏条目复用合并同类项汇总
    a_pos: list[int] = []
    b_pos: list[int] = []
    offsets = [0]
    for i, k in pairs_sorted:
        for a, b in combos_of[(i, k)]:
            a_pos.append(int(row_offset[i]) + a)
            b_pos.append(int(row_offset[k]) + b)
        offsets.append(len(a_pos))
    if a_pos:
        totals = _score_combos_sparse(
            ent_ptr, ent_cols, ent_vals, len(registry.specs), we, w,
            np.array(a_pos, dtype=np.int64), np.array(b_pos, dtype=np.int64),
        )
        starts = np.array(offsets[:-1], dtype=np.int64)
        valid = np.diff(np.array(offsets, dtype=np.int64)) > 0
        grouped_max = np.maximum.reduceat(
            totals, np.minimum(starts, len(totals) - 1)
        )
        best[valid] = np.maximum(best[valid], grouped_max[valid])
    best = np.maximum(best, 0.0)

    scored: list[tuple[float, int, int]] = []
    for idx, (i, k) in enumerate(pairs_sorted):
        m_i = float(gains_rows[i].max())
        m_k = float(gains_rows[k].max())
        gamma = max(m_i, m_k, float(best[idx])) - m_i - m_k
        if gamma > budget.gamma_tol:
            scored.append((float(gamma), i, k))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    used: set[int] = set()
    chosen: list[tuple[int, int, float]] = []
    for gamma, i, k in scored:
        if len(chosen) >= budget.max_pairs:
            break
        if i in used or k in used:
            continue
        used.add(i)
        used.add(k)
        chosen.append((i, k, gamma))

    # 只注册挑中配对的交换赢家，注册次序按对排序确定，与打分次序无关
    swap_rows_sel: list[tuple[str, ...]] = []
    for i, k in sorted((i, k) for i, k, _ in chosen):
        for _, u, v in swaps_of[(i, k)]:
            swap_rows_sel.append(u)
            swap_rows_sel.append(v)
    swap_ids = registry.register_many(swap_rows_sel)
    joint_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cursor = 0
    for i, k in sorted((i, k) for i, k, _ in chosen):
        n_swaps = len(swaps_of[(i, k)])
        joint = [
            (int(swap_ids[cursor + 2 * t]), int(swap_ids[cursor + 2 * t + 1]))
            for t in range(n_swaps)
        ]
        cursor += 2 * n_swaps
        joint_of[(i, k)] = joint + [
            (ids_rows[i][a], ids_rows[k][b]) for a, b in combos_of[(i, k)]
        ]
    workload = registry.build_workload(target, weights)

    lead_of: dict[int, tuple[int, int, float]] = {i: (i, k, g) for i, k, g in chosen}
    supports: list[BlockSupport] = []
    for i in range(n):
        if i in lead_of:
            i0, k0, _ = lead_of[i]
            si, sk = int(s[i0]), int(s[k0])
            outcomes: list[tuple[int, int]] = []
            # 完整保留两边全部单边动作，文档要求不能删掉有益单行方向
            outcomes += [(u, sk) for u in ids_rows[i0][1:]]
            outcomes += [(si, v) for v in ids_rows[k0][1:]]
            outcomes += joint_of[(i0, k0)]
            supports.append(
                BlockSupport((i0, k0), tuple(outcomes), (1.0,) * len(outcomes))
            )
        elif i in used:
            continue  # 该行是某对的搭档，块已由领头行生成
        else:
            supports.append(single_supports[i])
    validate_partition(supports, n)
    return PairedMenu(workload, supports, chosen)


def make_paired_provider(
    registry: StateRegistry,
    target,
    weights,
    menu_rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
    pairing_budget: PairingBudget | None = None,
):
    """带配对的候选提供器，平时纯单行菜单，冻结重试轮才启用配对。

    卡住的判据由引擎给出，frozen_streak 为连续冻结次数，
    大于 0 说明上一轮整张菜单没有一个正增益动作，此时配对上场。
    """

    def provider(state_ids, round_index: int, frozen_streak: int = 0):
        table = tuples_from_ids(registry, state_ids)
        singles = generate_edit_supports(
            table, registry.schema, registry, menu_rng, budget, joint_field_sets
        )
        if frozen_streak > 0:
            menu = build_paired_supports(
                state_ids, singles, registry, target, weights, menu_rng, pairing_budget
            )
            return menu.workload, menu.supports
        workload = registry.build_workload(target, weights)
        return workload, singles

    return provider


def build_rescue_menu(
    registry: StateRegistry,
    state_ids,
    target,
    weights,
    rng: np.random.Generator,
    rescue_rows: int,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
    pairing_budget: PairingBudget | None = None,
    use_gpu: bool = False,
):
    """冻结救援轮的子集配对菜单，临时小注册表隔离特征物化。

    惰性长跑里行级配对要物化全部历史状态的特征矩阵必爆内存，
    救援轮新开小注册表，只装全表当前行与抽中行的候选结果，
    算完整体丢弃，全局注册表的惰性欠账自始至终不清偿，
    单次救援内存只随救援行数走，与已跑轮数无关。
    抽 rescue_rows 行做行级配对其余行保持，伪目标 y' 取全表残差
    加子集贡献，子表残差恒等于全表卡点残差，动作增益与损失
    口径与全表一致，正方向结论是全表配对的保守下界。
    返回小注册表，抽中行小编号，抽中行的全表行下标，小负载，分块菜单。
    """
    if rescue_rows < 1:
        raise ValueError("救援行数必须为正")
    s = np.asarray(state_ids).astype(np.int64, copy=False)
    small = StateRegistry(registry.schema, registry.specs)
    small.gpu_featurize = use_gpu  # 物化查表 gather 搬卡，整数逐位同 CPU
    small.enable_counts_cache()  # 初装算过的命中计数给稀疏打分复用
    small_all = small.register_table(tuples_from_ids(registry, s))
    base = small.build_workload(target, weights)
    resid_full = table_residual(base, small_all)
    n_sub = min(int(rescue_rows), len(s))
    sub_idx = np.sort(rng.choice(len(s), size=n_sub, replace=False))
    sub_ids = small_all[sub_idx]
    y_prime = resid_full + base.features[sub_ids].sum(axis=0)
    singles = generate_edit_supports(
        tuples_from_ids(small, sub_ids), small.schema, small, rng,
        budget, joint_field_sets,
    )
    menu = build_paired_supports(
        sub_ids, singles, small, y_prime, weights, rng, pairing_budget,
    )
    return small, sub_ids, sub_idx, menu.workload, menu.supports
