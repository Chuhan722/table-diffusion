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


def _gpu_pack(features, weights, we):
    """上传特征与权重到 GPU，返回打分包，进场先归还池内缓存腾显存。"""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    return cp, cp.asarray(features), cp.asarray(weights), cp.asarray(we)


def _score_flat_deltas(features, flat_ids, flat_base, we, w, gpu_pack, block=25000):
    """平铺动作的增量矩阵与增益，GPU 路径分块算完回传，数学与 CPU 同式。"""
    if gpu_pack is None:
        d_flat = features[flat_ids] - features[flat_base]
        g_flat = d_flat @ we - np.sum(d_flat * d_flat * w, axis=1) / 2
        return d_flat, g_flat
    cp, feats_g, w_g, we_g = gpu_pack
    m = len(flat_ids)
    d_flat = np.empty((m, features.shape[1]), dtype=np.float64)
    g_flat = np.empty(m, dtype=np.float64)
    for lo in range(0, m, block):
        hi = min(lo + block, m)
        d_g = feats_g[cp.asarray(flat_ids[lo:hi])] - feats_g[
            cp.asarray(flat_base[lo:hi])
        ]
        g_flat[lo:hi] = cp.asnumpy(d_g @ we_g - ((d_g * d_g) @ w_g) / 2)
        d_flat[lo:hi] = cp.asnumpy(d_g)
        del d_g
    return d_flat, g_flat


def _score_joint_totals(features, us, vs, base_i, base_k, we, w, gpu_pack,
                        block=10000):
    """联合动作总分 gu 加 gv 减交叉项，GPU 路径分块算完回传，数学同式。"""
    if gpu_pack is None:
        du = features[us] - features[base_i]
        dv = features[vs] - features[base_k]
        gu = du @ we - ((du * du) @ w) / 2
        gv = dv @ we - ((dv * dv) @ w) / 2
        cross = (du * dv) @ w
        return gu + gv - cross
    cp, feats_g, w_g, we_g = gpu_pack
    m = len(us)
    totals = np.empty(m, dtype=np.float64)
    for lo in range(0, m, block):
        hi = min(lo + block, m)
        du_g = feats_g[cp.asarray(us[lo:hi])] - feats_g[cp.asarray(base_i[lo:hi])]
        dv_g = feats_g[cp.asarray(vs[lo:hi])] - feats_g[cp.asarray(base_k[lo:hi])]
        gu = du_g @ we_g - ((du_g * du_g) @ w_g) / 2
        gv = dv_g @ we_g - ((dv_g * dv_g) @ w_g) / 2
        cross = (du_g * dv_g) @ w_g
        totals[lo:hi] = cp.asnumpy(gu + gv - cross)
        del du_g, dv_g, gu, gv, cross
    return totals


def _gpu_release(gpu_pack):
    """打分包用完归还显存，让批量核随后照常吃满内存池。"""
    if gpu_pack is not None:
        gpu_pack[0].get_default_memory_pool().free_all_blocks()


@dataclass
class PairedMenu:
    """配对产出，负载快照，完整分块菜单，以及配对诊断信息。"""

    workload: Workload
    supports: list[BlockSupport]
    pairs: list[tuple[int, int, float]]  # (行 i, 行 k, Gamma)


def _action_keys(delta: NDArray[np.float64]) -> list[tuple[int, int]]:
    """一个动作的影响清单键，碰到的每个查询配上加减方向。"""
    return [(int(q), 1 if delta[q] > 0 else -1) for q in np.nonzero(delta)[0]]


def _select_representatives(
    deltas: NDArray[np.float64],
    gains: NDArray[np.float64],
    budget: PairingBudget,
) -> list[int]:
    """按增益降序贪心占方向组，保证代表动作方向多样。

    文档警告不能只留增益最高的几个，互补对里的动作单独看常是负分，
    这里每个方向组只由一个动作代表，负分动作只要方向组空着照样入选。
    """
    order = np.argsort(-gains[1:], kind="stable") + 1  # 跳过索引 0 的保持
    taken: set[tuple[int, int]] = set()
    reps: list[int] = []
    for j in order:
        j = int(j)
        keys = _action_keys(deltas[j])
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
    deltas: NDArray[np.float64],
    ledger: dict[tuple[int, int], list[int]],
    num_rows: int,
    rng: np.random.Generator,
    budget: PairingBudget,
) -> list[int]:
    """翻登记本按你多我少凑搭档，再补少量随机搭档防登记规则堵死。"""
    partners: list[int] = []
    for j in reps:
        for q, sign in _action_keys(deltas[j]):
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
    ids_i: list[int],
    ids_k: list[int],
    rng: np.random.Generator,
    budget: PairingBudget,
) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], list[tuple[int, int]]]:
    """每对的联合动作素材，一半交换一半组合。

    交换互换两行某字段的值，该字段总计数不变，只改与其他字段的搭配，
    交换产生的新行元组先收集不注册，由调用方全量批量注册，
    组合让两行各出一个代表动作同时动，按单行增益之和取最好的几个。
    """
    swaps: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for j in rng.permutation(len(row_i)):
        if len(swaps) >= budget.swap_actions:
            break
        j = int(j)
        if row_i[j] == row_k[j]:
            continue
        u = list(row_i)
        v = list(row_k)
        u[j], v[j] = row_k[j], row_i[j]
        swaps.append((tuple(u), tuple(v)))
    combos = [
        (float(gains_i[a] + gains_k[b]), a, b)
        for a in reps_i
        for b in reps_k
    ]
    combos.sort(key=lambda t: (-t[0], t[1], t[2]))
    combo_ids = [
        (ids_i[a], ids_k[b]) for _, a, b in combos[: budget.combine_actions]
    ]
    return swaps, combo_ids


def build_paired_supports(
    state_ids,
    single_supports: list[BlockSupport],
    registry: StateRegistry,
    target,
    weights,
    rng: np.random.Generator,
    budget: PairingBudget | None = None,
    use_gpu: bool = False,
) -> PairedMenu:
    """从每行单行菜单出发完成牵线打分成对，输出完整分块菜单。

    步骤，算每个动作的影响清单与增益，每行挑方向各异的代表动作，
    建查询加方向的登记本，按相反方向检索搭档，逐对拼联合动作，
    Gamma 为正的对按分数贪心绑定，每行至多属于一个块，
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

    # 每行含保持的增量与增益，索引 0 恒为保持，增益恒 0
    # 全部行平铺一次矢量化，单行块增量即两行特征之差，与逐行 block_deltas 逐位一致
    ids_rows: list[list[int]] = []
    counts: list[int] = []
    flat_ids: list[int] = []
    flat_base: list[int] = []
    for i, sp in enumerate(single_supports):
        row_ids = [int(s[i])] + [int(o[0]) for o in sp.outcomes]
        ids_rows.append(row_ids)
        counts.append(len(row_ids))
        flat_ids.extend(row_ids)
        flat_base.extend([int(s[i])] * len(row_ids))
    flat_ids_arr = np.array(flat_ids, dtype=np.int64)
    flat_base_arr = np.array(flat_base, dtype=np.int64)
    we_vec = workload.weights * residual
    pack1 = _gpu_pack(workload.features, workload.weights, we_vec) if use_gpu else None
    d_flat, g_flat = _score_flat_deltas(
        workload.features, flat_ids_arr, flat_base_arr,
        we_vec, workload.weights, pack1,
    )
    _gpu_release(pack1)
    pack1 = None
    deltas_rows: list[NDArray[np.float64]] = []
    gains_rows: list[NDArray[np.float64]] = []
    pos = 0
    for c in counts:
        deltas_rows.append(d_flat[pos : pos + c])
        gains_rows.append(g_flat[pos : pos + c])
        pos += c

    reps_rows = [
        _select_representatives(deltas_rows[i], gains_rows[i], budget) for i in range(n)
    ]

    # 登记本，键为查询编号加方向，值为登记过该方向代表动作的行
    ledger: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        for j in reps_rows[i]:
            for key in _action_keys(deltas_rows[i][j]):
                ledger.setdefault(key, []).append(i)

    pair_set: set[tuple[int, int]] = set()
    for i in range(n):
        for k in _partners_for_row(
            i, reps_rows[i], deltas_rows[i], ledger, n, rng, budget
        ):
            pair_set.add((min(i, k), max(i, k)))

    # 先收集全部联合动作素材，交换行统一批量注册，注册次序与逐个注册一致
    pairs_sorted = sorted(pair_set)
    swaps_of: dict[tuple[int, int], list] = {}
    combos_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    swap_rows: list[tuple[str, ...]] = []
    for i, k in pairs_sorted:
        swaps, combo_ids = _joint_actions_for_pair(
            registry.state_tuple(int(s[i])),
            registry.state_tuple(int(s[k])),
            reps_rows[i],
            reps_rows[k],
            gains_rows[i],
            gains_rows[k],
            ids_rows[i],
            ids_rows[k],
            rng,
            budget,
        )
        swaps_of[(i, k)] = swaps
        combos_of[(i, k)] = combo_ids
        for u, v in swaps:
            swap_rows.append(u)
            swap_rows.append(v)
    swap_ids = registry.register_many(swap_rows)
    joint_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cursor = 0
    for i, k in pairs_sorted:
        n_swaps = len(swaps_of[(i, k)])
        joint = [
            (int(swap_ids[cursor + 2 * t]), int(swap_ids[cursor + 2 * t + 1]))
            for t in range(n_swaps)
        ]
        cursor += 2 * n_swaps
        joint_of[(i, k)] = joint + combos_of[(i, k)]

    workload = registry.build_workload(target, weights)
    feats = workload.features
    w = workload.weights
    residual = table_residual(workload, s)
    we = w * residual

    # 全部联合动作平铺一次矢量化打分，再按对分组取最大
    us: list[int] = []
    vs: list[int] = []
    base_i: list[int] = []
    base_k: list[int] = []
    offsets = [0]
    for i, k in pairs_sorted:
        for u, v in joint_of[(i, k)]:
            us.append(u)
            vs.append(v)
            base_i.append(int(s[i]))
            base_k.append(int(s[k]))
        offsets.append(len(us))
    best = np.zeros(len(pairs_sorted), dtype=np.float64)
    if us:
        pack2 = _gpu_pack(feats, w, we) if use_gpu else None
        totals = _score_joint_totals(
            feats,
            np.array(us, dtype=np.int64), np.array(vs, dtype=np.int64),
            np.array(base_i, dtype=np.int64), np.array(base_k, dtype=np.int64),
            we, w, pack2,
        )
        _gpu_release(pack2)
        pack2 = None
        starts = np.array(offsets[:-1], dtype=np.int64)
        valid = np.diff(np.array(offsets, dtype=np.int64)) > 0
        grouped_max = np.maximum.reduceat(
            totals, np.minimum(starts, len(totals) - 1)
        )
        best[valid] = grouped_max[valid]
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
        use_gpu=use_gpu,
    )
    return small, sub_ids, sub_idx, menu.workload, menu.supports
