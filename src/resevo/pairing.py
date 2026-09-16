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
from .gain import block_deltas, block_gains
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
    registry: StateRegistry,
    rng: np.random.Generator,
    budget: PairingBudget,
) -> list[tuple[int, int]]:
    """每对的联合动作，一半交换一半组合，新状态在此注册。

    交换互换两行某字段的值，该字段总计数不变，只改与其他字段的搭配，
    组合让两行各出一个代表动作同时动，按单行增益之和取最好的几个。
    """
    joint: list[tuple[int, int]] = []
    for j in rng.permutation(len(row_i)):
        if len(joint) >= budget.swap_actions:
            break
        j = int(j)
        if row_i[j] == row_k[j]:
            continue
        u = list(row_i)
        v = list(row_k)
        u[j], v[j] = row_k[j], row_i[j]
        joint.append((registry.register(tuple(u)), registry.register(tuple(v))))
    combos = [
        (float(gains_i[a] + gains_k[b]), a, b)
        for a in reps_i
        for b in reps_k
    ]
    combos.sort(key=lambda t: (-t[0], t[1], t[2]))
    for _, a, b in combos[: budget.combine_actions]:
        joint.append((ids_i[a], ids_k[b]))
    return joint


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
    deltas_rows: list[NDArray[np.float64]] = []
    gains_rows: list[NDArray[np.float64]] = []
    ids_rows: list[list[int]] = []
    for i, sp in enumerate(single_supports):
        outcomes = ((int(s[i]),),) + tuple(sp.outcomes)
        d = block_deltas(workload.features, outcomes[0], outcomes)
        g = block_gains(d, residual, workload.weights)
        deltas_rows.append(d)
        gains_rows.append(g)
        ids_rows.append([int(o[0]) for o in outcomes])

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

    # 先生成全部联合动作并注册新状态，再统一重建负载打分
    joint_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, k in sorted(pair_set):
        joint_of[(i, k)] = _joint_actions_for_pair(
            registry.state_tuple(int(s[i])),
            registry.state_tuple(int(s[k])),
            reps_rows[i],
            reps_rows[k],
            gains_rows[i],
            gains_rows[k],
            ids_rows[i],
            ids_rows[k],
            registry,
            rng,
            budget,
        )

    workload = registry.build_workload(target, weights)
    feats = workload.features
    w = workload.weights
    residual = table_residual(workload, s)
    we = w * residual

    scored: list[tuple[float, int, int]] = []
    for i, k in sorted(pair_set):
        # 单行增益基于旧快照计算，旧状态贡献行前缀不变，数值依然有效
        m_i = float(gains_rows[i].max())
        m_k = float(gains_rows[k].max())
        best_joint = 0.0
        for u, v in joint_of[(i, k)]:
            du = feats[u] - feats[int(s[i])]
            dv = feats[v] - feats[int(s[k])]
            gu = float(du @ we - np.sum(du * du * w) / 2)
            gv = float(dv @ we - np.sum(dv * dv * w) / 2)
            cross = float(np.sum(du * w * dv))
            best_joint = max(best_joint, gu + gv - cross)
        gamma = max(m_i, m_k, best_joint) - m_i - m_k
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
