"""有限编辑候选生成器，规模化第一版的候选来源。

按补充设计文档的预算表生成每行的合法后继菜单，
单属性块修改至多 8 字段乘 2 替代值，供体复制至多 8，
联合属性修改至多 4，支持探索至多 4，保持原样由规范化阶段恒加一次。
三条硬规则，残差与目标绝不参与生成，本函数签名不接收它们，
每条生成路径参考质量 1 份，同一后继的路径质量在规范化时自动合并，
不按增益删除任何候选，负增益零增益路径原样保留。
最简占位决策，属性块等于单字段，替代值取该字段观测值域，
数值字段用观测值当有限网格，合法性检查恒真因为 test 数据没有跨字段约束。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import BlockSupport
from .dataset import StateRegistry, TableSchema


@dataclass(frozen=True)
class EditBudget:
    """候选生成预算，数字是明确的计算预算不是理论最优值。"""

    max_edit_fields: int = 8  # 单块修改每轮抽的字段数上限
    max_values_per_field: int = 2  # 每字段替代值上限
    donor_copies: int = 8  # 供体复制路径数
    joint_edits: int = 4  # 联合修改路径数
    explore_edits: int = 4  # 支持探索路径数

    def max_nonstay_paths(self) -> int:
        return (
            self.max_edit_fields * self.max_values_per_field
            + self.donor_copies
            + self.joint_edits
            + self.explore_edits
        )


def _sample_alternative(
    domain: tuple[str, ...], current: str, rng: np.random.Generator
) -> str | None:
    """从值域里均匀抽一个不同于当前值的替代值，值域退化时返回 None。"""
    alternatives = [v for v in domain if v != current]
    if not alternatives:
        return None
    return alternatives[int(rng.integers(len(alternatives)))]


def _single_field_edits(
    current: tuple[str, ...],
    schema: TableSchema,
    budget: EditBudget,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """单属性块修改，均匀无放回抽字段，每字段抽至多两个不同替代值。"""
    num_fields = schema.num_fields
    chosen = rng.permutation(num_fields)[: budget.max_edit_fields]
    results = []
    for j in chosen:
        j = int(j)
        alternatives = [v for v in schema.domains[j] if v != current[j]]
        take = min(budget.max_values_per_field, len(alternatives))
        if take == 0:
            continue
        picks = rng.permutation(len(alternatives))[:take]
        for p in picks:
            edited = list(current)
            edited[j] = alternatives[int(p)]
            results.append(tuple(edited))
    return results


def _donor_copies(
    current: tuple[str, ...],
    table: list[tuple[str, ...]],
    schema: TableSchema,
    budget: EditBudget,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """供体复制，从当前合成表均匀抽供体行，复制其一到两个字段的值。"""
    results = []
    for _ in range(budget.donor_copies):
        donor = table[int(rng.integers(len(table)))]
        k = int(rng.integers(1, 3))  # 每次复制 1 到 2 个字段
        fields = rng.permutation(schema.num_fields)[:k]
        edited = list(current)
        for j in fields:
            edited[int(j)] = donor[int(j)]
        results.append(tuple(edited))
    return results


def _joint_edits(
    current: tuple[str, ...],
    schema: TableSchema,
    joint_field_sets: list[tuple[int, ...]],
    budget: EditBudget,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """联合属性修改，从查询涉及的多字段集合抽一个，其字段同时换新值。"""
    if not joint_field_sets:
        return []
    results = []
    for _ in range(budget.joint_edits):
        fields = joint_field_sets[int(rng.integers(len(joint_field_sets)))]
        edited = list(current)
        changed = False
        for j in fields:
            new_value = _sample_alternative(schema.domains[j], current[j], rng)
            if new_value is not None:
                edited[j] = new_value
                changed = True
        if changed:
            results.append(tuple(edited))
    return results


def _explore_edits(
    current: tuple[str, ...],
    schema: TableSchema,
    budget: EditBudget,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """支持探索，均匀抽一到三个字段独立换新值，可产生表里不存在的状态。

    探索发生在支持构造阶段，绝不在抽出下一代后追加变异。
    """
    results = []
    for _ in range(budget.explore_edits):
        k = int(rng.integers(1, 4))  # 每次改 1 到 3 个字段
        fields = rng.permutation(schema.num_fields)[:k]
        edited = list(current)
        changed = False
        for j in fields:
            j = int(j)
            new_value = _sample_alternative(schema.domains[j], current[j], rng)
            if new_value is not None:
                edited[j] = new_value
                changed = True
        if changed:
            results.append(tuple(edited))
    return results


def generate_edit_supports(
    table: list[tuple[str, ...]],
    schema: TableSchema,
    registry: StateRegistry,
    rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
) -> list[BlockSupport]:
    """为整张表生成本轮的编辑候选菜单，每行一个单行块。

    签名不接收残差损失与目标，从接口上杜绝残差参与候选生成。
    每条生成路径迁移率 1 份，同一后继由规范化阶段合并质量，
    落回源状态的路径由规范化阶段自然丢弃，负增益路径全部保留。
    新状态在这里注册进注册表，注册表按需扩充贡献矩阵。
    """
    if budget is None:
        budget = EditBudget()
    joint_sets = [fs for fs in (joint_field_sets or []) if len(fs) >= 2]
    row_paths: list[list[tuple[str, ...]]] = []
    for current in table:
        paths: list[tuple[str, ...]] = []
        paths += _single_field_edits(current, schema, budget, rng)
        paths += _donor_copies(current, table, schema, budget, rng)
        paths += _joint_edits(current, schema, joint_sets, budget, rng)
        paths += _explore_edits(current, schema, budget, rng)
        assert len(paths) <= budget.max_nonstay_paths()
        row_paths.append(paths)
    # 全表候选一次批量注册，编号次序与逐个注册完全一致，
    # 候选都是表行的字段编辑，特征走基行命中计数增量，逐位同全量重算
    flat_ids = registry.register_edited_many(
        [table[i] for i, paths in enumerate(row_paths) for _ in paths],
        [p for paths in row_paths for p in paths],
    )
    supports = []
    pos = 0
    for i, paths in enumerate(row_paths):
        k = len(paths)
        outcomes = tuple((int(sid),) for sid in flat_ids[pos : pos + k])
        pos += k
        mobility = (1.0,) * len(outcomes)
        supports.append(BlockSupport((i,), outcomes, mobility))
    return supports


def tuples_from_ids(registry: StateRegistry, state_ids) -> list[tuple[str, ...]]:
    """把状态编号向量还原成字段值元组表，供体复制与菜单刷新用。"""
    return [registry.state_tuple(int(s)) for s in state_ids]


def make_edit_provider(
    registry: StateRegistry,
    target,
    weights,
    menu_rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
):
    """打包成 evolve 的候选提供器，菜单随机数与抽样随机数分开。

    顺序要点，先生成菜单把新状态注册进注册表，再构造本轮负载，
    这样负载的贡献矩阵才包含本轮全部候选状态。
    """

    def provider(state_ids, round_index: int, frozen_streak: int = 0):
        table = tuples_from_ids(registry, state_ids)
        supports = generate_edit_supports(
            table, registry.schema, registry, menu_rng, budget, joint_field_sets
        )
        workload = registry.build_workload(target, weights)
        return workload, supports

    return provider
