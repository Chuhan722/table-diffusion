"""评价指标，严格按 AIM 与 GSD 两篇文献的定义实现。

AIM 式，McKenna 等 VLDB 2022，workload 取 k 阶边缘表集合，
单个边缘误差为整张边缘表的 L1 距离除以真实表行数，再对全部边缘取平均，
Average Workload Error = (1/|W|) sum_r ||M_r(D)-M_r(S)||_1 / n。

GSD 式，Liu Vietri Wu ICML 2023，统计查询取比例口径 q(D)=(1/|D|) sum q(x)，
Max Error = max |q(S)-q(D)|，Average Error = mean |q(S)-q(D)|。

评价与生成严格分离，本模块只吃行元组列表，不接触引擎与提供器，
保留查询只允许在这里出现，绝不作为生成输入。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .dataset import QuerySpec, TableSchema, evaluate_query


def marginal_l1_error(
    real_rows: list[tuple[str, ...]],
    synth_rows: list[tuple[str, ...]],
    attrs: tuple[int, ...],
) -> float:
    """AIM 式单边缘误差，边缘计数表的 L1 距离除以真实表行数。"""
    if not real_rows or not synth_rows:
        raise ValueError("表不能为空")
    real_counts = Counter(tuple(r[j] for j in attrs) for r in real_rows)
    synth_counts = Counter(tuple(r[j] for j in attrs) for r in synth_rows)
    cells = set(real_counts) | set(synth_counts)
    l1 = sum(abs(real_counts.get(c, 0) - synth_counts.get(c, 0)) for c in cells)
    return l1 / len(real_rows)


@dataclass(frozen=True)
class AimReport:
    """AIM 式 workload 误差报告，per_marginal 与属性组合对齐。"""

    order: int
    average_error: float
    max_error: float
    per_marginal: tuple[float, ...]
    attr_sets: tuple[tuple[int, ...], ...]


def aim_workload_error(
    real_rows: list[tuple[str, ...]],
    synth_rows: list[tuple[str, ...]],
    schema: TableSchema,
    order: int,
) -> AimReport:
    """AIM 式平均 workload 误差，workload 取全部 order 阶属性组合的边缘表。"""
    if not (1 <= order <= schema.num_fields):
        raise ValueError("边缘阶数必须落在字段数范围内")
    attr_sets = tuple(combinations(range(schema.num_fields), order))
    errors = tuple(
        marginal_l1_error(real_rows, synth_rows, attrs) for attrs in attr_sets
    )
    return AimReport(
        order,
        float(np.mean(errors)),
        float(np.max(errors)),
        errors,
        attr_sets,
    )


@dataclass(frozen=True)
class GsdReport:
    """GSD 式统计查询误差报告，比例口径。"""

    query_count: int
    average_error: float
    max_error: float
    per_query: tuple[float, ...]


def gsd_query_errors(
    specs: list[QuerySpec],
    synth_rows: list[tuple[str, ...]],
    schema: TableSchema,
    real_total_rows: int,
) -> GsdReport:
    """GSD 式误差，真实答案取 spec.result 除以真实行数，合成答案按合成表比例。"""
    if not specs:
        raise ValueError("查询集合不能为空")
    if real_total_rows <= 0 or not synth_rows:
        raise ValueError("行数必须为正")
    errors = []
    n_synth = len(synth_rows)
    for spec in specs:
        synth_answer = sum(evaluate_query(spec, r, schema) for r in synth_rows) / n_synth
        real_answer = spec.result / real_total_rows
        errors.append(abs(synth_answer - real_answer))
    arr = np.asarray(errors)
    return GsdReport(len(specs), float(arr.mean()), float(arr.max()), tuple(errors))


def load_heldout_queries(json_path: str) -> list[QuerySpec]:
    """读保留查询，评价专用，绝不作为生成输入。"""
    import json

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("result_unit") != "records":
        raise ValueError("只支持行数计数口径")
    return [
        QuerySpec(str(q["id"]), tuple(q["conditions"]), float(q["result"]))
        for q in payload["queries"]
    ]


def canonical_condition_set(spec: QuerySpec) -> frozenset:
    """查询的语义规范形式，用于校验评价集与生成集零交集。

    半空间的规范形式是去掉零分项后的 (字段, 取值, 整数分) 集合加阈值，
    零分项在求值里与缺项同义，去掉后语义等价判定才不受写法影响。
    """
    out = []
    for c in spec.conditions:
        if c["operator"] == "between":
            out.append((c["attribute"], "between", float(c["lower"]), float(c["upper"])))
        elif c["operator"] == "halfspace":
            entries = frozenset(
                (f, str(v), int(s))
                for f, tab in c["scores"].items()
                for v, s in tab.items()
                if int(s) != 0
            )
            out.append(("halfspace", entries, int(c["threshold"])))
        else:
            out.append((c["attribute"], c["operator"], str(c.get("value"))))
    return frozenset(out)


def assert_disjoint_workloads(
    measured: list[QuerySpec], heldout: list[QuerySpec]
) -> None:
    """评价集混进生成集是评价体系的硬漏洞，加载时一次性拒绝。"""
    m = {canonical_condition_set(s) for s in measured}
    h = {canonical_condition_set(s) for s in heldout}
    overlap = m & h
    if overlap:
        raise ValueError(f"保留查询与生成查询语义重叠 {len(overlap)} 条")


@dataclass(frozen=True)
class CompositeReport:
    """综合误差分，四组子指标统一到 0 到 1 后等权平均，越小越好。

    统一口径，GSD 式比例误差本身落在 0 到 1，
    AIM 式边缘 L1 除以行数再除以 2 恰为总变差距离 TVD，也落在 0 到 1，
    四组等权，measured 拟合，heldout 泛化，二阶与三阶边缘结构各占四分之一。
    """

    score: float
    measured_error: float
    heldout_error: float
    tvd_2way: float
    tvd_3way: float


def composite_score(
    measured_report: GsdReport,
    heldout_report: GsdReport,
    aim2_report: AimReport,
    aim3_report: AimReport,
) -> CompositeReport:
    """四组子指标合成一个综合误差分。"""
    parts = (
        measured_report.average_error,
        heldout_report.average_error,
        aim2_report.average_error / 2,
        aim3_report.average_error / 2,
    )
    if any(not (0.0 <= p <= 1.0) for p in parts):
        raise ValueError("子指标越界，检查归一化口径")
    return CompositeReport(float(np.mean(parts)), *parts)
