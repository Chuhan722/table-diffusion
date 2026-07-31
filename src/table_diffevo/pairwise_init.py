"""
基于二阶测量的最大熵初始化。

这个模块只使用公开 schema、记录数和已经测量好的查询答案，不读取原始数据。
它先从完整的二属性等值查询中恢复二阶边缘，再用迭代比例拟合（IPF）在有限
类别状态空间上求一个与这些边缘一致的最大熵分布，最后从该分布抽样得到 S_0。

当前实现有意只覆盖“全部属性均为类别属性、状态空间可枚举”的场景。这样可以
先把 nltcs 上已经验证有效的改进做成边界清楚、容易审计的功能；数值属性和超大
状态空间不会静默近似，而是明确报错并提示回退到原有初始化方法。

隐私边界
--------
运行时输入中的 ``target`` 应当是已经发布/测量的统计量。IPF 与抽样都属于后处理，
不会额外读取原始表。不过仓库目前仍是无噪声原型；将来接入 DP 噪声后，不同二阶
边缘可能互相不一致，此时应先做统一的一致性投影。当前代码会返回未收敛诊断，
不会把近似结果误报为精确最大熵解。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table_diffevo.schema import Schema


def _category_index(value: Any, values: Sequence[Any]) -> Optional[int]:
    """把查询值映射到 schema 类别下标；兼容 JSON 中常见的字符串/数值差异。"""
    for i, candidate in enumerate(values):
        if value == candidate:
            return i

    # 查询加载器在真实表评价时也会按列类型做转换。这里没有真实表 dtype，因而仅在
    # 字符串表示能唯一匹配时回退，避免把两个不同类别误合并。
    matches = [i for i, candidate in enumerate(values) if str(value) == str(candidate)]
    if len(matches) == 1:
        return matches[0]
    return None


def _enumerate_states(cardinalities: Sequence[int]) -> np.ndarray:
    """枚举笛卡尔状态，返回每个属性的类别下标。"""
    n_states = int(np.prod(cardinalities, dtype=np.int64))
    flat = np.arange(n_states, dtype=np.int64)
    states = np.empty((n_states, len(cardinalities)), dtype=np.int32)

    stride = n_states
    for i, cardinality in enumerate(cardinalities):
        stride //= cardinality
        states[:, i] = (flat // stride) % cardinality
    return states


def _extract_pair_targets(
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    schema: Schema,
    n_records: int,
) -> Tuple[List[Tuple[int, int, np.ndarray]], Dict[str, int]]:
    """
    从二属性等值查询提取可用的完整二阶边缘。

    一个 K_i×K_j 边缘允许缺一个单元；该单元由公开总人数减去其余单元恢复。
    缺两个及以上单元的属性对会被跳过，因为仅靠现有测量无法唯一恢复。
    """
    names = schema.attribute_names()
    name_to_index = {name: i for i, name in enumerate(names)}
    domains = [list(attr.values) for attr in schema.attributes]

    sums: Dict[Tuple[int, int], np.ndarray] = {}
    observations: Dict[Tuple[int, int], np.ndarray] = {}

    for query, measured in zip(queries, target):
        conditions = query.get("conditions", [])
        if len(conditions) != 2:
            continue
        if any(cond.get("operator") != "==" for cond in conditions):
            continue

        first_name = conditions[0].get("attribute")
        second_name = conditions[1].get("attribute")
        if first_name not in name_to_index or second_name not in name_to_index:
            continue
        first = name_to_index[first_name]
        second = name_to_index[second_name]
        if first == second:
            continue

        if first < second:
            i, j = first, second
            cond_i, cond_j = conditions
        else:
            i, j = second, first
            cond_i, cond_j = conditions[1], conditions[0]

        value_i = _category_index(cond_i.get("value"), domains[i])
        value_j = _category_index(cond_j.get("value"), domains[j])
        if value_i is None or value_j is None:
            continue

        key = (i, j)
        n_cells = len(domains[i]) * len(domains[j])
        if key not in sums:
            sums[key] = np.zeros(n_cells, dtype=float)
            observations[key] = np.zeros(n_cells, dtype=int)
        cell = value_i * len(domains[j]) + value_j
        sums[key][cell] += float(measured)
        observations[key][cell] += 1

    pair_targets: List[Tuple[int, int, np.ndarray]] = []
    reconstructed_cells = 0
    skipped_pairs = 0

    for key in sorted(sums):
        seen = observations[key] > 0
        missing = int((~seen).sum())
        if missing > 1:
            skipped_pairs += 1
            continue

        counts = np.zeros_like(sums[key])
        counts[seen] = sums[key][seen] / observations[key][seen]
        if missing == 1:
            counts[~seen] = float(n_records) - counts[seen].sum()
            reconstructed_cells += 1

        # 为未来带噪测量保留最小的稳健处理：负计数截断、每个属性对单独归一化。
        # 跨属性对的一致性仍需由收敛诊断判断，不能在这里假装已经解决。
        counts = np.clip(counts, 0.0, None)
        total = float(counts.sum())
        if not np.isfinite(total) or total <= 0.0:
            skipped_pairs += 1
            continue
        pair_targets.append((key[0], key[1], counts / total))

    diagnostics = {
        "candidate_pairs": len(sums),
        "usable_pairs": len(pair_targets),
        "skipped_pairs": skipped_pairs,
        "reconstructed_cells": reconstructed_cells,
    }
    return pair_targets, diagnostics


def _pair_errors(
    probability: np.ndarray,
    pair_targets: List[Tuple[int, int, np.ndarray]],
    pair_codes: List[np.ndarray],
) -> Tuple[float, float]:
    """返回所有二阶单元的最大绝对误差与平均绝对误差（概率口径）。"""
    errors = []
    for (_, _, target), codes in zip(pair_targets, pair_codes):
        current = np.bincount(codes, weights=probability, minlength=len(target))
        errors.append(np.abs(current - target))
    joined = np.concatenate(errors)
    return float(joined.max()), float(joined.mean())


def init_from_pairwise_maxent(
    n_records: int,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    max_states: int = 1_000_000,
    max_sweeps: int = 200,
    tol: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    用二阶查询测量拟合最大熵分布并抽样生成初始表。

    Parameters
    ----------
    n_records : int
        合成记录数，也是恢复“唯一缺失二阶单元”时使用的公开总人数。
    schema : Schema
        当前仅支持所有属性均为 categorical。
    queries, target
        查询定义及其已测量计数。只采用含两个 ``==`` 条件的查询。
    rng : np.random.Generator or None
        最终从最大熵分布抽样所用随机数生成器。
    max_states : int, default 1_000_000
        可枚举状态数上限；超过时明确报错，避免意外耗尽内存。
    max_sweeps : int, default 200
        IPF 最大完整扫描轮数。
    tol : float, default 1e-8
        所有二阶单元概率的最大绝对误差收敛阈值。

    Returns
    -------
    (table, diagnostics)
        ``table`` 为抽样得到的 S_0；``diagnostics`` 记录状态数、使用的属性对、
        IPF 收敛误差和耗时等信息。
    """
    if n_records <= 0:
        raise ValueError(f"n_records 必须 > 0，得到 {n_records}")
    if isinstance(max_states, bool) or not isinstance(max_states, (int, np.integer)):
        raise ValueError(f"max_states 必须是正整数，得到 {max_states!r}")
    if max_states <= 0:
        raise ValueError(f"max_states 必须是正整数，得到 {max_states}")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, (int, np.integer)):
        raise ValueError(f"max_sweeps 必须是正整数，得到 {max_sweeps!r}")
    if max_sweeps <= 0:
        raise ValueError(f"max_sweeps 必须是正整数，得到 {max_sweeps}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol 必须是正的有限数，得到 {tol}")

    target = np.asarray(target, dtype=float)
    if len(target) != len(queries):
        raise ValueError(
            f"target 长度 ({len(target)}) 与查询数 ({len(queries)}) 不一致"
        )
    if not np.isfinite(target).all():
        raise ValueError("target 必须全部是有限数")

    non_categorical = [attr.name for attr in schema.attributes if not attr.is_categorical()]
    if non_categorical:
        raise ValueError(
            "pairwise_maxent 当前仅支持全部为 categorical 的 schema；"
            f"非类别属性: {non_categorical}。请改用 marginal 或 random 初始化。"
        )

    domains = [list(attr.values) for attr in schema.attributes]
    if any(len(values) == 0 for values in domains):
        raise ValueError("每个类别属性至少需要一个合法值")

    cardinalities = [len(values) for values in domains]
    n_states = 1
    for cardinality in cardinalities:
        n_states *= cardinality
        if n_states > max_states:
            raise ValueError(
                f"pairwise_maxent 状态空间 {n_states} 超过上限 {max_states}；"
                "请提高上限或改用 marginal/random 初始化。"
            )

    pair_targets, extraction_diag = _extract_pair_targets(
        queries, target, schema, n_records
    )
    if not pair_targets:
        raise ValueError(
            "没有可用的完整二阶等值边缘；每个属性对至多只能缺一个单元。"
        )

    if rng is None:
        rng = np.random.default_rng()

    fit_start = time.perf_counter()
    states = _enumerate_states(cardinalities)
    probability = np.full(n_states, 1.0 / n_states, dtype=float)
    pair_codes = [
        states[:, i] * cardinalities[j] + states[:, j]
        for i, j, _ in pair_targets
    ]

    best_probability = probability.copy()
    best_max_error, best_mean_error = _pair_errors(
        probability, pair_targets, pair_codes
    )
    sweeps_run = 0

    for sweep in range(1, max_sweeps + 1):
        for (_, _, desired), codes in zip(pair_targets, pair_codes):
            current = np.bincount(
                codes, weights=probability, minlength=len(desired)
            )
            ratio = np.ones_like(desired)
            nonzero = current > 0.0
            ratio[nonzero] = desired[nonzero] / current[nonzero]
            probability *= ratio[codes]
            total = float(probability.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("二阶测量互相冲突，IPF 概率质量退化为 0")
            probability /= total

        sweeps_run = sweep
        max_error, mean_error = _pair_errors(probability, pair_targets, pair_codes)
        if (max_error, mean_error) < (best_max_error, best_mean_error):
            best_probability = probability.copy()
            best_max_error = max_error
            best_mean_error = mean_error
        if max_error <= tol:
            break

    probability = best_probability
    sampled_state_ids = rng.choice(n_states, size=n_records, p=probability)
    sampled_codes = states[sampled_state_ids]
    sampled_probability = np.bincount(
        sampled_state_ids, minlength=n_states
    ).astype(float) / n_records
    sampled_max_error, sampled_mean_error = _pair_errors(
        sampled_probability, pair_targets, pair_codes
    )
    columns = {
        attr.name: np.asarray(domains[i])[sampled_codes[:, i]]
        for i, attr in enumerate(schema.attributes)
    }
    table = pd.DataFrame(columns, columns=schema.attribute_names())

    diagnostics: Dict[str, Any] = {
        "method": "pairwise_maxent",
        "n_states": int(n_states),
        **extraction_diag,
        "sweeps_run": int(sweeps_run),
        "converged": bool(best_max_error <= tol),
        "max_pair_error": float(best_max_error),
        "mean_pair_error": float(best_mean_error),
        "sampled_max_pair_error": float(sampled_max_error),
        "sampled_mean_pair_error": float(sampled_mean_error),
        "tol": float(tol),
        "max_sweeps": int(max_sweeps),
        "sampled_unique_states": int(np.unique(sampled_state_ids).size),
        "elapsed_sec": float(time.perf_counter() - fit_start),
    }
    return table, diagnostics
