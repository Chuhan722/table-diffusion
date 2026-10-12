"""测量后处理：噪声水平以内的格子视为未测出，剩余质量按低阶信息分配。

动机见工作笔记第五十七步第 14、15 条。低 ε 下高基数字段的一阶直方图
每格噪声 σ 比多数真计数还大，一半格子测成负数。原样当目标，演化会
精确贴到该噪声直方图的欧氏投影上（负格归零、碰巧测高的格子保留），
小格子全错；PGM 从均匀起步有限迭代，等价于向最大熵收缩，故此处占优。
本模块把这步收缩写成显式规则：

    |y| < z·σ 的格子视为"未测出"（与零分不开，无资格分高低）；
    其余格子保留 max(y, 0)；
    总行数减去保留格子之和，剩余质量在未测出格子间分配——
    一阶均分（无更低阶信息），二阶按清洗后一阶的乘积分（没证据就当独立）。

只改考卷里的目标数字，不碰真实数据，是已加噪输出的后处理，零隐私预算；
总行数与 PGM 的 --known-total 同口径视为公开。σ 小时几乎没有格子落入
z·σ 以内，规则自动退化为恒等，高 ε 不受影响。z 默认 2（统计惯例）。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .dataset import QuerySpec, TableSchema
from .initialization import _project_simplex


def pool_vector(
    y: NDArray[np.float64],
    sigma: NDArray[np.float64],
    total: float,
    z: float,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """核心规则。未测出格子为空时恒等返回；保留量已超总数时退回欧氏投影。"""
    y = np.asarray(y, dtype=np.float64)
    known = np.abs(y) >= z * np.asarray(sigma, dtype=np.float64)
    if known.all():
        return y.copy()
    out = np.zeros_like(y)
    out[known] = np.maximum(y[known], 0.0)
    rest = float(total) - out[known].sum()
    if rest < 0.0:
        return _project_simplex(y, float(total))
    w = np.ones(int((~known).sum())) if weights is None else np.asarray(weights, dtype=np.float64)[~known]
    if w.sum() <= 0.0:
        w = np.ones_like(w)
    out[~known] = rest * w / w.sum()
    return out


def _eq_cells(specs: list[QuerySpec], schema: TableSchema, order: int) -> dict:
    """按字段组归并恰 order 个等值条件的题，返回 {字段下标元组: {值元组: [题下标]}}。"""
    cells: dict[tuple[int, ...], dict[tuple[str, ...], list[int]]] = {}
    for i, spec in enumerate(specs):
        if len(spec.conditions) != order or any(c["operator"] != "==" for c in spec.conditions):
            continue
        js = tuple(schema.field_index(c["attribute"]) for c in spec.conditions)
        vs = tuple(str(c["value"]) for c in spec.conditions)
        if len(set(js)) != order:
            continue
        order_idx = np.argsort(js)
        key = tuple(js[k] for k in order_idx)
        val = tuple(vs[k] for k in order_idx)
        cells.setdefault(key, {}).setdefault(val, []).append(i)
    return cells


def _gather(specs, idx_map: dict, grid: list[tuple[str, ...]]):
    """按网格顺序取每格的均值结果与均值 σ，缺格或缺 σ 返回 None。"""
    y = np.empty(len(grid))
    s = np.empty(len(grid))
    for g, val in enumerate(grid):
        ids = idx_map.get(val)
        if not ids or any(specs[i].sigma is None for i in ids):
            return None, None
        y[g] = float(np.mean([specs[i].result for i in ids]))
        s[g] = float(np.mean([specs[i].sigma for i in ids]))
    return y, s


def covered_fields(specs: list[QuerySpec], schema: TableSchema) -> set[int]:
    """出现在任一 ≥2 条件题里的字段：它们的一阶由更高阶测量决定。"""
    out: set[int] = set()
    for spec in specs:
        if len(spec.conditions) >= 2:
            out.update(schema.field_index(c["attribute"]) for c in spec.conditions)
    return out


def pool_oneway(
    specs: list[QuerySpec], schema: TableSchema, total: int, z: float, scope: str = "all",
) -> tuple[list[QuerySpec], dict[int, NDArray[np.float64]], dict]:
    """一阶题清洗，整域覆盖且带 σ 的字段才处理，返回新题单、各字段清洗后计数、统计。

    scope="uncovered" 只清洗没被任何 ≥2 阶题覆盖的字段（第五十七步第 17 条：被覆盖字段的一阶
    由二阶测量决定，清洗多余且带均分偏差）；"all" 全部字段。
    """
    if scope not in ("all", "uncovered"):
        raise ValueError(f"scope 须为 all 或 uncovered，得到 {scope!r}")
    skip = covered_fields(specs, schema) if scope == "uncovered" else set()
    out = list(specs)
    cleaned: dict[int, NDArray[np.float64]] = {}
    stats = {"fields": 0, "cells": 0, "pooled": 0}
    for (j,), idx_map in _eq_cells(specs, schema, 1).items():
        if j in skip:
            continue
        grid = [(v,) for v in schema.domains[j]]
        y, s = _gather(specs, idx_map, grid)
        if y is None:
            continue
        c = pool_vector(y, s, total, z)
        cleaned[j] = c
        stats["fields"] += 1
        stats["cells"] += len(grid)
        stats["pooled"] += int((np.abs(y) < z * s).sum())
        for g, val in enumerate(grid):
            for i in idx_map[val]:
                out[i] = replace(out[i], result=float(c[g]))
    return out, cleaned, stats


def pool_pairs(
    specs: list[QuerySpec], schema: TableSchema, total: int, z: float,
    oneway: dict[int, NDArray[np.float64]],
) -> tuple[list[QuerySpec], dict]:
    """二阶题清洗，未测出格子按清洗后一阶乘积分；某字段无清洗一阶则该对均分。"""
    out = list(specs)
    stats = {"pairs": 0, "cells": 0, "pooled": 0}
    for (j, k), idx_map in _eq_cells(specs, schema, 2).items():
        grid = [(a, b) for a in schema.domains[j] for b in schema.domains[k]]
        y, s = _gather(specs, idx_map, grid)
        if y is None:
            continue
        weights = None
        if j in oneway and k in oneway:
            pj = oneway[j] / max(oneway[j].sum(), 1e-12)
            pk = oneway[k] / max(oneway[k].sum(), 1e-12)
            weights = np.outer(pj, pk).ravel()
        c = pool_vector(y, s, total, z, weights)
        stats["pairs"] += 1
        stats["cells"] += len(grid)
        stats["pooled"] += int((np.abs(y) < z * s).sum())
        for g, val in enumerate(grid):
            for i in idx_map[val]:
                out[i] = replace(out[i], result=float(c[g]))
    return out, stats
