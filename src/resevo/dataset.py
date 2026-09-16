"""真实表格数据与计数查询负载的加载。

数据文件直接复制进本仓 data 目录，绝不 import 旧仓代码。
查询是谓词型计数查询，结果单位是行数，贡献 a(x) 恒为 0 或 1，
支持三种算子，等于比较字符串值，between 与大于等于把值转成数值比较。
状态用惰性注册表编码，编辑候选可产生表里不存在的组合状态，
注册表按需给新状态求值全部查询并追加贡献矩阵一行。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np

from .state import Workload, make_workload


@dataclass(frozen=True)
class TableSchema:
    """字段名与每字段的观测值域，数值字段用观测值当有限网格，最简占位。"""

    fields: tuple[str, ...]
    domains: tuple[tuple[str, ...], ...]

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    def field_index(self, name: str) -> int:
        return self.fields.index(name)


@dataclass(frozen=True)
class QuerySpec:
    """一个计数查询，conditions 为结构化条件列表，result 为真实表上的答案。"""

    query_id: str
    conditions: tuple[dict, ...]
    result: float


def load_table(csv_path: str) -> tuple[TableSchema, list[tuple[str, ...]]]:
    """读 CSV 成行元组列表并构造观测值域，utf-8-sig 处理 BOM。"""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [tuple(r) for r in reader]
    if not rows:
        raise ValueError("数据表不能为空")
    if any(len(r) != len(header) for r in rows):
        raise ValueError("存在列数不一致的行")
    domains = tuple(
        tuple(sorted({r[j] for r in rows})) for j in range(len(header))
    )
    return TableSchema(tuple(header), domains), rows


def load_queries(json_path: str) -> list[QuerySpec]:
    """读查询 JSON，只接受行数计数口径。"""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("result_unit") != "records":
        raise ValueError("只支持行数计数口径的查询负载")
    specs = []
    for q in payload["queries"]:
        specs.append(
            QuerySpec(str(q["id"]), tuple(q["conditions"]), float(q["result"]))
        )
    return specs


def evaluate_condition(cond: dict, value: str) -> bool:
    """单条件求值，等于比字符串，数值算子转 float 比较。"""
    op = cond["operator"]
    if op == "==":
        return value == str(cond["value"])
    if op == "between":
        v = float(value)
        return float(cond["lower"]) <= v <= float(cond["upper"])
    if op == ">=":
        return float(value) >= float(cond["value"])
    raise ValueError(f"不支持的算子 {op}")


def evaluate_query(
    spec: QuerySpec, row: tuple[str, ...], schema: TableSchema
) -> float:
    """谓词计数贡献，全部条件同时满足记 1 否则记 0。"""
    for cond in spec.conditions:
        j = schema.field_index(cond["attribute"])
        if not evaluate_condition(cond, row[j]):
            return 0.0
    return 1.0


def query_field_sets(specs: list[QuerySpec], schema: TableSchema) -> list[tuple[int, ...]]:
    """每个查询依赖的字段索引集合，联合修改来源用它找关联字段。"""
    return [
        tuple(sorted({schema.field_index(c["attribute"]) for c in spec.conditions}))
        for spec in specs
    ]


class StateRegistry:
    """状态元组到编号的惰性注册表，新状态按需求值全部查询。

    features 行只增不改，已发编号永不变，
    因此扩表后旧状态的贡献行前缀完全一致，跨轮损失可直接比较。
    贡献矩阵预分配按需翻倍，取负载时给前段只读视图不做全量拷贝，
    有限性校验在注册每行时做一次，之后不再重复扫全矩阵。
    """

    _INITIAL_CAPACITY = 1024

    def __init__(self, schema: TableSchema, specs: list[QuerySpec]):
        self.schema = schema
        self.specs = specs
        self._index: dict[tuple[str, ...], int] = {}
        self._tuples: list[tuple[str, ...]] = []
        self._count = 0
        self._features = np.empty((0, len(specs)), dtype=np.float64)
        # 字段到受影响查询列的依赖索引，编辑增量求值只碰这些列
        field_sets = query_field_sets(specs, schema)
        buckets: list[list[int]] = [[] for _ in range(schema.num_fields)]
        for q, fields in enumerate(field_sets):
            for j in fields:
                buckets[j].append(q)
        self._field_queries = tuple(
            np.array(qs, dtype=np.int64) for qs in buckets
        )

    @property
    def num_states(self) -> int:
        return self._count

    def state_tuple(self, state_id: int) -> tuple[str, ...]:
        return self._tuples[state_id]

    def _grow(self) -> None:
        """容量翻倍扩容，旧行原样拷入新矩阵，摊销代价常数。"""
        new_capacity = max(self._INITIAL_CAPACITY, 2 * self._features.shape[0])
        grown = np.empty((new_capacity, len(self.specs)), dtype=np.float64)
        grown[: self._count] = self._features[: self._count]
        self._features = grown

    def _insert(self, key: tuple[str, ...], feats: np.ndarray) -> int:
        """把校验完的状态行写入注册表并发号，register 与 register_edit 共用。"""
        if self._count == self._features.shape[0]:
            self._grow()
        state_id = self._count
        self._features[state_id] = feats
        self._count += 1
        self._index[key] = state_id
        self._tuples.append(key)
        return state_id

    def register(self, row: tuple[str, ...]) -> int:
        """注册一个状态元组，已存在直接返回编号，新状态求值全部查询。"""
        key = tuple(row)
        found = self._index.get(key)
        if found is not None:
            return found
        if len(key) != self.schema.num_fields:
            raise ValueError("状态元组字段数与 schema 不一致")
        feats = np.array(
            [evaluate_query(spec, key, self.schema) for spec in self.specs],
            dtype=np.float64,
        )
        if not np.isfinite(feats).all():
            raise ValueError("查询贡献必须全部有限")
        return self._insert(key, feats)

    def register_table(self, rows: list[tuple[str, ...]]) -> np.ndarray:
        """注册整张表并返回状态编号向量。"""
        return np.array([self.register(r) for r in rows], dtype=np.int64)

    def register_edit(self, base_id: int, row: tuple[str, ...]) -> int:
        """注册一个由已注册状态编辑而来的状态，按依赖索引增量求值。

        与 register 的结果逐位一致，只是省去结构上不受影响的查询，
        没改的字段不涉及的查询列直接从来源行复制，
        受影响列重新求值，两条路径的贡献都是同一 evaluate_query 的精确输出。
        """
        if base_id < 0 or base_id >= self._count:
            raise ValueError("编辑来源状态编号越界")
        key = tuple(row)
        found = self._index.get(key)
        if found is not None:
            return found
        if len(key) != self.schema.num_fields:
            raise ValueError("状态元组字段数与 schema 不一致")
        base = self._tuples[base_id]
        changed = [j for j in range(self.schema.num_fields) if key[j] != base[j]]
        if not changed:
            raise AssertionError("元组与来源相同却不在索引里，注册表内部不一致")
        if len(changed) == 1:
            affected = self._field_queries[changed[0]]
        else:
            affected = np.unique(
                np.concatenate([self._field_queries[j] for j in changed])
            )
        feats = self._features[base_id].copy()
        for q in affected:
            feats[q] = evaluate_query(self.specs[int(q)], key, self.schema)
        if not np.isfinite(feats[affected]).all():
            raise ValueError("查询贡献必须全部有限")
        return self._insert(key, feats)

    def build_workload(self, target, weights) -> Workload:
        """用已注册状态贡献矩阵的前段只读视图构造负载，不拷贝。

        视图底层矩阵后续只会追加新行，已发编号的行内容永不改写，
        因此旧负载持有的视图跨轮仍然有效且数值不变。
        """
        if self._count == 0:
            raise ValueError("注册表为空，先注册状态")
        view = self._features[: self._count]
        view.setflags(write=False)
        return make_workload(view, target, weights, features_prevalidated=True)


def target_from_specs(specs: list[QuerySpec]) -> np.ndarray:
    """目标向量直接取查询负载里的真实答案。"""
    return np.array([spec.result for spec in specs], dtype=np.float64)
