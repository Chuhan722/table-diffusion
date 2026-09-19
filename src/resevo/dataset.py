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
    """谓词计数贡献，全部条件同时满足记 1 否则记 0。

    半空间条件是跨字段的整数加权分过线判断，
    每字段每取值一个整数分，行的总分不小于阈值才通过，全程整型无浮点误差。
    """
    for cond in spec.conditions:
        if cond["operator"] == "halfspace":
            total = 0
            for fname, table in cond["scores"].items():
                total += int(table.get(row[schema.field_index(fname)], 0))
            if total < int(cond["threshold"]):
                return 0.0
            continue
        j = schema.field_index(cond["attribute"])
        if not evaluate_condition(cond, row[j]):
            return 0.0
    return 1.0


def _condition_fields(cond: dict, schema: TableSchema) -> set[int]:
    """单条件涉及的字段索引集合，半空间条件覆盖它所有带分字段。"""
    if cond["operator"] == "halfspace":
        return {schema.field_index(f) for f in cond["scores"]}
    return {schema.field_index(cond["attribute"])}


def query_field_sets(specs: list[QuerySpec], schema: TableSchema) -> list[tuple[int, ...]]:
    """每个查询依赖的字段索引集合，联合修改来源用它找关联字段。"""
    out = []
    for spec in specs:
        fields: set[int] = set()
        for c in spec.conditions:
            fields |= _condition_fields(c, schema)
        out.append(tuple(sorted(fields)))
    return out


def compile_conditions(
    specs: list[QuerySpec], schema: TableSchema
) -> tuple[tuple[tuple, ...], ...]:
    """把查询条件预编译成字段索引加算子码，求值热路径不再做字段名查找。

    每条件编译为 (算子码, 载荷一, 载荷二, 下界, 上界)，
    算子码 0 等值比串，1 介于闭区间，2 大于等于，与 evaluate_condition 语义一致，
    算子码 3 半空间，载荷一是按字段排序的 (字段索引, 取值到整数分映射) 元组，
    载荷二是整数阈值，行的总分不小于阈值才通过。
    半空间查询必须单条件成查询，等值通道与分数通道不混在同一条查询里。
    """
    compiled = []
    for spec in specs:
        conds = []
        for c in spec.conditions:
            op = c["operator"]
            if op == "==":
                j = schema.field_index(c["attribute"])
                conds.append((0, j, str(c["value"]), 0.0, 0.0))
            elif op == "between":
                j = schema.field_index(c["attribute"])
                conds.append((1, j, "", float(c["lower"]), float(c["upper"])))
            elif op == ">=":
                j = schema.field_index(c["attribute"])
                conds.append((2, j, "", float(c["value"]), 0.0))
            elif op == "halfspace":
                if len(spec.conditions) != 1:
                    raise ValueError("半空间查询必须单条件成查询")
                tables = tuple(
                    sorted(
                        (
                            schema.field_index(f),
                            {str(v): int(s) for v, s in tab.items()},
                        )
                        for f, tab in c["scores"].items()
                    )
                )
                conds.append((3, tables, int(c["threshold"]), 0.0, 0.0))
            else:
                raise ValueError(f"不支持的算子 {op}")
        compiled.append(tuple(conds))
    return tuple(compiled)


def evaluate_compiled(conds: tuple[tuple, ...], row: tuple[str, ...]) -> float:
    """预编译条件的谓词计数贡献，与 evaluate_query 精确同值。"""
    for code, a, b, lo, hi in conds:
        if code == 0:
            if row[a] != b:
                return 0.0
        elif code == 1:
            v = float(row[a])
            if not (lo <= v <= hi):
                return 0.0
        elif code == 2:
            if float(row[a]) < lo:
                return 0.0
        else:
            total = 0
            for j, table in a:
                total += table.get(row[j], 0)
            if total < b:
                return 0.0
    return 1.0


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
        # 查询条件预编译，注册求值热路径不再做字段名查找
        self._compiled = compile_conditions(specs, schema)
        # 字段到受影响查询列的依赖索引，编辑增量求值只碰这些列
        field_sets = query_field_sets(specs, schema)
        buckets: list[list[int]] = [[] for _ in range(schema.num_fields)]
        for q, fields in enumerate(field_sets):
            for j in fields:
                buckets[j].append(q)
        self._field_queries = tuple(
            np.array(qs, dtype=np.int64) for qs in buckets
        )
        # 批量特征器缓存，惰性构建，见 _build_featurizer
        self._feat_ready = False
        self._val_cache: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        # 每字段全域取值查表，稀疏打分与物化共用，惰性构建一次
        self._field_tabs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # 惰性特征模式，注册只记元组编号，特征推迟到取负载时统一补算
        self._lazy = False
        self._feat_synced = 0
        # GPU 物化开关，命中计数查表 gather 搬卡上做，整数运算逐位同 CPU
        self.gpu_featurize = False
        # 命中计数缓存，救援小注册表开启后基行计数免重算
        self._cnt_cache: dict[tuple[str, ...], int] | None = None
        self._cnt_rows: list[np.ndarray] = []
        self._score_rows: list[np.ndarray] = []

    @property
    def num_states(self) -> int:
        return self._count

    @property
    def field_queries(self) -> tuple[np.ndarray, ...]:
        """字段到受影响查询列的依赖索引，批量菜单与增量注册共用。"""
        return self._field_queries

    @property
    def compiled(self) -> tuple[tuple[tuple, ...], ...]:
        """预编译查询条件，批量结构表构造时按域值求通过表。"""
        return self._compiled

    def state_tuple(self, state_id: int) -> tuple[str, ...]:
        return self._tuples[state_id]

    def _grow(self) -> None:
        """容量翻倍扩容，已算好的特征行原样拷入新矩阵，摊销代价常数。"""
        new_capacity = max(self._INITIAL_CAPACITY, 2 * self._features.shape[0])
        grown = np.empty((new_capacity, len(self.specs)), dtype=np.float64)
        grown[: self._feat_synced] = self._features[: self._feat_synced]
        self._features = grown

    def set_lazy_features(self, flag: bool) -> None:
        """开关惰性特征模式，关闭时立刻补算全部欠账特征行。

        惰性下注册只登记元组与编号，编号发放次序与急切模式逐个一致，
        特征在首次取负载或显式关闭时按首次注册顺序统一矢量化补算，
        数值与急切路径的逐行求值逐位相同，GPU 后端整轮不读特征时零开销。
        """
        self._lazy = bool(flag)
        if not self._lazy:
            self._ensure_features()

    def _ensure_features(self) -> None:
        """补算全部欠账特征行，分块矢量化限住临时内存。"""
        if self._feat_synced >= self._count:
            return
        while self._features.shape[0] < self._count:
            self._grow()
        chunk = 8192
        for lo in range(self._feat_synced, self._count, chunk):
            hi = min(lo + chunk, self._count)
            self._features[lo:hi] = self._features_for_rows(self._tuples[lo:hi])
        self._feat_synced = self._count

    def _insert_block(
        self, keys: list[tuple[str, ...]], feats: np.ndarray
    ) -> None:
        """一批新状态整块入表，容量一次扩到位，特征切片赋值免逐行拷贝。

        编号发放次序与逐行 _insert 完全一致，keys 必须都是未注册的新键。
        """
        m = len(keys)
        need = self._count + m
        if need > self._features.shape[0]:
            cap = max(self._INITIAL_CAPACITY, self._features.shape[0])
            while cap < need:
                cap *= 2
            grown = np.empty((cap, len(self.specs)), dtype=np.float64)
            grown[: self._feat_synced] = self._features[: self._feat_synced]
            self._features = grown
        self._features[self._count : need] = feats
        for key in keys:
            self._index[key] = self._count
            self._tuples.append(key)
            self._count += 1
        self._feat_synced = self._count

    def _insert(self, key: tuple[str, ...], feats: np.ndarray) -> int:
        """把校验完的状态行写入注册表并发号，register 与 register_edit 共用。"""
        if self._count == self._features.shape[0]:
            self._grow()
        state_id = self._count
        self._features[state_id] = feats
        self._count += 1
        self._feat_synced = self._count
        self._index[key] = state_id
        self._tuples.append(key)
        return state_id

    def _insert_lazy(self, key: tuple[str, ...]) -> int:
        """惰性登记状态行，只发号不算特征，欠账由 _ensure_features 清偿。"""
        state_id = self._count
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
        if self._lazy:
            return self._insert_lazy(key)
        feats = np.array(
            [evaluate_compiled(conds, key) for conds in self._compiled],
            dtype=np.float64,
        )
        if not np.isfinite(feats).all():
            raise ValueError("查询贡献必须全部有限")
        return self._insert(key, feats)

    def register_table(self, rows: list[tuple[str, ...]]) -> np.ndarray:
        """注册整张表并返回状态编号向量。"""
        return self.register_many(rows)

    def _build_featurizer(self) -> None:
        """按字段整理条件清单，供取值命中向量惰性求值，批量注册用。

        等值通道，每字段列出涉及它的 (查询列, 算子码, 载荷) 条件，
        取值命中向量按查询列累加命中条件数，行特征为命中数恰等于条件总数，
        半空间列条件总数记毒值 -1 防等值通道误命中，
        分数通道，每字段列出 (半空间序号, 取值到分数表)，
        行特征为各字段分数之和不小于阈值，与 evaluate_compiled 精确同值。
        """
        num_q = len(self.specs)
        ncond = np.zeros(num_q, dtype=np.int16)
        eq_conds: list[list[tuple[int, tuple]]] = [
            [] for _ in range(self.schema.num_fields)
        ]
        hs_cols: list[int] = []
        hs_thresholds: list[int] = []
        hs_tables: list[list[tuple[int, dict[str, int]]]] = [
            [] for _ in range(self.schema.num_fields)
        ]
        for h, conds in enumerate(self._compiled):
            for code, a, b, lo, hi in conds:
                if code == 3:
                    ncond[h] = -1
                    t = len(hs_cols)
                    hs_cols.append(h)
                    hs_thresholds.append(int(b))
                    for j, table in a:
                        hs_tables[j].append((t, table))
                else:
                    ncond[h] += 1
                    eq_conds[a].append((h, (code, b, lo, hi)))
        self._feat_ncond = ncond
        self._feat_eq_conds = eq_conds
        self._feat_hs_cols = np.array(hs_cols, dtype=np.int64)
        self._feat_hs_thresholds = np.array(hs_thresholds, dtype=np.int64)
        self._feat_hs_tables = hs_tables
        self._val_cache = [dict() for _ in range(self.schema.num_fields)]
        self._feat_ready = True

    def _value_vectors(self, j: int, value: str) -> tuple[np.ndarray, np.ndarray]:
        """字段 j 取 value 时的命中条件数向量与半空间分数向量，带缓存。"""
        cached = self._val_cache[j].get(value)
        if cached is not None:
            return cached
        cnt = np.zeros(len(self.specs), dtype=np.int16)
        for h, (code, b, lo, hi) in self._feat_eq_conds[j]:
            if code == 0:
                hit = value == b
            elif code == 1:
                hit = lo <= float(value) <= hi
            else:
                hit = float(value) >= lo
            if hit:
                cnt[h] += 1
        score = np.zeros(len(self._feat_hs_cols), dtype=np.int64)
        for t, table in self._feat_hs_tables[j]:
            score[t] = table.get(value, 0)
        pair = (cnt, score)
        self._val_cache[j][value] = pair
        return pair

    def counts_scores_for_rows(
        self, rows: list[tuple[str, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """一批状态行的命中计数与半空间分数中间量，矢量化按字段查表累加。

        特征由 counts 与 ncond 比较得出，稀疏差分打分拿中间量
        做单字段增量，与全量重算逐位一致，
        GPU 开关打开时查表 gather 搬卡上算完回传，整数运算无舍入。
        """
        if not self._feat_ready:
            self._build_featurizer()
        n = len(rows)
        num_q = len(self.specs)
        xp = np
        if self.gpu_featurize:
            import cupy as xp  # noqa: F811
        cnt = xp.zeros((n, num_q), dtype=np.int16)
        score = xp.zeros((n, len(self._feat_hs_cols)), dtype=np.int64)
        rows_arr = np.array(rows, dtype=str)
        for j in range(self.schema.num_fields):
            vals_arr, cnt_tab, score_tab = self.field_value_tables(j)
            codes = np.searchsorted(vals_arr, rows_arr[:, j])
            if xp is not np:
                cnt_tab = xp.asarray(cnt_tab)
                codes = xp.asarray(codes)
            cnt += cnt_tab[codes]
            if len(self._feat_hs_cols):
                if xp is not np:
                    score_tab = xp.asarray(score_tab)
                score += score_tab[codes]
        if xp is not np:
            cnt = xp.asnumpy(cnt)
            score = xp.asnumpy(score)
        if self._cnt_cache is not None:
            for r, row in enumerate(rows):
                if row not in self._cnt_cache:
                    self._cnt_cache[row] = len(self._cnt_rows)
                    self._cnt_rows.append(cnt[r])
                    self._score_rows.append(score[r])
        return cnt, score

    def enable_counts_cache(self) -> None:
        """开启命中计数缓存，之后算过计数的行元组可直接复用。

        只给救援小注册表用，全局注册表不开防内存无界增长。
        """
        if self._cnt_cache is None:
            self._cnt_cache = {}

    def counts_scores_cached(
        self, rows: list[tuple[str, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """带缓存的命中计数查询，命中直接取，未命中批量补算并入缓存。"""
        if self._cnt_cache is None:
            return self.counts_scores_for_rows(rows)
        miss = [row for row in rows if row not in self._cnt_cache]
        if miss:
            self.counts_scores_for_rows(miss)
        idx = [self._cnt_cache[row] for row in rows]
        cnt = np.stack([self._cnt_rows[p] for p in idx])
        score = np.stack([self._score_rows[p] for p in idx])
        return cnt, score

    def edit_feature_pack(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """稀疏差分打分所需特征器结构，条件总数与半空间列及阈值。"""
        if not self._feat_ready:
            self._build_featurizer()
        return self._feat_ncond, self._feat_hs_cols, self._feat_hs_thresholds

    def value_vectors(self, j: int, value: str) -> tuple[np.ndarray, np.ndarray]:
        """字段 j 取 value 的命中条件数向量与半空间分数向量，公开口。"""
        if not self._feat_ready:
            self._build_featurizer()
        return self._value_vectors(j, value)

    def field_value_tables(
        self, j: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """字段 j 全域取值查表，排序值数组与命中计数表与半空间分数表。

        一次救援内查表内容不变，惰性构建一次后各处纯索引取用，
        取值编码用 searchsorted 矢量化，免去每次重建小表与字典查找。
        """
        tabs = self._field_tabs.get(j)
        if tabs is None:
            if not self._feat_ready:
                self._build_featurizer()
            dom = sorted(set(self.schema.domains[j]))
            vals_arr = np.array(dom, dtype=str)
            cnt_tab = np.stack([self._value_vectors(j, v)[0] for v in dom])
            score_tab = np.stack([self._value_vectors(j, v)[1] for v in dom])
            tabs = (vals_arr, cnt_tab, score_tab)
            self._field_tabs[j] = tabs
        return tabs

    def _features_for_rows(self, rows: list[tuple[str, ...]]) -> np.ndarray:
        """一批新状态行的贡献矩阵，矢量化按字段查表累加。

        与 evaluate_compiled 逐行逐查询求值逐位一致，
        谓词取值只有 0 与 1，构造上必然有限，无需再扫有限性。
        """
        cnt, score = self.counts_scores_for_rows(rows)
        feats = (cnt == self._feat_ncond[None, :]).astype(np.float64)
        if len(self._feat_hs_cols):
            feats[:, self._feat_hs_cols] = (
                score >= self._feat_hs_thresholds[None, :]
            ).astype(np.float64)
        return feats

    def register_many(self, rows: list[tuple[str, ...]]) -> np.ndarray:
        """批量注册状态行，编号发放次序与逐个 register 完全一致。

        重复行与已注册行直接给旧编号，真正的新状态按首次出现顺序
        统一矢量化求特征后依序入表，返回与输入等长的编号向量。
        """
        ids = np.empty(len(rows), dtype=np.int64)
        new_keys: list[tuple[str, ...]] = []
        new_pos: dict[tuple[str, ...], int] = {}
        for r, row in enumerate(rows):
            key = tuple(row)
            found = self._index.get(key)
            if found is not None:
                ids[r] = found
                continue
            p = new_pos.get(key)
            if p is None:
                if len(key) != self.schema.num_fields:
                    raise ValueError("状态元组字段数与 schema 不一致")
                p = len(new_keys)
                new_pos[key] = p
                new_keys.append(key)
            ids[r] = self._count + p
        if new_keys:
            if self._lazy:
                for key in new_keys:
                    self._insert_lazy(key)
            else:
                feats = self._features_for_rows(new_keys)
                self._insert_block(new_keys, feats)
        return ids

    def register_edited_many(
        self, base_rows: list[tuple[str, ...]], rows: list[tuple[str, ...]]
    ) -> np.ndarray:
        """批量注册编辑行，特征由基行命中计数增量得出。

        base_rows 与 rows 等长，每个编辑行给出它出发的基行元组，
        编号发放次序与 register_many 完全一致，
        新行命中数为基行命中数加变动字段取值向量之差，整数运算，
        特征由命中数与条件总数比较得出，与全量重算逐位相同。
        """
        if len(base_rows) != len(rows):
            raise ValueError("基行列表与编辑行列表长度不一致")
        ids = np.empty(len(rows), dtype=np.int64)
        new_keys: list[tuple[str, ...]] = []
        new_base: list[tuple[str, ...]] = []
        new_pos: dict[tuple[str, ...], int] = {}
        for r, row in enumerate(rows):
            key = tuple(row)
            found = self._index.get(key)
            if found is not None:
                ids[r] = found
                continue
            p = new_pos.get(key)
            if p is None:
                if len(key) != self.schema.num_fields:
                    raise ValueError("状态元组字段数与 schema 不一致")
                p = len(new_keys)
                new_pos[key] = p
                new_keys.append(key)
                new_base.append(tuple(base_rows[r]))
            ids[r] = self._count + p
        if not new_keys:
            return ids
        if self._lazy:
            for key in new_keys:
                self._insert_lazy(key)
            return ids
        feats = self._features_for_edited(new_base, new_keys)
        self._insert_block(new_keys, feats)
        return ids

    def _features_for_edited(
        self,
        base_rows: list[tuple[str, ...]],
        rows: list[tuple[str, ...]],
    ) -> np.ndarray:
        """编辑行的贡献矩阵，基行命中数按变动字段稀疏增量。

        基行去重后统一算命中数与半空间分数，编辑行 gather 基行整行，
        变动字段按字段分组把取值向量之差加到依赖查询列上，
        最后与条件总数比较出特征，整数路径与全量重算逐位一致。
        """
        if not self._feat_ready:
            self._build_featurizer()
        uniq_bases: list[tuple[str, ...]] = []
        bpos: dict[tuple[str, ...], int] = {}
        bidx = np.empty(len(rows), dtype=np.int64)
        for r, b in enumerate(base_rows):
            p = bpos.get(b)
            if p is None:
                p = len(uniq_bases)
                bpos[b] = p
                uniq_bases.append(b)
            bidx[r] = p
        cnt_b, score_b = self.counts_scores_cached(uniq_bases)
        cnt = cnt_b[bidx]
        score = score_b[bidx]
        # 变动按字段分组，取值向量差只加到该字段依赖的查询列
        by_field: dict[int, list[tuple[int, str, str]]] = {}
        for r, (b, key) in enumerate(zip(base_rows, rows)):
            for j in range(self.schema.num_fields):
                if key[j] != b[j]:
                    by_field.setdefault(j, []).append((r, b[j], key[j]))
        for j, group in by_field.items():
            cols = self._field_queries[j]
            vals_arr, cnt_tab, score_tab = self.field_value_tables(j)
            rows_g = np.array([g[0] for g in group], dtype=np.int64)
            old_c = np.searchsorted(
                vals_arr, np.array([g[1] for g in group], dtype=str)
            )
            new_c = np.searchsorted(
                vals_arr, np.array([g[2] for g in group], dtype=str)
            )
            if len(cols):
                cnt[np.ix_(rows_g, cols)] += (
                    cnt_tab[new_c][:, cols] - cnt_tab[old_c][:, cols]
                )
            if len(self._feat_hs_cols):
                score[rows_g] += score_tab[new_c] - score_tab[old_c]
        feats = (cnt == self._feat_ncond[None, :]).astype(np.float64)
        if len(self._feat_hs_cols):
            feats[:, self._feat_hs_cols] = (
                score >= self._feat_hs_thresholds[None, :]
            ).astype(np.float64)
        return feats

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
        if self._lazy:
            return self._insert_lazy(key)
        if len(changed) == 1:
            affected = self._field_queries[changed[0]]
        else:
            affected = np.unique(
                np.concatenate([self._field_queries[j] for j in changed])
            )
        feats = self._features[base_id].copy()
        for q in affected:
            feats[q] = evaluate_compiled(self._compiled[int(q)], key)
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
        self._ensure_features()  # 惰性模式欠账在此清偿，急切模式零开销
        view = self._features[: self._count]
        view.setflags(write=False)
        return make_workload(view, target, weights, features_prevalidated=True)


def target_from_specs(specs: list[QuerySpec]) -> np.ndarray:
    """目标向量直接取查询负载里的真实答案。"""
    return np.array([spec.result for spec in specs], dtype=np.float64)
