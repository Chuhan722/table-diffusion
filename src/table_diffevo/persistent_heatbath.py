"""持久化表状态上的 workload 能量热浴扩散研究原型。

本模块直接把合成表作为 Markov 状态。每个随机扫描微步选择一个表坐标，枚举该
属性的公开合法值，并从 workload 能量的完整有限温条件分布中重采样。采样后立即
增量更新表、查询答案和平方 loss；不使用 donor、临时复制 mask、候选接受或回滚。

固定有限逆温下，本核是以玻尔兹曼分布为平稳分布的单坐标平衡态采样器，不是以
``argmin`` 为收敛目标的优化器。当前协议既不退火，也不按 best 状态选择输出；
强相关 workload 上的单坐标更新还可能与主线的块级更新具有不同混合性质。

平方能量模式服务 Issue #32，目标对齐的 normalized L1 模式服务 Issue #38；两者
都不接入默认生成器，也不承诺稳定公共 API。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.objective import compute_loss
from table_diffevo.queries import eval_query_mask, evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


MAX_DOMAIN_SIZE = 100_000
IDENTITY_TOLERANCE = 1e-12
ENERGY_MODE_SQUARED = "squared"
ENERGY_MODE_NORMALIZED_L1 = "normalized_l1"
ENERGY_MODES = frozenset({ENERGY_MODE_SQUARED, ENERGY_MODE_NORMALIZED_L1})


@dataclass
class PersistentHeatbathState:
    """会被微步原地更新的持久化表状态。"""

    table: pd.DataFrame
    query_answers: np.ndarray
    loss: float

    def copy(self) -> "PersistentHeatbathState":
        """返回不共享表或查询答案存储的状态副本。"""
        return PersistentHeatbathState(
            table=self.table.copy(deep=True),
            query_answers=self.query_answers.copy(),
            loss=float(self.loss),
        )


@dataclass(frozen=True)
class HeatbathConditional:
    """固定状态和一个表坐标后的完整合法值条件分布。"""

    row_index: int
    attribute_index: int
    attribute: str
    values: Tuple[Any, ...]
    current_value_index: int
    source_attribute_names: Tuple[str, ...]
    source_row_values: Tuple[Any, ...]
    source_query_answers: Tuple[int, ...]
    source_loss: float
    energy_mode: str
    source_energy: float
    query_deltas: np.ndarray
    candidate_losses: np.ndarray
    gains: np.ndarray
    candidate_energies: np.ndarray
    energy_gains: np.ndarray
    scaled_log_weights: np.ndarray
    probabilities: np.ndarray
    expected_loss: float
    reference_expected_loss: float
    expected_gain_over_reference: float
    expected_energy: float
    reference_expected_energy: float
    expected_energy_gain_over_reference: float
    entropy: float
    maximum_entropy: float
    normalized_entropy: float
    uphill_probability_mass: float
    candidate_state_evaluations: int
    query_indicator_evaluations: int


def _validate_nonnegative_finite(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} 必须是非负有限数值，得到 {value!r}")
    return float(value)


def _validate_positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} 必须是正整数，得到 {value!r}")
    return int(value)


def _validate_index(value: int, upper: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < upper
    ):
        raise ValueError(f"{name} 必须在 [0, {upper}) 内，得到 {value!r}")
    return int(value)


def _validate_energy_mode(value: str) -> str:
    if not isinstance(value, str) or value not in ENERGY_MODES:
        raise ValueError(
            "energy_mode 必须是 "
            f"{sorted(ENERGY_MODES)} 之一，得到 {value!r}"
        )
    return value


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def legal_attribute_values(block: AttributeBlock) -> Tuple[Any, ...]:
    """返回一个 schema 属性块的有限公开合法域。"""
    if not isinstance(block, AttributeBlock):
        raise ValueError("block 必须是 AttributeBlock")
    if block.is_numeric():
        if block.range is None or len(block.range) != 2:
            raise ValueError(f"数值属性 {block.name!r} 必须提供两端点 range")
        low_raw, high_raw = block.range
        if (
            isinstance(low_raw, (bool, np.bool_))
            or isinstance(high_raw, (bool, np.bool_))
            or not isinstance(low_raw, (int, float, np.integer, np.floating))
            or not isinstance(high_raw, (int, float, np.integer, np.floating))
            or not np.isfinite(low_raw)
            or not np.isfinite(high_raw)
            or float(low_raw) != int(low_raw)
            or float(high_raw) != int(high_raw)
        ):
            raise ValueError(
                f"数值属性 {block.name!r} 的 range 必须是有限整数端点"
            )
        low, high = int(low_raw), int(high_raw)
        if low > high:
            raise ValueError(f"数值属性 {block.name!r} 的 range 下界大于上界")
        if high - low + 1 > MAX_DOMAIN_SIZE:
            raise ValueError(
                f"属性 {block.name!r} 的合法域大小超过护栏 {MAX_DOMAIN_SIZE}"
            )
        return tuple(range(low, high + 1))

    if not block.is_categorical():
        raise ValueError(f"属性 {block.name!r} 的类型 {block.type!r} 不受支持")
    if block.values is None or len(block.values) == 0:
        raise ValueError(f"类别属性 {block.name!r} 的合法域不能为空")
    values = tuple(block.values)
    if len(values) > MAX_DOMAIN_SIZE:
        raise ValueError(
            f"属性 {block.name!r} 的合法域大小超过护栏 {MAX_DOMAIN_SIZE}"
        )
    for value in values:
        if value is None:
            raise ValueError(f"类别属性 {block.name!r} 的合法值不能是 None")
        try:
            comparison = value == value
            stable = (
                isinstance(comparison, (bool, np.bool_))
                and bool(comparison)
            )
            hash(value)
        except Exception as error:
            raise ValueError(
                f"类别属性 {block.name!r} 的合法值必须可哈希且可稳定比较"
            ) from error
        if not stable or (
            isinstance(value, (float, np.floating))
            and not np.isfinite(value)
        ):
            raise ValueError(
                f"类别属性 {block.name!r} 不能包含 NaN、无穷或模糊值"
            )
    if len(set(values)) != len(values):
        raise ValueError(f"类别属性 {block.name!r} 的合法值不能重复")
    return values


def _validate_schema(schema: Schema) -> Tuple[str, ...]:
    if not isinstance(schema, Schema):
        raise ValueError("schema 必须是 Schema")
    if schema.n_blocks() <= 0:
        raise ValueError("schema 必须至少包含一个属性")
    names = tuple(schema.attribute_names())
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("schema 属性名必须是非空字符串")
    if len(set(names)) != len(names):
        raise ValueError("schema 属性名不能重复")
    for block in schema.attributes:
        if not isinstance(block, AttributeBlock):
            raise ValueError("schema.attributes 必须只包含 AttributeBlock")
        legal_attribute_values(block)
    return names


def _validate_queries(
    queries: List[Dict[str, Any]], schema: Schema
) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(queries, list):
        raise ValueError("queries 必须是列表")
    attribute_names = _validate_schema(schema)
    positions = {name: index for index, name in enumerate(attribute_names)}
    active_by_attribute = [[] for _ in attribute_names]
    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"queries[{query_index}] 必须是字典")
        conditions = query.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError(
                f"queries[{query_index}].conditions 必须是列表"
            )
        query_attributes = set()
        for condition_index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                raise ValueError(
                    f"queries[{query_index}].conditions[{condition_index}] "
                    "必须是字典"
                )
            attribute = condition.get("attribute")
            if attribute not in positions:
                raise ValueError(
                    f"queries[{query_index}] 包含未知属性 {attribute!r}"
                )
            operator = condition.get("operator")
            if operator not in ("==", ">=", "between"):
                raise ValueError(
                    f"queries[{query_index}] 包含不支持的操作符 {operator!r}"
                )
            if operator in ("==", ">=") and "value" not in condition:
                raise ValueError(f"queries[{query_index}] 的条件缺少 value")
            if operator == "between" and not {
                "lower", "upper"
            }.issubset(condition):
                raise ValueError(
                    f"queries[{query_index}] 的 between 条件缺少端点"
                )
            query_attributes.add(attribute)
        for attribute in query_attributes:
            active_by_attribute[positions[attribute]].append(query_index)
    return tuple(tuple(indices) for indices in active_by_attribute)


def _target_values(target: np.ndarray, n_queries: int) -> np.ndarray:
    raw = np.asarray(target)
    if raw.shape != (n_queries,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"target 必须是长度 {n_queries} 的有限数值一维数组")
    values = raw.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"target 必须是长度 {n_queries} 的有限数值一维数组")
    return values


def _validate_table(table: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("table 必须是 pandas DataFrame")
    if len(table) == 0:
        raise ValueError("table 必须至少包含一行")
    names = list(_validate_schema(schema))
    missing = [name for name in names if name not in table.columns]
    if missing:
        raise ValueError(f"table 缺少 schema 属性列: {missing}")
    checked = table.loc[:, names].reset_index(drop=True).copy()
    for block in schema.attributes:
        domain = legal_attribute_values(block)
        column = checked[block.name].to_numpy()
        if block.is_numeric():
            valid = np.asarray([
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                and float(value) == int(value)
                and int(value) in domain
                for value in column
            ], dtype=bool)
        else:
            domain_set = set(domain)
            valid_items = []
            for value in column:
                try:
                    valid_items.append(value in domain_set)
                except (TypeError, ValueError):
                    valid_items.append(False)
            valid = np.asarray(valid_items, dtype=bool)
        if not bool(np.all(valid)):
            rows = np.flatnonzero(~valid).tolist()
            raise ValueError(
                f"table 的属性 {block.name!r} 含非法值，行索引 {rows[:5]}"
            )
    return checked


def initialize_persistent_heatbath_state(
    table: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
) -> PersistentHeatbathState:
    """验证输入并从整表评价构造持久化状态。"""
    checked = _validate_table(table, schema)
    _validate_queries(queries, schema)
    target_array = _target_values(target, len(queries))
    answers = evaluate_table(checked, queries).astype(np.int64)
    loss = float(compute_loss(target_array, answers))
    if not np.isfinite(loss):
        raise ValueError("初始 workload loss 超出 float64 可表示范围")
    return PersistentHeatbathState(checked, answers, loss)


def _validate_state(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
) -> Tuple[np.ndarray, Tuple[Tuple[int, ...], ...]]:
    if not isinstance(state, PersistentHeatbathState):
        raise ValueError("state 必须是 PersistentHeatbathState")
    if not isinstance(state.table, pd.DataFrame) or len(state.table) == 0:
        raise ValueError("state.table 必须是非空 pandas DataFrame")
    names = _validate_schema(schema)
    if tuple(state.table.columns) != names:
        raise ValueError("state.table 的列及顺序必须与 schema 完全一致")
    expected_index = pd.RangeIndex(len(state.table))
    if not isinstance(state.table.index, pd.RangeIndex) or not (
        state.table.index.equals(expected_index)
    ):
        raise ValueError("state.table 必须使用从 0 开始的连续 RangeIndex")
    active_queries = _validate_queries(queries, schema)
    target_array = _target_values(target, len(queries))
    answers = np.asarray(state.query_answers)
    if answers.shape != (len(queries),) or answers.dtype.kind not in "iu":
        raise ValueError(
            f"state.query_answers 必须是长度 {len(queries)} 的整数一维数组"
        )
    if np.any(answers < 0) or np.any(answers > len(state.table)):
        raise ValueError("state.query_answers 必须位于 [0, n_records]")
    if (
        isinstance(state.loss, (bool, np.bool_))
        or not isinstance(state.loss, (int, float, np.integer, np.floating))
        or not np.isfinite(state.loss)
        or state.loss < 0.0
    ):
        raise ValueError("state.loss 必须是有限非负数值")
    recomputed = float(compute_loss(target_array, answers))
    if abs(float(state.loss) - recomputed) > IDENTITY_TOLERANCE:
        raise ValueError(
            "state.loss 与 target/query_answers 不一致: "
            f"{state.loss} vs {recomputed}"
        )
    return target_array, active_queries


def verify_persistent_heatbath_state(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    *,
    tolerance: float = IDENTITY_TOLERANCE,
) -> Dict[str, Any]:
    """用整表查询独立复核合法值、计数和 workload loss。"""
    tol = _validate_nonnegative_finite(tolerance, "tolerance")
    target_array, _ = _validate_state(state, schema, queries, target)
    checked = _validate_table(state.table, schema)
    recomputed_answers = evaluate_table(checked, queries).astype(np.int64)
    recorded_answers = np.asarray(state.query_answers, dtype=np.int64)
    answer_error = (
        int(np.max(np.abs(recomputed_answers - recorded_answers)))
        if len(recomputed_answers) else 0
    )
    recomputed_loss = float(compute_loss(target_array, recomputed_answers))
    loss_error = abs(recomputed_loss - float(state.loss))
    if answer_error != 0 or loss_error > tol:
        raise RuntimeError(
            "持久状态整表复核失败: "
            f"query_answer_max_abs_error={answer_error}, "
            f"loss_abs_error={loss_error}"
        )
    return {
        "query_answer_max_abs_error": answer_error,
        "loss_abs_error": float(loss_error),
        "recomputed_loss": recomputed_loss,
    }


def _candidate_query_deltas(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    active_queries: Tuple[Tuple[int, ...], ...],
    row_index: int,
    attribute_index: int,
) -> Tuple[Tuple[Any, ...], int, np.ndarray, int]:
    block = schema.attributes[attribute_index]
    values = legal_attribute_values(block)
    current = state.table.iat[row_index, attribute_index]
    matches = [index for index, value in enumerate(values) if value == current]
    if len(matches) != 1:
        raise ValueError(
            f"state.table[{row_index}, {block.name!r}]={current!r} "
            "不在唯一合法域位置"
        )
    current_index = matches[0]
    row = state.table.iloc[[row_index]].reset_index(drop=True)
    candidates = pd.concat([row] * len(values), ignore_index=True)
    candidates[block.name] = list(values)
    deltas = np.zeros((len(values), len(queries)), dtype=np.int8)
    query_indices = active_queries[attribute_index]
    for query_index in query_indices:
        indicators = eval_query_mask(
            candidates, queries[query_index]
        ).astype(np.int8, copy=False)
        deltas[:, query_index] = indicators - indicators[current_index]
    if np.any(deltas[current_index] != 0):
        raise RuntimeError("当前合法值的查询增量必须严格为零")
    return values, current_index, deltas, len(values) * len(query_indices)


def heatbath_probabilities(
    candidate_losses: np.ndarray,
    n_records: int,
    inverse_temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回稳定归一化概率和相对最大值的 log weight。"""
    n = _validate_positive_integer(n_records, "n_records")
    beta = _validate_nonnegative_finite(
        inverse_temperature, "inverse_temperature"
    )
    raw_losses = np.asarray(candidate_losses)
    if (
        raw_losses.ndim != 1
        or len(raw_losses) == 0
        or raw_losses.dtype.kind not in "iuf"
    ):
        raise ValueError("candidate_losses 必须是非空有限数值一维数组")
    losses = raw_losses.astype(float, copy=False)
    if not np.all(np.isfinite(losses)) or np.any(losses < 0.0):
        raise ValueError("candidate_losses 必须是非空有限非负数值一维数组")
    if beta == 0.0:
        centered = np.zeros_like(losses)
    else:
        minimum = float(np.min(losses))
        with np.errstate(over="ignore", invalid="ignore"):
            centered = -beta * (losses - minimum) / n
        if not np.all(np.isfinite(centered)):
            raise ValueError("条件 log weight 超出 float64 可表示范围")
    with np.errstate(under="ignore", invalid="ignore"):
        weights = np.exp(centered)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("条件概率发生下溢或不再具有严格双向支持")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("条件概率归一化常数无效")
    probabilities = weights / total
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or (
            len(probabilities) > 1
            and np.any(probabilities >= 1.0)
        )
        or abs(float(np.sum(probabilities)) - 1.0) > IDENTITY_TOLERANCE
    ):
        raise ValueError("条件概率无法表示为严格正的有限分布")
    return _readonly(probabilities), _readonly(centered)


def boltzmann_probabilities(
    candidate_energies: np.ndarray,
    inverse_energy_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 ``exp(-inverse_energy_scale * energy)`` 稳定归一化。

    与 :func:`heatbath_probabilities` 不同，本函数不隐式再除以记录数。
    normalized L1 已包含 ``1/N``，因此必须使用这个显式能量接口，避免重复
    归一化。平方模式继续走历史函数，以保持概率与随机轨迹精确回归。
    """
    beta = _validate_nonnegative_finite(
        inverse_energy_scale, "inverse_energy_scale"
    )
    raw = np.asarray(candidate_energies)
    if raw.ndim != 1 or len(raw) == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("candidate_energies 必须是非空有限数值一维数组")
    energies = raw.astype(float, copy=False)
    if not np.all(np.isfinite(energies)) or np.any(energies < 0.0):
        raise ValueError(
            "candidate_energies 必须是非空有限非负数值一维数组"
        )
    if beta == 0.0:
        centered = np.zeros_like(energies)
    else:
        minimum = float(np.min(energies))
        with np.errstate(over="ignore", invalid="ignore"):
            centered = -beta * (energies - minimum)
        if not np.all(np.isfinite(centered)):
            raise ValueError("条件 log weight 超出 float64 可表示范围")
    with np.errstate(under="ignore", invalid="ignore"):
        weights = np.exp(centered)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("条件概率发生下溢或不再具有严格双向支持")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("条件概率归一化常数无效")
    probabilities = weights / total
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or (
            len(probabilities) > 1
            and np.any(probabilities >= 1.0)
        )
        or abs(float(np.sum(probabilities)) - 1.0) > IDENTITY_TOLERANCE
    ):
        raise ValueError("条件概率无法表示为严格正的有限分布")
    return _readonly(probabilities), _readonly(centered)


def build_persistent_heatbath_conditional(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    *,
    row_index: int,
    attribute_index: int,
    inverse_temperature: float,
    energy_mode: str = ENERGY_MODE_SQUARED,
) -> HeatbathConditional:
    """构造固定表状态与坐标下的完整 workload 热浴条件分布。"""
    beta = _validate_nonnegative_finite(
        inverse_temperature, "inverse_temperature"
    )
    mode = _validate_energy_mode(energy_mode)
    target_array, active_queries = _validate_state(
        state, schema, queries, target
    )
    row = _validate_index(row_index, len(state.table), "row_index")
    attribute = _validate_index(
        attribute_index, schema.n_blocks(), "attribute_index"
    )
    values, current_index, deltas, query_evaluations = (
        _candidate_query_deltas(
            state,
            schema,
            queries,
            active_queries,
            row,
            attribute,
        )
    )
    residual = target_array - state.query_answers.astype(float, copy=False)
    delta_float = deltas.astype(float, copy=False)
    gains = (
        delta_float @ residual
        - 0.5 * np.sum(delta_float ** 2, axis=1)
    )
    candidate_losses = float(state.loss) - gains
    if (
        not np.all(np.isfinite(candidate_losses))
        or np.any(candidate_losses < -IDENTITY_TOLERANCE)
    ):
        raise ValueError("候选 workload loss 无效")
    candidate_losses = np.maximum(candidate_losses, 0.0)
    if (
        gains[current_index] != 0.0
        or candidate_losses[current_index] != float(state.loss)
    ):
        raise RuntimeError("当前值候选必须严格保持原 workload 状态")

    if mode == ENERGY_MODE_SQUARED:
        source_energy = float(state.loss) / len(state.table)
        candidate_energies = candidate_losses / len(state.table)
        energy_gains = gains / len(state.table)
        # 保留历史运算顺序，确保 Issue #32 的概率和随机轨迹精确回归。
        probabilities, centered = heatbath_probabilities(
            candidate_losses, len(state.table), beta
        )
    else:
        if len(queries) == 0:
            raise ValueError("normalized_l1 能量要求至少一个查询")
        candidate_residuals = residual[None, :] - delta_float
        candidate_energies = (
            np.mean(np.abs(candidate_residuals), axis=1) / len(state.table)
        )
        source_energy = float(
            np.mean(np.abs(residual)) / len(state.table)
        )
        energy_gains = source_energy - candidate_energies
        probabilities, centered = boltzmann_probabilities(
            candidate_energies, beta
        )
    if (
        not np.all(np.isfinite(candidate_energies))
        or np.any(candidate_energies < 0.0)
        or not np.all(np.isfinite(energy_gains))
        or not np.isfinite(source_energy)
        or source_energy < 0.0
    ):
        raise ValueError("候选 workload 能量无效")
    if (
        energy_gains[current_index] != 0.0
        or candidate_energies[current_index] != source_energy
    ):
        raise RuntimeError("当前值候选必须严格保持原 workload 能量")

    minimum_loss = float(np.min(candidate_losses))
    loss_offsets = candidate_losses - minimum_loss
    expected_offset = float(np.dot(probabilities, loss_offsets))
    reference_offset = float(np.mean(loss_offsets))
    expected_gain_over_reference = reference_offset - expected_offset
    expected_loss = minimum_loss + expected_offset
    reference_expected_loss = minimum_loss + reference_offset
    if not np.isfinite(expected_loss) or not np.isfinite(
        reference_expected_loss
    ):
        raise ValueError("条件期望 loss 超出 float64 可表示范围")
    if (
        mode == ENERGY_MODE_SQUARED
        and expected_gain_over_reference < -IDENTITY_TOLERANCE
    ):
        raise RuntimeError("有限温条件期望 loss 高于 beta=0 参考扩散")

    minimum_energy = float(np.min(candidate_energies))
    energy_offsets = candidate_energies - minimum_energy
    expected_energy_offset = float(np.dot(probabilities, energy_offsets))
    reference_energy_offset = float(np.mean(energy_offsets))
    expected_energy_gain_over_reference = (
        reference_energy_offset - expected_energy_offset
    )
    expected_energy = minimum_energy + expected_energy_offset
    reference_expected_energy = minimum_energy + reference_energy_offset
    if not np.isfinite(expected_energy) or not np.isfinite(
        reference_expected_energy
    ):
        raise ValueError("条件期望能量超出 float64 可表示范围")
    if expected_energy_gain_over_reference < -IDENTITY_TOLERANCE:
        raise RuntimeError("有限温条件期望能量高于 beta=0 参考扩散")
    entropy = float(-np.dot(probabilities, np.log(probabilities)))
    maximum_entropy = float(np.log(len(values)))
    normalized_entropy = (
        entropy / maximum_entropy if maximum_entropy > 0.0 else 1.0
    )
    if not -IDENTITY_TOLERANCE <= normalized_entropy <= (
        1.0 + IDENTITY_TOLERANCE
    ):
        raise RuntimeError("条件熵超出有限域理论范围")
    if mode == ENERGY_MODE_SQUARED:
        # 保留历史诊断阈值的量纲与运算，避免只因新增通用能量字段而改变旧输出。
        uphill = candidate_losses > float(state.loss) + IDENTITY_TOLERANCE
    else:
        uphill = candidate_energies > source_energy + IDENTITY_TOLERANCE
    uphill_mass = float(np.sum(probabilities[uphill]))
    names = tuple(schema.attribute_names())
    return HeatbathConditional(
        row_index=row,
        attribute_index=attribute,
        attribute=schema.attributes[attribute].name,
        values=values,
        current_value_index=current_index,
        source_attribute_names=names,
        source_row_values=tuple(
            state.table.iloc[row].loc[list(names)].tolist()
        ),
        source_query_answers=tuple(
            int(value) for value in state.query_answers.tolist()
        ),
        source_loss=float(state.loss),
        energy_mode=mode,
        source_energy=float(source_energy),
        query_deltas=_readonly(deltas),
        candidate_losses=_readonly(candidate_losses),
        gains=_readonly(gains),
        candidate_energies=_readonly(candidate_energies),
        energy_gains=_readonly(energy_gains),
        scaled_log_weights=centered,
        probabilities=probabilities,
        expected_loss=expected_loss,
        reference_expected_loss=reference_expected_loss,
        expected_gain_over_reference=float(expected_gain_over_reference),
        expected_energy=float(expected_energy),
        reference_expected_energy=float(reference_expected_energy),
        expected_energy_gain_over_reference=float(
            expected_energy_gain_over_reference
        ),
        entropy=entropy,
        maximum_entropy=maximum_entropy,
        normalized_entropy=float(normalized_entropy),
        uphill_probability_mass=uphill_mass,
        candidate_state_evaluations=len(values),
        query_indicator_evaluations=query_evaluations,
    )


def sample_heatbath_index(
    conditional: HeatbathConditional,
    *,
    rng: Optional[np.random.Generator] = None,
    gumbels: Optional[np.ndarray] = None,
) -> int:
    """用 Gumbel-max 从条件分布采样一个合法值索引。"""
    if not isinstance(conditional, HeatbathConditional):
        raise ValueError("conditional 必须是 HeatbathConditional")
    if gumbels is None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("未提供 gumbels 时 rng 必须是 np.random.Generator")
        noise = rng.gumbel(size=len(conditional.values))
    else:
        raw = np.asarray(gumbels)
        if raw.shape != (len(conditional.values),) or raw.dtype.kind not in "iuf":
            raise ValueError(
                f"gumbels 必须是长度 {len(conditional.values)} 的有限数值数组"
            )
        noise = raw.astype(float, copy=False)
        if not np.all(np.isfinite(noise)):
            raise ValueError(
                f"gumbels 必须是长度 {len(conditional.values)} 的有限数值数组"
            )
    return int(np.argmax(np.log(conditional.probabilities) + noise))


def apply_heatbath_choice(
    state: PersistentHeatbathState,
    conditional: HeatbathConditional,
    choice_index: int,
) -> Dict[str, Any]:
    """把一个已采样合法值原地写入持久状态并返回微步诊断。"""
    if not isinstance(state, PersistentHeatbathState):
        raise ValueError("state 必须是 PersistentHeatbathState")
    if not isinstance(conditional, HeatbathConditional):
        raise ValueError("conditional 必须是 HeatbathConditional")
    if not isinstance(state.table, pd.DataFrame):
        raise ValueError("state.table 必须是 pandas DataFrame")
    if tuple(state.table.columns) != conditional.source_attribute_names:
        raise ValueError("state.table 的列已不再匹配 conditional 源状态")
    if len(state.table) <= conditional.row_index:
        raise ValueError("state.table 的行数已不再匹配 conditional 源状态")
    source_row = tuple(
        state.table.iloc[conditional.row_index].loc[
            list(conditional.source_attribute_names)
        ].tolist()
    )
    if source_row != conditional.source_row_values:
        raise ValueError("state 对应行已不再匹配 conditional 源状态")
    answers = np.asarray(state.query_answers)
    if (
        answers.shape != (len(conditional.source_query_answers),)
        or answers.dtype.kind not in "iu"
        or tuple(int(value) for value in answers.tolist())
        != conditional.source_query_answers
    ):
        raise ValueError("state.query_answers 已不再匹配 conditional 源状态")
    if float(state.loss) != conditional.source_loss:
        raise ValueError("state.loss 已不再匹配 conditional 源状态")
    choice = _validate_index(
        choice_index, len(conditional.values), "choice_index"
    )
    row = conditional.row_index
    old_value = state.table.iat[row, conditional.attribute_index]
    expected_old = conditional.values[conditional.current_value_index]
    if old_value != expected_old:
        raise ValueError("state 已不再是构造 conditional 时的当前状态")
    before_loss = float(state.loss)
    new_value = conditional.values[choice]
    delta = conditional.query_deltas[choice].astype(np.int64, copy=False)
    updated_answers = answers.astype(np.int64, copy=False) + delta
    if np.any(updated_answers < 0) or np.any(updated_answers > len(state.table)):
        raise RuntimeError("增量更新后的查询答案超出合法计数范围")
    state.table.iat[row, conditional.attribute_index] = new_value
    state.query_answers = updated_answers
    state.loss = float(conditional.candidate_losses[choice])
    return {
        "row_index": row,
        "attribute_index": conditional.attribute_index,
        "attribute": conditional.attribute,
        "old_value": old_value,
        "new_value": new_value,
        "choice_index": choice,
        "chosen_probability": float(conditional.probabilities[choice]),
        "changed": bool(new_value != old_value),
        "loss_before": before_loss,
        "loss_after": float(state.loss),
        "gain": float(before_loss - state.loss),
        "energy_mode": conditional.energy_mode,
        "energy_before": conditional.source_energy,
        "energy_after": float(conditional.candidate_energies[choice]),
        "energy_gain": float(conditional.energy_gains[choice]),
        "expected_loss": conditional.expected_loss,
        "reference_expected_loss": conditional.reference_expected_loss,
        "expected_gain_over_reference": (
            conditional.expected_gain_over_reference
        ),
        "expected_energy": conditional.expected_energy,
        "reference_expected_energy": (
            conditional.reference_expected_energy
        ),
        "expected_energy_gain_over_reference": (
            conditional.expected_energy_gain_over_reference
        ),
        "conditional_entropy": conditional.entropy,
        "conditional_maximum_entropy": conditional.maximum_entropy,
        "conditional_normalized_entropy": conditional.normalized_entropy,
        "uphill_probability_mass": conditional.uphill_probability_mass,
        "probability_min": float(np.min(conditional.probabilities)),
        "probability_max": float(np.max(conditional.probabilities)),
        "candidate_state_evaluations": conditional.candidate_state_evaluations,
        "query_indicator_evaluations": conditional.query_indicator_evaluations,
        "query_delta": delta.copy(),
    }


def persistent_heatbath_step(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    inverse_temperature: float,
    rng: np.random.Generator,
    *,
    coordinate_index: Optional[int] = None,
    gumbels: Optional[np.ndarray] = None,
    energy_mode: str = ENERGY_MODE_SQUARED,
) -> Dict[str, Any]:
    """执行一个随机扫描微步并原地更新 ``state``。"""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    beta = _validate_nonnegative_finite(
        inverse_temperature, "inverse_temperature"
    )
    mode = _validate_energy_mode(energy_mode)
    _validate_state(state, schema, queries, target)
    n_coordinates = len(state.table) * schema.n_blocks()
    if coordinate_index is None:
        coordinate = int(rng.integers(0, n_coordinates))
    else:
        coordinate = _validate_index(
            coordinate_index, n_coordinates, "coordinate_index"
        )
    row_index, attribute_index = divmod(coordinate, schema.n_blocks())
    conditional = build_persistent_heatbath_conditional(
        state,
        schema,
        queries,
        target,
        row_index=row_index,
        attribute_index=attribute_index,
        inverse_temperature=beta,
        energy_mode=mode,
    )
    choice = sample_heatbath_index(
        conditional,
        rng=rng if gumbels is None else None,
        gumbels=gumbels,
    )
    diagnostics = apply_heatbath_choice(state, conditional, choice)
    diagnostics.update({
        "coordinate_index": coordinate,
        "inverse_temperature": beta,
    })
    return diagnostics


def initial_gain_rms_scale(
    state: PersistentHeatbathState,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: np.ndarray,
    *,
    energy_mode: str = ENERGY_MODE_SQUARED,
) -> Dict[str, Any]:
    """枚举初始状态全部合法单坐标编辑，返回非零能量 gain 的 RMS。

    平方模式的能量 gain 是历史定义 ``squared_loss_gain/N``；L1 模式
    直接使用 normalized L1 的变化。默认平方路径保持 Issue #32 的运算顺序。
    """
    mode = _validate_energy_mode(energy_mode)
    target_array, active_queries = _validate_state(
        state, schema, queries, target
    )
    if mode == ENERGY_MODE_NORMALIZED_L1 and len(queries) == 0:
        raise ValueError("normalized_l1 能量要求至少一个查询")
    residual = target_array - state.query_answers.astype(float, copy=False)
    source_l1_energy = (
        float(np.mean(np.abs(residual)) / len(state.table))
        if mode == ENERGY_MODE_NORMALIZED_L1
        else None
    )
    scaled_gains = []
    candidate_evaluations = 0
    query_evaluations = 0
    for row_index in range(len(state.table)):
        for attribute_index in range(schema.n_blocks()):
            values, _, deltas, active_count = _candidate_query_deltas(
                state,
                schema,
                queries,
                active_queries,
                row_index,
                attribute_index,
            )
            delta_float = deltas.astype(float, copy=False)
            if mode == ENERGY_MODE_SQUARED:
                gains = (
                    delta_float @ residual
                    - 0.5 * np.sum(delta_float ** 2, axis=1)
                ) / len(state.table)
            else:
                candidate_energies = (
                    np.mean(
                        np.abs(residual[None, :] - delta_float), axis=1
                    )
                    / len(state.table)
                )
                gains = source_l1_energy - candidate_energies
            nonzero = gains[gains != 0.0]
            scaled_gains.extend(nonzero.tolist())
            candidate_evaluations += len(values)
            query_evaluations += active_count
    values = np.asarray(scaled_gains, dtype=float)
    scale = float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.0
    if not np.isfinite(scale):
        raise ValueError("初始 gain RMS 超出 float64 可表示范围")
    return {
        "scale": scale,
        "nonzero_gain_count": int(len(values)),
        "candidate_state_evaluations": int(candidate_evaluations),
        "query_indicator_evaluations": int(query_evaluations),
    }
