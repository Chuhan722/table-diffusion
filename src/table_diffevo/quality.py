"""Issue #53 current-state quality evaluation contract.

The functions in this module are deliberately offline and deterministic.  They
evaluate a materialized current table, never participate in generation control,
and never read source data by path.  A caller that requests reference-support
diagnostics must load the reference table only after generation has finished.
"""

from collections import Counter
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table_diffevo.metrics import compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


QUALITY_CONTRACT_VERSION = "issue53-stage0-v1"


def canonical_query_payload(query: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical semantic payload of a conjunction query.

    Query metadata such as ``id``, ``expression``, ``type`` and ``result`` is
    intentionally ignored.  Conjunction order is semantically irrelevant, so
    conditions are ordered by canonical JSON before hashing.
    """
    if not isinstance(query, dict):
        raise ValueError("query 必须是字典")
    conditions = query.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("query.conditions 必须是非空列表")

    canonical_conditions = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("query.conditions 中每项都必须是字典")
        try:
            encoded = json.dumps(
                condition,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("query condition 必须可严格 JSON 序列化") from exc
        canonical_conditions.append((encoded, json.loads(encoded)))
    canonical_conditions.sort(key=lambda item: item[0])
    return {"conditions": [item[1] for item in canonical_conditions]}


def query_fingerprint(query: Dict[str, Any]) -> str:
    """Return a SHA-256 identity based only on query semantics."""
    payload = canonical_query_payload(query)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _query_set_identity(
    queries: Sequence[Dict[str, Any]],
    name: str,
) -> Tuple[List[str], str]:
    if not isinstance(queries, (list, tuple)) or not queries:
        raise ValueError(f"{name} queries 必须是非空列表")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(f"{name} queries 包含重复语义查询")
    identity = hashlib.sha256(
        "\n".join(fingerprints).encode("ascii")
    ).hexdigest()
    return fingerprints, identity


def validate_query_partition(
    measured_queries: Sequence[Dict[str, Any]],
    heldout_queries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate duplicate-free, disjoint measured and held-out workloads."""
    measured, measured_identity = _query_set_identity(
        measured_queries, "measured"
    )
    heldout, heldout_identity = _query_set_identity(
        heldout_queries, "heldout"
    )
    overlap = set(measured).intersection(heldout)
    if overlap:
        raise ValueError(
            "measured 与 heldout queries 必须语义不相交，"
            f"发现 {len(overlap)} 个重复查询"
        )
    return {
        "measured_query_count": len(measured),
        "heldout_query_count": len(heldout),
        "measured_query_identity_sha256": measured_identity,
        "heldout_query_identity_sha256": heldout_identity,
        "overlap_count": 0,
    }


def query_error_metrics(
    target: Sequence[float],
    current: Sequence[float],
    n_records: int,
) -> Dict[str, Any]:
    """Compute the frozen per-query L1 distribution and squared diagnostic."""
    target_values = np.asarray(target, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if target_values.ndim != 1 or current_values.ndim != 1:
        raise ValueError("target 与 current 必须是一维数组")
    if target_values.shape != current_values.shape:
        raise ValueError(
            "target 与 current 形状不一致: "
            f"{target_values.shape} vs {current_values.shape}"
        )
    if len(target_values) == 0:
        raise ValueError("查询集合不能为空")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("target 必须全部为有限数值")
    if not np.all(np.isfinite(current_values)):
        raise ValueError("current 必须全部为有限数值")
    if isinstance(n_records, bool) or not isinstance(
        n_records, (int, np.integer)
    ) or n_records <= 0:
        raise ValueError("n_records 必须是正整数")

    per_query = np.abs(target_values - current_values) / int(n_records)
    return {
        "query_count": int(len(per_query)),
        "normalized_l1_mean": float(np.mean(per_query)),
        "normalized_l1_median": float(np.median(per_query)),
        "normalized_l1_p90": float(np.percentile(per_query, 90)),
        "normalized_l1_max": float(np.max(per_query)),
        "squared_loss_diagnostic_only": float(
            compute_squared_loss(target_values, current_values)
        ),
    }


def _validate_columns(frame: pd.DataFrame, schema: Schema, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} 必须是 pandas DataFrame")
    expected = schema.attribute_names()
    observed = list(frame.columns)
    if observed != expected:
        raise ValueError(
            f"{name} 列必须与 schema 完全一致，得到 {observed}，期望 {expected}"
        )
    if len(frame) == 0:
        raise ValueError(f"{name} 不能为空表")


def _attribute_valid_mask(
    series: pd.Series,
    block: AttributeBlock,
) -> np.ndarray:
    present = series.notna().to_numpy(dtype=bool)
    if block.is_categorical():
        allowed = series.isin(block.values).to_numpy(dtype=bool)
        return present & allowed
    if not block.is_numeric():
        raise ValueError(f"schema 属性 {block.name!r} 类型无效: {block.type!r}")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return np.zeros(len(series), dtype=bool)
    values = series.to_numpy(dtype=float, na_value=np.nan)
    low, high = map(float, block.range)
    return present & np.isfinite(values) & (values >= low) & (values <= high)


def schema_validity_metrics(
    frame: pd.DataFrame,
    schema: Schema,
) -> Dict[str, Any]:
    """Report schema validity without repairing invalid values."""
    _validate_columns(frame, schema, "current_table")
    valid_rows = np.ones(len(frame), dtype=bool)
    per_attribute = {}
    for block in schema.attributes:
        mask = _attribute_valid_mask(frame[block.name], block)
        valid_rows &= mask
        invalid_count = int((~mask).sum())
        per_attribute[block.name] = {
            "type": block.type,
            "invalid_count": invalid_count,
            "valid_rate": float(mask.mean()),
        }
    valid_count = int(valid_rows.sum())
    return {
        "row_count": int(len(frame)),
        "valid_row_count": valid_count,
        "valid_row_rate": float(valid_count / len(frame)),
        "invalid_row_count": int(len(frame) - valid_count),
        "per_attribute": per_attribute,
    }


def _entropy_from_counts(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=float)
    if len(values) == 0 or float(values.sum()) <= 0.0:
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _public_domain_size(block: AttributeBlock) -> int:
    if block.is_categorical():
        return len(block.values)
    low, high = block.range
    return int(high) - int(low) + 1


def diversity_metrics(frame: pd.DataFrame, schema: Schema) -> Dict[str, Any]:
    """Compute source-free row and per-attribute diversity diagnostics."""
    _validate_columns(frame, schema, "current_table")
    row_counts = frame.value_counts(dropna=False, sort=False).to_numpy()
    row_entropy = _entropy_from_counts(row_counts)
    effective_rows = float(np.exp(row_entropy))

    per_attribute = {}
    support_ratios = []
    for block in schema.attributes:
        mask = _attribute_valid_mask(frame[block.name], block)
        valid = frame.loc[mask, block.name]
        counts = valid.value_counts(dropna=False, sort=False).to_numpy()
        entropy = _entropy_from_counts(counts)
        effective_support = float(np.exp(entropy)) if len(valid) else 0.0
        domain_size = _public_domain_size(block)
        ratio = (
            float(effective_support / domain_size)
            if domain_size > 0 else 0.0
        )
        support_ratios.append(ratio)
        per_attribute[block.name] = {
            "valid_value_count": int(len(valid)),
            "public_domain_size": int(domain_size),
            "empirical_entropy": entropy,
            "effective_support": effective_support,
            "effective_support_ratio": ratio,
        }

    unique_count = int(len(row_counts))
    return {
        "row_count": int(len(frame)),
        "unique_row_count": unique_count,
        "unique_row_rate": float(unique_count / len(frame)),
        "duplicate_rate": float(1.0 - unique_count / len(frame)),
        "empirical_row_entropy": row_entropy,
        "effective_unique_rows": effective_rows,
        "effective_unique_row_ratio": float(effective_rows / len(frame)),
        "attribute_effective_support_ratio_mean": float(
            np.mean(support_ratios)
        ),
        "attribute_effective_support_ratio_min": float(
            np.min(support_ratios)
        ),
        "per_attribute": per_attribute,
    }


def _row_keys(frame: pd.DataFrame, schema: Schema) -> List[Tuple[Any, ...]]:
    keys = []
    blocks = schema.attributes
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value, block in zip(row, blocks):
            if pd.isna(value):
                values.append(("missing", None))
            elif block.is_numeric():
                try:
                    values.append(("numeric", float(value)))
                except (TypeError, ValueError):
                    values.append(("invalid_numeric", repr(value)))
            else:
                values.append(("categorical", str(value)))
        keys.append(tuple(values))
    return keys


def reference_support_metrics(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
) -> Dict[str, Any]:
    """Compare raw empirical joint support in an explicit offline step."""
    _validate_columns(reference, schema, "reference_table")
    _validate_columns(synthetic, schema, "current_table")
    reference_counts = Counter(_row_keys(reference, schema))
    synthetic_counts = Counter(_row_keys(synthetic, schema))
    reference_support = set(reference_counts)
    synthetic_support = set(synthetic_counts)
    overlap = reference_support.intersection(synthetic_support)
    synthetic_in_reference = sum(
        count for key, count in synthetic_counts.items()
        if key in reference_support
    )
    reference_covered = sum(
        count for key, count in reference_counts.items()
        if key in synthetic_support
    )
    return {
        "reference_row_count": int(len(reference)),
        "synthetic_row_count": int(len(synthetic)),
        "reference_unique_rows": int(len(reference_support)),
        "synthetic_unique_rows": int(len(synthetic_support)),
        "support_overlap_unique": int(len(overlap)),
        "synthetic_mass_in_reference_support": float(
            synthetic_in_reference / len(synthetic)
        ),
        "reference_mass_covered": float(
            reference_covered / len(reference)
        ),
    }


def _table_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def evaluate_quality_snapshot(
    current_table: pd.DataFrame,
    schema: Schema,
    measured_queries: Sequence[Dict[str, Any]],
    measured_target: Sequence[float],
    heldout_queries: Sequence[Dict[str, Any]],
    heldout_target: Sequence[float],
    *,
    reference_table: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate one materialized current state under the Stage 0 contract."""
    _validate_columns(current_table, schema, "current_table")
    partition = validate_query_partition(measured_queries, heldout_queries)
    heldout_orders = np.asarray([
        len(query["conditions"]) for query in heldout_queries
    ], dtype=int)
    if set(heldout_orders.tolist()) != {3, 4}:
        raise ValueError("heldout queries 必须同时且只包含 3-way 与 4-way")

    measured_answers = evaluate_table(current_table, list(measured_queries))
    heldout_answers = evaluate_table(current_table, list(heldout_queries))
    heldout_target_values = np.asarray(heldout_target, dtype=float)
    if heldout_target_values.shape != heldout_answers.shape:
        raise ValueError(
            "heldout target 与 queries 数量不一致: "
            f"{heldout_target_values.shape} vs {heldout_answers.shape}"
        )

    heldout_metrics = {}
    for order in (3, 4):
        selected = heldout_orders == order
        heldout_metrics[f"{order}way"] = query_error_metrics(
            heldout_target_values[selected],
            heldout_answers[selected],
            len(current_table),
        )
    heldout_metrics["combined"] = query_error_metrics(
        heldout_target_values,
        heldout_answers,
        len(current_table),
    )

    result = {
        "contract_version": QUALITY_CONTRACT_VERSION,
        "state_role": "current",
        "table_sha256": _table_sha256(current_table),
        "n_records": int(len(current_table)),
        "query_partition": partition,
        "measured": query_error_metrics(
            measured_target,
            measured_answers,
            len(current_table),
        ),
        "heldout": heldout_metrics,
        "validity": schema_validity_metrics(current_table, schema),
        "diversity": diversity_metrics(current_table, schema),
        "reference_support_offline_only": (
            reference_support_metrics(
                reference_table, current_table, schema
            )
            if reference_table is not None else None
        ),
    }
    # Fail closed now rather than allowing a later writer to emit NaN/Infinity.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
