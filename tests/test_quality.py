import copy
import json

import numpy as np
import pandas as pd
import pytest

from table_diffevo.quality import (
    diversity_metrics,
    evaluate_quality_snapshot,
    query_error_metrics,
    query_fingerprint,
    reference_support_metrics,
    schema_validity_metrics,
    validate_query_partition,
)
from table_diffevo.schema import AttributeBlock, Schema


def _binary_schema(names=("a", "b", "c", "d")):
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in names
    ])


def _cell_query(**values):
    return {
        "conditions": [
            {"attribute": attribute, "operator": "==", "value": value}
            for attribute, value in values.items()
        ]
    }


def test_query_fingerprint_ignores_metadata_and_condition_order():
    first = {
        "id": "first",
        "result": 99,
        "conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 0},
        ],
    }
    reordered = {
        "id": "second",
        "result": 0,
        "conditions": list(reversed(first["conditions"])),
    }
    changed = copy.deepcopy(first)
    changed["conditions"][0]["value"] = 0

    assert query_fingerprint(first) == query_fingerprint(reordered)
    assert query_fingerprint(first) != query_fingerprint(changed)


def test_query_partition_fails_closed_on_duplicates_and_overlap():
    measured = [_cell_query(a=1)]
    with pytest.raises(ValueError, match="重复语义"):
        validate_query_partition(measured * 2, [_cell_query(b=1)])
    with pytest.raises(ValueError, match="语义不相交"):
        validate_query_partition(measured, copy.deepcopy(measured))


def test_query_error_metrics_match_frozen_formulas():
    metrics = query_error_metrics(
        target=[4, 0, 2],
        current=[2, 1, 2],
        n_records=4,
    )
    per_query = np.asarray([0.5, 0.25, 0.0])
    assert metrics["query_count"] == 3
    assert metrics["normalized_l1_mean"] == pytest.approx(per_query.mean())
    assert metrics["normalized_l1_median"] == pytest.approx(0.25)
    assert metrics["normalized_l1_p90"] == pytest.approx(
        np.percentile(per_query, 90)
    )
    assert metrics["normalized_l1_max"] == pytest.approx(0.5)
    assert metrics["squared_loss_diagnostic_only"] == pytest.approx(2.5)


def test_schema_validity_reports_categorical_numeric_and_missing_values():
    schema = Schema([
        AttributeBlock(
            name="category", type="categorical", description="", values=["x", "y"]
        ),
        AttributeBlock(
            name="number", type="numeric", description="", range=[1, 3]
        ),
    ])
    frame = pd.DataFrame({
        "category": ["x", "bad", None, "y"],
        "number": [1.0, 4.0, 2.0, np.nan],
    })
    metrics = schema_validity_metrics(frame, schema)

    assert metrics["valid_row_count"] == 1
    assert metrics["valid_row_rate"] == pytest.approx(0.25)
    assert metrics["invalid_row_count"] == 3
    assert metrics["per_attribute"]["category"]["invalid_count"] == 2
    assert metrics["per_attribute"]["number"]["invalid_count"] == 2

    with pytest.raises(ValueError, match="列必须与 schema 完全一致"):
        schema_validity_metrics(frame[["number", "category"]], schema)


def test_diversity_metrics_cover_unique_and_collapsed_tables():
    schema = _binary_schema(("a", "b"))
    unique = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
    })
    collapsed = pd.DataFrame({"a": [0] * 4, "b": [0] * 4})

    unique_metrics = diversity_metrics(unique, schema)
    assert unique_metrics["unique_row_count"] == 4
    assert unique_metrics["unique_row_rate"] == 1.0
    assert unique_metrics["duplicate_rate"] == 0.0
    assert unique_metrics["effective_unique_rows"] == pytest.approx(4.0)
    assert unique_metrics["effective_unique_row_ratio"] == pytest.approx(1.0)
    assert unique_metrics["attribute_effective_support_ratio_min"] == 1.0

    collapsed_metrics = diversity_metrics(collapsed, schema)
    assert collapsed_metrics["unique_row_count"] == 1
    assert collapsed_metrics["unique_row_rate"] == pytest.approx(0.25)
    assert collapsed_metrics["duplicate_rate"] == pytest.approx(0.75)
    assert collapsed_metrics["effective_unique_rows"] == pytest.approx(1.0)
    assert collapsed_metrics["effective_unique_row_ratio"] == pytest.approx(
        0.25
    )
    assert collapsed_metrics["attribute_effective_support_ratio_min"] == 0.5


def test_reference_support_reports_partial_overlap_mass():
    schema = Schema([
        AttributeBlock(
            name="x",
            type="categorical",
            description="",
            values=["a", "b", "c"],
        )
    ])
    reference = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    synthetic = pd.DataFrame({"x": ["a", "a", "c", "c"]})
    metrics = reference_support_metrics(reference, synthetic, schema)

    assert metrics["reference_unique_rows"] == 2
    assert metrics["synthetic_unique_rows"] == 2
    assert metrics["support_overlap_unique"] == 1
    assert metrics["synthetic_mass_in_reference_support"] == pytest.approx(0.5)
    assert metrics["reference_mass_covered"] == pytest.approx(0.5)


def test_quality_snapshot_is_current_only_deterministic_and_json_safe():
    schema = _binary_schema()
    current = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
        "c": [0, 1, 1, 0],
        "d": [1, 1, 0, 0],
    })
    original = current.copy(deep=True)
    measured = [_cell_query(a=1)]
    heldout = [
        _cell_query(a=1, b=0, c=1),
        _cell_query(a=1, b=0, c=1, d=0),
    ]
    result = evaluate_quality_snapshot(
        current,
        schema,
        measured,
        measured_target=[2],
        heldout_queries=heldout,
        heldout_target=[1, 1],
        reference_table=original,
    )

    pd.testing.assert_frame_equal(current, original)
    assert result["state_role"] == "current"
    assert result["measured"]["normalized_l1_mean"] == 0.0
    assert result["heldout"]["3way"]["query_count"] == 1
    assert result["heldout"]["4way"]["query_count"] == 1
    assert result["heldout"]["combined"]["query_count"] == 2
    assert result["validity"]["valid_row_rate"] == 1.0
    assert result["reference_support_offline_only"][
        "synthetic_mass_in_reference_support"
    ] == 1.0
    json.dumps(result, allow_nan=False)


def test_quality_snapshot_rejects_incomplete_heldout_orders():
    schema = _binary_schema()
    current = pd.DataFrame({name: [0, 1] for name in ("a", "b", "c", "d")})
    with pytest.raises(ValueError, match="3-way 与 4-way"):
        evaluate_quality_snapshot(
            current,
            schema,
            [_cell_query(a=1)],
            [1],
            [_cell_query(a=1, b=1, c=1)],
            [1],
        )
