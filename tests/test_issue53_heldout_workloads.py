import json

import pandas as pd
import pytest

from scripts import build_issue53_heldout_workloads as builder
from table_diffevo.quality import query_fingerprint


def _small_marginals():
    return {
        "n_records": 4,
        "attributes": {
            name: {
                "type": "categorical",
                "values": [0, 1],
                "counts": [2, 2],
            }
            for name in ("a", "b", "c", "d")
        },
    }


def test_result_blind_selection_is_deterministic_and_disjoint():
    measured = [{
        "conditions": [
            {"attribute": "a", "operator": "==", "value": 0},
            {"attribute": "b", "operator": "==", "value": 0},
            {"attribute": "c", "operator": "==", "value": 0},
        ]
    }]
    first = builder.select_heldout_queries(
        "tiny",
        _small_marginals(),
        measured,
        per_order_limit=3,
    )
    second = builder.select_heldout_queries(
        "tiny",
        _small_marginals(),
        measured,
        per_order_limit=3,
    )

    assert first == second
    assert first["selected_counts"] == {"3": 3, "4": 3}
    measured_fingerprint = query_fingerprint(measured[0])
    assert measured_fingerprint not in {
        query["fingerprint_sha256"] for query in first["queries"]
    }


def test_attaching_different_answers_does_not_change_selection():
    selection = builder.select_heldout_queries(
        "tiny",
        _small_marginals(),
        [{"conditions": [
            {"attribute": "a", "operator": "==", "value": 0}
        ]}],
        per_order_limit=2,
    )["queries"]
    first_source = pd.DataFrame({
        "a": [0, 0, 0, 0],
        "b": [0, 0, 0, 0],
        "c": [0, 0, 0, 0],
        "d": [0, 0, 0, 0],
    })
    second_source = pd.DataFrame({
        "a": [1, 1, 1, 1],
        "b": [1, 1, 1, 1],
        "c": [1, 1, 1, 1],
        "d": [1, 1, 1, 1],
    })
    first = builder._decorate_with_results(selection, first_source)
    second = builder._decorate_with_results(selection, second_source)

    assert [query_fingerprint(query) for query in first] == [
        query_fingerprint(query) for query in second
    ]
    assert [query["selection_sha256"] for query in first] == [
        query["selection_sha256"] for query in second
    ]
    assert [query["result"] for query in first] != [
        query["result"] for query in second
    ]


@pytest.mark.parametrize("dataset", sorted(builder.DATASETS))
def test_formal_heldout_file_is_exact_deterministic_rebuild(dataset):
    specification = builder.DATASETS[dataset]
    output = specification["output"]
    if not output.exists():
        pytest.skip("正式 held-out 文件尚未物化")
    rebuilt = builder.build_dataset_workload(dataset, specification)
    with output.open(encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored == rebuilt
    assert stored["query_count"] == 1024
    assert stored["construction"]["selected_counts"] == {
        "3": 512,
        "4": 512,
    }
