from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import freeze_issue53_test_query_workload_ab as freeze

ROOT = Path(__file__).resolve().parents[1]


def _load_formal_inputs():
    with (ROOT / freeze.MARGINALS_PATH).open(encoding="utf-8") as handle:
        marginals = json.load(handle)
    with (ROOT / freeze.WORKLOAD_A_PATH).open(encoding="utf-8") as handle:
        old_queries = json.load(handle)["queries"]
    return marginals, old_queries


@pytest.fixture(scope="module")
def formal_freeze():
    marginals, old_queries = _load_formal_inputs()
    return freeze.freeze_query_identities(marginals, old_queries)


def _fingerprints(queries):
    return [freeze.query_fingerprint(query) for query in queries]


def test_selection_is_deterministic_and_ignores_all_answer_like_values():
    marginals, old_queries = _load_formal_inputs()
    changed = copy.deepcopy(marginals)
    for specification in changed["attributes"].values():
        specification["counts"] = [
            10_000 + index for index, _ in enumerate(specification["counts"])
        ]
    changed_queries = copy.deepcopy(old_queries)
    for index, query in enumerate(changed_queries):
        query["result"] = -10_000 - index

    baseline = freeze.freeze_query_identities(marginals, old_queries)
    mutated = freeze.freeze_query_identities(changed, changed_queries)

    assert {
        order: _fingerprints(queries)
        for order, queries in baseline["new_queries"].items()
    } == {
        order: _fingerprints(queries)
        for order, queries in mutated["new_queries"].items()
    }
    assert {
        name: freeze.query_set_identity(queries)
        for name, queries in baseline["evaluation_groups"].items()
    } == {
        name: freeze.query_set_identity(queries)
        for name, queries in mutated["evaluation_groups"].items()
    }


def test_workload_b_is_exactly_30_15_5_and_contains_no_one_way(formal_freeze):
    workload_b = formal_freeze["workload_b"]

    assert len(workload_b) == 50
    assert freeze._order_counts(workload_b) == {2: 30, 3: 15, 4: 5}
    assert freeze.query_set_identity(workload_b) == (
        "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
    )
    assert [query["id"] for query in workload_b[:20]] == [
        f"D{index:02d}" for index in range(1, 21)
    ]
    assert [query["id"] for query in workload_b[30:35]] == [
        f"T{index:02d}" for index in range(1, 6)
    ]
    freeze._assert_result_free(workload_b)


def test_new_query_selection_counts_and_identities_are_frozen(formal_freeze):
    selected = formal_freeze["new_queries"]

    assert {order: len(queries) for order, queries in selected.items()} == {
        2: 10,
        3: 10,
        4: 5,
    }
    assert {
        order: freeze.query_set_identity(queries)
        for order, queries in selected.items()
    } == {
        2: "c87b8dd421c21b799e218c204eb5f3d87c708e3eb40b3fccef3404a32d751681",
        3: "c422642908450206e82bef8b0c6aca474b6e9af61824e833cbcd408a5720fcdd",
        4: "394f9dddd68c38638d81c10f0c3f06d7f2159cafec2c4a772d5f2e856cecbdc6",
    }
    for order, queries in selected.items():
        assert [query["selection_rank"] for query in queries] == list(
            range(1, len(queries) + 1)
        )
        assert all(freeze.query_order(query) == order for query in queries)


def test_common_evaluation_groups_are_exact_and_disjoint(formal_freeze):
    groups = formal_freeze["evaluation_groups"]
    workload_a = formal_freeze["workload_a"]
    workload_b = formal_freeze["workload_b"]

    assert {name: len(queries) for name, queries in groups.items()} == (
        freeze.EXPECTED_EVALUATION_COUNTS
    )
    assert {
        name: freeze.query_set_identity(queries)
        for name, queries in groups.items()
    } == {
        "one_way_safety": (
            "b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c"
        ),
        "common_unseen_2way": (
            "fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2"
        ),
        "fixed_heldout_3way": (
            "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a"
        ),
        "fixed_heldout_4way": (
            "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171"
        ),
    }
    workload_union = {
        freeze.query_fingerprint(query) for query in [*workload_a, *workload_b]
    }
    common_two = set(_fingerprints(groups["common_unseen_2way"]))
    assert workload_union.isdisjoint(common_two)
    for name in ("fixed_heldout_3way", "fixed_heldout_4way"):
        assert workload_union.isdisjoint(_fingerprints(groups[name]))


def test_fixed_heldout_rebuild_matches_stored_query_identities(formal_freeze):
    with (
        ROOT / "configs/test_300x10/heldout_issue53_v1.json"
    ).open(encoding="utf-8") as handle:
        stored = json.load(handle)["queries"]

    rebuilt = [
        *formal_freeze["evaluation_groups"]["fixed_heldout_3way"],
        *formal_freeze["evaluation_groups"]["fixed_heldout_4way"],
    ]
    assert _fingerprints(rebuilt) == _fingerprints(stored)
    assert freeze.query_set_identity(rebuilt) == (
        freeze.EXPECTED_FIXED_HELDOUT_ALL_IDENTITY
    )


def test_formal_identity_artifact_is_exact_result_free_rebuild():
    rebuilt = freeze.build_payload(ROOT)
    with (ROOT / freeze.OUTPUT_PATH).open(encoding="utf-8") as handle:
        stored = json.load(handle)

    assert stored == rebuilt
    assert freeze._sha256_file(ROOT / freeze.PROTOCOL_DOC) == (
        stored["protocol_doc_sha256"]
    )
    assert stored["privacy_boundary"] == {
        "identity_frozen_before_raw_reference_load": True,
        "privacy_budget_consumed": False,
        "query_answers_present": False,
        "raw_reference_data_accessed": False,
    }
    freeze._assert_result_free(stored)


def test_formal_builder_never_opens_a_csv(monkeypatch):
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".csv":
            raise AssertionError(f"identity freeze attempted CSV access: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    payload = freeze.build_payload(ROOT)
    assert payload["selection"]["uses_raw_reference"] is False
