from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import freeze_issue53_test_query_workload_ab as freeze
from scripts import materialize_issue53_test_query_workload_b as materialize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_payload():
    return materialize.build_payload(ROOT)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_pre_reference_audit_never_opens_csv(monkeypatch):
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".csv":
            raise AssertionError(f"pre-reference audit opened CSV: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    audit = materialize._audit_identities_before_reference(ROOT)
    assert audit["identity_frozen_before_reference_load"] is True
    assert audit["workload_b_identity_sha256"] == (
        materialize.WORKLOAD_B_IDENTITY_SHA256
    )


def test_stdlib_evaluator_exactly_reproduces_all_old_workload_answers():
    rows = materialize._load_reference(ROOT / materialize.REFERENCE_PATH)
    old_queries = _load_json(ROOT / freeze.WORKLOAD_A_PATH)["queries"]

    observed = materialize.evaluate_queries(rows, old_queries)
    expected = [query["result"] for query in old_queries]

    assert observed == expected
    assert len(observed) == 50


def test_formal_workload_b_is_exact_deterministic_rebuild(formal_payload):
    stored = _load_json(ROOT / materialize.OUTPUT_PATH)

    assert stored == formal_payload
    assert freeze._sha256_file(ROOT / materialize.OUTPUT_PATH) == (
        "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
    )
    assert stored["construction"]["identity_artifact_sha256"] == (
        materialize.IDENTITY_ARTIFACT_SHA256
    )
    assert stored["construction"]["reference_sha256"] == (
        materialize.REFERENCE_SHA256
    )
    assert stored["construction"]["raw_reference_data_accessed"] is True
    assert stored["construction"]["privacy_budget_consumed"] is False


def test_answer_attachment_preserves_every_query_identity(formal_payload):
    identity = _load_json(ROOT / freeze.OUTPUT_PATH)
    frozen_queries = identity["workload_b"]["queries"]
    answered_queries = formal_payload["queries"]

    assert len(frozen_queries) == len(answered_queries) == 50
    assert freeze.query_set_identity(frozen_queries) == (
        materialize.WORKLOAD_B_IDENTITY_SHA256
    )
    assert freeze.query_set_identity(answered_queries) == (
        materialize.WORKLOAD_B_IDENTITY_SHA256
    )
    assert [
        {key: value for key, value in query.items() if key != "result"}
        for query in answered_queries
    ] == frozen_queries
    assert all(
        isinstance(query["result"], int) and not isinstance(query["result"], bool)
        for query in answered_queries
    )


def test_answered_workload_remains_exactly_30_15_5(formal_payload):
    queries = formal_payload["queries"]

    assert freeze._order_counts(queries) == {2: 30, 3: 15, 4: 5}
    assert all(freeze.query_order(query) >= 2 for query in queries)
    assert formal_payload["query_count"] == 50
    assert formal_payload["record_count"] == 300


def test_target_vector_identity_is_frozen_and_results_do_not_affect_queries(
    formal_payload,
):
    queries = formal_payload["queries"]
    answers = [query["result"] for query in queries]

    assert materialize._target_vector_sha256(answers) == (
        "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
    )
    changed = copy.deepcopy(queries)
    for query in changed:
        query["result"] += 1
    assert freeze.query_set_identity(changed) == freeze.query_set_identity(queries)
