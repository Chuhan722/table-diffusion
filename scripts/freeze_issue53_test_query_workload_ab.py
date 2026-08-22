#!/usr/bin/env python3
"""Freeze the result-blind Issue #53 test query-workload A/B identities.

This module intentionally uses only the Python standard library.  It reads the
public marginal-domain description and the existing measured-query semantics,
but it never imports a table loader, opens the raw reference CSV, evaluates a
query, or copies a ``result`` field into the frozen artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "issue53-test-query-workload-ab-v1"
SELECTION_NAMESPACE = "issue53-test-query-workload-ab-v1"
FIXED_HELDOUT_NAMESPACE = "issue53-heldout-v1"
DATASET = "test_300x10"
N_RECORDS = 300

MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
WORKLOAD_A_PATH = Path("configs/test_300x10/measured_50query.json")
PROTOCOL_DOC = Path("docs/设计/Issue53_test查询workload_AB结果前冻结协议.md")
OUTPUT_PATH = Path("configs/test_300x10/issue53_query_workload_ab_v1.json")

EXPECTED_INPUT_SHA256 = {
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
    "workload_a": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
}
EXPECTED_WORKLOAD_A_IDENTITY = (
    "cbb501f5c2f8c230b6d68d85baf40be7b17be713d41c5b97f54ac30457e90fc8"
)
EXPECTED_FIXED_HELDOUT_IDENTITIES = {
    3: "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a",
    4: "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171",
}
EXPECTED_FIXED_HELDOUT_ALL_IDENTITY = (
    "ab989cccd57586d41b7074e9d86324f207924f92450e08ebf924622cf14f65e8"
)

NEW_QUERY_COUNTS = {2: 10, 3: 10, 4: 5}
EXPECTED_PUBLIC_COUNTS = {2: 548, 3: 5056, 4: 30450}
EXPECTED_OLD_PUBLIC_OVERLAP = {2: 17, 3: 5, 4: 0}
EXPECTED_FIXED_HELDOUT_COUNTS = {3: 512, 4: 512}
EXPECTED_NEW_CANDIDATE_COUNTS = {2: 531, 3: 4539, 4: 29938}
EXPECTED_WORKLOAD_COUNTS = {
    "A": {1: 25, 2: 20, 3: 5},
    "B": {2: 30, 3: 15, 4: 5},
}
EXPECTED_EVALUATION_COUNTS = {
    "one_way_safety": 25,
    "common_unseen_2way": 521,
    "fixed_heldout_3way": 512,
    "fixed_heldout_4way": 512,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def canonical_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    """Return the same result-blind semantic payload as quality.py."""

    if not isinstance(query, dict):
        raise TypeError("query 必须是字典")
    conditions = query.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("query.conditions 必须是非空列表")
    canonical_conditions = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise TypeError("query.conditions 中每项都必须是字典")
        encoded = json.dumps(
            condition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical_conditions.append((encoded, json.loads(encoded)))
    canonical_conditions.sort(key=lambda item: item[0])
    return {"conditions": [item[1] for item in canonical_conditions]}


def query_fingerprint(query: dict[str, Any]) -> str:
    return hashlib.sha256(_strict_json_bytes(canonical_query_payload(query))).hexdigest()


def query_set_identity(queries: Sequence[dict[str, Any]]) -> str:
    if not queries:
        raise ValueError("查询集合不能为空")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("查询集合包含重复语义查询")
    return hashlib.sha256("\n".join(fingerprints).encode("ascii")).hexdigest()


def query_order(query: dict[str, Any]) -> int:
    conditions = query.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("query.conditions 必须是非空列表")
    attributes = [condition.get("attribute") for condition in conditions]
    if any(not isinstance(attribute, str) or not attribute for attribute in attributes):
        raise ValueError("每个 condition 必须包含非空 attribute")
    if len(attributes) != len(set(attributes)):
        raise ValueError("同一查询不得重复约束同一属性")
    return len(attributes)


def _copy_conditions(query: dict[str, Any]) -> list[dict[str, Any]]:
    canonical_query_payload(query)
    return [dict(condition) for condition in query["conditions"]]


def _attribute_cells(
    attribute: str,
    specification: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use domain identities only; deliberately never inspect ``counts``."""

    attribute_type = specification.get("type")
    if attribute_type == "categorical":
        values = specification.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{attribute}.values 必须是非空列表")
        return [
            {"attribute": attribute, "operator": "==", "value": value}
            for value in values
        ]
    if attribute_type == "numeric":
        bins = specification.get("bins")
        if not isinstance(bins, list) or not bins:
            raise ValueError(f"{attribute}.bins 必须是非空列表")
        cells = []
        for interval in bins:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{attribute}.bins 每项必须是 [lower, upper]")
            cells.append({
                "attribute": attribute,
                "operator": "between",
                "lower": interval[0],
                "upper": interval[1],
            })
        return cells
    raise ValueError(f"属性 {attribute!r} 的 type 无效：{attribute_type!r}")


def enumerate_public_cell_queries(
    marginals: dict[str, Any],
    order: int,
) -> list[dict[str, Any]]:
    attributes = marginals.get("attributes")
    if not isinstance(attributes, dict) or not attributes:
        raise ValueError("marginals.attributes 必须是非空字典")
    if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
        raise ValueError("order 必须是正整数")
    if order > len(attributes):
        return []

    queries = []
    for selected_attributes in itertools.combinations(attributes, order):
        domains = [
            _attribute_cells(attribute, attributes[attribute])
            for attribute in selected_attributes
        ]
        for conditions in itertools.product(*domains):
            queries.append({
                "conditions": [dict(condition) for condition in conditions]
            })
    return queries


def _selection_key(
    namespace: str,
    dataset: str,
    order: int,
    fingerprint: str,
) -> str:
    payload = "\0".join((namespace, dataset, str(order), fingerprint)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _condition_expression(condition: dict[str, Any]) -> str:
    operator = condition.get("operator")
    if operator == "==":
        return f"{condition['attribute']} == {condition['value']}"
    if operator == "between":
        return (
            f"{condition['lower']} <= {condition['attribute']} <= "
            f"{condition['upper']}"
        )
    if operator == ">=":
        return f"{condition['attribute']} >= {condition['value']}"
    raise ValueError(f"不支持的 query operator：{operator!r}")


def _expression(query: dict[str, Any]) -> str:
    return " AND ".join(
        _condition_expression(condition) for condition in query["conditions"]
    )


def _order_counts(queries: Sequence[dict[str, Any]]) -> dict[int, int]:
    return dict(sorted(Counter(query_order(query) for query in queries).items()))


def _strip_existing_query(
    query: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    conditions = _copy_conditions(query)
    stripped = {
        "id": query.get("id"),
        "type": query.get("type"),
        "order": query_order(query),
        "source": source,
        "expression": query.get("expression") or _expression(query),
        "fingerprint_sha256": query_fingerprint(query),
        "conditions": conditions,
    }
    if not isinstance(stripped["id"], str) or not stripped["id"]:
        raise ValueError("existing query id 缺失")
    if not isinstance(stripped["type"], str) or not stripped["type"]:
        raise ValueError("existing query type 缺失")
    return stripped


def _select_fixed_heldout(
    marginals: dict[str, Any],
    old_queries: Sequence[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    """Rebuild the already frozen v1 held-out identities without its answers."""

    old_fingerprints = {query_fingerprint(query) for query in old_queries}
    groups: dict[int, list[dict[str, Any]]] = {}
    candidate_counts: dict[int, int] = {}
    for order, limit in EXPECTED_FIXED_HELDOUT_COUNTS.items():
        candidates = []
        seen = set()
        for query in enumerate_public_cell_queries(marginals, order):
            fingerprint = query_fingerprint(query)
            if fingerprint in old_fingerprints or fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append((
                _selection_key(
                    FIXED_HELDOUT_NAMESPACE,
                    DATASET,
                    order,
                    fingerprint,
                ),
                fingerprint,
                query,
            ))
        candidates.sort(key=lambda item: (item[0], item[1]))
        candidate_counts[order] = len(candidates)
        chosen = []
        for rank, (selection_hash, fingerprint, query) in enumerate(
            candidates[:limit], start=1
        ):
            chosen.append({
                "id": f"H{order}_{rank:04d}",
                "type": f"heldout_{order}way",
                "order": order,
                "source": "fixed_issue53_heldout_v1_rebuild",
                "expression": _expression(query),
                "selection_rank": rank,
                "selection_sha256": selection_hash,
                "fingerprint_sha256": fingerprint,
                "conditions": _copy_conditions(query),
            })
        groups[order] = chosen
    return groups, candidate_counts


def _select_new_queries(
    marginals: dict[str, Any],
    old_queries: Sequence[dict[str, Any]],
    fixed_heldout: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    old_fingerprints = {query_fingerprint(query) for query in old_queries}
    fixed_fingerprints = {
        order: {query_fingerprint(query) for query in queries}
        for order, queries in fixed_heldout.items()
    }
    selected: dict[int, list[dict[str, Any]]] = {}
    candidate_counts: dict[int, int] = {}
    type_by_order = {2: "double", 3: "triple", 4: "quadruple"}
    for order, limit in NEW_QUERY_COUNTS.items():
        excluded = old_fingerprints | fixed_fingerprints.get(order, set())
        candidates = []
        seen = set()
        for query in enumerate_public_cell_queries(marginals, order):
            fingerprint = query_fingerprint(query)
            if fingerprint in excluded or fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append((
                _selection_key(
                    SELECTION_NAMESPACE,
                    DATASET,
                    order,
                    fingerprint,
                ),
                fingerprint,
                query,
            ))
        candidates.sort(key=lambda item: (item[0], item[1]))
        candidate_counts[order] = len(candidates)
        chosen = []
        for rank, (selection_hash, fingerprint, query) in enumerate(
            candidates[:limit], start=1
        ):
            chosen.append({
                "id": f"N{order}_{rank:02d}",
                "type": type_by_order[order],
                "order": order,
                "source": "result_blind_public_cell",
                "expression": _expression(query),
                "selection_rank": rank,
                "selection_sha256": selection_hash,
                "fingerprint_sha256": fingerprint,
                "conditions": _copy_conditions(query),
            })
        selected[order] = chosen
    return selected, candidate_counts


def _freeze_common_unseen_2way(
    marginals: dict[str, Any],
    workload_a: Sequence[dict[str, Any]],
    workload_b: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    measured_union = {
        query_fingerprint(query) for query in [*workload_a, *workload_b]
    }
    selected = []
    for query in enumerate_public_cell_queries(marginals, 2):
        fingerprint = query_fingerprint(query)
        if fingerprint in measured_union:
            continue
        selected.append({
            "id": f"E2_{len(selected) + 1:04d}",
            "type": "common_unseen_2way_cell",
            "order": 2,
            "source": "public_cells_excluding_semantic_union_A_B",
            "expression": _expression(query),
            "fingerprint_sha256": fingerprint,
            "conditions": _copy_conditions(query),
        })
    return selected


def _assert_disjoint(
    first: Iterable[dict[str, Any]],
    second: Iterable[dict[str, Any]],
    description: str,
) -> None:
    first_set = {query_fingerprint(query) for query in first}
    second_set = {query_fingerprint(query) for query in second}
    overlap = first_set & second_set
    if overlap:
        raise RuntimeError(f"{description} 存在 {len(overlap)} 条语义重叠")


def _assert_result_free(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "result":
                raise RuntimeError(f"结果前冻结产物不得包含 result：{path}.{key}")
            _assert_result_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_result_free(child, f"{path}[{index}]")


def freeze_query_identities(
    marginals: dict[str, Any],
    old_queries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return query identities without inspecting marginal counts or answers."""

    if marginals.get("n_records") != N_RECORDS:
        raise RuntimeError("marginals n_records 漂移")
    if len(old_queries) != 50:
        raise RuntimeError("workload A 必须恰好包含 50 条查询")
    if query_set_identity(old_queries) != EXPECTED_WORKLOAD_A_IDENTITY:
        raise RuntimeError("workload A 查询语义或顺序漂移")

    public_counts = {
        order: len(enumerate_public_cell_queries(marginals, order))
        for order in NEW_QUERY_COUNTS
    }
    if public_counts != EXPECTED_PUBLIC_COUNTS:
        raise RuntimeError(
            f"公开 cell 数量漂移：expected={EXPECTED_PUBLIC_COUNTS}, "
            f"observed={public_counts}"
        )

    old_fingerprints = {query_fingerprint(query) for query in old_queries}
    old_public_overlap = {
        order: sum(
            query_fingerprint(query) in old_fingerprints
            for query in enumerate_public_cell_queries(marginals, order)
        )
        for order in NEW_QUERY_COUNTS
    }
    if old_public_overlap != EXPECTED_OLD_PUBLIC_OVERLAP:
        raise RuntimeError("workload A 与公开 cell 的精确重叠数漂移")

    fixed_heldout, heldout_candidate_counts = _select_fixed_heldout(
        marginals,
        old_queries,
    )
    heldout_identities = {
        order: query_set_identity(queries)
        for order, queries in fixed_heldout.items()
    }
    if heldout_identities != EXPECTED_FIXED_HELDOUT_IDENTITIES:
        raise RuntimeError("固定 held-out 身份重建漂移")
    fixed_all = [
        query
        for order in EXPECTED_FIXED_HELDOUT_COUNTS
        for query in fixed_heldout[order]
    ]
    if query_set_identity(fixed_all) != EXPECTED_FIXED_HELDOUT_ALL_IDENTITY:
        raise RuntimeError("固定 held-out 合集身份重建漂移")

    new_queries, new_candidate_counts = _select_new_queries(
        marginals,
        old_queries,
        fixed_heldout,
    )
    if new_candidate_counts != EXPECTED_NEW_CANDIDATE_COUNTS:
        raise RuntimeError(
            f"新查询候选数漂移：expected={EXPECTED_NEW_CANDIDATE_COUNTS}, "
            f"observed={new_candidate_counts}"
        )
    if {order: len(items) for order, items in new_queries.items()} != NEW_QUERY_COUNTS:
        raise RuntimeError("新查询选取数量漂移")

    workload_a = [
        _strip_existing_query(query, source="existing_workload_A")
        for query in old_queries
    ]
    retained = {
        order: [
            _strip_existing_query(query, source="retained_from_workload_A")
            for query in old_queries
            if query_order(query) == order
        ]
        for order in (2, 3)
    }
    workload_b = [
        *retained[2],
        *new_queries[2],
        *retained[3],
        *new_queries[3],
        *new_queries[4],
    ]
    observed_workload_counts = {
        "A": _order_counts(workload_a),
        "B": _order_counts(workload_b),
    }
    if observed_workload_counts != EXPECTED_WORKLOAD_COUNTS:
        raise RuntimeError(
            f"A/B 阶数构成漂移：expected={EXPECTED_WORKLOAD_COUNTS}, "
            f"observed={observed_workload_counts}"
        )
    query_set_identity(workload_b)

    one_way_safety = [
        _strip_existing_query(query, source="workload_A_one_way_safety")
        for query in old_queries
        if query_order(query) == 1
    ]
    common_unseen_2way = _freeze_common_unseen_2way(
        marginals,
        workload_a,
        workload_b,
    )
    evaluation_groups = {
        "one_way_safety": one_way_safety,
        "common_unseen_2way": common_unseen_2way,
        "fixed_heldout_3way": fixed_heldout[3],
        "fixed_heldout_4way": fixed_heldout[4],
    }
    observed_evaluation_counts = {
        name: len(queries) for name, queries in evaluation_groups.items()
    }
    if observed_evaluation_counts != EXPECTED_EVALUATION_COUNTS:
        raise RuntimeError(
            f"公共评价组数量漂移：expected={EXPECTED_EVALUATION_COUNTS}, "
            f"observed={observed_evaluation_counts}"
        )

    _assert_disjoint(workload_a, common_unseen_2way, "A 与公共未见 2-way")
    _assert_disjoint(workload_b, common_unseen_2way, "B 与公共未见 2-way")
    for order in (3, 4):
        _assert_disjoint(workload_a, fixed_heldout[order], f"A 与 held-out {order}-way")
        _assert_disjoint(workload_b, fixed_heldout[order], f"B 与 held-out {order}-way")

    frozen = {
        "workload_a": workload_a,
        "workload_b": workload_b,
        "new_queries": new_queries,
        "evaluation_groups": evaluation_groups,
        "audit": {
            "public_counts": public_counts,
            "old_public_overlap": old_public_overlap,
            "heldout_candidate_counts": heldout_candidate_counts,
            "new_candidate_counts": new_candidate_counts,
            "workload_order_counts": observed_workload_counts,
            "evaluation_group_counts": observed_evaluation_counts,
        },
    }
    _assert_result_free(frozen)
    return frozen


def _group_metadata(queries: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    return {
        "query_count": len(queries),
        "query_identity_sha256": query_set_identity(queries),
        "policy": policy,
    }


def build_payload(root: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if root is None else root
    input_paths = {
        "marginals": root / MARGINALS_PATH,
        "workload_a": root / WORKLOAD_A_PATH,
    }
    input_sha256 = {
        name: _sha256_file(path) for name, path in input_paths.items()
    }
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"冻结输入 SHA 漂移：expected={EXPECTED_INPUT_SHA256}, "
            f"observed={input_sha256}"
        )
    protocol_doc_sha256 = _sha256_file(root / PROTOCOL_DOC)
    marginals = _load_json(input_paths["marginals"])
    measured_payload = _load_json(input_paths["workload_a"])
    old_queries = measured_payload.get("queries")
    if not isinstance(old_queries, list):
        raise TypeError("workload A queries 必须是列表")
    frozen = freeze_query_identities(marginals, old_queries)

    evaluation_groups = frozen["evaluation_groups"]
    payload = {
        "contract_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "record_count": N_RECORDS,
        "description": (
            "Issue #53 result-blind test query-workload A/B identity freeze; "
            "contains no query answers"
        ),
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": protocol_doc_sha256,
        "input_sha256": input_sha256,
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "uses_domain_types_values_bins_only": True,
            "uses_marginal_counts": False,
            "uses_query_results": False,
            "uses_raw_reference": False,
            "uses_terminal_errors": False,
            "fixed_heldout_namespace": FIXED_HELDOUT_NAMESPACE,
            "public_cell_counts": {
                str(key): value for key, value in frozen["audit"]["public_counts"].items()
            },
            "old_public_overlap_counts": {
                str(key): value
                for key, value in frozen["audit"]["old_public_overlap"].items()
            },
            "fixed_heldout_counts": {
                str(key): value for key, value in EXPECTED_FIXED_HELDOUT_COUNTS.items()
            },
            "new_candidate_counts": {
                str(key): value
                for key, value in frozen["audit"]["new_candidate_counts"].items()
            },
            "new_selected_counts": {
                str(key): value for key, value in NEW_QUERY_COUNTS.items()
            },
        },
        "workload_a": {
            "source": str(WORKLOAD_A_PATH),
            "query_count": len(frozen["workload_a"]),
            "order_counts": {
                str(key): value
                for key, value in frozen["audit"]["workload_order_counts"]["A"].items()
            },
            "query_identity_sha256": query_set_identity(frozen["workload_a"]),
        },
        "workload_b": {
            "answer_state": "identities_only_no_result_fields",
            "query_count": len(frozen["workload_b"]),
            "order_counts": {
                str(key): value
                for key, value in frozen["audit"]["workload_order_counts"]["B"].items()
            },
            "query_identity_sha256": query_set_identity(frozen["workload_b"]),
            "queries": frozen["workload_b"],
        },
        "evaluation_groups": {
            "one_way_safety": _group_metadata(
                evaluation_groups["one_way_safety"],
                "existing_25_one_way_queries_safety_only",
            ),
            "common_unseen_2way": _group_metadata(
                evaluation_groups["common_unseen_2way"],
                "all_548_public_cells_excluding_semantic_union_of_A_and_B",
            ),
            "fixed_heldout_3way": _group_metadata(
                evaluation_groups["fixed_heldout_3way"],
                "rebuild_existing_issue53_heldout_v1_without_answers",
            ),
            "fixed_heldout_4way": _group_metadata(
                evaluation_groups["fixed_heldout_4way"],
                "rebuild_existing_issue53_heldout_v1_without_answers",
            ),
            "cross_group_aggregate_allowed": False,
        },
        "privacy_boundary": {
            "identity_frozen_before_raw_reference_load": True,
            "raw_reference_data_accessed": False,
            "query_answers_present": False,
            "privacy_budget_consumed": False,
        },
    }
    _assert_result_free(payload)
    return payload


def serialize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialize_payload(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="确定性重建并与已存在的正式身份文件逐字段比较",
    )
    args = parser.parse_args()
    root = _repo_root()
    payload = build_payload(root)
    output = root / OUTPUT_PATH
    if args.verify_existing:
        if _load_json(output) != payload:
            raise RuntimeError(f"正式身份文件无法确定性重建：{output}")
        action = "verified"
    else:
        _write_new(output, payload)
        action = "wrote"
    print(
        f"{action} {OUTPUT_PATH} sha256={_sha256_file(output)} "
        f"workload_b={payload['workload_b']['query_identity_sha256']}"
    )


if __name__ == "__main__":
    main()
