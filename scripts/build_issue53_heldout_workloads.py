#!/usr/bin/env python3
"""Build deterministic, result-blind held-out workloads for Issue #53.

Selection uses only public domains and measured-query identities.  The source
table is loaded only after the selected query identities have been frozen in
memory, and is then used solely to attach exact answers for offline evaluation.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from table_diffevo.quality import (
    query_fingerprint,
    validate_query_partition,
)
from table_diffevo.queries import evaluate_table, load_data, load_queries
from table_diffevo.schema import load_schema


NAMESPACE = "issue53-heldout-v1"
ORDERS = (3, 4)
PER_ORDER_LIMIT = 512

DATASETS = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "measured": Path("configs/test_300x10/measured_50query.json"),
        "source": Path("data/test_300x10/test_300x10.csv"),
        "output": Path("configs/test_300x10/heldout_issue53_v1.json"),
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "measured": Path("configs/nltcs/measured_1000query.json"),
        "source": Path("data/nltcs/nltcs.csv"),
        "output": Path("configs/nltcs/heldout_issue53_v1.json"),
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _attribute_cells(
    attribute: str,
    specification: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if specification["type"] == "categorical":
        return [
            {
                "attribute": attribute,
                "operator": "==",
                "value": value,
            }
            for value in specification["values"]
        ]
    if specification["type"] == "numeric":
        return [
            {
                "attribute": attribute,
                "operator": "between",
                "lower": lower,
                "upper": upper,
            }
            for lower, upper in specification["bins"]
        ]
    raise ValueError(
        f"属性 {attribute!r} 的 marginals type 无效: "
        f"{specification['type']!r}"
    )


def enumerate_public_cell_queries(
    marginals: Dict[str, Any],
    order: int,
) -> List[Dict[str, Any]]:
    """Enumerate conjunction cells using only public marginal domains."""
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
    dataset: str,
    order: int,
    fingerprint: str,
) -> str:
    payload = "\0".join(
        (NAMESPACE, dataset, str(order), fingerprint)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_heldout_queries(
    dataset: str,
    marginals: Dict[str, Any],
    measured_queries: Sequence[Dict[str, Any]],
    *,
    orders: Sequence[int] = ORDERS,
    per_order_limit: int = PER_ORDER_LIMIT,
) -> Dict[str, Any]:
    """Freeze selected query identities without reading any query answers."""
    if not dataset:
        raise ValueError("dataset 不能为空")
    if (
        isinstance(per_order_limit, bool)
        or not isinstance(per_order_limit, int)
        or per_order_limit <= 0
    ):
        raise ValueError("per_order_limit 必须是正整数")
    measured_fingerprints = [
        query_fingerprint(query) for query in measured_queries
    ]
    if len(set(measured_fingerprints)) != len(measured_fingerprints):
        raise ValueError("measured queries 包含重复语义查询")
    measured_set = set(measured_fingerprints)

    selected = []
    candidate_counts = {}
    selected_counts = {}
    for order in orders:
        candidates = []
        seen = set()
        for query in enumerate_public_cell_queries(marginals, int(order)):
            fingerprint = query_fingerprint(query)
            if fingerprint in measured_set or fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append((
                _selection_key(dataset, int(order), fingerprint),
                fingerprint,
                query,
            ))
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[:per_order_limit]
        candidate_counts[str(order)] = len(candidates)
        selected_counts[str(order)] = len(chosen)
        for rank, (selection_hash, fingerprint, query) in enumerate(
            chosen, start=1
        ):
            selected.append({
                "order": int(order),
                "selection_rank": rank,
                "selection_sha256": selection_hash,
                "fingerprint_sha256": fingerprint,
                "conditions": query["conditions"],
            })

    partition = validate_query_partition(measured_queries, selected)
    return {
        "queries": selected,
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "partition": partition,
    }


def _condition_expression(condition: Dict[str, Any]) -> str:
    if condition["operator"] == "==":
        return f"{condition['attribute']} == {condition['value']}"
    if condition["operator"] == "between":
        return (
            f"{condition['lower']} <= {condition['attribute']} <= "
            f"{condition['upper']}"
        )
    raise ValueError(f"held-out condition 算子无效: {condition['operator']}")


def _decorate_with_results(
    selected_queries: Sequence[Dict[str, Any]],
    source,
) -> List[Dict[str, Any]]:
    answers = evaluate_table(source, list(selected_queries))
    result = []
    for query, answer in zip(selected_queries, answers):
        order = int(query["order"])
        rank = int(query["selection_rank"])
        conditions = [dict(item) for item in query["conditions"]]
        result.append({
            "id": f"H{order}_{rank:04d}",
            "type": f"heldout_{order}way",
            "order": order,
            "expression": " AND ".join(
                _condition_expression(condition)
                for condition in conditions
            ),
            "selection_rank": rank,
            "selection_sha256": query["selection_sha256"],
            "fingerprint_sha256": query["fingerprint_sha256"],
            "conditions": conditions,
            "result": int(answer),
        })
    return result


def build_dataset_workload(
    dataset: str,
    specification: Dict[str, Path],
) -> Dict[str, Any]:
    """Build one complete payload, keeping selection before source loading."""
    marginals = _load_json(specification["marginals"])
    measured_queries = load_queries(str(specification["measured"]))
    selection = select_heldout_queries(
        dataset,
        marginals,
        measured_queries,
    )

    # Privacy boundary: query identities are now frozen.  Only answers below
    # depend on the source table.
    schema = load_schema(str(specification["schema"]))
    source = load_data(str(specification["source"]))
    if list(source.columns) != schema.attribute_names():
        raise ValueError(f"{dataset} source 列与 schema 不一致")
    expected_records = int(marginals["n_records"])
    if len(source) != expected_records:
        raise ValueError(
            f"{dataset} source 行数 {len(source)} 与 marginals "
            f"n_records={expected_records} 不一致"
        )
    queries = _decorate_with_results(selection["queries"], source)
    query_identity = hashlib.sha256(
        "\n".join(query["fingerprint_sha256"] for query in queries)
        .encode("ascii")
    ).hexdigest()
    return {
        "dataset": dataset,
        "record_count": expected_records,
        "query_count": len(queries),
        "result_unit": "records",
        "description": (
            "Issue #53 result-blind held-out 3-way/4-way exact query answers; "
            "evaluation only, never a generation input"
        ),
        "construction": {
            "namespace": NAMESPACE,
            "orders": list(ORDERS),
            "per_order_limit": PER_ORDER_LIMIT,
            "candidate_counts": selection["candidate_counts"],
            "selected_counts": selection["selected_counts"],
            "query_identity_sha256": query_identity,
            "measured_query_count": selection["partition"][
                "measured_query_count"
            ],
            "measured_query_identity_sha256": selection["partition"][
                "measured_query_identity_sha256"
            ],
            "heldout_query_identity_sha256": selection["partition"][
                "heldout_query_identity_sha256"
            ],
            "input_sha256": {
                "schema": _sha256_file(specification["schema"]),
                "marginals": _sha256_file(specification["marginals"]),
                "measured_queries": _sha256_file(specification["measured"]),
                "source": _sha256_file(specification["source"]),
            },
        },
        "queries": queries,
    }


def _serialize_payload(payload: Dict[str, Any]) -> str:
    """Serialize valid JSON with one auditable query per line."""
    queries = payload["queries"]
    metadata = {
        key: value for key, value in payload.items() if key != "queries"
    }
    lines = ["{"]
    for key, value in metadata.items():
        encoded_key = json.dumps(key, ensure_ascii=False)
        encoded_value = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(f"  {encoded_key}: {encoded_value},")
    lines.append('  "queries": [')
    for index, query in enumerate(queries):
        suffix = "," if index + 1 < len(queries) else ""
        encoded_query = json.dumps(
            query,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(f"    {encoded_query}{suffix}")
    lines.extend(["  ]", "}"])
    return "\n".join(lines) + "\n"


def _write_payload(
    path: Path,
    payload: Dict[str, Any],
    *,
    overwrite_identical: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_identical:
        if not path.exists() or _load_json(path) != payload:
            raise RuntimeError(
                f"只允许重排与确定性重建内容逐字段相同的文件：{path}"
            )
        path.write_text(_serialize_payload(payload), encoding="utf-8")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(_serialize_payload(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="确定性重建并与已存在的正式文件逐字段比较",
    )
    parser.add_argument(
        "--rewrite-identical",
        action="store_true",
        help="仅在逐字段相同时按每条 query 一行重排已有文件",
    )
    args = parser.parse_args()
    if args.verify_existing and args.rewrite_identical:
        parser.error("--verify-existing 与 --rewrite-identical 不能同时使用")

    for dataset in args.datasets:
        specification = DATASETS[dataset]
        payload = build_dataset_workload(dataset, specification)
        output = specification["output"]
        if args.verify_existing:
            if _load_json(output) != payload:
                raise RuntimeError(f"{output} 无法由冻结构造器确定性重建")
            print(f"verified {output} sha256={_sha256_file(output)}")
        elif args.rewrite_identical:
            _write_payload(output, payload, overwrite_identical=True)
            print(f"rewrote {output} sha256={_sha256_file(output)}")
        else:
            _write_payload(output, payload)
            print(f"wrote {output} sha256={_sha256_file(output)}")


if __name__ == "__main__":
    main()
