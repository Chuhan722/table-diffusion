#!/usr/bin/env python3
"""Attach reference answers to the already frozen Issue #53 workload B.

The query identities are rebuilt and audited before this module opens the raw
reference CSV.  Reference values may attach answers, but they cannot select,
remove, reorder, or otherwise change a query identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import freeze_issue53_test_query_workload_ab as freeze
else:
    import freeze_issue53_test_query_workload_ab as freeze


CONTRACT_VERSION = "issue53-test-query-workload-b-answers-v1"
IDENTITY_ARTIFACT_SHA256 = (
    "a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36"
)
PROTOCOL_DOC_SHA256 = (
    "291c591ba5408e046005b24122bfe602bf8a97f7c175ee45e59f81daf96b44b6"
)
WORKLOAD_B_IDENTITY_SHA256 = (
    "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
)
REFERENCE_PATH = Path("data/test_300x10/test_300x10.csv")
REFERENCE_SHA256 = (
    "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
)
OUTPUT_PATH = Path("configs/test_300x10/measured_50query_30_15_5.json")
REFERENCE_COLUMNS = (
    "age",
    "education",
    "employment",
    "income",
    "marital",
    "children",
    "housing",
    "vehicle",
    "health",
    "region",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _load_reference(path: Path) -> list[dict[str, str]]:
    """Load the one authorized raw reference after identity audit."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REFERENCE_COLUMNS:
            raise RuntimeError(
                "reference 列身份漂移："
                f"expected={REFERENCE_COLUMNS}, observed={reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    if len(rows) != freeze.N_RECORDS:
        raise RuntimeError(
            f"reference 行数漂移：expected={freeze.N_RECORDS}, observed={len(rows)}"
        )
    if any(set(row) != set(REFERENCE_COLUMNS) for row in rows):
        raise RuntimeError("reference row 列不完整")
    return rows


def _numeric(value: str, attribute: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"数值条件读取到非数值 reference：{attribute}={value!r}"
        ) from exc


def _condition_matches(
    row: dict[str, str],
    condition: dict[str, Any],
) -> bool:
    attribute = condition.get("attribute")
    if attribute not in row:
        raise ValueError(f"查询引用未知属性：{attribute!r}")
    operator = condition.get("operator")
    observed = row[attribute]
    if operator == "==":
        return observed == str(condition["value"])
    if operator == "between":
        numeric = _numeric(observed, attribute)
        return float(condition["lower"]) <= numeric <= float(condition["upper"])
    if operator == ">=":
        return _numeric(observed, attribute) >= float(condition["value"])
    raise ValueError(f"不支持的查询算子：{operator!r}")


def evaluate_query(
    rows: Sequence[dict[str, str]],
    query: dict[str, Any],
) -> int:
    freeze.query_order(query)
    return sum(
        all(_condition_matches(row, condition) for condition in query["conditions"])
        for row in rows
    )


def evaluate_queries(
    rows: Sequence[dict[str, str]],
    queries: Sequence[dict[str, Any]],
) -> list[int]:
    return [evaluate_query(rows, query) for query in queries]


def _target_vector_sha256(values: Sequence[int]) -> str:
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("target vector 必须是非空整数列表")
    return hashlib.sha256(freeze._strict_json_bytes(list(values))).hexdigest()


def _audit_identities_before_reference(root: Path) -> dict[str, Any]:
    identity_path = root / freeze.OUTPUT_PATH
    observed_identity_sha = freeze._sha256_file(identity_path)
    if observed_identity_sha != IDENTITY_ARTIFACT_SHA256:
        raise RuntimeError(
            "identity artifact SHA 漂移："
            f"expected={IDENTITY_ARTIFACT_SHA256}, observed={observed_identity_sha}"
        )
    if freeze._sha256_file(root / freeze.PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("结果前协议文档 SHA 漂移")
    stored_identity = _load_json(identity_path)
    rebuilt_identity = freeze.build_payload(root)
    if stored_identity != rebuilt_identity:
        raise RuntimeError("identity artifact 无法在 reference 访问前确定性重建")
    if stored_identity.get("privacy_boundary", {}).get("query_answers_present"):
        raise RuntimeError("identity artifact 不应包含 query answers")
    workload_b = stored_identity.get("workload_b", {}).get("queries")
    if not isinstance(workload_b, list) or len(workload_b) != 50:
        raise RuntimeError("identity artifact 的 workload B 不完整")
    freeze._assert_result_free(workload_b)
    observed_workload_identity = freeze.query_set_identity(workload_b)
    if observed_workload_identity != WORKLOAD_B_IDENTITY_SHA256:
        raise RuntimeError("workload B 结果前身份漂移")
    if freeze._order_counts(workload_b) != {2: 30, 3: 15, 4: 5}:
        raise RuntimeError("workload B 阶数构成漂移")
    return {
        "identity_artifact": stored_identity,
        "workload_b": workload_b,
        "identity_artifact_sha256": observed_identity_sha,
        "workload_b_identity_sha256": observed_workload_identity,
        "identity_frozen_before_reference_load": True,
    }


def _audit_old_answers(
    root: Path,
    rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    old_payload = _load_json(root / freeze.WORKLOAD_A_PATH)
    old_queries = old_payload.get("queries")
    if not isinstance(old_queries, list) or len(old_queries) != 50:
        raise RuntimeError("workload A queries 不完整")
    expected = []
    for index, query in enumerate(old_queries):
        value = query.get("result")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"workload A result 不是整数：index={index}")
        expected.append(value)
    observed = evaluate_queries(rows, old_queries)
    if observed != expected:
        mismatches = [
            {
                "index": index,
                "id": old_queries[index].get("id"),
                "expected": expected[index],
                "observed": observed[index],
            }
            for index in range(len(expected))
            if expected[index] != observed[index]
        ]
        raise RuntimeError(
            "reference evaluator 无法精确复现旧 workload A 答案："
            f"{mismatches[:5]}"
        )
    return {
        "query_count": len(old_queries),
        "all_answers_exact": True,
        "target_vector_sha256": _target_vector_sha256(observed),
    }


def _attach_answers(
    workload_b: Sequence[dict[str, Any]],
    answers: Sequence[int],
) -> list[dict[str, Any]]:
    if len(workload_b) != len(answers):
        raise ValueError("workload B 与 answer vector 长度不一致")
    attached = []
    for query, answer in zip(workload_b, answers, strict=True):
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise TypeError("query answer 必须是整数")
        decorated = {key: value for key, value in query.items()}
        decorated["conditions"] = [dict(item) for item in query["conditions"]]
        decorated["result"] = answer
        attached.append(decorated)
    if freeze.query_set_identity(attached) != freeze.query_set_identity(workload_b):
        raise RuntimeError("附加答案改变了 workload B 查询身份")
    return attached


def build_payload(root: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if root is None else root

    # Privacy boundary: every query identity is rebuilt and frozen first.
    pre_reference = _audit_identities_before_reference(root)

    reference_path = root / REFERENCE_PATH
    observed_reference_sha = freeze._sha256_file(reference_path)
    if observed_reference_sha != REFERENCE_SHA256:
        raise RuntimeError(
            "raw reference SHA 漂移："
            f"expected={REFERENCE_SHA256}, observed={observed_reference_sha}"
        )
    rows = _load_reference(reference_path)
    old_answer_audit = _audit_old_answers(root, rows)

    workload_b = pre_reference["workload_b"]
    answers = evaluate_queries(rows, workload_b)
    attached = _attach_answers(workload_b, answers)
    identity_after = freeze.query_set_identity(attached)
    if identity_after != WORKLOAD_B_IDENTITY_SHA256:
        raise RuntimeError("workload B 附加答案后身份漂移")

    payload = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "test_300x10.csv",
        "record_count": freeze.N_RECORDS,
        "query_count": len(attached),
        "result_unit": "records",
        "description": (
            "Issue #53 frozen 30x2-way + 15x3-way + 5x4-way measured "
            "generation workload with reference answers"
        ),
        "construction": {
            "protocol_doc": str(freeze.PROTOCOL_DOC),
            "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
            "identity_artifact": str(freeze.OUTPUT_PATH),
            "identity_artifact_sha256": pre_reference[
                "identity_artifact_sha256"
            ],
            "reference": str(REFERENCE_PATH),
            "reference_sha256": observed_reference_sha,
            "identity_frozen_before_reference_load": pre_reference[
                "identity_frozen_before_reference_load"
            ],
            "query_identity_before_answers_sha256": pre_reference[
                "workload_b_identity_sha256"
            ],
            "query_identity_after_answers_sha256": identity_after,
            "identity_preserved_after_answer_attachment": True,
            "old_workload_a_answer_reproduction": old_answer_audit,
            "target_vector_sha256": _target_vector_sha256(answers),
            "selection_used_reference_answers": False,
            "selection_used_marginal_counts": False,
            "selection_used_terminal_errors": False,
            "raw_reference_data_accessed": True,
            "privacy_budget_consumed": False,
        },
        "queries": attached,
    }
    if freeze.query_set_identity(payload["queries"]) != WORKLOAD_B_IDENTITY_SHA256:
        raise RuntimeError("正式 workload B 输出身份漂移")
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
        help="重放固定 reference 答案并与正式 workload B 文件逐字段比较",
    )
    args = parser.parse_args()
    root = _repo_root()
    payload = build_payload(root)
    output = root / OUTPUT_PATH
    if args.verify_existing:
        if _load_json(output) != payload:
            raise RuntimeError(f"正式 workload B 无法确定性重建：{output}")
        action = "verified"
    else:
        _write_new(output, payload)
        action = "wrote"
    print(
        f"{action} {OUTPUT_PATH} sha256={freeze._sha256_file(output)} "
        f"identity={freeze.query_set_identity(payload['queries'])} "
        f"target={payload['construction']['target_vector_sha256']}"
    )


if __name__ == "__main__":
    main()
