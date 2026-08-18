"""Read-only ordered held-out diagnostic for Issue #53 ``test_300x10``.

The query identities are frozen from public domains and existing workload
identities before the reference table is loaded.  The analysis then evaluates
only the nine already-materialized terminal tables from the P=6 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import build_issue53_heldout_workloads as heldout_builder
from scripts import compare_issue53_residual_geometry_earlystop as source
from scripts import run_issue53_p6_dataset_smoke as base
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.quality import query_fingerprint, validate_query_partition
from table_diffevo.queries import evaluate_table, load_data, load_queries
from table_diffevo.schema import load_schema

CONTRACT_VERSION = "issue53-test-ordered-heldout-diagnostic-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_test分阶heldout只读诊断协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "a78d37ecfaa29ebcad87a8bb29dcbbecd2663ebc03780e14db90f47abc228937"
)
PROTOCOL_DOC_COMMIT = "d427db68b927375a58e87ea8b172476e1ed5dcbd"

SOURCE_REPORT = source.OUTPUT_DIR / "report.json"
SOURCE_REPORT_SHA256 = (
    "241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143"
)
SOURCE_EXECUTION_COMMIT = "fe8fb797a718bf0e9a89668d46fbd5726c1c3082"
OUTPUT_DIR = Path("outputs/issue53_test_ordered_heldout_diagnostic_v1")

N_RECORDS = 300
SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
MEASURED_PATH = Path("configs/test_300x10/measured_50query.json")
HELDOUT_PATH = Path("configs/test_300x10/heldout_issue53_v1.json")
REFERENCE_PATH = Path("data/test_300x10/test_300x10.csv")
INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
    "measured_queries": (
        "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88"
    ),
    "heldout_queries": (
        "300bffea1f3d9105ad8f1840d50a900616115659065efec35b3c02f7a38cc1e0"
    ),
    "reference": (
        "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
    ),
}
INPUT_PATHS = {
    "schema": SCHEMA_PATH,
    "marginals": MARGINALS_PATH,
    "measured_queries": MEASURED_PATH,
    "heldout_queries": HELDOUT_PATH,
    "reference": REFERENCE_PATH,
}

GROUP_ORDER = (
    "measured_1way",
    "measured_2way",
    "measured_3way",
    "unmeasured_2way_all",
    "heldout_3way_512",
    "heldout_4way_512",
)
EXPECTED_GROUP_COUNTS = {
    "measured_1way": 25,
    "measured_2way": 20,
    "measured_3way": 5,
    "unmeasured_2way_all": 531,
    "heldout_3way_512": 512,
    "heldout_4way_512": 512,
}
UNMEASURED_GROUPS = (
    "unmeasured_2way_all",
    "heldout_3way_512",
    "heldout_4way_512",
)
PAIRWISE_COMPARISONS = (
    ("sqrt_relative", "absolute"),
    ("relative", "sqrt_relative"),
    ("relative", "absolute"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _query_order(query: dict[str, Any]) -> int:
    conditions = query.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("query.conditions 必须是非空列表")
    attributes = [condition.get("attribute") for condition in conditions]
    if any(not isinstance(attribute, str) or not attribute for attribute in attributes):
        raise ValueError("每个 query condition 都必须有非空 attribute")
    if len(set(attributes)) != len(attributes):
        raise ValueError("同一查询不得重复约束同一属性")
    return len(attributes)


def _query_set_identity(queries: Sequence[dict[str, Any]]) -> str:
    if not queries:
        raise ValueError("查询集合不能为空")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("查询集合包含重复语义查询")
    return hashlib.sha256("\n".join(fingerprints).encode("ascii")).hexdigest()


def freeze_unmeasured_2way(
    marginals: dict[str, Any],
    measured_queries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Select all canonical unmeasured 2-way cells without query answers."""

    all_queries = heldout_builder.enumerate_public_cell_queries(marginals, 2)
    all_fingerprints = [query_fingerprint(query) for query in all_queries]
    if len(set(all_fingerprints)) != len(all_fingerprints):
        raise RuntimeError("公开域枚举产生重复 2-way 查询")

    measured_fingerprints = [
        query_fingerprint(query) for query in measured_queries
    ]
    if len(set(measured_fingerprints)) != len(measured_fingerprints):
        raise ValueError("measured queries 包含重复语义查询")
    measured_set = set(measured_fingerprints)

    selected = []
    overlap_count = 0
    for query, fingerprint in zip(all_queries, all_fingerprints, strict=True):
        if fingerprint in measured_set:
            overlap_count += 1
            continue
        selected.append(
            {
                "id": f"U2_{len(selected) + 1:04d}",
                "type": "unmeasured_2way_cell",
                "order": 2,
                "fingerprint_sha256": fingerprint,
                "conditions": [dict(item) for item in query["conditions"]],
            }
        )

    partition = validate_query_partition(measured_queries, selected)
    return {
        "queries": selected,
        "all_public_2way_count": len(all_queries),
        "exact_measured_overlap_count": overlap_count,
        "selected_count": len(selected),
        "query_identity_sha256": _query_set_identity(selected),
        "partition": partition,
        "selection_used_reference_answers": False,
        "selection_used_terminal_errors": False,
    }


def _audit_pre_reference_input_files(root: Path) -> dict[str, str]:
    observed = {}
    for name, relative_path in INPUT_PATHS.items():
        if name == "reference":
            continue
        actual = base._sha256_file(root / relative_path)
        expected = INPUT_SHA256[name]
        if actual != expected:
            raise RuntimeError(
                f"{name} SHA 漂移：expected={expected}, observed={actual}"
            )
        observed[name] = actual
    protocol_actual = base._sha256_file(root / PROTOCOL_DOC)
    if protocol_actual != PROTOCOL_DOC_SHA256:
        raise RuntimeError("诊断协议文档 SHA 漂移")
    observed["protocol_doc"] = protocol_actual
    return observed


def _audit_reference_file_after_identity_freeze(root: Path) -> str:
    actual = base._sha256_file(root / REFERENCE_PATH)
    expected = INPUT_SHA256["reference"]
    if actual != expected:
        raise RuntimeError(
            f"reference SHA 漂移：expected={expected}, observed={actual}"
        )
    return actual


def _audit_source_report(root: Path) -> dict[str, Any]:
    report_path = root / SOURCE_REPORT
    observed_sha = base._sha256_file(report_path)
    if observed_sha != SOURCE_REPORT_SHA256:
        raise RuntimeError(
            "source report SHA 漂移："
            f"expected={SOURCE_REPORT_SHA256}, observed={observed_sha}"
        )
    report = _load_json(report_path)
    if report.get("contract_version") != source.PROTOCOL_VERSION:
        raise RuntimeError("source contract version 漂移")
    if report.get("protocol_sha256") != source.FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("source protocol SHA 漂移")
    if report.get("execution_git_commit") != SOURCE_EXECUTION_COMMIT:
        raise RuntimeError("source execution commit 漂移")
    if report.get("case_count") != 18 or len(report.get("raw_results", [])) != 18:
        raise RuntimeError("source 18-case 矩阵不完整")
    if report.get("raw_reference_data_accessed"):
        raise RuntimeError("source 主实验不应读取 raw reference")
    if report.get("privacy_budget_consumed"):
        raise RuntimeError("source 主实验不应额外消耗隐私预算")
    expected_cases = {
        (dataset, seed, arm)
        for dataset in source.DATASETS
        for seed in source.SEEDS
        for arm in source.ARMS
    }
    observed_cases = {
        (row["dataset"], int(row["seed"]), row["arm"])
        for row in report["raw_results"]
    }
    if observed_cases != expected_cases or len(observed_cases) != 18:
        raise RuntimeError("source case 身份不完整或重复")
    return report


def _validate_stored_heldout(
    payload: dict[str, Any],
    marginals: dict[str, Any],
    measured_queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if payload.get("dataset") != "test_300x10":
        raise RuntimeError("held-out dataset 身份漂移")
    if payload.get("record_count") != N_RECORDS or payload.get("query_count") != 1024:
        raise RuntimeError("held-out record/query count 漂移")
    construction = payload.get("construction")
    if not isinstance(construction, dict):
        raise TypeError("held-out construction 必须是对象")
    if construction.get("namespace") != heldout_builder.NAMESPACE:
        raise RuntimeError("held-out namespace 漂移")
    if construction.get("orders") != [3, 4]:
        raise RuntimeError("held-out orders 漂移")
    if construction.get("per_order_limit") != 512:
        raise RuntimeError("held-out per-order limit 漂移")
    if construction.get("candidate_counts") != {"3": 5051, "4": 30450}:
        raise RuntimeError("held-out candidate counts 漂移")
    if construction.get("selected_counts") != {"3": 512, "4": 512}:
        raise RuntimeError("held-out selected counts 漂移")
    if construction.get("input_sha256") != {
        "schema": INPUT_SHA256["schema"],
        "marginals": INPUT_SHA256["marginals"],
        "measured_queries": INPUT_SHA256["measured_queries"],
        "source": INPUT_SHA256["reference"],
    }:
        raise RuntimeError("held-out input SHA metadata 漂移")

    stored = payload.get("queries")
    if not isinstance(stored, list) or len(stored) != 1024:
        raise RuntimeError("held-out queries 不完整")
    rebuilt = heldout_builder.select_heldout_queries(
        "test_300x10",
        marginals,
        measured_queries,
    )
    if rebuilt["candidate_counts"] != construction["candidate_counts"]:
        raise RuntimeError("held-out result-blind candidate rebuild 漂移")
    if rebuilt["selected_counts"] != construction["selected_counts"]:
        raise RuntimeError("held-out result-blind selection rebuild 漂移")

    comparison_keys = (
        "order",
        "selection_rank",
        "selection_sha256",
        "fingerprint_sha256",
        "conditions",
    )
    for index, (observed, expected) in enumerate(
        zip(stored, rebuilt["queries"], strict=True)
    ):
        if {key: observed.get(key) for key in comparison_keys} != {
            key: expected.get(key) for key in comparison_keys
        }:
            raise RuntimeError(f"held-out query identity rebuild 漂移：index={index}")
        if observed.get("fingerprint_sha256") != query_fingerprint(observed):
            raise RuntimeError(f"held-out fingerprint 漂移：index={index}")

    identity = _query_set_identity(stored)
    if identity != construction.get("query_identity_sha256"):
        raise RuntimeError("held-out query identity SHA 漂移")
    if identity != construction.get("heldout_query_identity_sha256"):
        raise RuntimeError("held-out partition identity SHA 漂移")
    partition = validate_query_partition(measured_queries, stored)
    if partition["heldout_query_identity_sha256"] != identity:
        raise RuntimeError("held-out partition rebuild identity 漂移")
    return stored, {
        "query_identity_sha256": identity,
        "candidate_counts": rebuilt["candidate_counts"],
        "selected_counts": rebuilt["selected_counts"],
        "result_blind_identity_rebuild_equal": True,
        "measured_overlap_count": partition["overlap_count"],
    }


def freeze_query_groups(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Freeze every group identity without opening the reference CSV."""

    marginals = _load_json(root / MARGINALS_PATH)
    if marginals.get("n_records") != N_RECORDS:
        raise RuntimeError("marginals n_records 漂移")
    measured = load_queries(str(root / MEASURED_PATH))
    if len(measured) != 50:
        raise RuntimeError("measured query count 漂移")

    measured_by_order = {order: [] for order in (1, 2, 3)}
    for query in measured:
        order = _query_order(query)
        if order not in measured_by_order:
            raise RuntimeError(f"measured query 出现未冻结阶数：{order}")
        measured_by_order[order].append(query)

    unmeasured = freeze_unmeasured_2way(marginals, measured)
    if unmeasured["all_public_2way_count"] != 548:
        raise RuntimeError("公开 2-way cell 总数漂移")
    if unmeasured["exact_measured_overlap_count"] != 17:
        raise RuntimeError("公开 2-way 与 measured 精确重叠数漂移")
    if unmeasured["selected_count"] != 531:
        raise RuntimeError("未测量 2-way 数量漂移")

    heldout_payload = _load_json(root / HELDOUT_PATH)
    heldout, heldout_audit = _validate_stored_heldout(
        heldout_payload,
        marginals,
        measured,
    )
    heldout_by_order = {
        order: [query for query in heldout if _query_order(query) == order]
        for order in (3, 4)
    }
    groups = {
        "measured_1way": measured_by_order[1],
        "measured_2way": measured_by_order[2],
        "measured_3way": measured_by_order[3],
        "unmeasured_2way_all": unmeasured["queries"],
        "heldout_3way_512": heldout_by_order[3],
        "heldout_4way_512": heldout_by_order[4],
    }
    observed_counts = {name: len(queries) for name, queries in groups.items()}
    if observed_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(
            f"查询分组数量漂移：expected={EXPECTED_GROUP_COUNTS}, "
            f"observed={observed_counts}"
        )

    all_queries = [query for name in GROUP_ORDER for query in groups[name]]
    if len({query_fingerprint(query) for query in all_queries}) != len(all_queries):
        raise RuntimeError("六个查询组之间存在语义重复")
    return groups, {
        "identity_frozen_before_reference_load": True,
        "group_counts": observed_counts,
        "group_query_identity_sha256": {
            name: _query_set_identity(groups[name]) for name in GROUP_ORDER
        },
        "unmeasured_2way": {
            key: value for key, value in unmeasured.items() if key != "queries"
        },
        "heldout_3way_4way": heldout_audit,
    }


def _integer_results(queries: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    values = []
    for query in queries:
        value = query.get("result")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} query result 必须是整数计数")
        integer = int(value)
        if float(value) != integer or integer < 0:
            raise ValueError(f"{name} query result 必须是非负整数计数")
        values.append(integer)
    return np.asarray(values, dtype=int)


def attach_reference_answers(
    root: Path,
    frozen_groups: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    """Load raw reference only after identities exist, then attach/audit answers."""

    schema = load_schema(str(root / SCHEMA_PATH))
    reference = load_data(str(root / REFERENCE_PATH))
    expected_columns = schema.attribute_names()
    if list(reference.columns) != expected_columns:
        raise RuntimeError("reference 列与 schema 不一致")
    if len(reference) != N_RECORDS:
        raise RuntimeError("reference 行数漂移")

    result = {name: list(queries) for name, queries in frozen_groups.items()}
    answer_audit = {}
    for name in (
        "measured_1way",
        "measured_2way",
        "measured_3way",
        "heldout_3way_512",
        "heldout_4way_512",
    ):
        stored = _integer_results(result[name], name)
        recomputed = evaluate_table(reference, result[name])
        if not np.array_equal(stored, recomputed):
            raise RuntimeError(f"{name} stored answers 与 reference 复算不一致")
        answer_audit[name] = {
            "query_count": len(stored),
            "stored_answers_equal_reference": True,
        }

    frozen_unmeasured = result["unmeasured_2way_all"]
    answers = evaluate_table(reference, frozen_unmeasured)
    decorated = []
    for query, answer in zip(frozen_unmeasured, answers, strict=True):
        decorated.append({**query, "result": int(answer)})
    result["unmeasured_2way_all"] = decorated
    answer_audit["unmeasured_2way_all"] = {
        "query_count": len(decorated),
        "answers_attached_after_identity_freeze": True,
    }
    return result, answer_audit, expected_columns


def _flat_query_catalog(
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    catalog = []
    global_index = 0
    for group_name in GROUP_ORDER:
        for group_index, query in enumerate(groups[group_name]):
            catalog.append(
                {
                    "query_global_index": global_index,
                    "query_group": group_name,
                    "query_group_index": group_index,
                    "query_id": str(query["id"]),
                    "query_order": _query_order(query),
                    "query_fingerprint_sha256": query_fingerprint(query),
                    "target_count": int(query["result"]),
                    "query": query,
                }
            )
            global_index += 1
    return catalog


def _build_error_frame(
    root: Path,
    source_report: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    expected_columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    catalog = _flat_query_catalog(groups)
    queries = [row["query"] for row in catalog]
    targets = np.asarray([row["target_count"] for row in catalog], dtype=float)
    measured_mask = np.asarray(
        [row["query_group"].startswith("measured_") for row in catalog],
        dtype=bool,
    )
    if int(measured_mask.sum()) != 50:
        raise RuntimeError("measured catalog 数量漂移")

    raw_results = {
        (row["dataset"], int(row["seed"]), row["arm"]): row
        for row in source_report["raw_results"]
    }
    records = []
    table_audit = []
    for seed in source.SEEDS:
        for arm in source.ARMS:
            source_row = raw_results[("test_300x10", seed, arm)]
            table_path = (
                root
                / source.OUTPUT_DIR
                / f"seed_{seed}"
                / "test_300x10"
                / arm
                / "terminal_current.csv"
            )
            observed_sha = base._sha256_file(table_path)
            expected_sha = source_row["terminal_table_sha256"]
            if observed_sha != expected_sha:
                raise RuntimeError(f"seed={seed}/{arm} terminal table SHA 漂移")
            table = pd.read_csv(table_path)
            if len(table) != N_RECORDS or list(table.columns) != expected_columns:
                raise RuntimeError(f"seed={seed}/{arm} terminal table 形状漂移")
            answers = np.asarray(evaluate_table(table, queries), dtype=float)
            measured_l1 = compute_normalized_l1(
                targets[measured_mask],
                answers[measured_mask],
                N_RECORDS,
            )
            source_l1 = float(source_row["terminal_current_normalized_l1"])
            if not np.isclose(measured_l1, source_l1, rtol=0.0, atol=1e-15):
                raise RuntimeError(f"seed={seed}/{arm} measured L1 复算漂移")
            table_audit.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "terminal_table_sha256": observed_sha,
                    "source_measured_normalized_l1": source_l1,
                    "recomputed_measured_normalized_l1": float(measured_l1),
                }
            )
            for metadata, answer in zip(catalog, answers, strict=True):
                signed_error = int(answer) - metadata["target_count"]
                records.append(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key != "query"
                    }
                    | {
                        "seed": seed,
                        "arm": arm,
                        "terminal_answer": int(answer),
                        "signed_error": signed_error,
                        "abs_error": abs(signed_error),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    expected_rows = len(catalog) * len(source.SEEDS) * len(source.ARMS)
    if len(frame) != expected_rows:
        raise RuntimeError("query×seed×arm error frame 不完整")
    return frame.sort_values(
        ["query_global_index", "seed", "arm"]
    ).reset_index(drop=True), table_audit


def _pairwise_summary(frame: pd.DataFrame) -> dict[str, Any]:
    pivot = frame.pivot(
        index=["query_group_index", "seed"],
        columns="arm",
        values="abs_error",
    )
    if set(pivot.columns) != set(source.ARMS) or pivot.isna().any().any():
        raise RuntimeError("pairwise pivot 不完整")
    result = {}
    for candidate, baseline in PAIRWISE_COMPARISONS:
        delta = pivot[candidate] - pivot[baseline]
        per_seed = delta.groupby(level="seed").mean()
        key = f"{candidate}_minus_{baseline}"
        result[key] = {
            "mean_abs_error_delta_count": float(delta.mean()),
            "mean_abs_error_delta_normalized": float(delta.mean() / N_RECORDS),
            "query_seed_candidate_better_count": int((delta < 0).sum()),
            "query_seed_tie_count": int((delta == 0).sum()),
            "query_seed_candidate_worse_count": int((delta > 0).sum()),
            "paired_seed_mean_abs_error_delta_count": {
                str(int(seed)): float(value) for seed, value in per_seed.items()
            },
            "paired_seed_candidate_better_count": int((per_seed < 0).sum()),
            "paired_seed_tie_count": int((per_seed == 0).sum()),
            "paired_seed_candidate_worse_count": int((per_seed > 0).sum()),
        }
    return result


def summarize_group(frame: pd.DataFrame) -> dict[str, Any]:
    group_names = frame["query_group"].unique()
    if len(group_names) != 1:
        raise ValueError("summarize_group 只能接收一个查询组")
    query_count = int(frame["query_group_index"].nunique())
    seed_count = int(frame["seed"].nunique())
    expected_rows = query_count * seed_count * len(source.ARMS)
    if query_count <= 0 or seed_count != len(source.SEEDS) or len(frame) != expected_rows:
        raise RuntimeError("查询组 frame 不完整")

    arms = {}
    for arm in source.ARMS:
        arm_rows = frame[frame["arm"] == arm]
        per_seed = arm_rows.groupby("seed", sort=True)["abs_error"].mean()
        values = arm_rows["abs_error"].to_numpy(dtype=float)
        arms[arm] = {
            "mean_abs_error_count": float(np.mean(values)),
            "mean_abs_error_normalized": float(np.mean(values) / N_RECORDS),
            "median_abs_error_count": float(np.median(values)),
            "p90_abs_error_count": float(np.percentile(values, 90)),
            "max_abs_error_count": float(np.max(values)),
            "exact_match_rate": float(np.mean(values == 0)),
            "paired_seed_mean_abs_error_count": {
                str(int(seed)): float(value) for seed, value in per_seed.items()
            },
        }
    return {
        "query_count": query_count,
        "query_seed_count": query_count * seed_count,
        "arms": arms,
        "pairwise": _pairwise_summary(frame),
    }


def _directional_interpretation(group_reports: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for candidate, baseline in PAIRWISE_COMPARISONS:
        key = f"{candidate}_minus_{baseline}"
        measured_1way_delta = group_reports["measured_1way"]["pairwise"][key][
            "mean_abs_error_delta_count"
        ]
        unseen_deltas = {
            group: group_reports[group]["pairwise"][key][
                "mean_abs_error_delta_count"
            ]
            for group in UNMEASURED_GROUPS
        }
        values = list(unseen_deltas.values())
        if measured_1way_delta > 0 and all(value <= 0 for value in values):
            classification = "supports_measured_1way_dominance"
        elif all(value > 0 for value in values):
            classification = "supports_unmeasured_joint_query_weakness"
        else:
            classification = "mixed_no_universal_winner"
        result[key] = {
            "classification": classification,
            "measured_1way_mean_delta_count": measured_1way_delta,
            "unmeasured_group_mean_delta_count": unseen_deltas,
        }
    return result


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "mode": "post_result_read_only_ordered_heldout_diagnostic",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "source_report": str(SOURCE_REPORT),
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "output_dir": str(OUTPUT_DIR),
        "dataset": "test_300x10",
        "arms": list(source.ARMS),
        "seeds": list(source.SEEDS),
        "query_groups_in_report_order": list(GROUP_ORDER),
        "expected_group_counts": EXPECTED_GROUP_COUNTS,
        "unmeasured_2way_policy": "all_public_cells_excluding_exact_measured_overlap",
        "heldout_3way_4way_policy": "existing_result_blind_frozen_512_each",
        "pairwise_comparisons": [
            f"{candidate}_minus_{baseline}"
            for candidate, baseline in PAIRWISE_COMPARISONS
        ],
        "cross_group_aggregate_allowed": False,
        "new_generation_performed": False,
        "raw_reference_data_access_planned_after_identity_freeze": True,
        "canonical_selection_allowed": False,
        "scientific_overrides_allowed": False,
    }


def _unmeasured_payload(
    queries: list[dict[str, Any]],
    query_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": "test_300x10",
        "record_count": N_RECORDS,
        "query_count": len(queries),
        "description": (
            "All canonical public-domain 2-way cells not semantically identical "
            "to a measured query; answers attached only after identity freeze"
        ),
        "construction": {
            **query_audit["unmeasured_2way"],
            "input_sha256": {
                "marginals": INPUT_SHA256["marginals"],
                "measured_queries": INPUT_SHA256["measured_queries"],
                "reference": INPUT_SHA256["reference"],
            },
            "raw_reference_data_accessed_after_identity_freeze": True,
        },
        "queries": queries,
    }


def run_analysis(confirmed_source_report_sha256: str) -> Path:
    if confirmed_source_report_sha256 != SOURCE_REPORT_SHA256:
        raise ValueError("必须显式确认完整 source report SHA-256")
    root = base._repo_root()
    if base._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("诊断要求包含 untracked 在内的干净工作树")
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"诊断输出已存在，不覆盖：{destination}")

    input_audit = _audit_pre_reference_input_files(root)
    source_report = _audit_source_report(root)
    frozen_groups, query_audit = freeze_query_groups(root)
    input_audit["reference"] = _audit_reference_file_after_identity_freeze(root)
    groups, answer_audit, expected_columns = attach_reference_answers(
        root,
        frozen_groups,
    )
    errors, terminal_table_audit = _build_error_frame(
        root,
        source_report,
        groups,
        expected_columns,
    )
    group_reports = {
        name: summarize_group(errors[errors["query_group"] == name])
        for name in GROUP_ORDER
    }
    if set(group_reports) != set(GROUP_ORDER):
        raise RuntimeError("分组报告不完整")

    report = {
        **build_plan(),
        "analysis_git_commit": base._git_text(root, "rev-parse", "HEAD"),
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "source_protocol_sha256": source.FROZEN_PROTOCOL_SHA256,
        "post_result_diagnostic": True,
        "new_generation_performed": False,
        "query_identity_frozen_before_reference_load": True,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "input_sha256_audit": input_audit,
        "query_identity_audit": query_audit,
        "reference_answer_audit": answer_audit,
        "terminal_table_audit": terminal_table_audit,
        "groups": group_reports,
        "directional_interpretation": _directional_interpretation(group_reports),
        "cross_group_aggregate_present": False,
        "claim_scope": (
            "result_aware_development_mechanism_diagnostic_not_selection_evidence"
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".issue53-test-ordered-heldout.tmp-",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        error_path = temporary / "query_seed_errors.csv"
        errors.to_csv(error_path, index=False)
        unseen_path = temporary / "unmeasured_2way_queries.json"
        unseen_path.write_text(
            json.dumps(
                _unmeasured_payload(
                    groups["unmeasured_2way_all"],
                    query_audit,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report["artifacts"] = {
            "query_seed_errors": {
                "path": error_path.name,
                "row_count": len(errors),
                "sha256": base._sha256_file(error_path),
            },
            "unmeasured_2way_queries": {
                "path": unseen_path.name,
                "query_count": len(groups["unmeasured_2way_all"]),
                "sha256": base._sha256_file(unseen_path),
            },
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-source-report-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = run_analysis(args.confirm_source_report_sha)
    print(f"ordered held-out diagnostic -> {path}")
    print(f"report SHA-256 -> {base._sha256_file(path)}")


if __name__ == "__main__":
    main()
