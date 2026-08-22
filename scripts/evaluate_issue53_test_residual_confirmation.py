"""Evaluate the frozen Issue #53 test residual confirmation collection."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import analyze_issue53_test_ordered_heldout as workloads
from scripts import run_issue53_p6_dataset_smoke as base
from scripts import run_issue53_test_residual_confirmation as collection
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.queries import evaluate_table

EVALUATION_VERSION = "issue53-test-residual-confirmation-evaluation-v1"
EVALUATION_MODE = "evaluate_frozen_collection_after_query_identity_audit"
EVALUATION_REPORT = "evaluation_report.json"
ERROR_ARTIFACT = "query_seed_errors.csv"
GROUP_ORDER = workloads.GROUP_ORDER
PRIMARY_GROUPS = (
    "unmeasured_2way_all",
    "heldout_3way_512",
    "heldout_4way_512",
)
CANDIDATES = ("sqrt_relative", "relative")
EXPECTED_GROUP_COUNTS = workloads.EXPECTED_GROUP_COUNTS
EXPECTED_GROUP_IDENTITIES = {
    "measured_1way": (
        "b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c"
    ),
    "measured_2way": (
        "ea558bd958af3fa996925b159657973ff0d6a0dc873efbc0e0d41856f9e6887e"
    ),
    "measured_3way": (
        "cb2a96159985cf0a241e82ef6ea90475910e98bfee311c9778c33696bcd5aea2"
    ),
    "unmeasured_2way_all": (
        "7d88a2db88a4576bb54bed341a3a8ccfbfc11f368662ad3b513e8fa863b5647f"
    ),
    "heldout_3way_512": (
        "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a"
    ),
    "heldout_4way_512": (
        "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171"
    ),
}


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_or_reference_read",
        "collection_protocol_sha256": collection.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            collection.OUTPUT_DIR / collection.COLLECTION_REPORT
        ),
        "evaluation_report": str(collection.OUTPUT_DIR / EVALUATION_REPORT),
        "query_groups_in_report_order": list(GROUP_ORDER),
        "expected_group_counts": EXPECTED_GROUP_COUNTS,
        "expected_group_identity_sha256": EXPECTED_GROUP_IDENTITIES,
        "primary_groups": list(PRIMARY_GROUPS),
        "candidates_vs_absolute": list(CANDIDATES),
        "normal_completion_required": "15_of_15_A_or_B",
        "unseen_pareto_rule": {
            "all_primary_group_mean_delta_lte_zero": True,
            "at_least_one_primary_group_mean_delta_lt_zero": True,
            "same_improved_group_paired_seed_better_minimum": 4,
            "paired_seed_count": 5,
        },
        "measured_1way_safety_rule": "mean_delta_lte_zero",
        "cross_group_aggregate_allowed": False,
        "scientific_overrides_allowed": False,
        "raw_reference_access_only_after_query_identity_audit": True,
        "generation_started": False,
    }


def build_evaluation_preamble() -> dict[str, Any]:
    """Return the frozen plan fields with an evaluated-report mode."""

    return {
        **build_plan(),
        "mode": EVALUATION_MODE,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _audit_collection(
    root: Path,
    confirmed_collection_report_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    report_path = root / collection.OUTPUT_DIR / collection.COLLECTION_REPORT
    observed_sha = base._sha256_file(report_path)
    if observed_sha != confirmed_collection_report_sha256:
        raise ValueError(
            "collection report SHA 与显式确认值不一致："
            f"confirmed={confirmed_collection_report_sha256}, "
            f"observed={observed_sha}"
        )
    report = _load_json(report_path)
    if report.get("contract_version") != collection.PROTOCOL_VERSION:
        raise RuntimeError("collection contract version 漂移")
    if report.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("collection protocol SHA 漂移")
    if report.get("protocol") != collection.frozen_protocol_manifest():
        raise RuntimeError("collection protocol manifest 漂移")
    if report.get("case_count") != 15:
        raise RuntimeError("collection case count 不是 15")
    if report.get("raw_reference_data_accessed"):
        raise RuntimeError("collection 不得访问 raw reference")
    if report.get("privacy_budget_consumed"):
        raise RuntimeError("collection 不得消耗隐私预算")
    if report.get("parameter_retuning_performed"):
        raise RuntimeError("collection 不得结果后调参")
    current_commit = base._git_text(root, "rev-parse", "HEAD")
    execution_commit = report.get("execution_git_commit")
    if not isinstance(execution_commit, str):
        raise TypeError("collection execution commit 缺失")
    merge_base = base._git_text(
        root,
        "merge-base",
        execution_commit,
        current_commit,
    )
    if merge_base != execution_commit:
        raise RuntimeError("collection execution commit 不是评价提交的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 15:
        raise RuntimeError("collection raw results 不完整")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["arm"])
        if key in indexed:
            raise RuntimeError(f"collection case 重复：{key}")
        if key[0] not in collection.SEEDS or key[1] not in collection.ARMS:
            raise RuntimeError(f"collection case 不在冻结矩阵：{key}")
        if row["protocol_sha256"] != collection.FROZEN_PROTOCOL_SHA256:
            raise RuntimeError(f"collection case protocol 漂移：{key}")
        if row["git_commit"] != execution_commit:
            raise RuntimeError(f"collection case commit 漂移：{key}")
        if row["dataset"] != collection.DATASET:
            raise RuntimeError(f"collection dataset 漂移：{key}")
        if row["raw_reference_data_accessed"]:
            raise RuntimeError(f"collection case 读取 raw reference：{key}")
        table_path = (
            root
            / collection.OUTPUT_DIR
            / f"seed_{key[0]}"
            / collection.DATASET
            / key[1]
            / "terminal_current.csv"
        )
        if base._sha256_file(table_path) != row["terminal_table_sha256"]:
            raise RuntimeError(f"collection terminal table SHA 漂移：{key}")
        indexed[key] = row
    expected = {
        (seed, arm) for seed in collection.SEEDS for arm in collection.ARMS
    }
    if set(indexed) != expected:
        raise RuntimeError("collection 15-case 身份不完整")
    return report, indexed


def _audit_pre_reference_inputs(root: Path) -> dict[str, str]:
    observed = {}
    for name, path in workloads.INPUT_PATHS.items():
        if name == "reference":
            continue
        actual = base._sha256_file(root / path)
        expected = workloads.INPUT_SHA256[name]
        if actual != expected:
            raise RuntimeError(f"evaluation {name} SHA 漂移")
        observed[name] = actual
    protocol_actual = base._sha256_file(root / collection.PROTOCOL_DOC)
    if protocol_actual != collection.PROTOCOL_DOC_SHA256:
        raise RuntimeError("evaluation protocol doc SHA 漂移")
    observed["protocol_doc"] = protocol_actual
    return observed


def _freeze_and_audit_query_groups(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups, audit = workloads.freeze_query_groups(root)
    if audit["group_counts"] != EXPECTED_GROUP_COUNTS:
        raise RuntimeError("evaluation query group counts 漂移")
    if audit["group_query_identity_sha256"] != EXPECTED_GROUP_IDENTITIES:
        raise RuntimeError("evaluation query group identities 漂移")
    if not audit["identity_frozen_before_reference_load"]:
        raise RuntimeError("query identity 未在 reference load 前冻结")
    return groups, audit


def _audit_reference_after_identity_freeze(root: Path) -> str:
    actual = base._sha256_file(root / workloads.REFERENCE_PATH)
    expected = workloads.INPUT_SHA256["reference"]
    if actual != expected:
        raise RuntimeError("evaluation reference SHA 漂移")
    return actual


def _build_error_frame(
    root: Path,
    indexed_rows: dict[tuple[int, str], dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    expected_columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    catalog = workloads._flat_query_catalog(groups)
    queries = [row["query"] for row in catalog]
    targets = np.asarray([row["target_count"] for row in catalog], dtype=float)
    measured_mask = np.asarray(
        [row["query_group"].startswith("measured_") for row in catalog],
        dtype=bool,
    )
    if int(measured_mask.sum()) != 50:
        raise RuntimeError("evaluation measured query count 漂移")

    records = []
    table_audit = []
    for seed in collection.SEEDS:
        for arm in collection.ARMS:
            source_row = indexed_rows[(seed, arm)]
            table_path = (
                root
                / collection.OUTPUT_DIR
                / f"seed_{seed}"
                / collection.DATASET
                / arm
                / "terminal_current.csv"
            )
            table = pd.read_csv(table_path)
            if len(table) != workloads.N_RECORDS:
                raise RuntimeError(f"seed={seed}/{arm} terminal row count 漂移")
            if list(table.columns) != expected_columns:
                raise RuntimeError(f"seed={seed}/{arm} terminal columns 漂移")
            answers = np.asarray(evaluate_table(table, queries), dtype=float)
            measured_l1 = compute_normalized_l1(
                targets[measured_mask],
                answers[measured_mask],
                workloads.N_RECORDS,
            )
            source_l1 = float(source_row["terminal_current_normalized_l1"])
            if not np.isclose(measured_l1, source_l1, rtol=0.0, atol=1e-15):
                raise RuntimeError(f"seed={seed}/{arm} measured L1 复算漂移")
            table_audit.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "terminal_table_sha256": source_row["terminal_table_sha256"],
                    "source_measured_normalized_l1": source_l1,
                    "recomputed_measured_normalized_l1": float(measured_l1),
                }
            )
            if len(catalog) != len(answers):
                raise RuntimeError(f"seed={seed}/{arm} query answer 数量漂移")
            for metadata, answer in zip(catalog, answers):
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
    expected_rows = len(catalog) * len(collection.SEEDS) * len(collection.ARMS)
    if len(frame) != expected_rows:
        raise RuntimeError("evaluation query×seed×arm frame 不完整")
    return frame.sort_values(
        ["query_global_index", "seed", "arm"]
    ).reset_index(drop=True), table_audit


def _pairwise_vs_absolute(frame: pd.DataFrame) -> dict[str, Any]:
    pivot = frame.pivot(
        index=["query_group_index", "seed"],
        columns="arm",
        values="abs_error",
    )
    if set(pivot.columns) != set(collection.ARMS) or pivot.isna().any().any():
        raise RuntimeError("evaluation pairwise pivot 不完整")
    result = {}
    for candidate in CANDIDATES:
        delta = pivot[candidate] - pivot["absolute"]
        per_seed = delta.groupby(level="seed").mean()
        result[f"{candidate}_minus_absolute"] = {
            "mean_abs_error_delta_count": float(delta.mean()),
            "mean_abs_error_delta_normalized": float(
                delta.mean() / workloads.N_RECORDS
            ),
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
    expected_rows = query_count * seed_count * len(collection.ARMS)
    if query_count <= 0 or seed_count != 5 or len(frame) != expected_rows:
        raise RuntimeError("evaluation group frame 不完整")

    arms = {}
    for arm in collection.ARMS:
        rows = frame[frame["arm"] == arm]
        values = rows["abs_error"].to_numpy(dtype=float)
        per_seed = rows.groupby("seed", sort=True)["abs_error"].mean()
        arms[arm] = {
            "mean_abs_error_count": float(np.mean(values)),
            "mean_abs_error_normalized": float(
                np.mean(values) / workloads.N_RECORDS
            ),
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
        "pairwise_vs_absolute": _pairwise_vs_absolute(frame),
    }


def evaluate_frozen_gates(
    group_reports: dict[str, Any],
    *,
    normal_completion: bool,
) -> dict[str, Any]:
    candidates = {}
    for candidate in CANDIDATES:
        key = f"{candidate}_minus_absolute"
        primary = {
            group: group_reports[group]["pairwise_vs_absolute"][key]
            for group in PRIMARY_GROUPS
        }
        all_nonpositive = all(
            row["mean_abs_error_delta_count"] <= 0 for row in primary.values()
        )
        stable_improved_groups = [
            group
            for group, row in primary.items()
            if row["mean_abs_error_delta_count"] < 0
            and row["paired_seed_candidate_better_count"] >= 4
        ]
        unseen_pass = all_nonpositive and bool(stable_improved_groups)
        safety_row = group_reports["measured_1way"]["pairwise_vs_absolute"][key]
        safety_pass = safety_row["mean_abs_error_delta_count"] <= 0
        if not normal_completion:
            classification = "inconclusive_resource_cap"
        elif unseen_pass and safety_pass:
            classification = "supports_unified_test_candidate"
        elif unseen_pass:
            classification = "unseen_gain_with_measured_1way_tradeoff"
        else:
            classification = "mixed_no_unified_test_candidate"
        candidates[candidate] = {
            "classification": classification,
            "unseen_pareto_pass": unseen_pass,
            "all_primary_group_mean_delta_lte_zero": all_nonpositive,
            "stable_improved_primary_groups": stable_improved_groups,
            "measured_1way_safety_pass": safety_pass,
            "measured_1way_mean_delta_count": safety_row[
                "mean_abs_error_delta_count"
            ],
            "primary_groups": {
                group: {
                    "mean_abs_error_delta_count": row[
                        "mean_abs_error_delta_count"
                    ],
                    "paired_seed_candidate_better_count": row[
                        "paired_seed_candidate_better_count"
                    ],
                    "paired_seed_tie_count": row["paired_seed_tie_count"],
                    "paired_seed_candidate_worse_count": row[
                        "paired_seed_candidate_worse_count"
                    ],
                }
                for group, row in primary.items()
            },
        }
    supported = [
        candidate
        for candidate, row in candidates.items()
        if row["classification"] == "supports_unified_test_candidate"
    ]
    if not normal_completion:
        overall = "inconclusive_resource_cap"
    elif supported:
        overall = "supports_at_least_one_unified_test_candidate"
    else:
        overall = "no_unified_test_candidate_under_frozen_rule"
    return {
        "normal_completion_15_of_15": normal_completion,
        "candidates": candidates,
        "supported_unified_test_candidates": supported,
        "overall_classification": overall,
    }


def evaluate(confirmed_collection_report_sha256: str) -> Path:
    if len(confirmed_collection_report_sha256) != 64:
        raise ValueError("必须显式确认完整 collection report SHA-256")
    root = base._repo_root()
    if base._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式评价要求包含 untracked 在内的干净工作树")
    destination = root / collection.OUTPUT_DIR
    report_path = destination / EVALUATION_REPORT
    error_path = destination / ERROR_ARTIFACT
    if report_path.exists() or error_path.exists():
        raise FileExistsError("正式评价产物已存在，不覆盖")

    collection_report, indexed_rows = _audit_collection(
        root,
        confirmed_collection_report_sha256,
    )
    input_audit = _audit_pre_reference_inputs(root)
    frozen_groups, query_audit = _freeze_and_audit_query_groups(root)
    input_audit["reference"] = _audit_reference_after_identity_freeze(root)
    groups, answer_audit, expected_columns = workloads.attach_reference_answers(
        root,
        frozen_groups,
    )
    errors, table_audit = _build_error_frame(
        root,
        indexed_rows,
        groups,
        expected_columns,
    )
    group_reports = {
        group: summarize_group(errors[errors["query_group"] == group])
        for group in GROUP_ORDER
    }
    normal_completion = all(
        row["termination_reason"] in {"fit_target_reached", "early_stopped"}
        for row in indexed_rows.values()
    )
    gates = evaluate_frozen_gates(
        group_reports,
        normal_completion=normal_completion,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".query-seed-errors.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_error_path = Path(handle.name)
        errors.to_csv(handle, index=False)
    artifact_sha = base._sha256_file(temporary_error_path)

    report = {
        **build_evaluation_preamble(),
        "evaluation_git_commit": base._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_git_commit": collection_report[
            "execution_git_commit"
        ],
        "collection_summary": collection_report["summary"],
        "query_identity_frozen_before_reference_load": True,
        "input_sha256_audit": input_audit,
        "query_identity_audit": query_audit,
        "reference_answer_audit": answer_audit,
        "terminal_table_audit": table_audit,
        "groups": group_reports,
        "frozen_gate_evaluation": gates,
        "cross_group_aggregate_present": False,
        "new_generation_performed_by_evaluator": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "canonical_selection_performed": False,
        "claim_scope": "fresh_seed_test_confirmation_not_cross_dataset_canonical",
        "artifacts": {
            "query_seed_errors": {
                "path": ERROR_ARTIFACT,
                "row_count": len(errors),
                "sha256": artifact_sha,
            }
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".evaluation-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_report_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_error_path, error_path)
    os.replace(temporary_report_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-collection-report-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = evaluate(args.confirm_collection_report_sha)
    print(f"confirmation evaluation -> {path}")
    print(f"evaluation SHA-256 -> {base._sha256_file(path)}")


if __name__ == "__main__":
    main()
