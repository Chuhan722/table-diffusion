from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_issue53_test_query_workload_ab as evaluation

ROOT = Path(__file__).resolve().parents[1]


def _pair(delta: float, better: int = 4) -> dict:
    return {
        "mean_abs_error_delta_count": delta,
        "paired_seed_candidate_better_count": better,
        "paired_seed_tie_count": 5 - better,
        "paired_seed_candidate_worse_count": 0,
    }


def _group_reports() -> dict:
    reports = {}
    for group in evaluation.GROUP_ORDER:
        reports[group] = {
            "workload_b_minus_a_by_geometry": {
                geometry: _pair(0.0, 0)
                for geometry in evaluation.collection.GEOMETRIES
            },
            "workload_b_geometry_minus_absolute": {
                geometry: _pair(0.0, 0)
                for geometry in evaluation.CANDIDATE_GEOMETRIES
            },
        }
    return reports


def _all_normal() -> dict[tuple[str, str], bool]:
    return {
        (workload, geometry): True
        for workload in evaluation.collection.WORKLOADS
        for geometry in evaluation.collection.GEOMETRIES
    }


def test_plan_freezes_comparison_order_and_has_no_scientific_overrides(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must not read collection or reference")

    monkeypatch.setattr(evaluation, "_audit_collection", forbidden)
    monkeypatch.setattr(evaluation, "_attach_reference_answers", forbidden)

    plan = evaluation.build_plan()
    assert plan["comparison_order"] == [
        "within_each_geometry_workload_B_minus_A",
        "within_workload_B_candidate_geometry_minus_absolute",
    ]
    assert plan["primary_groups"] == list(evaluation.PRIMARY_GROUPS)
    assert plan["cross_group_aggregate_allowed"] is False
    assert plan["scientific_overrides_allowed"] is False
    assert plan["generation_started"] is False


def test_common_query_groups_freeze_before_any_csv_open(monkeypatch):
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".csv":
            raise AssertionError(f"query identity freeze opened CSV: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    groups, audit = evaluation._freeze_query_groups(ROOT)

    assert {name: len(queries) for name, queries in groups.items()} == (
        evaluation.EXPECTED_GROUP_COUNTS
    )
    assert audit["group_query_identity_sha256"] == (
        evaluation.EXPECTED_GROUP_IDENTITIES
    )
    assert audit["identity_frozen_before_reference_load"] is True


def test_reference_attachment_matches_existing_heldout_archive():
    groups, _ = evaluation._freeze_query_groups(ROOT)
    targets, audit = evaluation._attach_reference_answers(ROOT, groups)

    assert {name: len(values) for name, values in targets.items()} == (
        evaluation.EXPECTED_GROUP_COUNTS
    )
    assert audit["fixed_heldout_answers_match_existing_archive"] is True
    assert audit["raw_reference_data_accessed"] is True
    assert audit["privacy_budget_consumed"] is False


def _synthetic_group_rows() -> list[dict]:
    rows = []
    values = {
        ("A", "absolute"): 3,
        ("B", "absolute"): 2,
        ("A", "sqrt_relative"): 3,
        ("B", "sqrt_relative"): 1,
        ("A", "relative"): 3,
        ("B", "relative"): 4,
    }
    for query_index in range(2):
        for seed in evaluation.collection.SEEDS:
            for (workload, geometry), value in values.items():
                rows.append({
                    "query_group": "g",
                    "query_group_index": query_index,
                    "seed": seed,
                    "workload": workload,
                    "geometry": geometry,
                    "abs_error": value + query_index,
                })
    return rows


def test_group_summary_pairs_workloads_first_then_geometry_inside_b():
    summary = evaluation.summarize_group(_synthetic_group_rows())

    assert summary["query_count"] == 2
    assert summary["query_seed_count_per_cell"] == 10
    assert summary["workload_b_minus_a_by_geometry"]["absolute"][
        "mean_abs_error_delta_count"
    ] == -1.0
    assert summary["workload_b_minus_a_by_geometry"]["sqrt_relative"][
        "mean_abs_error_delta_count"
    ] == -2.0
    assert summary["workload_b_geometry_minus_absolute"]["sqrt_relative"][
        "mean_abs_error_delta_count"
    ] == -1.0
    assert "cross_group_total" not in summary


def test_frozen_gates_keep_workload_and_geometry_conclusions_separate():
    reports = _group_reports()
    for group in evaluation.PRIMARY_GROUPS:
        reports[group]["workload_b_minus_a_by_geometry"]["sqrt_relative"] = (
            _pair(-0.1, 4)
        )
        reports[group]["workload_b_geometry_minus_absolute"][
            "sqrt_relative"
        ] = _pair(-0.2, 5)
    reports["one_way_safety"]["workload_b_minus_a_by_geometry"][
        "sqrt_relative"
    ] = _pair(-0.01, 3)
    reports["one_way_safety"]["workload_b_geometry_minus_absolute"][
        "sqrt_relative"
    ] = _pair(-0.01, 3)
    reports["fixed_heldout_4way"]["workload_b_minus_a_by_geometry"][
        "relative"
    ] = _pair(0.1, 2)

    result = evaluation.evaluate_frozen_gates(reports, _all_normal())

    assert result["workload_b_vs_a_by_geometry"]["sqrt_relative"][
        "classification"
    ] == "supports_workload_B_under_geometry"
    assert result["workload_b_vs_a_by_geometry"]["relative"][
        "classification"
    ] == "mixed_no_workload_replacement"
    assert result["workload_effect_consistency"] == (
        "geometry_dependent_workload_effect"
    )
    assert result["workload_b_geometry_vs_absolute"]["sqrt_relative"][
        "classification"
    ] == "supports_geometry_under_workload_B"


def test_resource_cap_only_invalidates_corresponding_comparisons():
    normal = _all_normal()
    normal[("B", "relative")] = False

    result = evaluation.evaluate_frozen_gates(_group_reports(), normal)

    assert result["workload_b_vs_a_by_geometry"]["relative"][
        "classification"
    ] == "inconclusive_resource_cap"
    assert result["workload_b_geometry_vs_absolute"]["relative"][
        "classification"
    ] == "inconclusive_resource_cap"
    assert result["workload_b_vs_a_by_geometry"]["absolute"][
        "classification"
    ] != "inconclusive_resource_cap"
    assert result["workload_b_geometry_vs_absolute"]["sqrt_relative"][
        "classification"
    ] != "inconclusive_resource_cap"


def test_cli_has_no_threshold_or_scientific_overrides():
    parser = evaluation._build_parser()
    parsed = parser.parse_args([
        "evaluate",
        "--confirm-collection-report-sha",
        "a" * 64,
    ])
    assert vars(parsed) == {
        "command": "evaluate",
        "confirm_collection_report_sha": "a" * 64,
    }
    with pytest.raises(SystemExit):
        parser.parse_args([
            "evaluate",
            "--confirm-collection-report-sha",
            "a" * 64,
            "--minimum-seed-wins",
            "3",
        ])


def test_short_collection_sha_fails_before_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("short SHA must fail before environment access")

    monkeypatch.setattr(evaluation.collection, "_repo_root", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        evaluation.evaluate("abc")
