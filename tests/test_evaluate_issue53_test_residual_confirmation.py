from __future__ import annotations

import pandas as pd
import pytest

from scripts import evaluate_issue53_test_residual_confirmation as evaluation


def _pair(delta: float, better: int) -> dict:
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
            "pairwise_vs_absolute": {
                "sqrt_relative_minus_absolute": _pair(0.0, 0),
                "relative_minus_absolute": _pair(0.0, 0),
            }
        }
    return reports


def test_frozen_gate_supports_only_candidate_passing_unseen_and_safety():
    reports = _group_reports()
    for group in evaluation.PRIMARY_GROUPS:
        reports[group]["pairwise_vs_absolute"][
            "sqrt_relative_minus_absolute"
        ] = _pair(-0.1, 4)
    reports["measured_1way"]["pairwise_vs_absolute"][
        "sqrt_relative_minus_absolute"
    ] = _pair(-0.01, 3)
    reports["heldout_3way_512"]["pairwise_vs_absolute"][
        "relative_minus_absolute"
    ] = _pair(0.01, 2)

    result = evaluation.evaluate_frozen_gates(
        reports,
        normal_completion=True,
    )

    sqrt = result["candidates"]["sqrt_relative"]
    assert sqrt["unseen_pareto_pass"] is True
    assert sqrt["measured_1way_safety_pass"] is True
    assert sqrt["classification"] == "supports_unified_test_candidate"
    assert result["candidates"]["relative"]["classification"] == (
        "mixed_no_unified_test_candidate"
    )
    assert result["supported_unified_test_candidates"] == ["sqrt_relative"]


def test_frozen_gate_reports_measured_1way_tradeoff_and_resource_cap():
    reports = _group_reports()
    for group in evaluation.PRIMARY_GROUPS:
        reports[group]["pairwise_vs_absolute"][
            "sqrt_relative_minus_absolute"
        ] = _pair(-0.1, 4)
    reports["measured_1way"]["pairwise_vs_absolute"][
        "sqrt_relative_minus_absolute"
    ] = _pair(0.01, 2)

    result = evaluation.evaluate_frozen_gates(
        reports,
        normal_completion=True,
    )
    assert result["candidates"]["sqrt_relative"]["classification"] == (
        "unseen_gain_with_measured_1way_tradeoff"
    )
    assert result["overall_classification"] == (
        "no_unified_test_candidate_under_frozen_rule"
    )

    capped = evaluation.evaluate_frozen_gates(
        reports,
        normal_completion=False,
    )
    assert capped["overall_classification"] == "inconclusive_resource_cap"
    assert all(
        row["classification"] == "inconclusive_resource_cap"
        for row in capped["candidates"].values()
    )


def _summary_frame() -> pd.DataFrame:
    rows = []
    for query_index in range(2):
        for seed in (313, 314, 315, 316, 317):
            values = {
                "absolute": query_index + 2,
                "sqrt_relative": query_index + 1,
                "relative": query_index + 3,
            }
            for arm, error in values.items():
                rows.append(
                    {
                        "query_group": "g",
                        "query_group_index": query_index,
                        "seed": seed,
                        "arm": arm,
                        "abs_error": error,
                    }
                )
    return pd.DataFrame(rows)


def test_group_summary_uses_five_paired_seeds_without_total_score():
    summary = evaluation.summarize_group(_summary_frame())

    assert summary["query_count"] == 2
    assert summary["query_seed_count"] == 10
    assert set(summary) == {
        "query_count",
        "query_seed_count",
        "arms",
        "pairwise_vs_absolute",
    }
    sqrt = summary["pairwise_vs_absolute"]["sqrt_relative_minus_absolute"]
    assert sqrt["mean_abs_error_delta_count"] == -1.0
    assert sqrt["paired_seed_candidate_better_count"] == 5


def test_plan_and_cli_have_no_scientific_overrides():
    plan = evaluation.build_plan()
    assert plan["primary_groups"] == list(evaluation.PRIMARY_GROUPS)
    assert plan["cross_group_aggregate_allowed"] is False
    assert plan["scientific_overrides_allowed"] is False
    assert plan["raw_reference_access_only_after_query_identity_audit"] is True

    parser = evaluation._build_parser()
    parsed = parser.parse_args(
        ["evaluate", "--confirm-collection-report-sha", "a" * 64]
    )
    assert vars(parsed) == {
        "command": "evaluate",
        "confirm_collection_report_sha": "a" * 64,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "--confirm-collection-report-sha",
                "a" * 64,
                "--minimum-seed-wins",
                "3",
            ]
        )


def test_short_collection_sha_fails_before_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("short SHA must fail before reading environment")

    monkeypatch.setattr(evaluation.base, "_repo_root", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        evaluation.evaluate("abc")
