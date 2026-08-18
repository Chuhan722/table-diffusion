from __future__ import annotations

import pandas as pd
import pytest

from scripts import analyze_issue53_residual_geometry_queries as diagnostic


@pytest.mark.parametrize(
    ("target", "n_records", "expected"),
    [
        (0, 1000, "zero"),
        (7, 1000, "below_floor"),
        (8, 1000, "rare"),
        (49, 1000, "rare"),
        (50, 1000, "medium"),
        (199, 1000, "medium"),
        (200, 1000, "common"),
    ],
)
def test_frequency_band_boundaries(target, n_records, expected):
    assert diagnostic.frequency_band(target, n_records) == expected


def test_frequency_band_rejects_invalid_counts():
    with pytest.raises(ValueError, match="n_records"):
        diagnostic.frequency_band(1, 0)
    with pytest.raises(ValueError, match="不能为负"):
        diagnostic.frequency_band(-1, 100)


def _query(query_id: str, target: int, *attributes: str) -> dict:
    return {
        "id": query_id,
        "type": f"order_{len(attributes)}",
        "result": target,
        "conditions": [
            {"attribute": attribute, "operator": "==", "value": 1}
            for attribute in attributes
        ],
    }


def test_query_features_use_only_target_order_and_attribute_overlap():
    queries = [
        _query("q0", 10, "a"),
        _query("q1", 20, "a", "b"),
        _query("q2", 30, "b", "c"),
        _query("q3", 40, "d"),
    ]

    features, thresholds = diagnostic._query_features(queries, 100)

    assert [row["query_order"] for row in features] == [1, 2, 2, 1]
    assert [row["frequency_band"] for row in features] == [
        "medium",
        "common",
        "common",
        "common",
    ]
    assert features[0]["structural_overlap_mean_jaccard"] == pytest.approx(1 / 6)
    assert features[1]["structural_overlap_mean_jaccard"] == pytest.approx(5 / 18)
    assert features[2]["structural_overlap_mean_jaccard"] == pytest.approx(1 / 9)
    assert features[3]["structural_overlap_mean_jaccard"] == 0.0
    assert features[3]["structural_overlap_band"] == "low"
    assert features[1]["structural_overlap_band"] == "high"
    assert thresholds["q25"] < thresholds["q75"]


def test_fractional_win_credit_splits_ties():
    frame = pd.DataFrame(
        [
            {
                "dataset": "d",
                "query_index": 0,
                "seed": 1,
                "arm": "absolute",
                "abs_error": 1,
            },
            {
                "dataset": "d",
                "query_index": 0,
                "seed": 1,
                "arm": "sqrt_relative",
                "abs_error": 1,
            },
            {
                "dataset": "d",
                "query_index": 0,
                "seed": 1,
                "arm": "relative",
                "abs_error": 2,
            },
        ]
    )

    result = diagnostic._attach_fractional_win_credit(frame)

    credits = dict(zip(result["arm"], result["fractional_win_credit"], strict=True))
    assert credits == {"absolute": 0.5, "sqrt_relative": 0.5, "relative": 0.0}


def test_subset_summary_reports_contribution_wins_and_pairwise_counts():
    rows = []
    errors = {
        0: {
            1: {"absolute": 1, "sqrt_relative": 1, "relative": 2},
            2: {"absolute": 2, "sqrt_relative": 1, "relative": 0},
        },
        1: {
            1: {"absolute": 4, "sqrt_relative": 3, "relative": 2},
            2: {"absolute": 3, "sqrt_relative": 3, "relative": 4},
        },
    }
    for query_index, seeds in errors.items():
        for seed, arms in seeds.items():
            for arm, error in arms.items():
                rows.append(
                    {
                        "dataset": "d",
                        "query_index": query_index,
                        "seed": seed,
                        "arm": arm,
                        "abs_error": error,
                        "target_count": 10 + query_index,
                        "target_frequency": (10 + query_index) / 100,
                    }
                )
    frame = diagnostic._attach_fractional_win_credit(pd.DataFrame(rows))
    arm_sums = {
        arm: float(frame[frame["arm"] == arm]["abs_error"].sum())
        for arm in diagnostic.source.ARMS
    }

    summary = diagnostic._summarize_subset(
        frame,
        n_records=100,
        total_query_count=2,
        dataset_arm_abs_error_sums=arm_sums,
    )

    assert summary["query_count"] == 2
    assert summary["query_seed_count"] == 4
    assert summary["arms"]["absolute"]["mean_dataset_l1_contribution"] == 0.025
    assert summary["arms"]["absolute"]["share_of_arm_total_abs_error"] == 1.0
    pair = summary["pairwise"]["relative_minus_absolute"]
    assert pair["candidate_better_count"] == 2
    assert pair["tie_count"] == 0
    assert pair["candidate_worse_count"] == 2


def test_plan_and_cli_have_no_scientific_overrides():
    plan = diagnostic.build_plan()
    assert plan["mode"] == "post_result_read_only_diagnostic"
    assert plan["generation_started"] is False
    assert plan["raw_reference_data_accessed"] is False
    assert plan["canonical_selection_allowed"] is False
    assert plan["scientific_overrides_allowed"] is False

    parser = diagnostic._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--confirm-source-report-sha",
            diagnostic.SOURCE_REPORT_SHA256,
        ]
    )
    assert vars(args) == {
        "command": "run",
        "confirm_source_report_sha": diagnostic.SOURCE_REPORT_SHA256,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--confirm-source-report-sha",
                diagnostic.SOURCE_REPORT_SHA256,
                "--rare-frequency",
                "0.1",
            ]
        )


def test_wrong_source_sha_fails_before_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail before reading environment")

    monkeypatch.setattr(diagnostic.base, "_repo_root", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        diagnostic.run_analysis("0" * 64)
